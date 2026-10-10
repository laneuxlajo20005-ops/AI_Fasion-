"""Reproducible final comparison of the original and upgraded fashion systems.

Ranking metrics use held-out catalog rows as clearly labeled pseudo-profiles.
They are not user-interaction or satisfaction measurements.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from evaluation import (
    DEFAULT_INPUT,
    RELEVANCE_THRESHOLD,
    _evaluate_rankings,
    proxy_relevance_grades,
    random_category_baseline,
)
from fashion_preprocessing import validate_and_normalize
from recommendation_engine import FashionRecommendationEngine


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "final_evaluation"
CLASSIFICATION_FEATURES = ("gender", "location", "season")
K_VALUES = (5, 10)


def _make_classifier(random_state: int) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CLASSIFICATION_FEATURES),
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _preferences_from_profile(profile: pd.Series) -> dict[str, str]:
    return {
        "gender": str(profile["gender"]),
        "location": str(profile["location"]),
        "season": str(profile["season"]),
        "preferred_category": str(profile["category"]),
        "preferred_color": str(profile["color"]),
        "preferred_brand": str(profile["brand"]),
        "preferred_size": (
            "Unknown" if pd.isna(profile["size"]) else str(profile["size"])
        ),
    }


def _baseline_recommend(
    profile: pd.Series,
    predicted_category: str,
    candidates: pd.DataFrame,
    top_n: int,
    random_state: int,
) -> list[str]:
    predicted_profile = profile.copy()
    predicted_profile["category"] = predicted_category
    return random_category_baseline(
        predicted_profile,
        candidates,
        top_n,
        random_state,
    )


def _latency_summary(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=float)
    if not len(values):
        raise ValueError("Latency measurement requires at least one sample")
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def _plot_ranking_metrics(comparison: pd.DataFrame, path: Path) -> None:
    columns = (
        ("precision_at_k", "Precision@K"),
        ("recall_at_k", "Recall@K"),
        ("f1_at_k", "F1@K"),
        ("ndcg_at_k", "NDCG@K"),
        ("hit_rate_at_k", "Hit Rate@K"),
    )
    methods = list(comparison["method"].drop_duplicates())
    colors = ["#8993a4", "#e94560"]
    figure, axes = plt.subplots(1, len(columns), figsize=(17, 4.4), sharey=True)
    for axis, (column, title) in zip(axes, columns):
        width = 0.34
        ks = sorted(comparison["k"].unique())
        x = np.arange(len(ks))
        for offset, (method, color) in enumerate(zip(methods, colors)):
            values = (
                comparison.loc[comparison["method"] == method]
                .set_index("k")
                .loc[ks, column]
            )
            axis.bar(x + (offset - 0.5) * width, values, width, label=method, color=color)
        axis.set_title(title)
        axis.set_xticks(x, [str(k) for k in ks])
        axis.set_xlabel("K")
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Macro-average proxy metric")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle(
        "Held-out catalog-proxy ranking evaluation (not user-interaction data)",
        y=1.03,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 0.97))
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_confusion_matrix(
    actual: pd.Series,
    predicted: np.ndarray,
    labels: list[str],
    path: Path,
) -> None:
    matrix = confusion_matrix(actual, predicted, labels=labels)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Held-out products")
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted category",
        ylabel="Actual category",
        title="Original Random Forest category classification",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_latency(latency: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(latency))
    width = 0.34
    axis.bar(x - width / 2, latency["median_ms"], width, label="Median", color="#4f61f0")
    axis.bar(x + width / 2, latency["p95_ms"], width, label="95th percentile", color="#e94560")
    axis.set_xticks(x, latency["method"], rotation=10, ha="right")
    axis.set_ylabel("Milliseconds per recommendation request")
    axis.set_title("Local in-process inference and ranking latency")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    headers = list(frame.columns)
    rows: list[list[str]] = []
    for values in frame.itertuples(index=False, name=None):
        row: list[str] = []
        for column, value in zip(headers, values):
            if pd.isna(value):
                row.append("N/A")
            elif isinstance(value, (float, np.floating)):
                row.append(f"{value:.{float_digits}f}")
            else:
                row.append(str(value))
        rows.append(row)
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def _write_report(
    metadata: dict[str, Any],
    classification: pd.DataFrame,
    ranking: pd.DataFrame,
    latency: pd.DataFrame,
) -> str:
    return f"""# Final experimental evaluation: fashion recommendation system

## Executive summary

The supplied dataset contains **{metadata["source_rows"]} product records** and no
customer IDs, recommendation impressions, clicks, user-linked purchases, explicit
relevance judgments, weather history, or body-shape labels. Consequently, this
experiment reports held-out **product-row pseudo-profile** results, not observed
user outcomes. The ranking relevance definition is a transparent catalog-attribute
proxy, and it overlaps with the features used by the improved ranker. Treat its
results as a controlled software/catalog experiment, not proof of customer
satisfaction or generalization to real users.

## Experimental setup

- Data: `{metadata["input_file"]}`; stratified {metadata["train_fraction"]:.0%}/{metadata["test_fraction"]:.0%} train/test split, seed {metadata["random_state"]}.
- Training catalog: {metadata["train_rows"]} products; in-stock candidate catalog: {metadata["candidate_count"]}; held-out rows: {metadata["test_rows"]}.
- Product ID overlap between train and test: {metadata["train_test_product_id_overlap"]}.
- Baseline classifier: a reproduced Random Forest category predictor using only gender, location, and season (100 trees; one-hot categories). It is trained only on the training split.
- Baseline fidelity: the upgraded project no longer contains the original trained Random Forest artifact. This is a documented re-implementation of the audited baseline design, not a score from the historical serialized model.
- Baseline recommender: predict category, then seeded-randomly select in-stock products of that category.
- Improved recommender: rank in-stock training products using the existing content/profile scoring engine; pseudo-profile preferences are taken from each held-out row's category, season, gender, location, color, brand, and size.
- Proxy relevance: category match = 2; exact season, gender-compatible (Unisex also compatible with Men/Women), location, color, brand, and size = 1 each. Binary relevance threshold = {metadata["relevance_threshold"]}. NDCG uses the graded relevance. Ranking metrics are macro-averaged over {metadata["evaluated_profiles"]} pseudo-profiles; F1 is computed per profile before averaging.
- {metadata["skipped_profiles"]} rows had no proxy-relevant in-stock candidate and were omitted from ranking averages.
- No weather or body-shape adjustment was supplied to either system in this ranking experiment. The dataset has no weather records or body-shape ground truth, so those components are **not experimentally validated here**.

## Category-classification results

Classification applies only to the Random Forest component of the original baseline.
The upgraded recommender is a ranker, not a category classifier; its accuracy,
precision, recall, and F1 as a classifier are therefore **not applicable**, rather
than zero.

{_markdown_table(classification)}

## Top-K recommendation results

These are computed against the catalog proxy labels above, not user judgments.

{_markdown_table(ranking)}

## Local response-time results

Times measure local, in-process request handling for category inference plus
random selection (baseline), versus preference normalization and full top-{metadata["latency_k"]} catalog ranking (improved). The measurement uses
all {metadata["latency_profiles"]} held-out pseudo-profiles in one pass. It excludes Flask/HTTP overhead,
weather network calls, SQLite profile I/O, image upload, MediaPipe pose
estimation, browser rendering, and machine-independent deployment effects.

{_markdown_table(latency, 3)}

## Interpretation and model comparison

- The Random Forest classification scores indicate how well the audited baseline
  category-prediction task is recovered from gender, location, and season on the
  held-out catalog rows. This is a product-row classification result, not a
  recommendation-quality score.
- The baseline recommender uses a predicted category but randomizes product
  order, so it cannot consistently prioritize the pseudo-profile's other
  matching attributes. The improved ranker explicitly orders catalog products
  using preference/profile similarity and available product attributes.
- Higher ranking metrics for the improved system, if observed, show better
  alignment with this designed attribute proxy. The proxy reuses ranking inputs,
  so high values are expected and potentially optimistic; they do not establish
  real-world superiority.
- Runtime should be interpreted as local algorithm cost only. The improved
  method performs more work per request; whether its measured latency is
  acceptable depends on deployment resources and catalog size.
- Weather-aware ranking and body-shape-aware ranking may change results when
  inputs are supplied at runtime, but this data does not support a defensible
  offline effectiveness comparison for either component.

## Limitations

1. No interaction/relevance ground truth; pseudo-profiles are held-out product
   attributes, not people.
2. Proxy relevance overlaps with recommender inputs, creating target-feature
   alignment and optimistic evaluation risk.
3. No body-shape labels or fashion-styling outcome labels exist. Image-based
   styling proxies have no measured offline accuracy or recommendation benefit.
4. The catalog does not contain weather-at-purchase or garment material,
   breathability, insulation, or waterproof attributes. Context rules are
   documented heuristics, not experimentally validated user benefits.
5. A single seeded split and one random baseline do not establish statistical
   significance or robustness across catalogs and time.
6. Response timings depend on this machine, Python environment, warm state, and
   catalog size, and exclude network and UI costs.

## Future improvements

- Log consented impressions, ranked positions, clicks, saves, purchases, and
  returns with user/item IDs; define a time-aware train/validation/test protocol.
- Evaluate with held-out interaction data, per-user Precision/Recall/NDCG/Hit
  Rate, confidence intervals, and suitable popularity/category baselines.
- Run user studies or expert-labeled outfit relevance judgments to validate
  preference, weather, and body-shape styling assumptions.
- Collect trustworthy item material/performance metadata and time/location
  context before assessing weather effects.
- Measure latency under representative deployment load, including HTTP,
  database, weather, and optional image-processing paths.

## Reproducibility and output

Run `python final_evaluation.py` from the project folder. The CSV tables, JSON
metadata, report, and graphs are written to `data/final_evaluation/`. Split seed,
classifier seed, ranking protocol, and latency sample parameters are recorded in
the JSON metadata. Only calculated values appear in the tables.
"""


def run_final_evaluation(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Path]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")

    data = validate_and_normalize(pd.read_csv(input_path))
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["category"],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    classifier = _make_classifier(random_state)
    classifier.fit(train.loc[:, CLASSIFICATION_FEATURES], train["category"])
    class_labels = sorted(train["category"].unique().tolist())
    candidate_catalog = train.loc[train["stock_quantity"] > 0].copy()
    engine = FashionRecommendationEngine(train)
    predicted_categories: list[str] = []
    baseline_latency_samples: list[float] = []
    improved_latency_samples: list[float] = []
    relevance_by_profile: dict[str, dict[str, int]] = {}
    rankings: dict[str, dict[str, list[str]]] = {
        "Original RF + random selection": {},
        "Improved personalized content ranking": {},
    }
    skipped_profiles = 0
    for _, profile in test.iterrows():
        start = time.perf_counter_ns()
        predicted_category = str(
            classifier.predict(
                pd.DataFrame([profile.loc[list(CLASSIFICATION_FEATURES)]])
            )[0]
        )
        profile_id = str(profile["product_id"])
        baseline_ids = _baseline_recommend(
            profile,
            predicted_category,
            candidate_catalog,
            max(K_VALUES),
            random_state,
        )
        baseline_latency_samples.append(
            (time.perf_counter_ns() - start) / 1_000_000
        )
        predicted_categories.append(predicted_category)

        relevance = proxy_relevance_grades(profile, candidate_catalog)
        start = time.perf_counter_ns()
        preferences = _preferences_from_profile(profile)
        ranked_items = engine.recommend(preferences, top_n=max(K_VALUES))
        improved_latency_samples.append(
            (time.perf_counter_ns() - start) / 1_000_000
        )
        if not any(grade >= RELEVANCE_THRESHOLD for grade in relevance.values()):
            skipped_profiles += 1
            continue
        rankings["Original RF + random selection"][profile_id] = baseline_ids
        rankings["Improved personalized content ranking"][profile_id] = [
            item["product_id"] for item in ranked_items
        ]
        relevance_by_profile[profile_id] = relevance
    if not relevance_by_profile:
        raise ValueError("No eligible held-out profiles for ranking evaluation")

    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        test["category"],
        predicted_categories,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    classification = pd.DataFrame(
        [
            {
                "model": "Original Random Forest category classifier",
                "test_products": len(test),
                "accuracy": accuracy_score(test["category"], predicted_categories),
                "macro_precision": class_precision,
                "macro_recall": class_recall,
                "macro_f1": class_f1,
            },
            {
                "model": "Improved personalized content ranker",
                "test_products": len(test),
                "accuracy": np.nan,
                "macro_precision": np.nan,
                "macro_recall": np.nan,
                "macro_f1": np.nan,
            },
        ]
    )
    aggregate = _evaluate_rankings(
        rankings,
        relevance_by_profile,
        K_VALUES,
    )
    ranking_rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        for method in rankings:
            metrics = aggregate[(method, k)]
            ranking_rows.append(
                {
                    "method": method,
                    "k": k,
                    "pseudo_profiles": len(relevance_by_profile),
                    "precision_at_k": metrics[f"precision@{k}"],
                    "recall_at_k": metrics[f"recall@{k}"],
                    "f1_at_k": metrics[f"f1@{k}"],
                    "ndcg_at_k": metrics[f"ndcg@{k}"],
                    "hit_rate_at_k": metrics[f"hit_rate@{k}"],
                }
            )
    ranking = pd.DataFrame(ranking_rows)
    latency = pd.DataFrame(
        [
            {
                "method": "Original RF + random selection",
                **_latency_summary(baseline_latency_samples),
            },
            {
                "method": "Improved personalized content ranker",
                **_latency_summary(improved_latency_samples),
            },
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "classification": output_dir / "classification_metrics.csv",
        "ranking": output_dir / "ranking_metrics.csv",
        "latency": output_dir / "response_time_metrics.csv",
        "report_json": output_dir / "final_evaluation.json",
        "report_markdown": output_dir / "final_evaluation.md",
        "ranking_plot": output_dir / "ranking_metrics.png",
        "classification_plot": output_dir / "classification_confusion_matrix.png",
        "latency_plot": output_dir / "response_time_comparison.png",
    }
    classification.to_csv(paths["classification"], index=False)
    ranking.to_csv(paths["ranking"], index=False)
    latency.to_csv(paths["latency"], index=False)

    metadata: dict[str, Any] = {
        "evaluation_type": "held_out_catalog_proxy_and_local_latency",
        "real_user_interaction_data_available": False,
        "relevance_data_fabricated": False,
        "weather_and_body_shape_effectiveness_measured": False,
        "input_file": input_path.name,
        "source_rows": int(len(data)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_fraction": 1 - test_size,
        "test_fraction": test_size,
        "random_state": random_state,
        "train_test_product_id_overlap": int(
            len(set(train["product_id"]) & set(test["product_id"]))
        ),
        "candidate_count": int(len(candidate_catalog)),
        "classification_features": list(CLASSIFICATION_FEATURES),
        "historical_model_artifact_available": False,
        "baseline_is_reimplementation_of_audited_design": True,
        "classification_metrics_averaging": "macro; labels from training split; zero_division=0",
        "random_forest": {"n_estimators": 100, "random_state": random_state, "n_jobs": 1},
        "ranking_methods": list(rankings),
        "ranking_k_values": list(K_VALUES),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "graded_relevance_points": {
            "category": 2,
            "season": 1,
            "gender_compatibility": 1,
            "location": 1,
            "color": 1,
            "brand": 1,
            "size": 1,
        },
        "ranking_aggregation": "Macro average over held-out product-row pseudo-profiles; F1 computed per profile",
        "evaluated_profiles": len(relevance_by_profile),
        "skipped_profiles": skipped_profiles,
        "latency_scope": (
            "Local one-pass in-process inference/ranking after model and engine initialization; "
            "no discarded warm-up; excludes HTTP, weather network, "
            "SQLite, image processing, MediaPipe, and browser rendering"
        ),
        "latency_profiles": len(test),
        "latency_repetitions": 1,
        "latency_k": max(K_VALUES),
        "limitations": [
            "No user-item interaction history or explicit relevance judgments are present.",
            "Ranking pseudo-profiles are held-out product rows and are not real users.",
            "The proxy relevance uses attributes also supplied to the improved recommender, so results are optimistic and not independent.",
            "The improved ranker is not a category classifier; classification metrics are not applicable to it.",
            "The catalog contains no weather observations, body-shape labels, or evaluated styling outcomes.",
            "Latency is machine-, environment-, and catalog-size-specific and excludes external services and application transport.",
            "One seeded split does not measure uncertainty or prove statistical significance.",
        ],
    }
    paths["report_json"].write_text(
        json.dumps(metadata, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    paths["report_markdown"].write_text(
        _write_report(metadata, classification, ranking, latency),
        encoding="utf-8",
    )
    _plot_ranking_metrics(ranking, paths["ranking_plot"])
    _plot_confusion_matrix(
        test["category"],
        predicted_categories,
        class_labels,
        paths["classification_plot"],
    )
    _plot_latency(latency, paths["latency_plot"])
    return paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate the original RF/random system against the improved ranker."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    paths = run_final_evaluation(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("Results are catalog-proxy experiments, not user-interaction measurements.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
