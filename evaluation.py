"""Offline ranking metrics and an explicitly simulated catalog-evaluation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fashion_preprocessing import validate_and_normalize
from recommendation_engine import FashionRecommendationEngine


DEFAULT_INPUT = Path(__file__).resolve().parent / "fashion_boutique_fixed_location.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "evaluation"
DEFAULT_K_VALUES = (5, 10)
RELEVANCE_THRESHOLD = 4
RELEVANCE_WEIGHTS = {
    "category": 2,
    "season": 1,
    "gender_compatibility": 1,
    "location": 1,
    "color": 1,
    "brand": 1,
    "size": 1,
}


def _gender_matches(candidate_gender: pd.Series, profile_gender: str) -> pd.Series:
    exact = candidate_gender.astype(str) == profile_gender
    if profile_gender in {"Men", "Women"}:
        return exact | (candidate_gender.astype(str) == "Unisex")
    return exact


def proxy_relevance_grades(
    profile: pd.Series, candidate_catalog: pd.DataFrame
) -> dict[str, int]:
    """Assign graded proxy relevance from held-out catalog attributes, not user clicks."""
    candidate = candidate_catalog.loc[candidate_catalog["stock_quantity"] > 0].copy()
    if candidate.empty:
        return {}

    target_size = profile["size"]
    target_size = "Unknown" if pd.isna(target_size) or not str(target_size).strip() else str(target_size)
    candidate_size = candidate["size"].fillna("Unknown").astype(str)

    grade = (
        (candidate["category"].astype(str) == str(profile["category"])).astype(int)
        * RELEVANCE_WEIGHTS["category"]
        + (candidate["season"].astype(str) == str(profile["season"])).astype(int)
        * RELEVANCE_WEIGHTS["season"]
        + _gender_matches(candidate["gender"], str(profile["gender"])).astype(int)
        * RELEVANCE_WEIGHTS["gender_compatibility"]
        + (candidate["location"].astype(str) == str(profile["location"])).astype(int)
        * RELEVANCE_WEIGHTS["location"]
        + (candidate["color"].astype(str) == str(profile["color"])).astype(int)
        * RELEVANCE_WEIGHTS["color"]
        + (candidate["brand"].astype(str) == str(profile["brand"])).astype(int)
        * RELEVANCE_WEIGHTS["brand"]
        + (candidate_size == target_size).astype(int) * RELEVANCE_WEIGHTS["size"]
    )
    return {
        str(product_id): int(value)
        for product_id, value in zip(candidate["product_id"], grade)
    }


def ranking_metrics(
    ranked_product_ids: Iterable[str],
    relevance_grades: dict[str, int],
    k: int,
    relevance_threshold: int = RELEVANCE_THRESHOLD,
) -> dict[str, float]:
    """Calculate per-profile Precision/Recall/F1/Hit Rate/NDCG at K."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    ranked = list(ranked_product_ids)[:k]
    retrieved_relevant = sum(
        relevance_grades.get(str(product_id), 0) >= relevance_threshold
        for product_id in ranked
    )
    relevant_count = sum(
        grade >= relevance_threshold for grade in relevance_grades.values()
    )
    precision = retrieved_relevant / k
    recall = retrieved_relevant / relevant_count if relevant_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    hit_rate = float(retrieved_relevant > 0)

    def dcg(grades: Iterable[int]) -> float:
        return sum(
            (2**int(grade) - 1) / np.log2(rank + 2)
            for rank, grade in enumerate(grades)
        )

    observed_grades = [
        relevance_grades.get(str(product_id), 0) for product_id in ranked
    ]
    ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal_grades)
    ndcg = dcg(observed_grades) / ideal_dcg if ideal_dcg else 0.0
    return {
        f"precision@{k}": float(precision),
        f"recall@{k}": float(recall),
        f"f1@{k}": float(f1),
        f"ndcg@{k}": float(ndcg),
        f"hit_rate@{k}": hit_rate,
    }


def random_category_baseline(
    profile: pd.Series,
    candidate_catalog: pd.DataFrame,
    top_n: int,
    random_state: int,
) -> list[str]:
    """Return a reproducible random ranking within the preferred category."""
    candidates = candidate_catalog.loc[
        (candidate_catalog["stock_quantity"] > 0)
        & (candidate_catalog["category"] == profile["category"])
    ]
    product_ids = candidates["product_id"].astype(str).to_numpy()
    seed_material = f"{random_state}:{profile['product_id']}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    if len(product_ids):
        product_ids = generator.permutation(product_ids)
    return [str(product_id) for product_id in product_ids[:top_n]]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _evaluate_rankings(
    rankings: dict[str, list[str]],
    relevance_by_profile: dict[str, dict[str, int]],
    k_values: tuple[int, ...],
) -> dict[tuple[str, int], dict[str, float]]:
    aggregate: dict[tuple[str, int], dict[str, float]] = {}
    for method, profile_rankings in rankings.items():
        for k in k_values:
            metrics: dict[str, list[float]] = {
                f"precision@{k}": [],
                f"recall@{k}": [],
                f"f1@{k}": [],
                f"ndcg@{k}": [],
                f"hit_rate@{k}": [],
            }
            for profile_id, ordered_ids in profile_rankings.items():
                result = ranking_metrics(
                    ordered_ids,
                    relevance_by_profile[profile_id],
                    k,
                )
                for metric_name, value in result.items():
                    metrics[metric_name].append(value)
            aggregate[(method, k)] = {
                metric_name: _mean(values) for metric_name, values in metrics.items()
            }
    return aggregate


def _comparison_rows(
    aggregate: dict[tuple[str, int], dict[str, float]],
    profile_count: int,
    k_values: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in k_values:
        for method in ("Random/category baseline", "Content-based ranking"):
            values = aggregate[(method, k)]
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "profiles": profile_count,
                    "precision_at_k": values[f"precision@{k}"],
                    "recall_at_k": values[f"recall@{k}"],
                    "f1_at_k": values[f"f1@{k}"],
                    "ndcg_at_k": values[f"ndcg@{k}"],
                    "hit_rate_at_k": values[f"hit_rate@{k}"],
                }
            )
    return pd.DataFrame(rows)


def _plot_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    metrics = (
        ("precision_at_k", "Precision@K"),
        ("recall_at_k", "Recall@K"),
        ("f1_at_k", "F1@K"),
        ("ndcg_at_k", "NDCG@K"),
        ("hit_rate_at_k", "Hit Rate@K"),
    )
    figure, axes = plt.subplots(1, len(metrics), figsize=(17, 4.2), sharey=True)
    colors = {
        "Random/category baseline": "#8993a4",
        "Content-based ranking": "#e94560",
    }
    for axis, (column, title) in zip(axes, metrics):
        for method, group in comparison.groupby("method", sort=False):
            axis.plot(
                group["k"],
                group[column],
                marker="o",
                linewidth=2,
                label=method,
                color=colors[method],
            )
        axis.set_title(title)
        axis.set_xticks(sorted(comparison["k"].unique()))
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Macro-average score")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle(
        "Simulated catalog proxy evaluation (not user-interaction evaluation)",
        y=1.03,
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0.12, 1, 0.98))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _markdown_report(
    comparison: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    headers = [str(column) for column in comparison.columns]
    rows = [
        [
            f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        for row in comparison.itertuples(index=False, name=None)
    ]
    table = "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )
    return f"""# Fashion recommender offline evaluation

## Interpretation and limitations

This is a **simulated catalog-proxy evaluation**, not a measurement of user satisfaction.
The source catalog has {metadata["source_rows"]} product rows but no customer identifier,
user-item interactions, clicks, purchases linked to a customer, or recommendation
relevance judgments. Therefore genuine user-level Precision@K, Recall@K, F1@K,
NDCG@K, and Hit Rate@K are not computable from the supplied data.

No interaction or relevance data was fabricated. Instead, each held-out catalog row
is treated as a **pseudo-profile**, not a real person. Its available category, season,
gender, location, color, brand, and size fields define a documented attribute-match
proxy against in-stock products in the training catalog. This protocol measures
attribute retrieval behavior only; the held-out row is not evidence that a real
customer wanted matching products.

## Protocol

- Data split: {metadata["train_fraction"]:.0%} candidate catalog / {metadata["test_fraction"]:.0%} held-out pseudo-profiles.
- Split seed: {metadata["random_state"]}; stratified by category.
- Candidate inventory: training-split products with positive stock; both methods use this inventory.
- Baseline: deterministic seeded random ordering among in-stock products matching the pseudo-profile's category.
- Improved method: the existing content-based engine ranked using the pseudo-profile's gender, location, season, category, color, brand, and size.
- Proxy graded relevance: category match = 2 points; exact season, compatible gender, location, color, brand, and size = 1 point each. Men/Women profiles treat Unisex catalog products as gender-compatible.
- Binary relevance for Precision, Recall, F1, and Hit Rate: proxy grade >= {metadata["relevance_threshold"]}.
- NDCG uses the full graded proxy score as gain. Metrics are averaged per pseudo-profile (macro average); F1 is computed per profile before averaging.
- K values: {", ".join(map(str, metadata["k_values"]))}. No model training or computer vision is involved.
- Evaluated pseudo-profiles: {metadata["evaluated_profiles"]}; skipped because no in-stock relevant candidate existed: {metadata["skipped_profiles"]}.

## Comparison

{table}

## Reproducibility

Run `python evaluation.py` from the project directory. Results, protocol
metadata, and the plot are generated under `data/evaluation/`. The random baseline
is seeded per pseudo-profile, so its ranking is stable for a fixed dataset and seed.
"""


def run_evaluation(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    test_size: float = 0.2,
    random_state: int = 42,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    max_profiles: int | None = None,
) -> dict[str, Path]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")
    if not k_values or any(
        isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in k_values
    ):
        raise ValueError("k_values must contain positive integers")
    if len(set(k_values)) != len(k_values):
        raise ValueError("k_values cannot contain duplicates")
    if max_profiles is not None and max_profiles < 1:
        raise ValueError("max_profiles must be a positive integer")

    data = validate_and_normalize(pd.read_csv(input_path))
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["category"],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    if max_profiles is not None:
        test = test.head(max_profiles).copy()

    engine = FashionRecommendationEngine(train)
    available_catalog = train.loc[train["stock_quantity"] > 0].copy()
    max_k = max(k_values)
    rankings: dict[str, dict[str, list[str]]] = {
        "Random/category baseline": {},
        "Content-based ranking": {},
    }
    relevance_by_profile: dict[str, dict[str, int]] = {}
    skipped_profiles = 0

    for _, profile in test.iterrows():
        profile_id = str(profile["product_id"])
        relevance = proxy_relevance_grades(profile, available_catalog)
        relevant_count = sum(
            grade >= RELEVANCE_THRESHOLD for grade in relevance.values()
        )
        if relevant_count == 0:
            skipped_profiles += 1
            continue
        preferences = {
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
        content_ranking = engine.recommend(preferences, top_n=max_k)
        rankings["Random/category baseline"][profile_id] = random_category_baseline(
            profile, available_catalog, max_k, random_state
        )
        rankings["Content-based ranking"][profile_id] = [
            item["product_id"] for item in content_ranking
        ]
        relevance_by_profile[profile_id] = relevance

    if not relevance_by_profile:
        raise ValueError(
            "No held-out pseudo-profiles had in-stock proxy-relevant candidates; "
            "evaluation cannot produce meaningful ranking metrics"
        )

    aggregate = _evaluate_rankings(
        rankings,
        relevance_by_profile,
        tuple(sorted(k_values)),
    )
    comparison = _comparison_rows(
        aggregate,
        len(relevance_by_profile),
        tuple(sorted(k_values)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "evaluation_comparison.csv"
    report_path = output_dir / "evaluation_report.json"
    markdown_path = output_dir / "evaluation_report.md"
    plot_path = output_dir / "evaluation_comparison.png"
    comparison.to_csv(comparison_path, index=False)

    metadata = {
        "evaluation_type": "simulated_catalog_proxy",
        "real_user_interaction_data_available": False,
        "relevance_data_fabricated": False,
        "source_rows": int(len(data)),
        "train_rows": int(len(train)),
        "test_pseudo_profile_rows": int(len(test)),
        "evaluated_profiles": len(relevance_by_profile),
        "skipped_profiles": skipped_profiles,
        "train_fraction": 1.0 - test_size,
        "test_fraction": test_size,
        "random_state": random_state,
        "stratified_by": "category",
        "train_test_product_id_overlap": int(
            len(set(train["product_id"]) & set(test["product_id"]))
        ),
        "candidate_count": int(len(available_catalog)),
        "k_values": list(sorted(k_values)),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "graded_relevance_points": RELEVANCE_WEIGHTS,
        "gender_compatibility": (
            "Exact gender match; Unisex candidates also match Men/Women pseudo-profiles"
        ),
        "aggregation": "Macro average over pseudo-profiles; F1 computed per profile",
        "baseline": (
            "Deterministic random ranking among in-stock products matching pseudo-profile category"
        ),
        "improved_method": (
            "Content-based ranking with category, season, gender, location, color, brand, and size preferences"
        ),
        "metrics": [
            "Precision@K",
            "Recall@K",
            "F1@K",
            "NDCG@K",
            "Hit Rate@K",
        ],
        "comparison": comparison.to_dict(orient="records"),
        "limitations": [
            "The source data contains no customer identifiers, recommendation impressions, clicks, or user-item interactions.",
            "Pseudo-profiles are held-out product rows and do not represent actual users.",
            "Proxy relevance is derived from matching catalog attributes, so the evaluation measures attribute retrieval, not user satisfaction.",
            "The relevance rule overlaps the preference fields used by the content recommender; results must not be interpreted as independent evidence of real-world effectiveness.",
        ],
    }
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(metadata, report_file, indent=2, allow_nan=False)
    _plot_comparison(comparison, plot_path)
    markdown_path.write_text(
        _markdown_report(comparison, metadata),
        encoding="utf-8",
    )
    return {
        "comparison": comparison_path,
        "report_json": report_path,
        "report_markdown": markdown_path,
        "plot": plot_path,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare the category baseline and content ranker using a labeled simulated catalog proxy."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="One or more cutoff values, e.g. --k 5 10",
    )
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=None,
        help="Optional cap for a quick run; omit for the full held-out split",
    )
    args = parser.parse_args()
    paths = run_evaluation(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        k_values=tuple(args.k),
        max_profiles=args.max_profiles,
    )
    comparison = pd.read_csv(paths["comparison"])
    print("Evaluation type: simulated catalog proxy; not user-interaction evaluation.")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
