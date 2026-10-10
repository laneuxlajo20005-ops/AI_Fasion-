"""Validate, split, and preprocess the fashion dataset without training a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from fashion_preprocessing import (
    MODEL_FEATURE_COLUMNS,
    FashionFeatureEngineer,
    build_preprocessing_pipeline,
    build_quality_report,
    validate_and_normalize,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "fashion_boutique_fixed_location.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "processed"


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Path]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")

    raw = pd.read_csv(input_path)
    data = validate_and_normalize(raw)
    report = build_quality_report(data)
    report["source_dtypes"] = {column: str(dtype) for column, dtype in raw.dtypes.items()}

    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["category"],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    pipeline = build_preprocessing_pipeline()
    pipeline.fit(train)
    feature_names = pipeline.named_steps["encoding"].get_feature_names_out()
    train_matrix = pipeline.transform(train)
    test_matrix = pipeline.transform(test)

    output_dir.mkdir(parents=True, exist_ok=True)
    engineered = pipeline.named_steps["feature_engineering"]
    train_engineered = engineered.transform(train)
    test_engineered = engineered.transform(test)

    train_engineered_path = output_dir / "train_engineered.csv"
    test_engineered_path = output_dir / "test_engineered.csv"
    x_train_path = output_dir / "X_train.csv"
    x_test_path = output_dir / "X_test.csv"
    y_train_path = output_dir / "y_train.csv"
    y_test_path = output_dir / "y_test.csv"
    report_path = output_dir / "data_quality_report.json"
    pipeline_path = output_dir / "fashion_preprocessor.joblib"

    train_engineered.to_csv(train_engineered_path, index=False)
    test_engineered.to_csv(test_engineered_path, index=False)
    for split, matrix, x_path, y_path in (
        (train, train_matrix, x_train_path, y_train_path),
        (test, test_matrix, x_test_path, y_test_path),
    ):
        encoded = pd.DataFrame(matrix, columns=feature_names, index=split.index)
        encoded.to_csv(x_path, index=False)
        split[["product_id", "category"]].reset_index(drop=True).to_csv(y_path, index=False)

    report["split"] = {
        "test_size_fraction": test_size,
        "random_state": random_state,
        "stratified_by": "category",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_category_counts": {
            str(key): int(value) for key, value in train["category"].value_counts().items()
        },
        "test_category_counts": {
            str(key): int(value) for key, value in test["category"].value_counts().items()
        },
        "train_test_product_id_overlap": int(
            len(set(train["product_id"]) & set(test["product_id"]))
        ),
        "encoded_feature_count": int(len(feature_names)),
        "model_feature_inputs": list(MODEL_FEATURE_COLUMNS),
        "excluded_from_model_features": [
            "category (target)",
            "product_id (identifier)",
            "purchase_date and derived calendar fields (historical transaction timing)",
            "customer_rating and rating_category (post-purchase outcome)",
            "is_returned and return_reason (post-purchase outcome)",
        ],
        "price_segment_thresholds_fitted_on_train": {
            "budget_max_original_price": float(engineered.price_thresholds_[0]),
            "mid_range_max_original_price": float(engineered.price_thresholds_[1]),
        },
    }
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)
    joblib.dump(pipeline, pipeline_path)

    return {
        "train_engineered": train_engineered_path,
        "test_engineered": test_engineered_path,
        "X_train": x_train_path,
        "X_test": x_test_path,
        "y_train": y_train_path,
        "y_test": y_test_path,
        "report": report_path,
        "pipeline": pipeline_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and preprocess the fashion catalog; does not train a model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for processed data, report, and fitted preprocessing pipeline",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of rows reserved for testing"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Seed for reproducible train/test split"
    )
    args = parser.parse_args()

    paths = prepare_dataset(args.input, args.output_dir, args.test_size, args.random_state)
    print("Data preparation completed (no recommendation model was trained).")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
