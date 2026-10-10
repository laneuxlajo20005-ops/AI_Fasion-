"""Reusable data validation and feature preprocessing for the fashion catalog."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


REQUIRED_COLUMNS = (
    "product_id",
    "category",
    "brand",
    "season",
    "size",
    "color",
    "original_price",
    "markdown_percentage",
    "current_price",
    "purchase_date",
    "stock_quantity",
    "customer_rating",
    "is_returned",
    "return_reason",
    "gender",
    "location",
)

ALLOWED_VALUES = {
    "category": {"Accessories", "Bottoms", "Dresses", "Outerwear", "Shoes", "Tops"},
    "season": {"Fall", "Spring", "Summer", "Winter"},
    "gender": {"Men", "Unisex", "Women"},
    "location": {"Rural", "Urban"},
}

STRING_COLUMNS = (
    "product_id",
    "category",
    "brand",
    "season",
    "size",
    "color",
    "return_reason",
    "gender",
    "location",
)
NUMERIC_COLUMNS = (
    "original_price",
    "markdown_percentage",
    "current_price",
    "stock_quantity",
    "customer_rating",
)
REQUIRED_NUMERIC_COLUMNS = (
    "original_price",
    "markdown_percentage",
    "current_price",
    "stock_quantity",
)

MODEL_CATEGORICAL_COLUMNS = (
    "season",
    "gender",
    "location",
    "brand",
    "color",
    "size",
    "price_segment",
    "stock_level",
)
MODEL_NUMERIC_COLUMNS = (
    "original_price",
    "current_price",
    "discount_amount",
    "discount_percentage",
    "stock_quantity",
)
MODEL_FEATURE_COLUMNS = MODEL_CATEGORICAL_COLUMNS + MODEL_NUMERIC_COLUMNS


def validate_and_normalize(data: pd.DataFrame) -> pd.DataFrame:
    """Validate catalog records and return normalized values without dropping rows."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    frame = data.loc[:, REQUIRED_COLUMNS].copy()
    for column in STRING_COLUMNS:
        frame[column] = frame[column].astype("string").str.strip()
        frame[column] = frame[column].replace("", pd.NA)

    for column in NUMERIC_COLUMNS:
        source = frame[column]
        converted = pd.to_numeric(source, errors="coerce")
        invalid = source.notna() & converted.isna()
        if invalid.any():
            raise ValueError(
                f"Column {column!r} contains {int(invalid.sum())} non-numeric value(s)"
            )
        frame[column] = converted

    for column in REQUIRED_NUMERIC_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"Column {column!r} contains missing values")

    date_source = frame["purchase_date"]
    frame["purchase_date"] = pd.to_datetime(date_source, errors="coerce")
    if frame["purchase_date"].isna().any():
        raise ValueError(
            "purchase_date contains "
            f"{int(frame['purchase_date'].isna().sum())} missing or invalid value(s)"
        )

    boolean_values = frame["is_returned"].astype("string").str.strip().str.lower()
    boolean_map = {"true": True, "false": False, "1": True, "0": False}
    frame["is_returned"] = boolean_values.map(boolean_map)
    if frame["is_returned"].isna().any():
        raise ValueError("is_returned contains missing or unrecognized boolean values")
    frame["is_returned"] = frame["is_returned"].astype(bool)

    for column, allowed in ALLOWED_VALUES.items():
        observed = set(frame[column].dropna().unique())
        invalid = sorted(observed - allowed)
        if frame[column].isna().any() or invalid:
            raise ValueError(
                f"Column {column!r} has missing or unsupported values: {invalid}"
            )

    for column in ("product_id", "brand", "color"):
        if frame[column].isna().any():
            raise ValueError(f"Column {column!r} contains missing or blank values")

    if frame["product_id"].duplicated().any():
        raise ValueError("product_id must be unique; duplicate IDs were found")
    if frame.duplicated().any():
        raise ValueError("Exact duplicate rows were found; resolve them explicitly")

    if (frame[["original_price", "current_price", "markdown_percentage", "stock_quantity"]] < 0).any().any():
        raise ValueError("Prices, markdown_percentage, and stock_quantity must be non-negative")
    if (frame["markdown_percentage"] > 100).any():
        raise ValueError("markdown_percentage must be between 0 and 100")
    if (frame["current_price"] > frame["original_price"] + 0.01).any():
        raise ValueError("current_price cannot exceed original_price")
    if (frame["customer_rating"].dropna().between(1, 5) == False).any():
        raise ValueError("customer_rating must be between 1 and 5 when present")

    calculated_discount = np.where(
        frame["original_price"] > 0,
        100 * (frame["original_price"] - frame["current_price"]) / frame["original_price"],
        0,
    )
    markdown_difference = np.abs(frame["markdown_percentage"] - calculated_discount)
    if (markdown_difference > 0.15).any():
        raise ValueError(
            "markdown_percentage disagrees with original_price/current_price "
            "by more than 0.15 percentage points"
        )

    return frame.reset_index(drop=True)


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def build_quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Create a JSON-serializable summary of missingness, values, and validity."""
    missing = {
        column: int(count)
        for column, count in frame.isna().sum().items()
    }
    numeric_summary: dict[str, dict[str, float | int]] = {}
    outlier_summary: dict[str, dict[str, float | int]] = {}
    for column in frame.select_dtypes(include="number").columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        numeric_summary[column] = {
            key: float(value)
            for key, value in values.describe(
                percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            ).items()
        }
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_summary[column] = {
            "iqr_lower_fence": float(lower),
            "iqr_upper_fence": float(upper),
            "flagged_count": int(((values < lower) | (values > upper)).sum()),
            "note": "Reported only; values are not removed or capped.",
        }

    date_values = pd.to_datetime(frame["purchase_date"], errors="coerce")
    calculated_discount = np.where(
        frame["original_price"] > 0,
        100 * (frame["original_price"] - frame["current_price"]) / frame["original_price"],
        0,
    )
    return_reasons = frame.loc[frame["is_returned"], "return_reason"].fillna("Unspecified")
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_product_ids": int(frame["product_id"].duplicated().sum()),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": missing,
        "categorical_values": {
            column: _value_counts(frame[column])
            for column in frame.select_dtypes(include=["object", "string", "bool"]).columns
            if column != "product_id"
        },
        "numeric_summary": numeric_summary,
        "iqr_outliers": outlier_summary,
        "date_analysis": {
            "invalid_dates": int(date_values.isna().sum()),
            "minimum": date_values.min().isoformat() if date_values.notna().any() else None,
            "maximum": date_values.max().isoformat() if date_values.notna().any() else None,
            "month_counts": _value_counts(date_values.dt.month),
            "year_counts": _value_counts(date_values.dt.year),
        },
        "season_analysis": {
            "counts": _value_counts(frame["season"]),
            "category_by_season": {
                str(index): {str(key): int(value) for key, value in row.items()}
                for index, row in pd.crosstab(frame["category"], frame["season"]).iterrows()
            },
        },
        "gender_analysis": {
            "counts": _value_counts(frame["gender"]),
            "category_by_gender": {
                str(index): {str(key): int(value) for key, value in row.items()}
                for index, row in pd.crosstab(frame["category"], frame["gender"]).iterrows()
            },
        },
        "location_analysis": {
            "counts": _value_counts(frame["location"]),
            "category_by_location": {
                str(index): {str(key): int(value) for key, value in row.items()}
                for index, row in pd.crosstab(frame["category"], frame["location"]).iterrows()
            },
        },
        "price_analysis": {
            "original_price_min": float(frame["original_price"].min()),
            "original_price_median": float(frame["original_price"].median()),
            "original_price_max": float(frame["original_price"].max()),
            "current_price_min": float(frame["current_price"].min()),
            "current_price_median": float(frame["current_price"].median()),
            "current_price_max": float(frame["current_price"].max()),
        },
        "discount_analysis": {
            "mean_markdown_percentage": float(frame["markdown_percentage"].mean()),
            "median_markdown_percentage": float(frame["markdown_percentage"].median()),
            "calculated_discount_amount_median": float(
                (frame["original_price"] - frame["current_price"]).median()
            ),
            "max_markdown_consistency_error_percentage_points": float(
                np.abs(frame["markdown_percentage"] - calculated_discount).max()
            ),
        },
        "rating_analysis": {
            "rated_count": int(frame["customer_rating"].notna().sum()),
            "unrated_count": int(frame["customer_rating"].isna().sum()),
            "mean": float(frame["customer_rating"].mean()),
            "median": float(frame["customer_rating"].median()),
            "min": float(frame["customer_rating"].min()),
            "max": float(frame["customer_rating"].max()),
        },
        "return_analysis": {
            "returned_count": int(frame["is_returned"].sum()),
            "not_returned_count": int((~frame["is_returned"]).sum()),
            "return_rate": float(frame["is_returned"].mean()),
            "reasons_among_returned": _value_counts(return_reasons),
            "returned_without_reason": int(
                (frame["is_returned"] & frame["return_reason"].isna()).sum()
            ),
            "not_returned_with_reason": int(
                ((~frame["is_returned"]) & frame["return_reason"].notna()).sum()
            ),
        },
        "stock_analysis": {
            "zero_stock_count": int((frame["stock_quantity"] == 0).sum()),
            "stock_level_counts": {
                "out_of_stock": int((frame["stock_quantity"] == 0).sum()),
                "low_1_to_10": int(frame["stock_quantity"].between(1, 10).sum()),
                "medium_11_to_30": int(frame["stock_quantity"].between(11, 30).sum()),
                "high_over_30": int((frame["stock_quantity"] > 30).sum()),
            },
        },
        "validation_checks": {
            "status": "passed; invalid inputs raise ValueError before output is written",
            "unexpected_domain_values": {column: [] for column in ALLOWED_VALUES},
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_product_ids": int(frame["product_id"].duplicated().sum()),
            "missing_required_values": {
                column: int(frame[column].isna().sum())
                for column in (
                    "product_id",
                    "category",
                    "brand",
                    "season",
                    "color",
                    "original_price",
                    "markdown_percentage",
                    "current_price",
                    "purchase_date",
                    "stock_quantity",
                    "is_returned",
                    "gender",
                    "location",
                )
            },
            "invalid_dates": int(date_values.isna().sum()),
            "negative_prices_discounts_or_stock": int(
                (
                    frame[
                        ["original_price", "current_price", "markdown_percentage", "stock_quantity"]
                    ]
                    < 0
                )
                .any(axis=1)
                .sum()
            ),
            "markdown_over_100": int((frame["markdown_percentage"] > 100).sum()),
            "current_price_above_original": int(
                (frame["current_price"] > frame["original_price"] + 0.01).sum()
            ),
            "rating_outside_1_to_5": int(
                (~frame["customer_rating"].dropna().between(1, 5)).sum()
            ),
            "markdown_mismatch_over_0_15_percentage_points": int(
                (np.abs(frame["markdown_percentage"] - calculated_discount) > 0.15).sum()
            ),
        },
    }


class FashionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add analysis features and train-fitted price bands to catalog rows."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FashionFeatureEngineer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FashionFeatureEngineer expects a pandas DataFrame")
        required = set(MODEL_CATEGORICAL_COLUMNS[:6]) | {
            "original_price",
            "current_price",
            "stock_quantity",
        }
        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(f"Feature input is missing required columns: {missing}")

        prices = pd.to_numeric(X["original_price"], errors="raise")
        if prices.isna().any() or prices.empty:
            raise ValueError("original_price must contain at least one non-missing value")
        self.price_thresholds_ = (
            float(prices.quantile(1 / 3)),
            float(prices.quantile(2 / 3)),
        )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "price_thresholds_"):
            raise RuntimeError("FashionFeatureEngineer must be fitted before transform")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FashionFeatureEngineer expects a pandas DataFrame")

        frame = X.copy()
        for column in MODEL_CATEGORICAL_COLUMNS[:6]:
            if column not in frame:
                raise ValueError(f"Feature input is missing required column: {column}")
            values = frame[column].astype("string").str.strip()
            frame[column] = values.replace("", pd.NA).fillna("Unknown").astype(str)

        original_price = pd.to_numeric(frame["original_price"], errors="raise")
        current_price = pd.to_numeric(frame["current_price"], errors="raise")
        stock = pd.to_numeric(frame["stock_quantity"], errors="raise")
        if original_price.isna().any() or current_price.isna().any() or stock.isna().any():
            raise ValueError("Prices and stock_quantity must be present for feature creation")

        frame["discount_amount"] = original_price - current_price
        frame["discount_percentage"] = np.divide(
            frame["discount_amount"] * 100,
            original_price,
            out=np.zeros(len(frame), dtype=float),
            where=original_price.to_numpy() != 0,
        )

        first_threshold, second_threshold = self.price_thresholds_
        frame["price_segment"] = np.select(
            [
                original_price <= first_threshold,
                original_price <= second_threshold,
            ],
            ["budget", "mid_range"],
            default="premium",
        )
        frame["stock_level"] = np.select(
            [
                stock == 0,
                stock <= 10,
                stock <= 30,
            ],
            ["out_of_stock", "low", "medium"],
            default="high",
        )

        if "customer_rating" in frame:
            rating = pd.to_numeric(frame["customer_rating"], errors="coerce")
            frame["rating_category"] = np.select(
                [rating.notna() & (rating < 3), rating.notna() & (rating < 4)],
                ["low", "average"],
                default="high",
            )
            frame.loc[rating.isna(), "rating_category"] = "unrated"
        else:
            frame["rating_category"] = "unrated"

        if "purchase_date" in frame:
            dates = pd.to_datetime(frame["purchase_date"], errors="coerce")
            frame["purchase_month"] = dates.dt.month.astype("Int64")
            frame["purchase_day"] = dates.dt.day.astype("Int64")
            frame["purchase_year"] = dates.dt.year.astype("Int64")
        else:
            frame["purchase_month"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
            frame["purchase_day"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
            frame["purchase_year"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")

        return frame


def build_preprocessing_pipeline() -> Pipeline:
    """Build the reusable feature-engineering and one-hot-encoding pipeline."""
    encoder = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(MODEL_CATEGORICAL_COLUMNS),
            ),
            ("numeric", "passthrough", list(MODEL_NUMERIC_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("feature_engineering", FashionFeatureEngineer()),
            ("encoding", encoder),
        ]
    )
