"""Content-based product ranking for the fashion catalog."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from context_processor import CONTEXT_SCORE_WEIGHT, score_weather_context
from fashion_preprocessing import validate_and_normalize


PROFILE_FIELDS = ("category", "season", "brand", "color", "size", "location")
PROFILE_FIELD_WEIGHTS = {
    "category": 0.25,
    "season": 0.20,
    "brand": 0.15,
    "color": 0.15,
    "size": 0.10,
    "location": 0.10,
}
REPRESENTATION_CATEGORICAL_FIELDS = (
    "category",
    "season",
    "gender",
    "brand",
    "color",
    "size",
    "location",
)
REPRESENTATION_NUMERIC_FIELDS = (
    "original_price",
    "current_price",
    "markdown_percentage",
    "customer_rating",
    "stock_quantity",
)
PREFERENCE_TO_FIELD = {
    "preferred_category": "category",
    "season": "season",
    "preferred_brand": "brand",
    "preferred_color": "color",
    "preferred_size": "size",
    "location": "location",
}
PREFERENCE_WEIGHTS = {
    "profile_similarity": 0.68,
    "gender_compatibility": 0.10,
    "rating_quality": 0.12,
    "budget_affordability": 0.08,
    "discount_value": 0.02,
}
STYLE_CATEGORY_PROXIES = {
    "Everyday": frozenset({"Tops", "Bottoms", "Shoes", "Accessories"}),
    "Occasion": frozenset({"Dresses", "Shoes", "Accessories"}),
    "Layered": frozenset({"Outerwear", "Tops", "Bottoms"}),
}
PROFILE_MATCH_WEIGHT = 0.20
BODY_SHAPE_MATCH_WEIGHT = 0.12
BODY_SHAPE_CATEGORY_PROXIES = {
    "Pear/triangle-style": frozenset({"Tops", "Outerwear", "Accessories"}),
    "Inverted-triangle-style": frozenset({"Bottoms", "Dresses", "Accessories"}),
    "Balanced shoulder/hip-style": frozenset(
        {"Dresses", "Tops", "Bottoms", "Accessories"}
    ),
}


def _optional_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


class FashionRecommendationEngine:
    """Rank real catalog products by content fit and explicit constraints."""

    def __init__(self, products: pd.DataFrame):
        self.products = validate_and_normalize(products)
        self.products = self.products.copy()
        self.products["size"] = self.products["size"].fillna("Unknown").astype(str)

        self._option_values: dict[str, tuple[str, ...]] = {}
        for field in PREFERENCE_TO_FIELD.values():
            allowed = set(self.products[field].dropna().astype(str).unique())
            if field == "size":
                allowed.add("Unknown")
            self._option_values[field] = tuple(sorted(allowed))
        self.gender_values = tuple(
            sorted(self.products["gender"].dropna().astype(str).unique())
        )
        self.style_values = tuple(STYLE_CATEGORY_PROXIES)
        self.body_shape_values = tuple(BODY_SHAPE_CATEGORY_PROXIES)

        self.profile_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
        profile_matrix = self.profile_encoder.fit_transform(self.products[list(PROFILE_FIELDS)])
        profile_names = self.profile_encoder.get_feature_names_out(PROFILE_FIELDS)
        profile_weights = np.ones(len(profile_names), dtype=float)
        for index, feature_name in enumerate(profile_names):
            field = feature_name.split("_", 1)[0]
            profile_weights[index] = np.sqrt(PROFILE_FIELD_WEIGHTS[field])
        self._profile_weights = profile_weights
        self._weighted_profile_matrix = profile_matrix * profile_weights

        self.representation_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
        representation_categorical = self.representation_encoder.fit_transform(
            self.products[list(REPRESENTATION_CATEGORICAL_FIELDS)]
        )
        self.representation_categorical_names = (
            self.representation_encoder.get_feature_names_out(
                REPRESENTATION_CATEGORICAL_FIELDS
            )
        )

        numeric_representation = self.products[list(REPRESENTATION_NUMERIC_FIELDS)].copy()
        rating_median = numeric_representation["customer_rating"].median()
        numeric_representation["customer_rating"] = numeric_representation[
            "customer_rating"
        ].fillna(rating_median)
        self.numeric_scaler = MinMaxScaler()
        scaled_numeric = self.numeric_scaler.fit_transform(numeric_representation)
        self.feature_names = tuple(self.representation_categorical_names) + tuple(
            REPRESENTATION_NUMERIC_FIELDS
        )
        self.product_feature_matrix = np.hstack(
            [representation_categorical, scaled_numeric]
        )
        self._rating_imputation = float(rating_median)

    def form_options(self) -> dict[str, tuple[str, ...]]:
        """Return data-backed choices for the Flask form."""
        return {
            "categories": self._option_values["category"],
            "seasons": self._option_values["season"],
            "brands": self._option_values["brand"],
            "colors": self._option_values["color"],
            "sizes": self._option_values["size"],
            "locations": self._option_values["location"],
            "genders": self.gender_values,
            "styles": self.style_values,
            "body_shapes": self.body_shape_values,
        }

    def _normalize_preferences(self, preferences: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        allowed_by_key = {
            "preferred_category": self._option_values["category"],
            "season": self._option_values["season"],
            "preferred_brand": self._option_values["brand"],
            "preferred_color": self._option_values["color"],
            "preferred_size": self._option_values["size"],
            "location": self._option_values["location"],
            "gender": self.gender_values,
        }
        for key, allowed in allowed_by_key.items():
            value = _optional_value(preferences.get(key))
            if value is not None and value not in allowed:
                raise ValueError(f"Unsupported {key.replace('_', ' ')}: {value}")
            normalized[key] = value

        multi_value_fields = {
            "preferred_categories": self._option_values["category"],
            "preferred_colors": self._option_values["color"],
            "preferred_brands": self._option_values["brand"],
            "preferred_styles": self.style_values,
            "body_shape": self.body_shape_values,
        }
        for key, allowed in multi_value_fields.items():
            raw_values = preferences.get(key, ())
            if isinstance(raw_values, str):
                raw_values = [raw_values] if raw_values.strip() else []
            if raw_values is None:
                raw_values = []
            if not isinstance(raw_values, (list, tuple, set)):
                raise ValueError(f"{key.replace('_', ' ')} must be a list of choices")
            values = list(dict.fromkeys(
                value
                for item in raw_values
                if (value := _optional_value(item)) is not None
            ))
            invalid = [value for value in values if value not in allowed]
            if invalid:
                raise ValueError(
                    f"Unsupported {key.replace('_', ' ')}: {', '.join(invalid)}"
                )
            normalized[key] = values
        if normalized["preferred_categories"] and normalized["preferred_category"] is None:
            normalized["preferred_category"] = normalized["preferred_categories"][0]

        price_bounds: dict[str, float | None] = {}
        for key in ("preferred_price_min", "preferred_price_max"):
            value = _optional_value(preferences.get(key))
            if value is None:
                price_bounds[key] = None
                continue
            try:
                parsed = float(value)
            except ValueError as error:
                raise ValueError(f"{key.replace('_', ' ')} must be numeric") from error
            if not np.isfinite(parsed) or parsed < 0:
                raise ValueError("Preferred price range must be finite and non-negative")
            price_bounds[key] = parsed
        if (
            price_bounds["preferred_price_min"] is not None
            and price_bounds["preferred_price_max"] is not None
            and price_bounds["preferred_price_min"] > price_bounds["preferred_price_max"]
        ):
            raise ValueError("Preferred minimum price cannot exceed maximum price")
        normalized.update(price_bounds)

        body_shape = _optional_value(preferences.get("body_shape"))
        if body_shape is not None and body_shape not in self.body_shape_values:
            raise ValueError(f"Unsupported body shape styling proxy: {body_shape}")
        normalized["body_shape"] = body_shape

        for key, minimum, maximum in (
            ("max_budget", 0.0, float("inf")),
            ("minimum_rating", 1.0, 5.0),
        ):
            value = _optional_value(preferences.get(key))
            if value is None:
                normalized[key] = None
                continue
            try:
                parsed = float(value)
            except ValueError as error:
                raise ValueError(f"{key.replace('_', ' ')} must be numeric") from error
            if not np.isfinite(parsed) or not minimum <= parsed <= maximum:
                if key == "max_budget":
                    raise ValueError("Maximum budget must be a finite, non-negative amount")
                raise ValueError("Minimum rating must be between 1 and 5")
            normalized[key] = parsed

        return normalized

    @staticmethod
    def _gender_compatibility(product_gender: str, preferred_gender: str | None) -> float:
        if preferred_gender is None:
            return 0.0
        if preferred_gender == "Unisex":
            return 1.0 if product_gender == "Unisex" else 0.75
        if product_gender == preferred_gender:
            return 1.0
        if product_gender == "Unisex":
            return 0.85
        return 0.0

    def _profile_similarity(self, preferences: dict[str, Any]) -> np.ndarray:
        fields_by_key = {
            **PREFERENCE_TO_FIELD,
            "preferred_categories": "category",
            "preferred_colors": "color",
            "preferred_brands": "brand",
        }
        query_values: dict[str, list[str]] = {
            field: [] for field in PROFILE_FIELDS
        }
        for preference_key, field in fields_by_key.items():
            value = preferences.get(preference_key)
            if isinstance(value, list):
                values = value
            elif value is not None:
                values = [value]
            else:
                values = []
            for selected in values:
                if selected not in query_values[field]:
                    query_values[field].append(selected)
        if not any(query_values.values()):
            return np.full(len(self.products), 0.5, dtype=float)

        categories = self.profile_encoder.categories_
        encoded_query = np.zeros((1, len(self._profile_weights)), dtype=float)
        column_offset = 0
        for field, allowed_values in zip(PROFILE_FIELDS, categories):
            selected = query_values[field]
            if selected:
                for value in selected:
                    matches = np.flatnonzero(allowed_values == value)
                    if len(matches):
                        encoded_query[0, column_offset + int(matches[0])] = 1.0
            column_offset += len(allowed_values)
        encoded_query *= self._profile_weights
        return cosine_similarity(
            self._weighted_profile_matrix,
            encoded_query,
        ).ravel()

    @staticmethod
    def _profile_fit(product: pd.Series, preferences: dict[str, Any]) -> float:
        dimension_scores: list[float] = []
        for key, field in (
            ("preferred_categories", "category"),
            ("preferred_colors", "color"),
            ("preferred_brands", "brand"),
        ):
            values = preferences[key]
            if values:
                dimension_scores.append(float(str(product[field]) in values))

        styles = preferences["preferred_styles"]
        if styles:
            suitable_categories = set().union(
                *(STYLE_CATEGORY_PROXIES[style] for style in styles)
            )
            dimension_scores.append(
                float(str(product["category"]) in suitable_categories)
            )
        minimum = preferences["preferred_price_min"]
        maximum = preferences["preferred_price_max"]
        if minimum is not None or maximum is not None:
            price = float(product["current_price"])
            lower = minimum if minimum is not None else 0.0
            upper = maximum if maximum is not None else max(
                float(product["original_price"]), lower + 50.0
            )
            if lower <= price <= upper:
                dimension_scores.append(1.0)
            else:
                distance = lower - price if price < lower else price - upper
                scale = max(upper - lower, 50.0)
                dimension_scores.append(max(0.0, 1.0 - distance / scale))
        return float(np.mean(dimension_scores)) if dimension_scores else 0.5

    def normalize_preferences(self, preferences: dict[str, Any]) -> dict[str, Any]:
        """Public profile validator shared by the recommendation route and profile store."""
        return self._normalize_preferences(preferences)

    @staticmethod
    def _preference_reasons(
        product: pd.Series, preferences: dict[str, Any]
    ) -> list[str]:
        reasons: list[str] = []
        for preference_key, field in PREFERENCE_TO_FIELD.items():
            preferred = preferences[preference_key]
            if preferred is not None and str(product[field]) == preferred:
                reasons.append(f"Matches your {field} preference ({preferred})")
        for key, field in (
            ("preferred_categories", "category"),
            ("preferred_colors", "color"),
            ("preferred_brands", "brand"),
        ):
            values = preferences[key]
            if values and str(product[field]) in values:
                reasons.append(f"Matches your {field} preference(s)")
        styles = preferences["preferred_styles"]
        if styles and str(product["category"]) in set().union(
            *(STYLE_CATEGORY_PROXIES[style] for style in styles)
        ):
            reasons.append(
                f"Category fits your {', '.join(styles)} style proxy preference"
            )
        body_shape = preferences["body_shape"]
        if body_shape is not None and str(product["category"]) in (
            BODY_SHAPE_CATEGORY_PROXIES[body_shape]
        ):
            reasons.append(
                f"Category fits the {body_shape} styling proxy"
            )
        minimum_price = preferences["preferred_price_min"]
        maximum_price = preferences["preferred_price_max"]
        if (minimum_price is not None or maximum_price is not None) and (
            minimum_price is None or product["current_price"] >= minimum_price
        ) and (maximum_price is None or product["current_price"] <= maximum_price):
            reasons.append("Within your preferred price range")
        gender = preferences["gender"]
        if gender is not None:
            product_gender = str(product["gender"])
            if product_gender == gender:
                reasons.append(f"Matches your gender preference ({gender})")
            elif product_gender == "Unisex" and gender != "Unisex":
                reasons.append(f"Unisex style suitable for your {gender} preference")
            elif gender == "Unisex":
                reasons.append("Included for your broad Unisex preference")
        rating = product["customer_rating"]
        if float(product["markdown_percentage"]) > 0:
            reasons.append(
                f"Discount contributes to ranking ({product['markdown_percentage']:.1f}% off)"
            )
        if pd.notna(rating) and rating >= 4:
            reasons.append(f"Strong customer rating ({rating:.1f}/5)")
        elif pd.isna(rating):
            reasons.append("Rating unavailable; catalog median used for ranking")
        return reasons[:5]

    def recommend(
        self,
        preferences: dict[str, Any],
        top_n: int = 10,
        weather_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to top_n available products, ordered by transparent relevance."""
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer")
        preferences = self._normalize_preferences(preferences)

        candidates = self.products["stock_quantity"] > 0
        if preferences["max_budget"] is not None:
            candidates &= self.products["current_price"] <= preferences["max_budget"]
        budget_safe_candidates = candidates.copy()
        fallback_minimum_rating = False
        if preferences["minimum_rating"] is not None:
            candidates &= self.products["customer_rating"].notna() & (
                self.products["customer_rating"] >= preferences["minimum_rating"]
            )
            if not candidates.any() and budget_safe_candidates.any():
                candidates = budget_safe_candidates
                fallback_minimum_rating = True

        candidate_positions = np.flatnonzero(candidates.to_numpy())
        if len(candidate_positions) == 0:
            return []

        profile_similarity = self._profile_similarity(preferences)
        weights: dict[str, float] = {
            "profile_similarity": PREFERENCE_WEIGHTS["profile_similarity"]
        }
        if preferences["gender"] is not None:
            weights["gender_compatibility"] = PREFERENCE_WEIGHTS[
                "gender_compatibility"
            ]
        weights["rating_quality"] = PREFERENCE_WEIGHTS["rating_quality"]
        if preferences["max_budget"] is not None:
            weights["budget_affordability"] = PREFERENCE_WEIGHTS[
                "budget_affordability"
            ]
        weights["discount_value"] = PREFERENCE_WEIGHTS["discount_value"]
        weight_total = sum(weights.values())

        ranked: list[dict[str, Any]] = []
        for position in candidate_positions:
            product = self.products.iloc[position]
            rating = product["customer_rating"]
            rating_score = (
                float(rating) / 5.0 if pd.notna(rating) else self._rating_imputation / 5.0
            )
            components = {
                "profile_similarity": float(profile_similarity[position]),
                "gender_compatibility": self._gender_compatibility(
                    str(product["gender"]), preferences["gender"]
                ),
                "rating_quality": rating_score,
                "budget_affordability": (
                    max(0.0, 1.0 - float(product["current_price"]) / preferences["max_budget"])
                    if preferences["max_budget"] not in (None, 0)
                    else 0.0
                ),
                "discount_value": float(product["markdown_percentage"]) / 100.0,
            }
            score = sum(weights[key] * components[key] for key in weights) / weight_total
            profile_dimensions_active = any(
                preferences[key]
                for key in (
                    "preferred_categories",
                    "preferred_colors",
                    "preferred_brands",
                    "preferred_styles",
                )
            ) or preferences["preferred_price_min"] is not None or preferences[
                "preferred_price_max"
            ] is not None
            if profile_dimensions_active:
                profile_fit = self._profile_fit(product, preferences)
                score = score * (1.0 - PROFILE_MATCH_WEIGHT) + (
                    profile_fit * PROFILE_MATCH_WEIGHT
                )
            else:
                profile_fit = None
            context_weight = (
                CONTEXT_SCORE_WEIGHT if weather_context is not None else 0.0
            )
            profile_weight = PROFILE_MATCH_WEIGHT if profile_dimensions_active else 0.0
            body_shape_active = preferences["body_shape"] is not None
            body_shape_fit = (
                float(
                    str(product["category"])
                    in BODY_SHAPE_CATEGORY_PROXIES[preferences["body_shape"]]
                )
                if body_shape_active
                else None
            )
            if body_shape_active:
                score = (
                    score * (1.0 - BODY_SHAPE_MATCH_WEIGHT)
                    + body_shape_fit * BODY_SHAPE_MATCH_WEIGHT
                )
            score_contributions = {
                key: round(
                    weights[key]
                    * components[key]
                    / weight_total
                    * (1.0 - profile_weight)
                    * (1.0 - (BODY_SHAPE_MATCH_WEIGHT if body_shape_active else 0.0))
                    * (1.0 - context_weight)
                    * 100,
                    2,
                )
                for key in weights
            }
            if profile_dimensions_active:
                score_contributions["profile_fit"] = round(
                    profile_fit
                    * profile_weight
                    * (1.0 - (BODY_SHAPE_MATCH_WEIGHT if body_shape_active else 0.0))
                    * (1.0 - context_weight)
                    * 100,
                    2,
                )
            if body_shape_active:
                score_contributions["body_shape_fit"] = round(
                    body_shape_fit
                    * BODY_SHAPE_MATCH_WEIGHT
                    * (1.0 - context_weight)
                    * 100,
                    2,
                )
            reasons = self._preference_reasons(product, preferences)
            if weather_context is not None:
                weather_score, weather_reasons = score_weather_context(
                    product,
                    weather_context,
                )
                score = (
                    score * (1.0 - CONTEXT_SCORE_WEIGHT)
                    + weather_score * CONTEXT_SCORE_WEIGHT
                )
                score_contributions["weather_context"] = round(
                    weather_score * context_weight * 100,
                    2,
                )
                reasons.append(
                    f"Weather-context proxy fit: {weather_score * 100:.1f}%"
                )
            else:
                weather_score = None
                weather_reasons = []

            final_score = round(score * 100, 2)
            contribution_delta = round(
                final_score - sum(score_contributions.values()),
                2,
            )
            if score_contributions:
                largest_component = max(
                    score_contributions,
                    key=score_contributions.get,
                )
                score_contributions[largest_component] = round(
                    score_contributions[largest_component] + contribution_delta,
                    2,
                )

            score_component_labels = {
                "profile_similarity": "Catalog preference similarity",
                "gender_compatibility": "Gender compatibility",
                "rating_quality": "Customer rating quality",
                "budget_affordability": "Budget affordability",
                "discount_value": "Discount value",
                "profile_fit": "Saved profile fit",
                "body_shape_fit": "Body-shape styling proxy fit",
                "weather_context": "Weather context fit",
            }
            score_explanations = []
            for component_name, contribution_points in score_contributions.items():
                if component_name in components:
                    component_score = components[component_name]
                elif component_name == "profile_fit":
                    component_score = profile_fit
                elif component_name == "body_shape_fit":
                    component_score = body_shape_fit
                else:
                    component_score = weather_score
                score_explanations.append(
                    {
                        "name": score_component_labels[component_name],
                        "score": round(float(component_score) * 100, 2),
                        "contribution_points": contribution_points,
                    }
                )
            score_explanations.sort(
                key=lambda factor: -factor["contribution_points"]
            )

            evidence = list(reasons)
            if pd.isna(rating) and not any(
                reason.startswith("Rating unavailable;") for reason in evidence
            ):
                evidence.append("Rating unavailable; catalog median used for ranking")
            eligibility_evidence = [
                f"In stock ({int(product['stock_quantity'])} available)"
            ]
            if preferences["max_budget"] is not None:
                eligibility_evidence.append(
                    f"Within maximum budget ({preferences['max_budget']:.2f})"
                )
            if preferences["minimum_rating"] is not None:
                if fallback_minimum_rating:
                    eligibility_evidence.append(
                        "Minimum rating filter relaxed: no in-stock product within "
                        "the budget met the threshold"
                    )
                else:
                    eligibility_evidence.append(
                        f"Meets minimum rating ({preferences['minimum_rating']:g}/5)"
                    )
            ranked.append(
                {
                    "product_id": str(product["product_id"]),
                    "category": str(product["category"]),
                    "brand": str(product["brand"]),
                    "season": str(product["season"]),
                    "gender": str(product["gender"]),
                    "location": str(product["location"]),
                    "color": str(product["color"]),
                    "size": str(product["size"]),
                    "original_price": float(product["original_price"]),
                    "current_price": float(product["current_price"]),
                    "discount_percentage": float(product["markdown_percentage"]),
                    "customer_rating": (
                        float(rating) if pd.notna(rating) else None
                    ),
                    "stock_quantity": int(product["stock_quantity"]),
                    "score": final_score,
                    "score_contributions": score_contributions,
                    "score_explanations": score_explanations,
                    "explanation_evidence": evidence,
                    "eligibility_evidence": eligibility_evidence,
                    "fallback_used": fallback_minimum_rating,
                    "weather_rationale": weather_reasons,
                    "weather_score": (
                        round(weather_score * 100, 2)
                        if weather_score is not None
                        else None
                    ),
                    "profile_fit": (
                        round(profile_fit * 100, 2)
                        if profile_fit is not None
                        else None
                    ),
                    "body_shape_fit": (
                        round(body_shape_fit * 100, 2)
                        if body_shape_fit is not None
                        else None
                    ),
                    "reasons": reasons,
                }
            )

        ranked.sort(
            key=lambda item: (
                -item["score"],
                -(item["customer_rating"] if item["customer_rating"] is not None else -1),
                item["current_price"],
                item["product_id"],
            )
        )
        return ranked[:top_n]
