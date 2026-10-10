import base64
import binascii
import json
import os
import secrets
import uuid
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename
import pandas as pd

from recommendation_engine import FashionRecommendationEngine
from fashion_preprocessing import validate_and_normalize
from context_processor import WeatherService
from database import DatabaseUnavailable, database_url_from_environment
from profile_store import ProfileStore
from wizard_store import WizardStore
from body_shape import (
    MAX_IMAGE_BYTES,
    BodyShapeError,
    BodyShapeEstimator,
    BodyShapeUnavailable,
)

IS_PRODUCTION = (
    os.environ.get("VERCEL") == "1"
    or os.environ.get("APP_ENV", "").strip().lower() == "production"
)
DATABASE_URL = database_url_from_environment()
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "").strip() or None
if IS_PRODUCTION and not FLASK_SECRET_KEY:
    raise RuntimeError(
        "FLASK_SECRET_KEY is required in production to preserve signed sessions."
    )

app = Flask(__name__)
app.config.update(
    SECRET_KEY=FLASK_SECRET_KEY or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=8 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_COOKIE_SECURE") == "1",
)

DATA_PATH = Path(__file__).resolve().parent / "fashion_boutique_fixed_location.csv"
EVALUATION_PATH = Path(__file__).resolve().parent / "data" / "final_evaluation"
CATALOG_LOAD_ERROR = None
try:
    df = validate_and_normalize(pd.read_csv(DATA_PATH))
except FileNotFoundError:
    df = pd.DataFrame()
    CATALOG_LOAD_ERROR = (
        f"The required catalog file '{DATA_PATH.name}' is missing. "
        "Restore the dataset and restart the application."
    )
recommendation_engine = (
    FashionRecommendationEngine(df) if CATALOG_LOAD_ERROR is None else None
)
weather_service = WeatherService()
DATABASE_PATH = Path(__file__).resolve().parent / "instance" / "fashion_profiles.sqlite3"
profile_store = ProfileStore(DATABASE_PATH, database_url=DATABASE_URL)
wizard_store = WizardStore(DATABASE_PATH, database_url=DATABASE_URL)
body_shape_estimator = BodyShapeEstimator()
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@app.errorhandler(DatabaseUnavailable)
def database_unavailable(_error):
    message = "The persistent database is temporarily unavailable. Please retry shortly."
    app.logger.error("A database operation failed; returning HTTP 503.")
    if request.path.startswith("/api/"):
        return jsonify(error=message), 503
    return render_template("service_unavailable.html", message=message), 503


@app.before_request
def require_catalog():
    if CATALOG_LOAD_ERROR and request.endpoint != "static":
        return render_template(
            "service_unavailable.html",
            message=CATALOG_LOAD_ERROR,
        ), 503


def _form_preferences() -> dict:
    return {
        "gender": request.form.get("gender"),
        "location": request.form.get("location"),
        "season": request.form.get("season"),
        "preferred_category": request.form.get("preferred_category"),
        "preferred_color": request.form.get("preferred_color"),
        "preferred_brand": request.form.get("preferred_brand"),
        "preferred_size": request.form.get("preferred_size"),
        "preferred_categories": request.form.getlist("preferred_categories"),
        "preferred_colors": request.form.getlist("preferred_colors"),
        "preferred_brands": request.form.getlist("preferred_brands"),
        "preferred_styles": request.form.getlist("preferred_styles"),
        "preferred_price_min": request.form.get("preferred_price_min"),
        "preferred_price_max": request.form.get("preferred_price_max"),
        "max_budget": request.form.get("max_budget"),
        "minimum_rating": request.form.get("minimum_rating"),
        "body_shape": request.form.get("body_shape"),
        "weather_location": request.form.get("weather_location", ""),
    }


def _get_wizard_state() -> tuple[str, dict]:
    session_id = session.get("wizard_session_id")
    state = None
    if isinstance(session_id, str):
        try:
            state = wizard_store.get(session_id)
        except ValueError:
            state = None
    if state is None:
        session_id = wizard_store.create()
        state = {}
        session["wizard_session_id"] = session_id
    return session_id, state


def _save_wizard_state(session_id: str, state: dict) -> None:
    wizard_store.save(session_id, state)


def _clear_analysis(state: dict) -> None:
    for key in (
        "weather_checked",
        "weather_context",
        "weather_message",
        "weather_cached",
        "recommendations",
        "recommendation_fallback",
        "analysis_ready",
    ):
        state.pop(key, None)


def _render_wizard_page(
    template_name: str,
    page_number: int,
    state: dict,
    **context,
):
    return render_template(
        template_name,
        page_number=page_number,
        personal=state.get("personal", {}),
        preferences=state.get("preferences", {}),
        body_estimate=state.get("body_estimate"),
        options=recommendation_engine.form_options(),
        **context,
    )


def _wizard_redirect_for_missing_step(state: dict, require_preferences: bool = True):
    if not state.get("personal"):
        return redirect(url_for("personal_details"))
    if require_preferences and not state.get("preferences"):
        return redirect(url_for("fashion_preferences"))
    return None


def _clear_body_estimate(state: dict) -> None:
    state["body_estimate"] = None
    preferences = dict(state.get("preferences", {}))
    preferences["body_shape"] = None
    try:
        state["preferences"] = recommendation_engine.normalize_preferences(
            preferences
        )
    except ValueError:
        preferences.pop("body_shape", None)
        state["preferences"] = recommendation_engine.normalize_preferences(
            preferences
        )
    _clear_analysis(state)


def _read_optional_body_image() -> bytes | None:
    upload = request.files.get("body_image")
    camera_data = request.form.get("camera_image", "").strip()
    has_upload = upload is not None and bool(upload.filename)
    if has_upload and camera_data:
        raise BodyShapeError("Choose either an uploaded image or a webcam capture, not both")
    if has_upload:
        safe_filename = secure_filename(upload.filename)
        extension = Path(safe_filename).suffix.lower()
        if extension not in ALLOWED_IMAGE_EXTENSIONS:
            raise BodyShapeError("Use a PNG, JPG, JPEG, or WEBP image")
        image_bytes = upload.stream.read(MAX_IMAGE_BYTES + 1)
    elif camera_data:
        prefix = "data:image/jpeg;base64,"
        if not camera_data.startswith(prefix):
            raise BodyShapeError("Webcam image format is invalid")
        encoded = camera_data[len(prefix):]
        if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
            raise BodyShapeError("Image must be 5 MB or smaller")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise BodyShapeError("Webcam image data is invalid") from error
    else:
        return None
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise BodyShapeError("Image must be 5 MB or smaller")
    if not image_bytes:
        raise BodyShapeError("The selected image is empty")
    return image_bytes


@app.get("/api/profile/<profile_id>")
def get_profile(profile_id: str):
    try:
        profile = profile_store.get(profile_id)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    if profile is None:
        return jsonify({"error": "Profile not found"}), 404
    return jsonify(profile)


@app.post("/api/profile")
def save_profile():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON profile document is required"}), 400
    try:
        profile_id = payload.get("profile_id")
        if not isinstance(profile_id, str):
            raise ValueError("Profile identifier is invalid")
        raw_preferences = payload.get("preferences", {})
        if not isinstance(raw_preferences, dict):
            raise ValueError("Profile preferences must be an object")
        preferences = recommendation_engine.normalize_preferences(raw_preferences)
        weather_location = raw_preferences.get("weather_location", "")
        if weather_location is None:
            weather_location = ""
        if not isinstance(weather_location, str):
            raise ValueError("Weather location must be text")
        weather_location = " ".join(weather_location.split())
        if len(weather_location) > 100:
            raise ValueError("Weather location must be 100 characters or fewer")
        preferences["weather_location"] = weather_location or None
        profile = profile_store.save(
            profile_id,
            payload.get("profile_name", ""),
            preferences,
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(profile), 200


@app.get("/")
def home():
    return render_template("home.html")


@app.get("/context")
def fashion_context():
    _, state = _get_wizard_state()
    personal = state.get("personal", {})
    preferences = state.get("preferences", {})
    weather_checked = state.get("weather_checked", False)
    weather_message = state.get("weather_message")
    if not weather_checked:
        weather_message = (
            "Complete your profile and continue to AI analysis to check current "
            "weather for your selected city."
        )
    return render_template(
        "context.html",
        personal=personal,
        preferences=preferences,
        weather_context=state.get("weather_context"),
        weather_message=weather_message,
        weather_cached=state.get("weather_cached", False),
    )


@app.get("/evaluation")
def evaluation_page():
    report_file = EVALUATION_PATH / "final_evaluation.json"
    ranking_file = EVALUATION_PATH / "ranking_metrics.csv"
    classification_file = EVALUATION_PATH / "classification_metrics.csv"
    latency_file = EVALUATION_PATH / "response_time_metrics.csv"
    evaluation_available = all(
        path.is_file()
        for path in (
            report_file,
            ranking_file,
            classification_file,
            latency_file,
        )
    )
    report = None
    ranking_metrics = []
    classification_metrics = []
    response_time_metrics = []
    if evaluation_available:
        with report_file.open(encoding="utf-8") as report_stream:
            report = json.load(report_stream)
        ranking_metrics = pd.read_csv(ranking_file).to_dict(orient="records")
        classification_metrics = pd.read_csv(classification_file).to_dict(
            orient="records"
        )
        response_time_metrics = pd.read_csv(latency_file).to_dict(orient="records")
    return render_template(
        "evaluation.html",
        evaluation_available=evaluation_available,
        report=report,
        ranking_metrics=ranking_metrics,
        classification_metrics=classification_metrics,
        response_time_metrics=response_time_metrics,
    )


@app.get("/about")
def about():
    return render_template("about.html")


@app.route("/personal-details", methods=["GET", "POST"])
def personal_details():
    session_id, state = _get_wizard_state()
    error = None
    values = dict(state.get("personal", {}))
    if request.method == "POST":
        name = " ".join(request.form.get("name", "").split())
        gender = request.form.get("gender", "").strip()
        city = " ".join(request.form.get("city", "").split())
        catalog_location = request.form.get("catalog_location", "").strip()
        if len(name) > 60:
            error = "Name must be 60 characters or fewer."
        elif not gender:
            error = "Select a gender preference to continue."
        elif gender not in recommendation_engine.gender_values:
            error = "Choose a gender option shown on this page."
        elif not city:
            error = "Enter a city so current weather can be checked."
        elif len(city) > 100:
            error = "City must be 100 characters or fewer."
        elif catalog_location and catalog_location not in recommendation_engine.form_options()["locations"]:
            error = "Choose a catalog location option shown on this page."
        else:
            new_values = {
                "name": name,
                "gender": gender,
                "city": city,
                "catalog_location": catalog_location,
            }
            if values != new_values:
                state["preferences"] = {}
                state["body_estimate"] = None
                _clear_analysis(state)
            state["personal"] = new_values
            _save_wizard_state(session_id, state)
            return redirect(url_for("fashion_preferences"))
        values = {
            "name": name,
            "gender": gender,
            "city": city,
            "catalog_location": catalog_location,
        }
    return _render_wizard_page(
        "personal_details.html",
        1,
        state,
        values=values,
        error=error,
    )


@app.route("/preferences", methods=["GET", "POST"])
def fashion_preferences():
    session_id, state = _get_wizard_state()
    missing_step = _wizard_redirect_for_missing_step(
        state,
        require_preferences=False,
    )
    if missing_step:
        return missing_step
    error = None
    values = dict(state.get("preferences", {}))
    if request.method == "POST":
        raw_preferences = {
            "gender": state["personal"]["gender"],
            "location": state["personal"].get("catalog_location") or None,
            "season": request.form.get("season"),
            "preferred_categories": request.form.getlist("preferred_categories"),
            "preferred_colors": request.form.getlist("preferred_colors"),
            "preferred_brands": request.form.getlist("preferred_brands"),
            "preferred_styles": request.form.getlist("preferred_styles"),
            "preferred_price_min": request.form.get("preferred_price_min"),
            "preferred_price_max": request.form.get("preferred_price_max"),
            "max_budget": request.form.get("max_budget"),
            "minimum_rating": request.form.get("minimum_rating"),
            "body_shape": (
                state.get("body_estimate") or {}
            ).get("category"),
        }
        try:
            normalized = recommendation_engine.normalize_preferences(raw_preferences)
        except ValueError as validation_error:
            error = str(validation_error)
            values = raw_preferences
        else:
            state["preferences"] = normalized
            _clear_analysis(state)
            _save_wizard_state(session_id, state)
            return redirect(url_for("photo_analysis"))
    return _render_wizard_page(
        "preferences.html",
        2,
        state,
        values=values,
        error=error,
    )


@app.route("/photo-analysis", methods=["GET", "POST"])
def photo_analysis():
    session_id, state = _get_wizard_state()
    missing_step = _wizard_redirect_for_missing_step(state)
    if missing_step:
        return missing_step
    error = None
    transient_estimate = None
    if request.method == "POST":
        action = request.form.get("action", "analyze")
        if action in {"skip", "remove"}:
            _clear_body_estimate(state)
            _save_wizard_state(session_id, state)
            if action == "skip":
                return redirect(url_for("ai_analysis"))
            return redirect(url_for("photo_analysis"))
        if action == "continue":
            return redirect(url_for("ai_analysis"))
        if action != "analyze":
            error = "Choose Analyze, Continue, or Skip Photo Analysis."
        else:
            try:
                image_bytes = _read_optional_body_image()
                if image_bytes is None:
                    raise BodyShapeError("Choose an image to analyze or skip photo analysis.")
                transient_estimate = body_shape_estimator.estimate(image_bytes).as_dict()
            except (BodyShapeError, BodyShapeUnavailable) as image_error:
                error = (
                    "We couldn't detect a full-body pose in this image. Please try another photo."
                    if "No person pose was detected" in str(image_error)
                    else str(image_error)
                )
            else:
                state["body_estimate"] = {
                    key: value
                    for key, value in transient_estimate.items()
                    if key != "landmark_image_data_uri"
                }
                preferences = dict(state["preferences"])
                preferences["body_shape"] = transient_estimate["category"]
                state["preferences"] = recommendation_engine.normalize_preferences(
                    preferences
                )
                _clear_analysis(state)
                _save_wizard_state(session_id, state)
    return _render_wizard_page(
        "photo_analysis.html",
        3,
        state,
        error=error,
        transient_estimate=transient_estimate,
    )


@app.get("/ai-analysis")
def ai_analysis():
    session_id, state = _get_wizard_state()
    missing_step = _wizard_redirect_for_missing_step(state)
    if missing_step:
        return missing_step
    if not state.get("analysis_ready"):
        personal = state["personal"]
        weather_lookup = weather_service.fetch_current(personal["city"])
        preferences = dict(state["preferences"])
        if state.get("body_estimate"):
            preferences["body_shape"] = state["body_estimate"]["category"]
        recommendations = recommendation_engine.recommend(
            preferences,
            top_n=10,
            weather_context=weather_lookup.context,
        )
        fallback_message = None
        if not recommendations:
            fallback_preferences = dict(preferences)
            fallback_preferences["max_budget"] = None
            fallback_preferences["minimum_rating"] = None
            recommendations = recommendation_engine.recommend(
                fallback_preferences,
                top_n=10,
                weather_context=weather_lookup.context,
            )
            if recommendations:
                fallback_message = (
                    "No in-stock products were available within your maximum budget. "
                    "Showing ranked in-stock catalog alternatives; prices may exceed "
                    "your maximum budget."
                )
        state.update(
            {
                "weather_checked": True,
                "weather_context": weather_lookup.context,
                "weather_message": weather_lookup.message,
                "weather_cached": weather_lookup.cached,
                "recommendations": recommendations,
                "recommendation_fallback": fallback_message,
                "analysis_ready": True,
            }
        )
        _save_wizard_state(session_id, state)
    return _render_wizard_page(
        "ai_analysis.html",
        4,
        state,
        weather_context=state.get("weather_context"),
        weather_message=state.get("weather_message"),
        weather_cached=state.get("weather_cached", False),
        recommendations=state.get("recommendations", []),
        recommendation_fallback=state.get("recommendation_fallback"),
    )


@app.get("/recommendations")
def recommendations():
    _, state = _get_wizard_state()
    missing_step = _wizard_redirect_for_missing_step(state)
    if missing_step:
        return missing_step
    if not state.get("analysis_ready"):
        return redirect(url_for("ai_analysis"))
    products = state.get("recommendations", [])
    categories = state["preferences"].get("preferred_categories") or []
    has_preference_dimensions = any(
        state["preferences"].get(key)
        for key in (
            "preferred_categories",
            "preferred_colors",
            "preferred_brands",
            "preferred_styles",
            "preferred_price_min",
            "preferred_price_max",
        )
    )
    if categories:
        alternatives = [
            product for product in products if product["category"] not in categories
        ]
    elif products:
        alternatives = [
            product for product in products if product["category"] != products[0]["category"]
        ]
    else:
        alternatives = []
    weather_picks = sorted(
        (product for product in products if product["weather_score"] is not None),
        key=lambda product: product["weather_score"],
        reverse=True,
    )[:4]
    preference_picks = (
        sorted(
            (
                product
                for product in products
                if product["profile_fit"] is not None and product["profile_fit"] > 0
            ),
            key=lambda product: product["profile_fit"],
            reverse=True,
        )[:4]
        if has_preference_dimensions
        else []
    )
    budget_candidates = [
        product
        for product in products
        if state["preferences"].get("max_budget") is None
        or product["current_price"] <= state["preferences"]["max_budget"]
    ]
    budget_picks = sorted(
        budget_candidates,
        key=lambda product: (
            product["current_price"],
            -product["score"],
            product["product_id"],
        ),
    )[:4]
    return _render_wizard_page(
        "recommendations.html",
        5,
        state,
        products=products,
        best_match=products[0] if products else None,
        weather_picks=weather_picks,
        preference_picks=preference_picks,
        alternatives=alternatives[:4],
        budget_picks=budget_picks,
        recommendation_fallback=state.get("recommendation_fallback"),
        weather_context=state.get("weather_context"),
        weather_message=state.get("weather_message"),
        weather_cached=state.get("weather_cached", False),
    )


@app.route("/start-over", methods=["GET", "POST"])
def start_over():
    session_id = session.get("wizard_session_id")
    if isinstance(session_id, str):
        try:
            wizard_store.delete(session_id)
        except ValueError:
            pass
    session.clear()
    return redirect(url_for("personal_details"))


@app.route("/predict", methods=["POST"])
def predict():
    preferences = _form_preferences()
    profile_id = request.form.get("profile_id", "")
    profile_name = request.form.get("profile_name", "")
    weather_lookup = None
    weather_context = None
    body_estimate = None
    body_shape_message = None
    weather_location = preferences["weather_location"].strip()
    try:
        try:
            body_image = _read_optional_body_image()
            if body_image is not None:
                preferences["body_shape"] = ""
                body_estimate = body_shape_estimator.estimate(body_image).as_dict()
                preferences["body_shape"] = body_estimate["category"]
        except (BodyShapeError, BodyShapeUnavailable) as error:
            body_shape_message = str(error)
        normalized_preferences = recommendation_engine.normalize_preferences(
            preferences
        )
        if profile_id:
            profile_store.save(profile_id, profile_name, normalized_preferences)
        if weather_location:
            weather_lookup = weather_service.fetch_current(weather_location)
            weather_context = weather_lookup.context
        ranked_products = recommendation_engine.recommend(
            normalized_preferences,
            top_n=10,
            weather_context=weather_context,
        )
    except ValueError as error:
        return (
            render_template(
                "market.html",
                options=recommendation_engine.form_options(),
                preferences=preferences,
                profile_id=profile_id,
                error=str(error),
                body_estimate=body_estimate,
                body_shape_message=body_shape_message,
                weather_context=weather_context,
                weather_message=(
                    weather_lookup.message if weather_lookup is not None else None
                ),
            ),
            400,
        )

    return render_template(
        "market.html",
        options=recommendation_engine.form_options(),
        preferences=preferences,
        profile_id=profile_id,
        recommendations=ranked_products,
        searched=True,
        weather_context=weather_context,
        weather_message=(
            weather_lookup.message if weather_lookup is not None else None
        ),
        weather_cached=weather_lookup.cached if weather_lookup is not None else False,
        body_estimate=body_estimate,
        body_shape_message=body_shape_message,
    )


@app.errorhandler(413)
def request_too_large(_error):
    if CATALOG_LOAD_ERROR:
        return render_template(
            "service_unavailable.html",
            message=CATALOG_LOAD_ERROR,
        ), 503
    if request.endpoint == "predict":
        return (
            render_template(
                "market.html",
                options=recommendation_engine.form_options(),
                preferences={},
                error="Request is too large. Use one image no larger than 5 MB.",
            ),
            413,
        )
    _, state = _get_wizard_state()
    return (
        _render_wizard_page(
            "photo_analysis.html",
            3,
            state,
            error="The image upload is too large. Use a PNG, JPG, JPEG, or WEBP image no larger than 5 MB.",
            transient_estimate=None,
        ),
        413,
    )


if __name__ == "__main__":
    app.run(debug=False)
