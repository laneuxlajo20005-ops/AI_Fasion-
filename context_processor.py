"""Current-weather lookup and transparent weather-to-fashion context rules."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CACHE_TTL_SECONDS = 600
GEOCODING_CACHE_TTL_SECONDS = 86400
FAILURE_CACHE_TTL_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 5
CONTEXT_SCORE_WEIGHT = 0.20

WMO_CONDITIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_RAIN_CODES = frozenset(
    set(range(51, 68)) | {80, 81, 82, 95, 96, 99}
)


@dataclass(frozen=True)
class WeatherLookup:
    """Weather result, with an explicit non-sensitive fallback message."""

    context: dict[str, Any] | None
    message: str | None = None
    source: str | None = None
    cached: bool = False


def _valid_city(city: str) -> str:
    normalized = " ".join(city.split())
    if not normalized:
        raise ValueError("Enter a city to use weather-aware recommendations")
    if len(normalized) > 100:
        raise ValueError("City name must be 100 characters or fewer")
    return normalized


def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        f"{url}?{urlencode(params)}",
        headers={"User-Agent": "FashionRecommendationProject/1.0"},
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Weather API returned an invalid response")
    return payload


class WeatherService:
    """Open-Meteo geocoding/current weather client with bounded TTL caches.

    Open-Meteo's public forecast/geocoding endpoints do not require an API key.
    If a different provider requiring credentials is configured in future, its
    key must be read from an environment variable rather than committed here.
    """

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._weather_cache: dict[str, tuple[float, WeatherLookup]] = {}
        self._geocoding_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def fetch_current(self, city: str) -> WeatherLookup:
        try:
            normalized_city = _valid_city(city)
        except ValueError as error:
            return WeatherLookup(None, str(error))

        cache_key = normalized_city.casefold()
        now = self._clock()
        with self._lock:
            cached = self._weather_cache.get(cache_key)
            if cached and cached[0] > now:
                result = cached[1]
                return WeatherLookup(
                    context=result.context,
                    message=result.message,
                    source=result.source,
                    cached=True,
                )
            if cached:
                del self._weather_cache[cache_key]

        try:
            place = self._geocode(normalized_city)
            payload = _request_json(
                FORECAST_URL,
                {
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,weather_code",
                    "timezone": "auto",
                    "forecast_days": 1,
                },
            )
            current = payload.get("current")
            if not isinstance(current, dict):
                raise ValueError("Weather API response has no current conditions")
            temperature = float(current["temperature_2m"])
            humidity = float(current["relative_humidity_2m"])
            weather_code = int(current["weather_code"])
            if (
                not math.isfinite(temperature)
                or not math.isfinite(humidity)
                or not 0 <= humidity <= 100
            ):
                raise ValueError("Weather API returned invalid current conditions")
            context = {
                "city": place["name"],
                "country": place.get("country", ""),
                "temperature_c": temperature,
                "humidity_percent": humidity,
                "weather_code": weather_code,
                "weather_condition": WMO_CONDITIONS.get(
                    weather_code, f"Weather code {weather_code}"
                ),
            }
            result = WeatherLookup(
                context=context,
                source="Open-Meteo current weather",
            )
            ttl = WEATHER_CACHE_TTL_SECONDS
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            result = WeatherLookup(
                context=None,
                message=(
                    "Live weather is unavailable "
                    f"({type(error).__name__}). Recommendations use your selected "
                    "fashion preferences and season without weather adjustments."
                ),
            )
            ttl = FAILURE_CACHE_TTL_SECONDS

        with self._lock:
            self._weather_cache[cache_key] = (self._clock() + ttl, result)
        return result

    def _geocode(self, city: str) -> dict[str, Any]:
        key = city.casefold()
        now = self._clock()
        with self._lock:
            cached = self._geocoding_cache.get(key)
            if cached and cached[0] > now:
                return cached[1]
            if cached:
                del self._geocoding_cache[key]

        payload = _request_json(
            GEOCODING_URL,
            {"name": city, "count": 1, "language": "en", "format": "json"},
        )
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError(f"Location {city!r} was not found by the weather service")
        place = results[0]
        latitude = float(place["latitude"])
        longitude = float(place["longitude"])
        if (
            not math.isfinite(latitude)
            or not math.isfinite(longitude)
            or not -90 <= latitude <= 90
            or not -180 <= longitude <= 180
        ):
            raise ValueError("Geocoding service returned invalid coordinates")
        normalized = {
            "name": str(place["name"]),
            "country": str(place.get("country", "")),
            "latitude": latitude,
            "longitude": longitude,
        }
        with self._lock:
            self._geocoding_cache[key] = (
                self._clock() + GEOCODING_CACHE_TTL_SECONDS,
                normalized,
            )
        return normalized


def _category_context_fit(category: str, season: str, signal: str) -> float:
    """Use catalog category/season only as explainable style proxies.

    No fabric, insulation, breathability, or waterproofness is present in the
    catalog; these rules must not be interpreted as product-material claims.
    """
    category = str(category)
    season = str(season)
    if signal == "hot":
        category_score = {
            "Tops": 1.0,
            "Dresses": 1.0,
            "Bottoms": 0.8,
            "Accessories": 0.7,
            "Shoes": 0.55,
            "Outerwear": 0.1,
        }.get(category, 0.5)
        season_score = {
            "Summer": 1.0,
            "Spring": 0.65,
            "Fall": 0.35,
            "Winter": 0.1,
        }.get(season, 0.5)
    elif signal == "cold":
        category_score = {
            "Outerwear": 1.0,
            "Bottoms": 0.65,
            "Shoes": 0.6,
            "Accessories": 0.55,
            "Tops": 0.4,
            "Dresses": 0.2,
        }.get(category, 0.5)
        season_score = {
            "Winter": 1.0,
            "Fall": 0.7,
            "Spring": 0.4,
            "Summer": 0.1,
        }.get(season, 0.5)
    elif signal == "rain":
        category_score = {
            "Shoes": 1.0,
            "Outerwear": 0.9,
            "Accessories": 0.45,
            "Bottoms": 0.4,
            "Tops": 0.35,
            "Dresses": 0.3,
        }.get(category, 0.5)
        season_score = 0.5
    elif signal == "humid":
        category_score = {
            "Tops": 0.85,
            "Dresses": 0.85,
            "Bottoms": 0.7,
            "Accessories": 0.55,
            "Shoes": 0.45,
            "Outerwear": 0.2,
        }.get(category, 0.5)
        season_score = 0.5
    else:
        return 0.5
    return (category_score + season_score) / 2


def score_weather_context(
    product: pd.Series | dict[str, Any], context: dict[str, Any]
) -> tuple[float, list[str]]:
    """Score a product from 0 to 1 and explain each active weather signal."""
    temperature = float(context["temperature_c"])
    humidity = float(context["humidity_percent"])
    weather_code = int(context["weather_code"])
    if (
        not math.isfinite(temperature)
        or not math.isfinite(humidity)
        or not 0 <= humidity <= 100
    ):
        raise ValueError("Weather context contains invalid temperature or humidity")

    signals: list[str] = []
    if temperature >= 28:
        signals.append("hot")
    if temperature <= 12:
        signals.append("cold")
    if weather_code in _RAIN_CODES:
        signals.append("rain")
    if humidity >= 75:
        signals.append("humid")

    if not signals:
        return 0.5, ["Weather is mild; no category-specific weather adjustment"]

    category = str(product["category"])
    season = str(product["season"])
    scores = [_category_context_fit(category, season, signal) for signal in signals]
    explanations = {
        "hot": (
            "Warm-weather proxy: favors Summer items and lighter catalog categories; "
            "fabric weight is not in the data"
        ),
        "cold": (
            "Cold-weather proxy: favors Winter items and Outerwear; insulation is not in the data"
        ),
        "rain": (
            "Rain proxy: favors Shoes and Outerwear categories; water resistance is not in the data"
        ),
        "humid": (
            "Humidity proxy: favors Tops and Dresses as lighter categories; breathability is not measured"
        ),
    }
    reasons = [explanations[signal] for signal in signals]
    return float(sum(scores) / len(scores)), reasons
