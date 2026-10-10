# Context-aware weather scoring

## Weather source and fallback

The app uses Open-Meteo's public Geocoding API to resolve the entered city and
its Forecast API current-weather fields (`temperature_2m`,
`relative_humidity_2m`, and `weather_code`). These endpoints do not require an
API key. No credential is embedded in the application. If a future provider
requires an API key, read it from an environment variable and never commit it.

Weather results are cached per normalized city for 10 minutes. Geocoding results
are cached for 24 hours, and failed weather lookups for 60 seconds to avoid
repeating calls during an outage. The HTTP timeout is five seconds per call.
When the provider is unavailable or a city cannot be resolved, the request
continues with the normal preference/season rank and the page explains that
weather was omitted; no weather values are guessed.

Open-Meteo attribution is shown in the interface. API documentation:
[Forecast API](https://open-meteo.com/en/docs) and
[Geocoding API](https://open-meteo.com/en/docs/geocoding-api).

## Catalog limitations and proxy features

The supplied catalog has category and season but no garment weight, fabric,
breathability, insulation, or water-resistance fields. The rules below are
therefore category/season proxies only. They must not be described as detecting
that a product is actually lightweight, breathable, warm, or waterproof.

| Signal | Threshold/source | Higher-scoring catalog proxies | Reason |
|---|---|---|---|
| Hot | Current temperature >= 28 °C | Summer season; Tops, Dresses, and Bottoms | Seasonal/category association with warm-weather outfits |
| Cold | Current temperature <= 12 °C | Winter season and Outerwear | Outerwear and winter season are plausible cold-weather choices |
| Rain | Open-Meteo WMO precipitation/thunderstorm codes | Shoes and Outerwear | Relevant categories, but the catalog cannot establish water resistance |
| High humidity | Relative humidity >= 75% | Tops and Dresses | Lighter category proxy only; breathability is not measured |

Each active signal contributes a compatibility score in [0, 1]. The score for
one signal is the mean of a documented category match and season association.
When multiple signals apply, their scores are averaged. Neutral conditions
produce 0.5 and do not favor a product category. Thresholds are transparent
initial heuristic policy values, not values learned from user outcomes.

## Final ranking score

With live weather:

`final_score = 0.80 * preference_score + 0.20 * weather_context_score`

Without live weather:

`final_score = preference_score`

The existing preference score is the normalized weighted score from the
content-based engine. The UI displays the weather compatibility component and
the engine adds a reason for every applicable signal. The 20% context weight
limits weather's influence while allowing it to shift close recommendations.

## Validation

Tests use synthetic weather contexts to exercise hot, cold, rain, and high
humidity rules, plus mocked API responses for cache/failure behavior. These are
test scenarios, not observations or hard-coded production weather values.
