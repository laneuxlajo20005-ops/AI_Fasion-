# Multi-page recommendation experience

## Flow and route map

The web application now starts at `/` and redirects to a five-step guided flow:

| Step | Route | Purpose |
|---|---|---|
| 1. Personal details | `/personal-details` | Requires gender preference and city; optional display name and dataset-supported Urban/Rural catalog location |
| 2. Fashion preferences | `/preferences` | Optional season, categories, styles, colors, brands, preferred price range, budget ceiling, and minimum rating |
| 3. Photo analysis | `/photo-analysis` | Optional image upload or browser webcam capture; analyze, retry, remove, continue, or skip |
| 4. AI analysis | `/ai-analysis` | Checks weather, prepares the profile/context inputs, and runs the existing ranker without an artificial loading delay |
| 5. Recommendations | `/recommendations` | Shows the best match, up to ten ranked items, and evidence-based subsets |
| Reset | `/start-over` | Deletes the current wizard state and clears the browser's signed session identifier |

Existing `/predict` and `/api/profile` routes remain available for the previous
single-request form and anonymous profile API. The root route now opens the
guided flow.

## State, validation, and privacy

The browser's signed Flask cookie contains only a random wizard-session UUID.
The corresponding personal details, normalized preferences, body-shape proxy,
weather result, and recommendation list are stored in a `fashion_wizard_sessions`
table. Local development uses SQLite; production uses PostgreSQL. Wizard
records expire after 24 hours; expired records are removed during database
initialization, state reads, and state writes. Start Over deletes the current
record.

When `DATABASE_URL` is set, both `fashion_wizard_sessions` and named
`fashion_profiles` use PostgreSQL instead. Production requires this variable
and never falls back to SQLite. The schema is initialized using versioned,
transactional migrations. Local development and isolated tests may use SQLite.
The Vercel runtime requirements omit optional MediaPipe/OpenCV packages to fit
the serverless bundle limit; image analysis remains optional and unavailable
there. Install `requirements-full.txt` for the full local computer-vision flow.

The wizard does not ask for age, height, or body measurements. Display name is
optional and is only used in page copy. Gender and city are required because the
guided profile uses gender compatibility and requests city weather. The
dataset's `location` feature is separate from the city: the optional catalog
location choice uses only dataset values such as `Urban` and `Rural`.

Photo bytes are read and analyzed in memory. They are not written to a public
folder or to SQLite. The pose module returns an annotated preview for the
immediate analysis response; that preview is not included in persisted wizard
state. Only the limited estimate, measurements, explanation, and caveat remain
in the temporary workflow record. Uploads are restricted to PNG/JPG/JPEG/WEBP,
5 MB, and the existing decoder rejects unreadable or corrupt images. Webcam
capture uses browser `getUserMedia`; the camera is stopped when capture or form
submission completes. If camera permission is denied, image upload remains
available.

Set `FLASK_SECRET_KEY` to a random secret for sessions that need to survive an
application restart. If it is not configured, the development process creates
a random key on startup, so existing browser sessions are intentionally
invalidated when that process restarts. `FLASK_COOKIE_SECURE=1` should only be
used behind HTTPS; it prevents the cookie from being sent over local HTTP.

## Recommendation dashboard

All product cards are created from the returned catalog rows and the existing
`recommendation_engine.py` output. The UI does not create product photographs
or scores. It presents actual price fields as catalog price units because the
dataset does not establish a currency. Each card includes the engine's
relevance score, product attributes, discount/rating/stock fields, explanation
evidence, score contributions, and eligibility evidence.

The named dashboard sections are derived from the same top-ranked set:

- **Your best match:** first result from the deterministic score order.
- **Top recommendations:** up to ten ranked products, with browser-side filters
  for category, maximum price, and rating, and sorting by score, price, or rating.
- **Weather-based picks:** products with a non-null weather score, ordered by
  that score.
- **Preference matches:** returned items with a non-zero engine `profile_fit`
  when explicit category/color/brand/style/price preferences were selected.
- **Other ranked alternatives:** different categories within the returned set.
- **Budget-friendly picks:** lowest-priced returned items that satisfy the
  user's maximum budget, if set.

The engine's existing minimum-rating fallback is retained and disclosed. If no
in-stock item is within the chosen maximum budget, the guided flow performs a
second ranking without the budget/rating thresholds and labels the results as
above-budget alternatives. It does not describe those fallback items as
budget-compliant. When no preferences or weather data exist, corresponding
sections explain that instead of asserting unsupported reasons.

## Reproduction and verification

From the project directory in PowerShell:

```powershell
$env:FLASK_SECRET_KEY = (python -c "import secrets; print(secrets.token_hex(32))")
python market.py
```

Open `http://127.0.0.1:5000/`. The current camera path requires a browser with
camera support and permission. The automated suite validates upload handling,
mocked pose results and failures, camera-image payload processing, skip, reset,
server-side state persistence, weather fallback/context, and the legacy route:

```powershell
python -m unittest tests.test_wizard_flow -v
python -m unittest discover -s tests -v
```

Automated webcam tests submit the same image payload produced by the browser;
they do not prove that a physical camera is available or that a particular
browser/device grants access. No successful real-person body-shape result or
user-outcome validation is claimed.
