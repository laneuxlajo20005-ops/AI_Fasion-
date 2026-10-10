# Hybrid Multimodal Context-Aware Personalized Fashion Recommendation System Using Machine Learning and Computer Vision

## Project documentation for the MSc Applied Data Science PG mini-project

**Implementation status key**

- **Implemented:** present in the Flask application and/or reusable project code.
- **Measured:** supported by the completed offline experiment described below.
- **Experimental / heuristic:** implemented as a prototype rule, but not validated against user outcomes or ground truth.
- **Future work:** proposed only; not represented as an implemented feature.

The title describes the project's direction. It does not imply that every modality
or recommendation feature has been experimentally validated. In particular, the
catalog has no interaction history, and the weather and pose-based styling
components have no outcome labels.

## 1. Abstract

This project upgrades an existing Flask fashion application from a
Random-Forest-category-prediction and random-product-selection workflow to a
modular fashion recommendation prototype. The implemented application accepts
catalog preferences and optional saved profile preferences, filters unavailable
products, and ranks in-stock products using a deterministic content-based score.
Optional context can include current weather obtained from Open-Meteo. An
optional OpenCV and MediaPipe Pose pipeline estimates a small set of 2D
proportions and maps them to a limited styling-proxy category. Each product
recommendation exposes its relevance score, score contributions, product
attributes, and evidence-based explanation.

The catalog contains 2,176 product records and product attributes, but no
customer identifiers, impression logs, click/purchase histories linked to
customers, or explicit relevance judgments. Therefore, genuine user-level
recommendation effectiveness cannot be measured. A reproducible 80/20
stratified hold-out experiment was conducted. A re-created Random Forest using
gender, location, and season achieved 0.5229 accuracy, 0.5226 macro precision,
0.5409 macro recall, and 0.5095 macro F1 for category prediction. A separate
catalog-proxy ranking experiment over 436 held-out product-row pseudo-profiles
showed higher Precision@K and NDCG@K for the improved ranker than for a
predicted-category plus random-selection baseline. Because proxy relevance is
constructed from attributes also used by the improved ranker, these results are
optimistic retrieval-proxy results, not evidence of user satisfaction.

In a local in-process timing experiment, median top-10 processing time was
23.144 ms for the re-created baseline path and 604.826 ms for the improved
content ranker. The latter has greater computational cost. Weather and
body-shape-aware ranking are implemented as transparent heuristics but have not
been evaluated for real-world effectiveness. The project demonstrates a
working, explainable prototype and a reproducible experimental methodology,
while identifying the additional data required for a user-grounded study.

## 2. Introduction

Fashion catalogs contain descriptive fields such as category, brand, season,
color, size, price, rating, availability, gender, and location. A
recommendation system can use these attributes to retrieve items relevant to a
request, rather than merely predicting one category and selecting products
without a relevance-based order.

The starting application used a Random Forest classifier to predict a category
from gender, location, and season. It then selected products from the predicted
category randomly. That design produced category-level suggestions but did not
rank products using the wider set of available attributes or persist user
preferences.

The upgraded system preserves the Flask application and catalog while adding
content-based ranking, optional anonymous profiles, current weather lookup,
optional pose-based styling proxies, score explanations, and an offline
evaluation framework. It is a hybrid in the sense that a deterministic
content-based ranker is combined with explicit profile, context, and optional
vision-derived preference signals. It is not a collaborative filtering system
and does not learn from user-item interactions.

## 3. Problem Statement

The original workflow predicts a single fashion category from a limited set of
inputs and then returns randomly selected products. It does not provide a
consistent ranking based on brand, color, size, price, discount, rating,
availability, saved preferences, live weather, or optional visual styling
information.

The project addresses this engineering problem: **How can the existing Flask
application be incrementally improved to rank available catalog products
according to explicit user preferences and optional context, while keeping the
ranking explainable and evaluating it without inventing interaction data?**

The data does not support the stronger claim that recommendations are preferred
by customers. This project therefore distinguishes a functioning
recommendation prototype from proof of customer-level effectiveness.

## 4. Motivation

The work is motivated by four practical concerns:

1. **Relevance:** random selection cannot consistently prioritize a product
   that matches more of the supplied preferences.
2. **Personalization:** reusable preference profiles support repeatable
   demonstrations of different recommendation outcomes.
3. **Context:** current weather can adjust ranking through transparent
   category-and-season associations where relevant catalog information exists.
4. **Accountability:** visible score components and limitations make the
   recommendations easier to explain during a technical demonstration or viva.

The project also addresses an important research-method concern: recommendation
metrics require relevance evidence. Where the dataset does not contain customer
responses, an attribute proxy can be used for experimentation only if it is
clearly labeled and its relationship to the algorithm is disclosed.

## 5. Objectives

### Achieved objectives

- Preserve and extend the existing Flask application.
- Validate the supplied product catalog and create reusable preprocessing
  artifacts.
- Replace the application's random product selection with a deterministic
  top-10 content-based ranking engine.
- Accept category, color, brand, size, gender, location, season, budget, rating,
  price-range, and style-proxy preferences.
- Support anonymous preference profiles persisted in SQLite.
- Retrieve live city weather from Open-Meteo, cache it, and omit weather
  scoring gracefully when lookup fails.
- Offer optional OpenCV and MediaPipe Pose-based styling estimates.
- Display product details, weighted score contributions, supporting evidence,
  and separate eligibility filters.
- Produce a reproducible offline classification, ranking-proxy, and local
  response-time experiment.

### Not achieved or not empirically established

- Learning recommendation preferences from clicks, purchases, or ratings
  linked to individual users.
- Demonstrating that recommendations improve customer satisfaction,
  conversion, or return rates.
- Validating weather recommendations against weather-linked outcomes.
- Validating the pose-based styling category against expert or user labels.
- Using computer-vision fashion-image similarity or a trained body-shape
  classifier.

## 6. Existing System

The audited original system was a Flask web application backed by a fashion
catalog CSV. Its recommendation flow was:

1. Collect gender, location, and season.
2. Encode those categorical inputs.
3. Use a Random Forest classifier to predict a fashion category.
4. Select product names from the predicted category at random.
5. Render the results in a Flask template.

The upgraded project no longer contains the historical trained Random Forest
artifact. For the final benchmark, the original design was re-created using the
documented inputs and a seeded 100-tree Random Forest. Results for this
re-creation must not be described as results from the original serialized
model.

## 7. Limitations of Existing System

- It used only a small subset of the catalog/user-request attributes for
  category prediction.
- Random product selection did not order candidate products by relevance.
- It did not retain reusable user preferences.
- It did not apply optional live weather or visual styling context.
- It did not return a score decomposition or explanations tied to product
  attributes.
- It had no interaction-based recommendation evaluation.
- The dataset itself does not provide the customer-event data needed to
  establish recommendation effectiveness.

## 8. Proposed System

The proposed and implemented application retains Flask and the existing
catalog. It combines:

- **Product content:** category, season, brand, color, size, location, gender,
  price, discount, customer rating, and stock.
- **Request preferences:** selected profile and catalog choices.
- **Saved profile:** a browser-held UUID referencing an anonymous SQLite
  preference record.
- **Optional current weather:** temperature, humidity, and weather code from
  Open-Meteo.
- **Optional pose-based estimate:** a limited style-proxy label from a
  transient image.
- **Ranking and explanation:** deterministic score, eligibility filters, and
  per-item explanation.

“Hybrid” describes the combination of content ranking with rule-based profile,
weather, and optional visual signals. There is no collaborative filtering,
learned ranking model, multimodal deep neural network, or fusion model trained
on user outcomes.

## 9. System Architecture

```text
Five-step browser wizard
  1. Personal details ───────┐
  2. Fashion preferences ────┼──> Signed cookie UUID + server-side SQLite state
  3. Optional image/webcam ──┘            │
       └── in-memory OpenCV/MediaPipe      │
  4. AI analysis <── Open-Meteo weather ───┤
       └── existing candidate filters and content-based scoring
  5. Recommendations <── ranked products, score factors, and evidence

Legacy single-request /predict and anonymous profile API remain available.
```

The system has distinct layers:

- **Presentation:** `templates/wizard_base.html` and the five wizard page
  templates provide responsive, step-by-step forms, optional upload/webcam
  controls, analysis summary, and recommendation sections. The existing
  `templates/market.html` remains for the legacy `/predict` route.
- **Web application:** `market.py`, with wizard and legacy routes, input
  validation, weather/profile/image integration, and template rendering.
- **Wizard state:** `wizard_store.py` stores a random session UUID's workflow in
  a 24-hour SQLite record; the browser cookie does not contain personal details.
- **Data and ranking:** `fashion_preprocessing.py` and
  `recommendation_engine.py`.
- **Context modules:** `context_processor.py` and `body_shape.py`.
- **Profile persistence:** `profile_store.py`, using SQLite.
- **Offline analysis:** `evaluation.py` and `final_evaluation.py`.

## 10. Data Flow

### Recommendation request

1. The user enters required gender and city plus an optional name and
   dataset-supported catalog location at `/personal-details`.
2. The user selects optional preferences at `/preferences`; Flask validates
   them against the catalog vocabulary and stores normalized values server-side.
3. At `/photo-analysis`, an optional upload or camera capture is decoded and
   analyzed in memory. Failed or skipped image analysis does not block the flow.
4. At `/ai-analysis`, Flask requests current city weather through Open-Meteo,
   applies its supported fallback if unavailable, then calls the existing ranker.
5. The ranker excludes out-of-stock products, products above the maximum budget,
   and products below a minimum rating when possible. Its existing explicit
   rating-only fallback remains within stock and budget constraints.
6. If no in-stock products are within the hard budget, the wizard can perform a
   second, clearly labeled above-budget fallback ranking rather than presenting
   an empty dashboard or claiming those items meet the budget.
7. The engine deterministically scores and returns up to ten products. The
   `/recommendations` dashboard uses those products for all sections; it does
   not create another ranking or fabricate product imagery.
8. Explanation evidence and score contributions are rendered from engine
   output. Browser controls only filter/reorder the displayed top-ten list.

### Offline evaluation

1. Validate all 2,176 product rows.
2. Split catalog rows into stratified 80% training and 20% held-out records
   using seed 42.
3. Train the re-created Random Forest only on training records.
4. Use held-out product rows as pseudo-profiles; they are not real users.
5. Rank only in-stock training-catalog items.
6. Score output using a documented attribute-match proxy derived from the
   held-out row and candidate item attributes.
7. Aggregate classification, ranking, and local timing results and write CSV,
   JSON, Markdown, and PNG artifacts.

The experimental split has zero product-ID overlap between training and test.

## 11. Dataset Description

The source file is `fashion_boutique_fixed_location.csv`. It has **2,176 rows
and 16 columns**:

| Column | Meaning/type |
|---|---|
| `product_id` | Unique product identifier |
| `category` | Product category; six observed categories |
| `brand` | Brand label |
| `season` | Fall, Spring, Summer, or Winter |
| `size` | Catalog size; some records are missing |
| `color` | Product color |
| `original_price` | Original price, numeric |
| `markdown_percentage` | Discount percentage, numeric |
| `current_price` | Current price, numeric |
| `purchase_date` | Date-like catalog transaction field |
| `stock_quantity` | Inventory quantity, numeric |
| `customer_rating` | Product rating from 1 to 5 when available |
| `is_returned` | Boolean-like return outcome |
| `return_reason` | Optional return-reason category |
| `gender` | Men, Unisex, or Women |
| `location` | Rural or Urban catalog value |

The table is a product/transaction-style dataset, not a user interaction
dataset. In particular, it has no customer identifier, session, impression,
click, recommendation display, or user-linked purchase history.

The recorded quality report identified 491 missing sizes, 362 missing ratings,
and 1,856 missing return reasons. It reported zero duplicate product IDs,
zero exact duplicate rows, and zero invalid purchase dates. There were 320
rows marked returned (14.71% of the catalog records); this is not a
recommendation outcome linked to a customer. The catalog is gender-imbalanced
(1,525 Unisex, 334 Men, 317 Women) and location-imbalanced (1,792 Rural, 384
Urban), which limits how broadly subgroup behavior can be interpreted.

## 12. Data Preprocessing

`validate_and_normalize` checks the required 16 columns, trims strings,
converts numeric fields and dates, normalizes boolean return values, verifies
allowed category/season/gender/location values, and rejects invalid numeric
ranges, duplicate IDs/rows, invalid discounts, and invalid ratings. It does
not silently remove outliers or fabricate missing values. Missing size,
rating, and return-reason values can remain where permitted by validation.

The preprocessing workflow (`prepare_data.py`) performs a seeded
category-stratified 80/20 split and fits the reusable `FashionFeatureEngineer`
and one-hot pipeline on training data only. Train-fitted original-price
tertiles define budget, mid-range, and premium segments. Unknown categories
are handled by `OneHotEncoder(handle_unknown="ignore")`.

It writes:

- engineered train/test CSVs;
- encoded `X_train` and `X_test`;
- separate `y_train` and `y_test` files containing ID and category;
- data-quality JSON report; and
- reusable `fashion_preprocessor.joblib`.

The preprocessing module does **not** train a predictive classifier.

## 13. Feature Engineering

Implemented features include:

- `discount_amount = original_price - current_price`;
- derived discount percentage;
- `price_segment`, with thresholds fitted on training data;
- `stock_level`: out-of-stock, low (1–10), medium (11–30), high (>30);
- `rating_category`: low (<3), average (3 to <4), high (>=4), or unrated;
- `purchase_month`, `purchase_day`, and `purchase_year`.

The reusable preprocessing model matrix includes:

- categorical: season, gender, location, brand, color, size, price segment,
  stock level;
- numeric: original price, current price, discount amount, discount
  percentage, and stock quantity.

For the preprocessing classification-style artifacts, target category,
product ID, date-derived fields, and post-purchase rating/return fields are
excluded from X. The final experimental Random Forest has its own smaller,
explicit feature set: gender, location, and season.

## 14. Machine Learning Methodology

### Random Forest classifier

The final experiment re-created the audited category-prediction design with
100 trees and one-hot encoding of gender, location, and season. It was trained
on the 1,740-row training split and evaluated on 436 held-out products. The
target was `category`.

Random Forest is an ensemble of decision trees. Each tree predicts a category,
and the forest combines tree outputs. It can model non-linear feature
interactions, but this dataset's three predictors are not sufficiently
informative to separate all categories reliably in the held-out evaluation.

### What is not a learned model

The improved recommendation score is deterministic. It is not trained on
clicks or purchases. The optional body-shape category is determined by
transparent geometric thresholds, not a supervised classifier; no labeled
body-shape training set exists.

## 15. Recommendation Algorithm

The engine normalizes preferences, constructs a weighted categorical
preference representation, uses cosine similarity for the active catalog
profile fields, and combines this with explicit numeric/compatibility
components. Gender compatibility is separately handled. Product rating,
discount, price affordability, profile fit, and optional context contribute
according to the implemented formula.

Base components and nominal weights when active:

| Component | Weight | Definition |
|---|---:|---|
| Profile/content similarity | 0.68 | Weighted one-hot cosine match across selected category, season, brand, color, size, and location fields |
| Gender compatibility | 0.10 | Compatibility rule; active only when a gender preference is supplied |
| Rating quality | 0.12 | Rating divided by five; missing catalog ratings use catalog median |
| Budget affordability | 0.08 | Relative budget headroom; active only when maximum budget is supplied |
| Discount value | 0.02 | Markdown percentage divided by 100 |

The active base weights are normalized. Stock, maximum budget, and minimum
rating can be hard eligibility filters: they remove candidates rather than
give ranking bonuses. The system sorts deterministically and returns at most
ten products.

If no product meets the requested minimum rating but budget-safe in-stock
products exist, the engine retries with only the rating threshold relaxed and
labels the fallback in the result. The budget and stock constraints remain
strict. If there is no in-stock product within budget, the system returns no
recommendations and displays an explanation rather than breaking the budget.

Where requested, subsequent blending is:

1. saved profile fit: 80% current score + 20% profile fit;
2. styling/body-shape proxy: 88% current score + 12% category-proxy fit;
3. live weather: 80% current score + 20% weather-context fit.

The score is a relative ranking score out of 100, **not a probability of
purchase or customer satisfaction**. Product data has no verified image
similarity or collaborative filtering component.

## 16. Context-Aware Recommendation

When a user enters a city, the application uses Open-Meteo Geocoding and
Forecast APIs to retrieve current temperature, humidity, and a weather code.
The implementation does not require an API key. Weather data is cached by
city for ten minutes, geocoding for 24 hours, and failures for 60 seconds;
requests use a five-second timeout. If lookup fails, the application explains
that weather was omitted and returns the normal ranking.

Weather scoring is a **heuristic category/season proxy**:

- hot (>= 28 °C): preference for Summer and Tops, Dresses, Bottoms;
- cold (<= 12 °C): preference for Winter and Outerwear;
- rain: preference for Shoes and Outerwear;
- humidity (>= 75%): preference for Tops and Dresses.

These rules do not establish that garments are lightweight, waterproof, warm,
or breathable. Catalog material and performance labels are absent. Unit tests
exercise synthetic hot/cold/rain/humidity contexts, and the weather service has
mocked cache/failure tests. These are software tests, not offline outcome
evaluation.

## 17. Personalization

The legacy profile API stores anonymous preference documents using a browser
UUID in SQLite at `instance/fashion_profiles.sqlite3`. The guided wizard uses a
separate random UUID held in a signed cookie and stores workflow details in the
server-side `fashion_wizard_sessions` table for up to 24 hours. Wizard data is
not automatically written to the persistent saved-profile table. No login,
identity verification, or multi-user authorization is implemented.

Supported preferences include gender, category, color, brand, size, season,
catalog location, preferred price range, maximum budget, minimum rating, and
style category proxies. Categories/colors/brands can be multi-selected.
Style labels map to documented category sets because the dataset has no style
annotations. Preference ranges and style labels therefore represent prototype
rules, not ground-truth fashion expertise.

Automated tests show distinct profile inputs can produce distinct rankings.
That is a functional check, not user-study evidence.

## 18. Computer Vision Module

An optional image upload or webcam capture is decoded with OpenCV and passed
to MediaPipe Pose. The pipeline checks landmark visibility and estimates
normalized 2D shoulder/hip widths and torso/leg proportions. Where available,
a segmentation-based width near the torso midpoint is only a silhouette proxy,
not a measured waist circumference.

Shoulder-to-hip width ratio thresholds map to:

- below 0.90: `Pear/triangle-style`;
- above 1.10: `Inverted-triangle-style`;
- between thresholds: `Balanced shoulder/hip-style`.

Only these styling-proxy labels are implemented. Hourglass and oval are not
estimated. The result may influence category ranking using an explicit proxy
mapping. It is not a medical, objective, or validated determination of body
type, garment fit, or suitability.

Images and measurements are processed transiently and are not persisted;
the selected proxy label may be saved in a profile. Image-based successful
estimation on a valid real user photo has not been established by the recorded
tests. Tests cover synthetic proportions and rejection of an image without a
detectable pose.

## 19. Explainability

Each recommendation shows:

- relevance score out of 100 (not a probability);
- category, brand, color, size, season, gender, location, price, discount,
  rating, and stock;
- raw component fit and its final-score contribution in points;
- evidence statements based on actual product/preference values; and
- eligibility constraints separately from ranking contributions.

Weather rationale identifies the rule/proxy used. The body-shape result is
labeled as a styling proxy. SHAP is not used because the recommendation
ranker is a deterministic formula rather than a learned ranking model; the
exact score decomposition is more direct.

## 20. System Implementation

| File/module | Role |
|---|---|
| `market.py` | Flask wizard and legacy routes; integration of catalog, profile, weather, pose, and recommendations |
| `wizard_store.py` | Short-lived server-side SQLite state for the guided workflow |
| `templates/wizard_base.html`, `personal_details.html`, `preferences.html`, `photo_analysis.html`, `ai_analysis.html`, `recommendations.html`, `_product_card.html` | Responsive five-step flow, photo controls, analysis summary, evidence-backed results sections |
| `templates/market.html` | Preserved legacy single-request page with product cards and explanations |
| `fashion_preprocessing.py` | Validation, quality analysis, feature engineering, reusable preprocessing pipeline |
| `prepare_data.py` | Train/test preparation artifact generation |
| `recommendation_engine.py` | Catalog preference validation, candidate filtering, scoring, top-N ranking, explanations |
| `context_processor.py` | Open-Meteo geocoding/weather access, caching, fallback, transparent proxy scoring |
| `profile_store.py` | SQLite profile validation and persistence |
| `body_shape.py` | Optional OpenCV/MediaPipe pose measurements and styling-proxy estimate |
| `evaluation.py` | Earlier simulated catalog proxy comparison to random-within-category |
| `final_evaluation.py` | Final re-created RF/random baseline comparison, rank metrics, local latency, graphs, report |

### Dependency map

| File | Function | Main dependencies | Interacts with | Potential issue / boundary |
|---|---|---|---|---|
| `market.py` | Flask wizard and legacy request validation, route orchestration, startup | Flask, pandas, project modules | wizard/profile DB, catalog, Open-Meteo, body estimator, ranker | Wizard cookie holds only a UUID; local server-side state expires after 24 hours |
| `wizard_store.py` | Temporary workflow state persistence | SQLite, JSON, UUID | wizard routes in `market.py` | Local demonstration state, no authentication |
| `templates/wizard_*.html`, `templates/recommendations.html` | Personal details, preferences, optional camera/upload, analysis and ranked results | Jinja, browser HTML/CSS/JS | Wizard routes and engine-produced result fields | Browser sorting/filtering applies only to returned items |
| `templates/market.html` | Legacy preference form, profile browser calls, upload/webcam, results and controls | Jinja, browser HTML/CSS/JS | `/predict`, profile API, rendered ranking data | Retained for compatibility |
| `fashion_preprocessing.py` | Dataset validation, quality report, feature engineering, reusable preprocessing | pandas, NumPy, scikit-learn | `prepare_data.py`, ranker, evaluation scripts | Validates required schema; permitted size/rating/reason missingness remains |
| `prepare_data.py` | Stratified preprocessing split and saved artifacts | pandas, joblib, scikit-learn | source CSV, preprocessor, `data/processed/` | Prepares model matrices but does not train/save a classifier |
| `recommendation_engine.py` | Candidate eligibility, similarity, scoring, top-N ordering, explanation | NumPy, pandas, scikit-learn, context processor | `market.py`, catalog, profile/shape/context signals | Heuristic weights; local scoring is slower than the random baseline |
| `context_processor.py` | Current-weather lookup, geocoding, cache, weather proxy | Python HTTP/JSON, pandas | Open-Meteo, ranker, Flask | Network can fail; fallback explicitly omits weather score |
| `body_shape.py` | Image decoding, pose landmarks, normalized proportions, styling proxy | OpenCV, MediaPipe, NumPy | Flask upload handler, ranker, template | Single-pose pipeline; no multi-person resolution or labeled validation |
| `profile_store.py` | Anonymous profile save/read | SQLite, Python JSON/UUID | Flask profile API, browser localStorage | UUID is a bearer identifier; no user authentication/authorization |
| `evaluation.py` | Initial random-within-category proxy experiment and metrics | pandas, NumPy, matplotlib, scikit-learn | catalog, ranker, `data/evaluation/` | Proxy labels overlap with features used by ranker |
| `final_evaluation.py` | Re-created Random Forest baseline, proxy rank comparison, timing, reports/plots | pandas, NumPy, scikit-learn, matplotlib, evaluation/ranker | CSV, classification pipeline, ranker, `data/final_evaluation/` | Historical RF artifact absent; context/vision effects not measured |
| `tests/*.py` | Unit and integration checks for the modules above | unittest, mocks, project runtime dependencies | Flask app and module APIs | Synthetic weather/pose checks validate code paths, not user effectiveness |
| `requirements.txt` | Declares direct runtime packages and compatible ranges | pip | all Python modules | Installed environment may have unrelated dependency conflicts |
| `fashion_boutique_fixed_location.csv` | Source catalog | pandas CSV reader | validation, Flask, preprocessing, evaluation | 2,176 rows; no user-item interactions or material/weather/body labels |

The final benchmark writes to `data/final_evaluation/`: classification and
ranking CSVs, response-time CSV, JSON metadata, Markdown report, ranking
metrics graph, classifier confusion matrix, and latency graph.

### Reproduction commands

From PowerShell in the project directory:

```powershell
Set-Location 'C:\Users\Administrator\Documents\market\market'
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:FLASK_SECRET_KEY = (python -c "import secrets; print(secrets.token_hex(32))")
python prepare_data.py
python final_evaluation.py
python -m unittest discover -s tests -v
python market.py
```

Python 3.11 is the tested recommendation for these pinned MediaPipe/OpenCV
versions. If PowerShell blocks activation, use
`Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate
again. The data-preparation command is optional for serving recommendations;
it creates reusable preprocessing artifacts but does not train a classifier.
The final evaluation command runs the documented catalog-proxy experiment and
writes its report/graphs. The test suite checks data preparation, ranking, API
fallback behavior, profile persistence, pose behavior, and Flask integration.
Weather uses Open-Meteo and does not require an API key or project environment
variable. SQLite is created at runtime under `instance/`. Open
`http://127.0.0.1:5000/` after starting the app. Webcam access requires a
supported browser context, generally localhost or HTTPS.
The `FLASK_SECRET_KEY` command above keeps the signed session identifier valid
across restarts in that PowerShell process; without the environment variable,
the local app creates an ephemeral key on startup. See
`docs/multi-page-user-flow.md` for route-by-route behavior and photo handling.

## 21. Experimental Setup

- Dataset: 2,176 product records.
- Split: stratified 80/20 by category; seed 42.
- Train/test: 1,740/436, with zero product-ID overlap.
- Candidate inventory: 1,705 in-stock training products.
- Baseline: re-created 100-tree Random Forest classifier using gender,
  location, and season, followed by seeded random selection in the predicted
  category.
- Improved: current content-based engine ranks products using preference values
  drawn from each held-out catalog row.
- Ranking profiles: 436 held-out product rows treated as pseudo-profiles.
- Timing: warmed local in-process measurement over those 436 profiles, one
  pass. HTTP, weather network, database, image, MediaPipe, and browser costs
  are excluded.

The historic serialized Random Forest model is not present in the upgraded
project. The evaluated classifier is a re-implementation of the audited
baseline design, not a measurement of that historical artifact.

## 22. Evaluation Metrics

### Classification

- Accuracy: fraction of held-out product categories classified correctly.
- Macro precision, recall, and F1: calculate per-category results, then give
  each category equal weight.

These apply to the Random Forest category classifier only. They are not
applicable to the improved ranker, which does not predict one category label.

### Ranking

Proxy relevance grade gives category a weight of two and season,
gender-compatibility, location, color, brand, and size one each. Binary
relevance uses grade >= 4. NDCG uses the graded relevance. Precision@K,
Recall@K, F1@K, NDCG@K, and Hit Rate@K are macro-averaged over pseudo-profiles;
F1 is computed per pseudo-profile before aggregation.

These ranking metrics are computable for the explicitly defined proxy; they
are **not real-user recommendation metrics** because no user-item relevance
history exists.

### Response time

Mean, median, and p95 local processing time were measured. This is not a
production end-to-end latency benchmark.

## 23. Results

### Classification results

| System/task | Test records | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Re-created original Random Forest category classifier | 436 | 0.5229 | 0.5226 | 0.5409 | 0.5095 |
| Improved content ranker | 436 | N/A | N/A | N/A | N/A |

### Ranking results against the catalog proxy

| Method | K | Precision@K | Recall@K | F1@K | NDCG@K | Hit Rate@K |
|---|---:|---:|---:|---:|---:|---:|
| RF + random selection | 5 | 0.4665 | 0.0078 | 0.0153 | 0.1504 | 0.6583 |
| Improved personalized content ranking | 5 | 1.0000 | 0.0163 | 0.0321 | 0.9830 | 1.0000 |
| RF + random selection | 10 | 0.4656 | 0.0155 | 0.0300 | 0.1675 | 0.7546 |
| Improved personalized content ranking | 10 | 1.0000 | 0.0327 | 0.0632 | 0.9885 | 1.0000 |

### Local response time

| Method | Mean | Median | p95 |
|---|---:|---:|---:|
| RF + random selection | 23.458 ms | 23.144 ms | 29.686 ms |
| Improved content ranker, top 10 | 630.800 ms | 604.826 ms | 768.161 ms |

### Graphs

- [Ranking metrics](../data/final_evaluation/ranking_metrics.png)
- [Random Forest confusion matrix](../data/final_evaluation/classification_confusion_matrix.png)
- [Local response time](../data/final_evaluation/response_time_comparison.png)

## 24. Discussion

The classifier's 0.5229 accuracy shows moderate category prediction on this
split using only gender, location, and season; its confusion matrix shows
category errors. This is a different task from product ranking and should not
be compared numerically with Precision@K or NDCG.

Against the designed proxy, the improved system produces better-ranked
attributes than category prediction followed by random ordering. At K=5,
proxy Precision rises from 0.4665 to 1.0000 and proxy NDCG from 0.1504 to
0.9830. At K=10, proxy Precision is 1.0000 versus 0.4656 and NDCG is 0.9885
versus 0.1675. These near-perfect values are expected to be optimistic because
the proxy relevance definition uses the same catalog attributes passed to the
ranker. They do not show that customers would prefer or buy the returned items.

Recall is low for both methods because many items across the candidate
inventory can meet the proxy relevance threshold while the system returns only
five or ten products. The improved system has a higher local median computation
time: about 26 times the baseline measurement. This is a genuine measured
trade-off in the tested local environment and indicates an optimization need
before scaling the catalog or serving load.

The experiment does not isolate the causal contribution of saved profile,
weather, or body-shape components. No weather or pose signal was supplied to
the comparative ranking test, and no corresponding ground truth exists.

## 25. Limitations

1. **No interaction ground truth:** no customer IDs, impressions, clicks,
   purchases, or explicit recommendations relevance judgments.
2. **Proxy overlap:** held-out product attributes both define pseudo-profiles
   and contribute to proxy labels and ranking; performance is optimistic and
   partly measures recovery of the proxy's own rules.
3. **Pseudo-profiles are not people:** results cannot be interpreted as user
   satisfaction or intent.
4. **Limited baseline fidelity:** the original serialized Random Forest model
   was unavailable; the benchmark re-created the documented baseline design.
5. **Single split:** no confidence interval, significance test, temporal
   generalization, or multi-seed robustness estimate.
6. **Weather unvalidated:** no weather-at-purchase or fabric properties exist;
   current rules use category/season proxies.
7. **Vision unvalidated:** no labeled body-shape data or validated styling
   outcomes exist; 2D pose measures are sensitive to viewpoint, clothing,
   occlusion, and pose.
8. **Latency scope:** timings exclude external services, persistence, HTTP,
   image processing, and front-end rendering; results depend on the test
   machine and implementation.
9. **Limited catalog:** 2,176 products and finite categorical vocabularies do
   not represent a large or changing commercial catalog.
10. **No collaborative filtering:** there are no interactions from which to
    learn latent user-item preferences.

## 26. Ethical Considerations

- **Privacy:** optional images are processed in memory and are not saved;
  profile UUIDs are persisted without login. Do not store sensitive profile
  attributes. UUID profiles are not access-controlled user accounts.
- **Computer-vision caveat:** a generated styling label is a heuristic,
  optional, and non-medical. It should never be presented as an objective
  assessment or used to deny a user choices.
- **Bias and representation:** the catalog has limited gender, location, and
  style vocabularies. The system may reproduce those constraints and does not
  establish fairness across groups.
- **Transparency:** scores are relative formula outputs, not purchase
  probabilities. Heuristic context matches and hard filters are labeled.
- **Consent and control:** image input is optional; users can rely on
  preferences without uploading an image.
- **Deployment security:** profile persistence has no authentication and is
  suitable only for a local demonstration unless access control, transport
  security, retention policy, and privacy controls are added.

## 27. Future Scope

1. Collect consented impression/click/save/purchase/return events with stable
   user and item identifiers and documented retention.
2. Use temporal or user-grouped evaluation splits to avoid leakage from
   repeated users and time.
3. Compare to category-only, popularity, and randomized baselines on genuine
   held-out relevance; report confidence intervals and per-user metrics.
4. Reduce ranking latency through cached vectorization, vectorized scoring,
   candidate retrieval, or indexing, then remeasure the deployed path.
5. Gather material, insulation, breathability, and water-resistance metadata
   to evaluate weather relevance meaningfully.
6. Validate body-shape styling proxies with voluntary expert/user feedback;
   assess errors across pose, clothing, lighting, and demographic groups.
7. Add proper authentication/authorization and profile deletion controls
   before public deployment.
8. Consider collaborative filtering or a learned ranker only after sufficient
   interaction data is collected.
9. Conduct a user study or controlled A/B test before claiming business or
   satisfaction improvements.

## 28. Conclusion

The project preserves an existing Flask fashion application and replaces
random product selection with a personalized, deterministic, explainable
content-based ranking workflow. It adds reusable preprocessing, anonymous
profiles, optional current weather, optional pose-derived styling proxies,
and visible score contributions. The final held-out experiment calculated
category classification, proxy ranking metrics, and local processing time
without inventing interaction labels.

The experiment supports a narrow conclusion: **the improved ranker orders
products much more consistently with the explicitly defined catalog-attribute
proxy than the re-created category-prediction plus random-selection baseline,
at a substantially higher local computation cost.** It does not prove that
the full weather/vision-aware system improves customer outcomes. Obtaining real
user feedback and relevant context data is the next necessary step for that
claim.

## 29. References

1. Breiman, L. (2001). “Random Forests.” *Machine Learning*, 45, 5–32.
   https://doi.org/10.1023/A:1010933404324
2. Ricci, F., Rokach, L., and Shapira, B. (eds.) (2022). *Recommender Systems
   Handbook*, 3rd ed. Springer. https://doi.org/10.1007/978-1-0716-2197-4
3. Järvelin, K. and Kekäläinen, J. (2002). “Cumulated gain-based evaluation of
   IR techniques.” *ACM Transactions on Information Systems*, 20(4), 422–446.
   https://doi.org/10.1145/582415.582418
4. Manning, C. D., Raghavan, P., and Schütze, H. (2008). *Introduction to
   Information Retrieval*. Cambridge University Press.
   https://nlp.stanford.edu/IR-book/
5. scikit-learn developers. *RandomForestClassifier* documentation.
   https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
6. scikit-learn developers. *Classification metrics* documentation.
   https://scikit-learn.org/stable/modules/model_evaluation.html
7. pandas developers. *pandas documentation*.
   https://pandas.pydata.org/docs/
8. Flask developers. *Flask documentation*.
   https://flask.palletsprojects.com/
9. Google AI Edge. *MediaPipe Pose Landmarker documentation*.
   https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
10. OpenCV developers. *OpenCV documentation*.
    https://docs.opencv.org/
11. Open-Meteo. *Weather Forecast API documentation*.
    https://open-meteo.com/en/docs
12. Open-Meteo. *Geocoding API documentation*.
    https://open-meteo.com/en/docs/geocoding-api

## Viva preparation: likely questions and defensible answers

### 1. What problem does your project solve?

The original Flask application predicted one product category and randomly
selected products from it. I replaced random selection with a ranked list
using available product attributes and explicit preferences. I do not claim
the dataset proves that users prefer those recommendations.

### 2. Why is the project called hybrid?

It combines content-based similarity with deterministic profile, weather, and
optional pose-derived styling signals. It is not collaborative filtering or a
trained multimodal model. The available data does not support those claims.

### 3. What does multimodal mean in this implementation?

The system accepts structured catalog/profile data and can optionally process
an image using pose landmarks. Weather is another external contextual signal.
These components are combined by explicit rules and score blending, not by a
joint deep-learning model.

### 4. Does the system learn from user purchases or clicks?

No. There are no user IDs or interaction logs in the supplied dataset. The
ranker is deterministic and uses item attributes and current request/profile
preferences.

### 5. Why use content-based recommendation?

Product descriptions are available, while user-item interactions are not.
Content-based methods can rank items by their recorded attributes without
inventing interaction data or requiring a collaborative model.

### 6. How is the similarity calculated?

The selected categorical preferences are encoded as weighted one-hot
components. Cosine similarity compares the preference vector with catalog
product profiles. Other terms, such as rating quality, affordability, and
discount value, are combined separately according to the documented score.

### 7. What does a 90/100 recommendation score mean?

It means the product has a relative score of 90 under the implemented ranking
formula. It is not a 90% probability of purchase, satisfaction, or fit. The
score is not calibrated from customer outcomes.

### 8. Why not use SHAP?

The ranker is not a learned prediction model; it is a deterministic formula.
Showing each component score and its weighted contribution is an exact
explanation of the calculation, whereas SHAP would add an approximate
model-agnostic layer without a learned ranker to explain.

### 9. How did you evaluate recommendations without user interactions?

I used held-out catalog rows as pseudo-profiles and defined a transparent
attribute-match relevance proxy. I compared top-K results against that proxy
and clearly label the metrics as simulated/catalog-proxy results, not user
evaluation.

### 10. Why are proxy Precision@5 and NDCG@5 so high?

The proxy defines relevance using attributes such as category, season, color,
brand, and size, which are also supplied to the content ranker. The ranker is
therefore optimized for attributes that also define the proxy labels. The
results demonstrate retrieval under that rule, but are optimistic and cannot
establish customer satisfaction.

### 11. Can you compare classification accuracy between the original and
improved systems?

Not directly. The Random Forest predicts one category and can be evaluated
using accuracy and classification metrics. The improved system outputs a
ranked product list, not a category class. Its classification scores are
therefore not applicable. Recommendation metrics are reported separately.

### 12. What does the Random Forest result show?

On 436 held-out catalog rows, the re-created baseline classifier achieved
0.5229 accuracy and 0.5095 macro F1 using gender, location, and season. That
is category-prediction performance on this split, not recommendation
effectiveness.

### 13. Did you evaluate the exact historical Random Forest model?

No. Its serialized model artifact was not available in the upgraded project.
The final evaluation re-created the audited design with the documented inputs
and a seeded 100-tree Random Forest. I disclose this difference in the report.

### 14. What is the measured cost of the improved ranker?

In one local in-process pass over 436 held-out records, median top-10 ranking
time was 604.826 ms, compared with 23.144 ms for category inference and random
selection. These figures exclude HTTP, weather, persistence, image processing,
and browser time and are specific to the test environment.

### 15. Is the weather recommendation scientifically validated?

No. It is implemented as a transparent prototype heuristic using live current
weather and category/season proxies. The catalog lacks garment material,
breathability, insulation, water resistance, and weather-linked outcomes.
Tests verify code behavior, not user benefit.

### 16. Does the vision module determine a person's true body type?

No. It estimates 2D pose proportions and maps a shoulder-to-hip ratio to a
limited styling-proxy label. Perspective, pose, clothing, and occlusion can
affect the measurement. It is not medical, objective, or validated body-type
classification.

### 17. Why are hourglass and oval not supported?

The implemented landmarks and optional segmentation do not reliably measure
waist shape or torso circumference. Returning those labels would imply more
information than the pipeline can infer defensibly.

### 18. Is body-shape-aware recommendation included in the benchmark?

No. The final comparative ranking experiment did not supply body-shape
preferences. The catalog has no labels or outcomes to evaluate such a feature.
It is implemented in the application as an optional styling heuristic only.

### 19. How do you avoid data leakage?

The final experiment uses a category-stratified 80/20 split with seed 42 and
zero product-ID overlap. The Random Forest is fit only on training rows. The
ranking candidate inventory is also the training split. However, the proxy
relevance reuses attributes given to the ranker, which is a separate
evaluation-validity limitation that I report explicitly.

### 20. How are missing values handled?

Validation preserves permitted missing fields rather than fabricating values.
Some fields, such as missing ratings, are handled explicitly by the ranker;
for its rating component a missing product rating uses the catalog median and
the explanation can state this. Required invalid values cause a validation
error.

### 21. How are user profiles stored?

The legacy saved-profile API uses a browser UUID in local storage and SQLite.
The new wizard uses a different random UUID in a signed cookie that points to
server-side SQLite workflow state, which expires after 24 hours. The cookie
does not contain the personal details. There is no login or authorization, so
this is suitable only for a local academic demonstration and needs security
controls before public deployment.

### 22. What are the main limitations?

The main limitations are the absence of user interaction ground truth, the
optimistic overlapping proxy evaluation, a limited product catalog, no
validated weather/body-shape outcomes, single-split evaluation, and higher
measured local ranking latency.

### 23. What would you do next?

Collect consented user-item events and evaluate using time-separated
interaction data. Add expert/user relevance judgments, validate weather and
styling assumptions with suitable item/context labels, profile the deployed
request path, optimize ranking, and add authentication and privacy controls
before public use.

### 24. What is your strongest defensible conclusion?

The implemented deterministic ranker aligns much better than predicted
category plus random selection with the project's explicitly defined
catalog-attribute proxy. That result does not prove customers prefer its
recommendations. The next step for that claim is collecting and evaluating
real, consented user feedback.
