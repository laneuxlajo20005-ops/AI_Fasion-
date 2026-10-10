# Project QA and run report

## Overall status

**WORKING AS A LOCAL RECOMMENDATION PROTOTYPE; NOT A USER-VALIDATED OR PRODUCTION SYSTEM.**

The Flask app starts, renders the preference UI, returns ranked products, handles
the exercised failure cases, and passes the automated suite. This status does
not imply that the ranking improves customer outcomes, that the original saved
Random Forest model is present, or that weather/body-shape effectiveness is
validated.

## Project dependency map

| File/resource | Function | Depends on | Interacts with | QA note |
|---|---|---|---|---|
| `market.py` | Guided and legacy Flask routes, validation, app orchestration and direct startup | Flask, pandas, project modules | CSV, templates, ranker, profile and wizard stores, weather, pose | Direct run has debug disabled; browser cookie contains only a session UUID |
| `wizard_store.py` | 24-hour server-side guided-workflow state | SQLite, JSON, UUID | `market.py`, `instance/fashion_profiles.sqlite3` | Personal details are not stored in the signed browser cookie |
| `templates/wizard_base.html` and wizard templates | Five-step responsive flow, upload/webcam, analysis summary, recommendation dashboard | Jinja, HTML/CSS/JS | Guided Flask routes and engine results | Uses real catalog values; no fabricated product photos |
| `templates/market.html` | Preserved legacy single-request form, profiles, optional upload/webcam, and ranked results | Jinja, HTML/CSS/JS | `/predict` and profile API | Retained for compatibility |
| `fashion_boutique_fixed_location.csv` | Source product catalog | pandas CSV parser | Preprocessing, app startup, ranker and evaluation | 2,176 rows; no user-item interaction history |
| `fashion_preprocessing.py` | Validation, quality analysis, feature engineering, train-fitted transform pipeline | pandas, NumPy, scikit-learn | Preparation, ranker, evaluation | The pipeline artifact is preprocessing, not a trained recommendation model |
| `prepare_data.py` | Train/test split and preprocessing artifact generation | joblib, pandas, scikit-learn | Source CSV and `data/processed/` | Runs successfully; does not train a classifier |
| `recommendation_engine.py` | Candidate filtering, weighted content similarity, scoring, top-N ordering, explanations | NumPy, pandas, scikit-learn, context processor | Flask and catalog | Deterministic ranker; rating-only fallback preserves budget and stock constraints |
| `context_processor.py` | Open-Meteo geocoding/weather, bounded cache, weather proxy scoring | Python HTTP/JSON, pandas | Open-Meteo, ranker, Flask | Live lookup succeeded; unit tests cover cache and failure fallback |
| `body_shape.py` | OpenCV decode, MediaPipe Pose landmarks, 2D proportions, styling-proxy assignment | OpenCV, MediaPipe, NumPy | Flask upload handler and ranker | No labeled training data; no multiple-person counting/rejection |
| `profile_store.py` | Anonymous preference profile persistence | SQLite, UUID and JSON | Flask API and browser localStorage | No login/authorization; local demonstration only |
| `evaluation.py` | Seeded random/category proxy evaluation | pandas, NumPy, matplotlib, scikit-learn | Catalog and ranker | Relevance is catalog-attribute proxy, not user feedback |
| `final_evaluation.py` | Re-created baseline RF, recommendation proxy metrics, latency and plots | pandas, NumPy, scikit-learn, matplotlib | Catalog, ranker, final evaluation artifacts | Historical RF artifact unavailable; contextual/vision benefits not measured |
| `tests/*.py` | Unit and Flask/module integration tests | `unittest`, project dependencies | All core Python modules and wizard routes | 46 tests pass, including eight guided-flow tests |
| `requirements.txt` | Direct Python dependency declarations | pip | All Python modules | Direct imported packages and declared versions were verified in this environment |
| `docs/*.md` | Methodology, limitations, user and technical documentation | Markdown | Project readers and evaluators | Includes the project report, scoring notes and this QA report |
| `data/processed/` | Reusable transformed splits and quality report | `prepare_data.py` | Preprocessing workflow | Regenerated successfully |
| `data/final_evaluation/` | Results, metadata, tables, graphs and report | `final_evaluation.py` | Evaluation protocol | Regenerated successfully from the current source |
| `instance/fashion_profiles.sqlite3` | Runtime anonymous profile and expiring wizard-state DB | SQLite | `profile_store.py`, `wizard_store.py` | Contains local workflow/profile data; not an authentication store |

There is no `static/` directory because presentation styles and scripts are
inline in the template. There is no `.env` file or required API key:
Open-Meteo's selected endpoints are used without a credential. There is no
runtime Random Forest model artifact in the application.

## QA status matrix

| Area | Status | Evidence / boundary |
|---|---|---|
| Application startup | **PASS** | `python market.py` started Flask on `127.0.0.1:5000` with debug off |
| Dataset | **PASS** | 2,176 rows and 16 required columns; zero duplicate rows/IDs and zero invalid dates |
| Preprocessing | **PASS** | `python prepare_data.py` succeeded; 1,740/436 split, zero product-ID overlap; tests pass |
| ML model | **PARTIAL** | The original RF design is re-created and measured in final evaluation; no historical model artifact and no RF prediction in the serving app |
| Recommendation engine | **PASS** | Top-10 products, no-exact-match and empty-budget cases exercised; no duplicate IDs in tests; rating fallback is explicit |
| Weather | **PASS WITH EXTERNAL-SERVICE CAVEAT** | A live Open-Meteo London lookup succeeded; cache/failure behavior is covered by tests; network remains external |
| Computer vision | **PARTIAL** | OpenCV/MediaPipe pipeline, synthetic measurement and no-person rejection tests pass; no valid real full-body photo test, no multiple-person detection, no labeled accuracy |
| Flask routes | **PASS** | Guided routes `/personal-details`, `/preferences`, `/photo-analysis`, `/ai-analysis`, `/recommendations`, `/start-over`; legacy `/predict` and profile API remain available |
| Frontend | **PASS FOR EXERCISED FLOWS** | Browser completed the five-page flow, displayed ten results, sorting worked, camera start/stop worked, at 375px no horizontal overflow, no JavaScript errors observed |
| End-to-end | **PASS FOR TESTED SCENARIOS** | App started, guided form submitted, optional CV skipped, live weather/ranking displayed and dashboard rendered; see new flow table |
| Test suite | **PASS** | `python -m unittest discover -s tests -v`: 46 tests passed |
| Python compilation | **PASS** | `python -m compileall -q .` completed |
| Dependencies | **PASS WITH SHARED-ENVIRONMENT WARNING** | Project imports and declared versions work; global `pip check` reports conflicts from unrelated installed packages (below) |

## End-to-end scenarios exercised

| Scenario | Result |
|---|---|
| Home page `GET /` | HTTP 302 to `/personal-details`; guided first page returns HTTP 200 |
| Woman, hot weather, budget 80 | HTTP 200; mocked hot weather context rendered with ranked results |
| Man, Fall, budget 150 | HTTP 200; ranked results rendered |
| Urban location | HTTP 200; ranked results rendered |
| No image | HTTP 200; normal recommendations rendered |
| Invalid image bytes | HTTP 200; friendly image error shown and recommendation flow continued |
| Very low budget (`0.01`) | HTTP 200; explicit no-in-budget-products message, no budget relaxation |
| Missing optional preferences | HTTP 200; default recommendations rendered |
| Valid preferences with no exact seven-field catalog match | 0 exact matches; 10 in-stock ranked alternatives returned |
| Minimum rating eliminates all in-budget products | Unit test verifies only the rating filter is relaxed and fallback is labeled; budget remains enforced |
| Upload larger than request limit | HTTP 413 with friendly size message |
| Profile API | Tests cover save/read, invalid preferences/ID, and missing profile |
| Browser product controls | Ten cards displayed; price sort produced ascending values |
| Mobile viewport | 375px viewport; no horizontal overflow and no page-level JavaScript errors |
| Live weather service | Current London conditions returned by Open-Meteo in this run |
| Personal details → preferences → photo page → skip → analysis → results | Browser journey completed; 10 real catalog cards rendered |
| Back-navigation and refresh | Automated test confirms inputs are retained across wizard steps and GET refreshes |
| Browser webcam permission | Live browser camera API became available; preview started and explicit Stop camera shut it down without capturing a person |
| Webcam capture, retake, and remove controls | Browser test used a synthetic empty MediaStream/canvas payload; capture preview, retake, recapture, and remove all worked without accessing camera imagery |
| Webcam payload and estimate removal | Automated mocked estimator test confirms captured payload reaches the estimator in memory, is not persisted, and Remove clears the stored estimate |
| Photo selection/removal | Browser selected two project-generated plots in sequence; previews appeared, Remove cleared each selection, and a second file could be selected |
| Start Over | Removes the server-side wizard row and clears the browser session; verified in an integration test |
| No in-budget results | Wizard explicitly labels ranked above-budget fallback products; budget-friendly section correctly remains empty |

The browser's first profile lookup returned 404 because a new UUID did not yet
have a saved profile; the UI handles that expected not-found state. The
favicon-related 404 was removed with an empty data-URI favicon. No successful
pose estimate on a real full-body person photo was established. The live
webcam was opened and stopped without capturing a person; capture/retake/remove
controls used a synthetic camera stream and no image of a person.

## Bugs found and fixes applied

| File | Problem and cause | Fix | Verification |
|---|---|---|---|
| `market.py` | Running the module enabled Flask debug mode by default, exposing the interactive debugger if bound beyond a trusted local environment | Changed direct startup to `debug=False` | Started with `python market.py`; server reported Debug mode off |
| `recommendation_engine.py` | A strict minimum rating could leave no products even when in-budget, in-stock alternatives existed | If the rating filter alone empties budget-safe candidates, retry without the rating threshold; never relax budget/stock; mark fallback evidence | Added unit test asserting fallback flag, evidence text and price constraint |
| `templates/market.html` | A no-result state described relaxing both budget and rating even though budget should remain a hard constraint; fallback was not explained in the UI | Added explicit labeled rating-only fallback message and truthful no-budget-result message | Flask scenario and browser rendering checks pass |
| `templates/market.html` | Browsers requested `/favicon.ico`, resulting in an avoidable 404 | Added a `data:,` favicon declaration | Reloaded live page; no new favicon request and no browser console errors |
| `final_evaluation.py` | The benchmark report called the one-pass measurement “warmed” and duplicated two JSON metadata keys | Corrected latency scope wording and removed duplicate metadata declarations | Re-ran final evaluation; report and JSON regenerated from current source |
| `tests/test_recommendation_engine.py` | No regression test for safe minimum-rating fallback | Added a test covering the narrow fallback without exceeding budget | Full suite passes |
| `tests/test_profile_store.py` | Missing/invalid profile read responses were not both asserted | Added checks for 404 unknown profile and 400 invalid UUID | Full suite passes |
| `market.py` | A shared route guard incorrectly sent the preferences page back to itself until preferences had already been saved | Split the personal-details guard from downstream preference guards | Guided-flow tests now cover every transition and pass |
| `templates/wizard_base.html`, `templates/photo_analysis.html` | Component display styles overrode native `[hidden]`, and an active camera had no explicit stop control | Added an explicit hidden-state rule and Stop camera / Remove captured photo controls | Live browser check confirmed initially hidden controls, camera stop, no capture, and no JS errors |

No files were removed.

## Guided multi-page flow verification

The application now starts at a five-step wizard and retains `/predict` for
the prior single-request workflow. Personal details and preferences persist
server-side in a 24-hour SQLite wizard record; only its UUID is stored in the
signed session cookie. Image bytes and the annotated landmark preview are not
persisted. See [multi-page-user-flow.md](./multi-page-user-flow.md) for routes,
privacy, image handling, budget fallback semantics, and run instructions.

The browser walkthrough reached ten recommendations after selecting gender,
city, location, season, category, style, color, brand, and budget. It also
opened and explicitly stopped the live webcam without capturing a person. A
separate browser run used a synthetic empty camera stream to exercise capture,
retake, recapture, and removal controls. Product sorting was verified as
ascending by price. At a 375px viewport, body horizontal overflow measured
0px; browser page-error and console-error collections were empty. No valid
full-body human photo was tested.

## Experimental outputs

The final evaluation was rerun after the ranking fallback change. It uses a
stratified 80/20 split (seed 42), 1,740 training products, 436 held-out
pseudo-profiles, and 1,705 in-stock training candidates.

### Category classification

| Re-created original Random Forest | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Test set, 436 catalog rows | 0.5229 | 0.5226 | 0.5409 | 0.5095 |

The upgraded app's ranker does not predict one class, so classification metrics
for it are **not applicable**, not zero.

### Ranking proxy metrics

| Method | K | Precision@K | Recall@K | F1@K | NDCG@K | Hit Rate@K |
|---|---:|---:|---:|---:|---:|---:|
| Re-created RF + random selection | 5 | 0.4665 | 0.0078 | 0.0153 | 0.1504 | 0.6583 |
| Personalized content ranker | 5 | 1.0000 | 0.0163 | 0.0321 | 0.9830 | 1.0000 |
| Re-created RF + random selection | 10 | 0.4656 | 0.0155 | 0.0300 | 0.1675 | 0.7546 |
| Personalized content ranker | 10 | 1.0000 | 0.0327 | 0.0632 | 0.9885 | 1.0000 |

These are not user-interaction metrics. Relevance is a synthetic catalog
attribute rule applied to held-out product rows treated as pseudo-profiles; the
same attributes also drive the improved ranker. Near-perfect precision/NDCG is
therefore optimistic and cannot establish customer preference.

### Local processing time

| Path | Mean | Median | p95 |
|---|---:|---:|---:|
| RF inference + random selection | 19.561 ms | 19.095 ms | 23.045 ms |
| Content ranker, top 10 | 552.068 ms | 541.218 ms | 640.625 ms |

This one-pass local in-process measurement includes category prediction and
random selection for the baseline, and normalization plus full-catalog top-10
ranking for the improved method. It excludes Flask/HTTP, weather network calls,
SQLite I/O, image upload/pose inference, and browser rendering. The measured
median ranker path is about 28.3 times slower on this machine; performance is a
clear trade-off and should be optimized before scaling.

### Graphs and raw output

- [Ranking metrics](../data/final_evaluation/ranking_metrics.png)
- [Classifier confusion matrix](../data/final_evaluation/classification_confusion_matrix.png)
- [Local timing](../data/final_evaluation/response_time_comparison.png)
- [Full experimental report](../data/final_evaluation/final_evaluation.md)
- [Machine-readable metadata](../data/final_evaluation/final_evaluation.json)

## Dependency/environment note

The active environment is Python 3.11.15, with Flask 3.1.3, NumPy 1.26.4,
pandas 3.0.3, scikit-learn 1.9.1, OpenCV 4.11.0, MediaPipe 0.10.21,
Matplotlib 3.11.2, and joblib 1.6.0. These versions fit the project's declared
version ranges, and all required project modules imported successfully.

`python -m pip check` reports two conflicts in the shared Python environment:

- `grpcio-status` requests protobuf >=5.26.1, while MediaPipe's installed
  dependency set uses protobuf 4.25.9;
- unrelated `trafilatura` requests lxml >=6.1.1, while lxml 5.4.0 is installed.

Neither `grpcio-status`, `trafilatura`, nor lxml is declared or imported by
this project. Do not change the project's MediaPipe/protobuf pins to solve
unrelated shared-environment packages. A clean virtual environment installed
from `requirements.txt` is recommended. Python imports, app startup, and the
full test suite all succeeded in the environment used for this QA.

## Remaining issues and honest boundaries

- **No deployed category classifier:** the Flask app uses the content ranker.
  The Random Forest is re-created and trained within the final experiment; the
  historical serialized model was unavailable and is not loaded by Flask.
- **No genuine recommender ground truth:** no user IDs, impressions, clicks,
  purchases, or relevance labels exist in the catalog.
- **Weather and body-shape impact:** both are implemented and can alter scores,
  but neither has outcome-based effectiveness results.
- **Pose limitations:** the pipeline processes one pose and does not reliably
  detect or reject group images. No labeled body-shape classifier exists and
  no successful real full-body user-image result was validated in this QA.
- **Latency:** full-catalog scoring is comparatively slow; this is not a
  production load test.
- **Profile security:** local UUID profile access has no authentication or
  authorization. The development server is for local demonstrations, not
  public deployment.
- **Population/catalog coverage:** the 2,176-row catalog is limited and
  imbalanced across some gender/location labels. Results are not evidence of
  fairness or broad retail generalization.

## Bugs not changed

No code was added to detect multiple people in an image because the current
MediaPipe Pose pipeline returns one pose and the project has no reliable
multi-person detector/model. Adding an unvalidated detector would create a
misleading guarantee. Group-photo rejection should be future work; current
guidance asks for one front-facing person.

## Clean Windows run procedure

Recommended Python: **3.11**. From a fresh PowerShell session:

```powershell
Set-Location 'C:\path\to\market'
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python prepare_data.py
python -m unittest discover -s tests -v
python final_evaluation.py
python market.py
```

Open `http://127.0.0.1:5000/`. `prepare_data.py` creates feature matrices and a
reusable preprocessing artifact; it does not train the app's recommender.
`final_evaluation.py` re-trains its evaluation-only baseline and regenerates
the experiment outputs. Open-Meteo requires no project API key. The optional
pose feature needs the installed OpenCV/MediaPipe dependencies and one clear,
front-facing full-body person image.

If PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate
the environment.
