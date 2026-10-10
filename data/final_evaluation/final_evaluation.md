# Final experimental evaluation: fashion recommendation system

## Executive summary

The supplied dataset contains **2176 product records** and no
customer IDs, recommendation impressions, clicks, user-linked purchases, explicit
relevance judgments, weather history, or body-shape labels. Consequently, this
experiment reports held-out **product-row pseudo-profile** results, not observed
user outcomes. The ranking relevance definition is a transparent catalog-attribute
proxy, and it overlaps with the features used by the improved ranker. Treat its
results as a controlled software/catalog experiment, not proof of customer
satisfaction or generalization to real users.

## Experimental setup

- Data: `fashion_boutique_fixed_location.csv`; stratified 80%/20% train/test split, seed 42.
- Training catalog: 1740 products; in-stock candidate catalog: 1705; held-out rows: 436.
- Product ID overlap between train and test: 0.
- Baseline classifier: a reproduced Random Forest category predictor using only gender, location, and season (100 trees; one-hot categories). It is trained only on the training split.
- Baseline fidelity: the upgraded project no longer contains the original trained Random Forest artifact. This is a documented re-implementation of the audited baseline design, not a score from the historical serialized model.
- Baseline recommender: predict category, then seeded-randomly select in-stock products of that category.
- Improved recommender: rank in-stock training products using the existing content/profile scoring engine; pseudo-profile preferences are taken from each held-out row's category, season, gender, location, color, brand, and size.
- Proxy relevance: category match = 2; exact season, gender-compatible (Unisex also compatible with Men/Women), location, color, brand, and size = 1 each. Binary relevance threshold = 4. NDCG uses the graded relevance. Ranking metrics are macro-averaged over 436 pseudo-profiles; F1 is computed per profile before averaging.
- 0 rows had no proxy-relevant in-stock candidate and were omitted from ranking averages.
- No weather or body-shape adjustment was supplied to either system in this ranking experiment. The dataset has no weather records or body-shape ground truth, so those components are **not experimentally validated here**.

## Category-classification results

Classification applies only to the Random Forest component of the original baseline.
The upgraded recommender is a ranker, not a category classifier; its accuracy,
precision, recall, and F1 as a classifier are therefore **not applicable**, rather
than zero.

| model | test_products | accuracy | macro_precision | macro_recall | macro_f1 |
| --- | --- | --- | --- | --- | --- |
| Original Random Forest category classifier | 436 | 0.5229 | 0.5226 | 0.5409 | 0.5095 |
| Improved personalized content ranker | 436 | N/A | N/A | N/A | N/A |

## Top-K recommendation results

These are computed against the catalog proxy labels above, not user judgments.

| method | k | pseudo_profiles | precision_at_k | recall_at_k | f1_at_k | ndcg_at_k | hit_rate_at_k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original RF + random selection | 5 | 436 | 0.4665 | 0.0078 | 0.0153 | 0.1504 | 0.6583 |
| Improved personalized content ranking | 5 | 436 | 1.0000 | 0.0163 | 0.0321 | 0.9830 | 1.0000 |
| Original RF + random selection | 10 | 436 | 0.4656 | 0.0155 | 0.0300 | 0.1675 | 0.7546 |
| Improved personalized content ranking | 10 | 436 | 1.0000 | 0.0327 | 0.0632 | 0.9885 | 1.0000 |

## Local response-time results

Times measure local, in-process request handling for category inference plus
random selection (baseline), versus preference normalization and full top-10 catalog ranking (improved). The measurement uses
all 436 held-out pseudo-profiles in one pass. It excludes Flask/HTTP overhead,
weather network calls, SQLite profile I/O, image upload, MediaPipe pose
estimation, browser rendering, and machine-independent deployment effects.

| method | mean_ms | median_ms | p95_ms |
| --- | --- | --- | --- |
| Original RF + random selection | 19.561 | 19.095 | 23.045 |
| Improved personalized content ranker | 552.068 | 541.218 | 640.625 |

## Interpretation and model comparison

- The Random Forest classification scores indicate how well the audited baseline
  category-prediction task is recovered from gender, location, and season on the
  held-out catalog rows. This is a product-row classification result, not a
  recommendation-quality score.
- The baseline recommender uses a predicted category but randomizes product
  order, so it cannot consistently prioritize the pseudo-profile's other
  matching attributes. The improved ranker explicitly orders catalog products
  using preference/profile similarity and available product attributes.
- Higher ranking metrics for the improved system, if observed, show better
  alignment with this designed attribute proxy. The proxy reuses ranking inputs,
  so high values are expected and potentially optimistic; they do not establish
  real-world superiority.
- Runtime should be interpreted as local algorithm cost only. The improved
  method performs more work per request; whether its measured latency is
  acceptable depends on deployment resources and catalog size.
- Weather-aware ranking and body-shape-aware ranking may change results when
  inputs are supplied at runtime, but this data does not support a defensible
  offline effectiveness comparison for either component.

## Limitations

1. No interaction/relevance ground truth; pseudo-profiles are held-out product
   attributes, not people.
2. Proxy relevance overlaps with recommender inputs, creating target-feature
   alignment and optimistic evaluation risk.
3. No body-shape labels or fashion-styling outcome labels exist. Image-based
   styling proxies have no measured offline accuracy or recommendation benefit.
4. The catalog does not contain weather-at-purchase or garment material,
   breathability, insulation, or waterproof attributes. Context rules are
   documented heuristics, not experimentally validated user benefits.
5. A single seeded split and one random baseline do not establish statistical
   significance or robustness across catalogs and time.
6. Response timings depend on this machine, Python environment, warm state, and
   catalog size, and exclude network and UI costs.

## Future improvements

- Log consented impressions, ranked positions, clicks, saves, purchases, and
  returns with user/item IDs; define a time-aware train/validation/test protocol.
- Evaluate with held-out interaction data, per-user Precision/Recall/NDCG/Hit
  Rate, confidence intervals, and suitable popularity/category baselines.
- Run user studies or expert-labeled outfit relevance judgments to validate
  preference, weather, and body-shape styling assumptions.
- Collect trustworthy item material/performance metadata and time/location
  context before assessing weather effects.
- Measure latency under representative deployment load, including HTTP,
  database, weather, and optional image-processing paths.

## Reproducibility and output

Run `python final_evaluation.py` from the project folder. The CSV tables, JSON
metadata, report, and graphs are written to `data/final_evaluation/`. Split seed,
classifier seed, ranking protocol, and latency sample parameters are recorded in
the JSON metadata. Only calculated values appear in the tables.
