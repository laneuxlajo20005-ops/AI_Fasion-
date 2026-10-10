# Fashion recommender offline evaluation

## Interpretation and limitations

This is a **simulated catalog-proxy evaluation**, not a measurement of user satisfaction.
The source catalog has 2176 product rows but no customer identifier,
user-item interactions, clicks, purchases linked to a customer, or recommendation
relevance judgments. Therefore genuine user-level Precision@K, Recall@K, F1@K,
NDCG@K, and Hit Rate@K are not computable from the supplied data.

No interaction or relevance data was fabricated. Instead, each held-out catalog row
is treated as a **pseudo-profile**, not a real person. Its available category, season,
gender, location, color, brand, and size fields define a documented attribute-match
proxy against in-stock products in the training catalog. This protocol measures
attribute retrieval behavior only; the held-out row is not evidence that a real
customer wanted matching products.

## Protocol

- Data split: 80% candidate catalog / 20% held-out pseudo-profiles.
- Split seed: 42; stratified by category.
- Candidate inventory: training-split products with positive stock; both methods use this inventory.
- Baseline: deterministic seeded random ordering among in-stock products matching the pseudo-profile's category.
- Improved method: the existing content-based engine ranked using the pseudo-profile's gender, location, season, category, color, brand, and size.
- Proxy graded relevance: category match = 2 points; exact season, compatible gender, location, color, brand, and size = 1 point each. Men/Women profiles treat Unisex catalog products as gender-compatible.
- Binary relevance for Precision, Recall, F1, and Hit Rate: proxy grade >= 4.
- NDCG uses the full graded proxy score as gain. Metrics are averaged per pseudo-profile (macro average); F1 is computed per profile before averaging.
- K values: 5, 10. No model training or computer vision is involved.
- Evaluated pseudo-profiles: 436; skipped because no in-stock relevant candidate existed: 0.

## Comparison

| method | k | profiles | precision_at_k | recall_at_k | f1_at_k | ndcg_at_k | hit_rate_at_k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random/category baseline | 5 | 436 | 0.8674 | 0.0136 | 0.0268 | 0.2423 | 0.9977 |
| Content-based ranking | 5 | 436 | 1.0000 | 0.0163 | 0.0321 | 0.9830 | 1.0000 |
| Random/category baseline | 10 | 436 | 0.8644 | 0.0271 | 0.0525 | 0.2723 | 1.0000 |
| Content-based ranking | 10 | 436 | 1.0000 | 0.0327 | 0.0632 | 0.9885 | 1.0000 |

## Reproducibility

Run `python evaluation.py` from the project directory. Results, protocol
metadata, and the plot are generated under `data/evaluation/`. The random baseline
is seeded per pseudo-profile, so its ranking is stable for a fixed dataset and seed.
