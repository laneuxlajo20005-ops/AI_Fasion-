# Explainable recommendation ranking

## What the displayed score means

Each product's displayed relevance score is the final deterministic ranking
formula multiplied by 100. It is a relative score for ordering this catalog,
not a probability that a customer will buy, like, or keep the item. It is not
calibrated against user feedback because the project has no user-item
interaction history or relevance labels.

The base content score is a normalized weighted sum of:

| Component | Weight when active | Meaning |
|---|---:|---|
| Catalog preference similarity | 0.68 | Weighted one-hot cosine similarity for selected category, season, brand, color, size, and location |
| Gender compatibility | 0.10 | Compatibility rule for the requested and catalog gender |
| Customer rating quality | 0.12 | Rating divided by five; a missing catalog rating uses the catalog median and is identified as imputed |
| Budget affordability | 0.08 | Relative headroom under the maximum budget; the maximum budget itself is also a hard eligibility filter |
| Discount value | 0.02 | Catalog markdown percentage divided by 100 |

Only the gender and budget terms are omitted when their corresponding input is
not provided; the remaining active weights are renormalized to sum to one.
Out-of-stock items, over-budget items, and items below the requested minimum
rating are excluded before ranking. The minimum rating is therefore a filter,
not an additional ranking bonus.

If the minimum-rating filter yields no products but in-stock products remain
within the requested budget, the engine retries with that rating filter
relaxed and marks each returned result accordingly. It never relaxes the
maximum-budget or stock constraint. If no in-stock product is within budget, it
returns an empty list and tells the user to adjust the budget; it does not
silently recommend an over-budget item.

Where enabled, subsequent blend stages match the implementation order:

1. `0.80 * base score + 0.20 * saved-profile fit`
2. If a styling/body-shape proxy was selected: `0.88 * current score + 0.12 * proxy fit`
3. If live weather is available: `0.80 * current score + 0.20 * weather-context fit`

These are deterministic rules, not learned coefficients. The catalog lacks
fabric and garment-performance attributes, so weather suitability remains the
documented category/season proxy. Body-shape labels remain optional styling
proxies and are not objective classifications.

The saved-profile fit averages the active exact-match dimensions among
preferred categories, colors, brands, mapped style categories, and the
preferred price range. Body-shape styling fit is kept as its own blend stage,
so it is not counted again inside saved-profile fit.

## Per-product explanation

The interface shows each active component's raw component score and its
post-blend contribution in points. The contributions are rounded so their sum
equals the displayed final score. This is a direct decomposition of the exact
formula, rather than an approximation. Separate evidence lines show observed
attribute matches and eligibility constraints (for example, within budget or
in stock); filters are not presented as score contributions.

SHAP is not used: there is no learned ranker to explain. Applying a model
agnostic attribution method to a hand-written weighted formula would be less
transparent than exposing its exact component values and contributions.

## Results controls

The result list can be filtered by category, maximum price, and minimum rating,
and sorted by relevance, price, or rating. These controls operate on the
already-ranked top ten and do not change the ranking formula or retrieve
additional candidates.
