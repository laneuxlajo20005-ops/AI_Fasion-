# Lightweight user profiles and personalization

## Profile storage and security scope

The legacy profile API stores saved preferences in SQLite at
`instance/fashion_profiles.sqlite3`; its browser UUID is kept in `localStorage`
and loads/updates that profile. The five-page wizard uses a separate random UUID
in a signed, HttpOnly Flask cookie and stores its temporary name, city, gender,
preferences, optional proportions, weather result, and rankings in a server-side
SQLite workflow table for up to 24 hours. The wizard does not automatically
create a persistent profile record. Start Over deletes the current workflow
record. There is no login or identity verification.

This is suitable only as a local PG mini-project demonstration, not an
internet-facing multi-user service. Wizard cookies contain only a random
identifier, but server-side state still requires appropriate access to the
machine/database. Do not store unnecessary personal information, and add
authentication/authorization, transport security, and an explicit deployment
retention policy before exposing profile data publicly.

The SQLite database is generated at runtime and should be backed up or excluded
from source control according to the deployment policy. User preferences are
validated against the actual catalog vocabulary before they are persisted.

## Profile fields and their use

- Gender, season, and catalog location (Rural/Urban) remain content-profile
  similarity inputs. Weather city is a separate context input.
- Categories, colors, and brands are multi-select preferences. Their selected
  values form a multi-hot user vector compared with the product's weighted
  one-hot content vector using cosine similarity.
- Preferred minimum/maximum price contributes a soft price-range fit: products
  inside the range score 1 for that dimension; products outside lose score as
  their distance from the boundary increases. The separately entered maximum
  budget remains a hard eligibility filter.
- Size remains a single catalog preference.
- Style is represented by explicit catalog-category proxies because the source
  has no style labels: Everyday maps to Tops/Bottoms/Shoes/Accessories; Occasion
  maps to Dresses/Shoes/Accessories; Layered maps to Outerwear/Tops/Bottoms.
  These are user-facing heuristic archetypes, not measured or inferred style
  annotations.

## Ranking score

If a profile adds style or preferred-price-range dimensions:

`base_preference_score = weighted existing preference components`

`personalized_score = 0.80 * base_preference_score + 0.20 * profile_fit`

`final_score = 0.80 * personalized_score + 0.20 * weather_fit` (when weather is available)

Otherwise, the existing preference score stays the base score. Existing season,
location, gender, category, color, brand, and size preferences remain in content
similarity, with arrays represented as multi-hot selections. Weather score
semantics and physical-property limitations are documented in
[context-aware-scoring.md](./context-aware-scoring.md).

Profile fit is the unweighted mean of the active dimensions: match against any
selected category, color, or brand; selected style proxy; and preferred price
range fit. When a body-shape styling proxy is present, the engine adds a
separate body-shape/category fit component at 12% of the pre-weather score.
Empty dimensions are excluded rather than counted as failures. Score
contributions are returned with each recommendation and are designed to sum to
the displayed final score (up to rounding).

## Example profiles

Two hypothetical profiles—Everyday selecting Tops/Bottoms, Blue, Uniqlo, and a
lower price range versus Occasion selecting Dresses, Red, Zara, and a higher
price range—will yield different profile similarity and fit components and
therefore different product rankings. The automated recommender test verifies
that the top-10 IDs differ and that recommendations match each profile's
preferred categories. This is a behavioral example, not measured user outcome.
