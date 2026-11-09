# Limits and next steps

- Ratings are synthetic, so the collaborative metrics say nothing about tourist taste; the harness is ready
  for real ratings.
- The problem statement mentions "current tourist location"; the content route handles a place name via
  `ContentModel.similar_places`, but the page does not expose it yet, and the latitude/longitude columns
  are unused. A "near me" route by haversine distance is the next step.
- TF-IDF without stemming on Indonesian text; category and city weights were set by hand, not learned.
- Single process, model in memory, no auth or caching; a production version would precompute per-user lists
  nightly and serve them from a store.
