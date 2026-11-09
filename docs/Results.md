# Results

From `models/metrics.json`, quick mode (identical to full; 2 s on this machine), seed 42:

| Metric | Value | Baseline | Meaning |
|---|---|---|---|
| RMSE, SVD rank 2 | 1.384 | 1.363 global mean | lower is better; a real dataset would show the model 10–20% under the baseline |
| RMSE, user kNN / item kNN | 1.384 / 1.382 | 1.383 user mean | neighbour votes add nothing here |
| hit-rate@10, hybrid | 5.4% | 6.5% random | share of 280 test users with a liked held-out place in their top 10 |
| precision@5, collaborative | 0.71% | 0.67% random | liked held-out places among the first five |

The honest reading: the rating file is synthetic noise (entropy 2.315 of 2.322 bits, per-place mean spread
0.308 vs 0.299 after shuffling), so collaborative filtering has nothing to learn and the numbers sit on the
baseline. The content route is the one that produces sensible, explainable picks because it uses the real
catalogue. The EDA conclusions (Bandung for nature, Keraton Surabaya top, amusement parks then nature
preserves most loved) match the solution notebook exactly.
