---
title: Tourism recommender (Capstone 2, Part 2)
group: capstone
role: Capstone
port: 8103
status: working
one_liner: Explores 437 Indonesian tourist spots and 10k user ratings, then recommends places to a known user or to someone who only states a category, city and budget, with a reason for every pick.
---
# Tourism recommender (Capstone 2, Part 2)

## Demo
<!-- demo:auto -->
![demo](demo/demo.gif)

Full walkthrough with captions: [demo/demo.mp4](demo/demo.mp4) (72 s, 1280x800, animated intro and end card). Run it yourself: `just setup && just up`, then open http://localhost:8103/.

Two-minute walkthrough:
1. Capstone 2, part 2: recommendations for user 1 on the hybrid route, shown on load
2. Change the user: user 42
3. Or describe your interests: nature preserve, Bandung, budget Rp 20,000
4. Find places: content-based ranking with a reason for every pick
5. Offline metrics: RMSE, hit-rate@10, precision@5
6. Findings and charts: the EDA behind the engine
7. Who rates: the age distribution of the 300 users
8. What is on offer: categories and each city's speciality
9. The ratings carry almost no signal: near-uniform 1 to 5, 0.008 correlation with the public rating
10. So collaborative filtering sits at the baseline; the content route is what carries the recommendations
<!-- /demo:auto -->

## Brief
A government tourism agency wants to market Indonesia's five big cities better. It has a catalogue of 437
attractions, 300 tourists' demographics and 10,000 star ratings, and asks two things: what do the data say
about who visits what, and can a recommender point each tourist to places they have not seen yet. "Solved"
is an analysis with clear answers (which city for nature, which spots and categories are loved) and a working
engine that returns ten unvisited places per user with an explanation.

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language and runtime | Python 3.11 via uv | one lockfile, same across the suite |
| Model or core logic | pandas + scikit-learn: cosine kNN, TruncatedSVD, TF-IDF content model, rank fusion | small data (300 x 437), fits in two seconds, fully inspectable |
| Interface | FastAPI + Jinja2 HTML, shared `common/ui.css` | no build step, scriptable ids for the demo video |
| Data | `tourism_with_id.xlsx` (437 places), `user.csv` (300), `tourism_rating.csv` (10,000 rows -> 9,597 pairs) | the course dataset, read-only from DATA_ROOT |
| Tests | pytest + FastAPI TestClient, 9 tests, offline | health, metrics, recommend contracts, page states |

## Libraries used
`pandas` (cleaning, group-bys, the user x place pivot), `numpy` (similarity matrices and predictions),
`scikit-learn` (cosine_similarity, TruncatedSVD, TfidfVectorizer), `scipy` (sparse stacking of the content
features), `openpyxl` (read the xlsx), `matplotlib` (7 PNG charts), `fastapi` + `uvicorn` (API and pages),
`jinja2` (templates), `pydantic` (request validation), `httpx` (demo script and TestClient), `pytest`,
`ruff`.

## How it was built
1. Read the solution notebook (62 cells) and the problem statement. Kept its cleaning exactly: drop the two
   `Unnamed` columns, `Coordinate` and `Time_Minutes`; strip and capitalise `Category`. Added one step it
   missed: 403 duplicate (user, place) rating pairs are averaged, which its pivot did implicitly.
2. Rebuilt the notebook's EDA as code that writes `models/eda.json` and seven charts: age bands, home cities,
   category counts, category per city, weighted ratings, loved cities and categories, and a new
   rating-signal chart. The "weighted average rating" is an IMDb-style Bayesian average (m = median votes per
   place), shown next to the notebook's plain mean; both put Keraton Surabaya first.
3. Recommender. The notebook's user-based Pearson/cosine neighbour scoring became a proper rating predictor
   (user kNN), joined by item kNN and TruncatedSVD on the mean-centred matrix. A content model was added
   because the problem statement asks for recommendations from a current place or interests: one-hot
   category and city, log price, TF-IDF of the description.
4. Evaluation: 20% random hold-out of the rating pairs; RMSE for every predictor and three baselines;
   hit-rate@10, precision@5 and recall@10 for each route against held-out ratings >= 4, with the random
   expectation printed beside them.
5. Found that no predictor beats the global mean on four seeds and a full sweep of k, SVD rank and
   shrinkage. Checked why: the ratings are uniform over 1–5, and shuffling them changes nothing about the
   spread of per-place means. Stopped tuning; added shrinkage to kNN so it is not worse than the baseline,
   and reported the result as is.
6. Interface: a user picker (user 1 preselected so the first load shows results), route switch, interests
   form with sample buttons, reason and source pill per place, metric tiles from `metrics.json`, and a
   findings page with the charts and the plain-words answers.
7. Tests and lint, then `just up` / `just demo` / `just down` as the proof path.

## Results
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

## Demo details
- Run: `just setup && just train && just up`, open `http://localhost:8103/`.
- Two-minute walkthrough: (1) the page opens on user 1 with ten picks already shown; open "Places they
  already rated" to see what is excluded; (2) set Route to "Content only", press Recommend, read the reasons
  ("Similar to Museum Bahari Jakarta which you rated 5/5; same category and city"); (3) press "Nature, Bandung,
  ≤20k" for an interests query; (4) open "Findings and charts" and scroll to the rating-signal chart.
- Video clip: `demo/clips/capstone2-tourism-recommender.webm` (recorded by `just demo-video` at the root).
- Sample inputs: user ids 1, 42, 286 (buttons on the page); interests "Cagar alam + Bandung + 20000" returns
  Bukit Jamur, Sanghyang Heuleut, Rainbow Garden, Curug Batu Templek, Tafso Barn. `just demo` prints the
  same calls from the shell.

## Limits and next steps
- Ratings are synthetic, so the collaborative metrics say nothing about tourist taste; the harness is ready
  for real ratings.
- The problem statement mentions "current tourist location"; the content route handles a place name via
  `ContentModel.similar_places`, but the page does not expose it yet, and the latitude/longitude columns
  are unused. A "near me" route by haversine distance is the next step.
- TF-IDF without stemming on Indonesian text; category and city weights were set by hand, not learned.
- Single process, model in memory, no auth or caching; a production version would precompute per-user lists
  nightly and serve them from a store.
