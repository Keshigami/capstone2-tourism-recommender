# How it was built

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

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language and runtime | Python 3.11 via uv | one lockfile, same across the suite |
| Model or core logic | pandas + scikit-learn: cosine kNN, TruncatedSVD, TF-IDF content model, rank fusion | small data (300 x 437), fits in two seconds, fully inspectable |
| Interface | FastAPI + Jinja2 HTML, shared `common/ui.css` | no build step, scriptable ids for the demo video |
| Data | `tourism_with_id.xlsx` (437 places), `user.csv` (300), `tourism_rating.csv` (10,000 rows -> 9,597 pairs) | the course dataset, read-only from DATA_ROOT |
| Tests | pytest + FastAPI TestClient, 9 tests, offline | health, metrics, recommend contracts, page states |

## Libraries
`pandas` (cleaning, group-bys, the user x place pivot), `numpy` (similarity matrices and predictions),
`scikit-learn` (cosine_similarity, TruncatedSVD, TfidfVectorizer), `scipy` (sparse stacking of the content
features), `openpyxl` (read the xlsx), `matplotlib` (7 PNG charts), `fastapi` + `uvicorn` (API and pages),
`jinja2` (templates), `pydantic` (request validation), `httpx` (demo script and TestClient), `pytest`,
`ruff`.
