<!-- public-top -->
# Tourism recommender (Capstone 2, Part 2)

Explores 437 Indonesian tourist spots and 10k user ratings, then recommends places to a known user or to someone who only states a category, city and budget, with a reason for every pick. Caltech CTME PGC AIML, capstone.

**Watch it first:** [the 90-second film](demo/film.mp4) (what it does, how it is built, the proof, the answer) or the [60-second cut](demo/film-60s.mp4). The plain walkthrough with captions is [demo/demo.mp4](demo/demo.mp4).

## Try it in three commands

```bash
git clone https://github.com/Keshigami/capstone2-tourism-recommender.git && cd capstone2-tourism-recommender
just setup        # creates the venv with uv (Python 3.11)
just up           # http://127.0.0.1:8103/  (just down to stop)
```

You need [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just) (`brew install uv just` on a Mac). The dataset (`data/`) and the trained models (`models/`) are in the repo, so the app runs as is; `just train` retrains. No API key is needed anywhere in this project. `just test` runs the lint and the offline tests.

Step by step: [docs/Getting-Started.md](docs/Getting-Started.md), then [how it was built](docs/How-It-Was-Built.md), [results](docs/Results.md), [limits](docs/Limits-and-Next-Steps.md) and [the film](docs/The-Film.md). The same pages are on the [wiki](https://github.com/Keshigami/capstone2-tourism-recommender/wiki).
<!-- /public-top -->
# Capstone 2, Part 2: Indonesian tourism EDA and recommender

## Demo walkthrough
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

Port **8103**. FastAPI service with an HTML front end: pick a user and see what they rated and ten places to
visit next, with a reason per place; or enter interests (category, city, budget) and get matching places.
A findings page reproduces the solution notebook's analysis with seven charts.

## What it does
- Cleans the three Part 2 files (`tourism_with_id.xlsx`, `tourism_rating.csv`, `user.csv`) the way the
  solution notebook does, plus averaging duplicate (user, place) rating pairs.
- Reproduces the notebook's EDA (users by age and home city, places by category and city, nature-preserve
  city, weighted ratings, most-loved places, cities and categories) as JSON findings and PNG charts.
- Trains two recommender routes and evaluates them on a 20% hold-out.
- Serves it all on `http://127.0.0.1:8103`.

## Data
`DATA_ROOT` defaults to `~/Caltech CTME/PGC AIML - Capstone/Datasets/Capstone 2/Part 2` and is read-only.
Cleaned copies land in `./data/clean/` (git-ignored) on `just train`.

| File | Rows | After cleaning |
|---|---|---|
| tourism_with_id.xlsx | 437 places, 13 columns | 10 columns: dropped `Unnamed: 11` (empty), `Unnamed: 12` (serial), `Coordinate` (duplicates Lat/Long), `Time_Minutes` (232/437 missing); Category trimmed and capitalised |
| user.csv | 300 users | added `city` and `province` from `Location` |
| tourism_rating.csv | 10,000 rows | 9,597 unique (user, place) pairs; 403 duplicate pairs averaged |

The user x place matrix is 7.3% filled (about 32 ratings per user, 22 per place).

## Findings (numbers from `models/eda.json`)
- **Raters**: 300 users aged 18–40 (mean 28.7, median 29); 28 home cities, led by Bekasi (39), Semarang (22),
  Lampung (20), Yogyakarta (20). 116 come from Jawa Barat, 46 from DKI Jakarta: domestic weekend tourists.
- **Catalogue**: 437 places in 5 cities and 6 categories. Taman hiburan / amusement parks 135, Budaya / culture
  117, Cagar alam / nature preserves 106, Bahari / marine 47, Tempat ibadah 17, Pusat perbelanjaan 15.
  Yogyakarta (126) and Bandung (124) hold 57% of places.
- **City specialities**: Bandung = nature preserves (54 of 106, the clear choice for a nature enthusiast);
  Jakarta = culture (32) and 10 of the 15 shopping centres; Yogyakarta = amusement parks (36) and 34 of the
  47 beaches; Semarang = nature (20); Surabaya = amusement parks and culture (18 each).
- **Prices**: 137 places are free; median ticket Rp 5,000; most expensive is Pulau Pelangi at Rp 900,000.
- **Most loved** (notebook definition: mean user rating > 3.5): 31 places. Highest mean rating is
  **Keraton Surabaya** (3.94), also first by Bayesian weighted rating (3.56). **Bandung** has the most loved
  spots (11), then Yogyakarta (10). **Amusement parks** are the most loved category (13), closely followed by
  nature preserves (9), matching the notebook's conclusion.
- **The ratings carry almost no signal.** Stars 1–5 occur 1,582 / 2,044 / 2,011 / 2,079 / 1,881 times
  (entropy 2.315 of a maximum 2.322 bits). The per-place mean spread is 0.308 observed vs 0.299 after
  shuffling all ratings at random; per-user 0.293 vs 0.245. Per-place user means correlate 0.008 with the
  sheet's own public rating. The rating file is synthetic noise around 3.07. `static/rating_signal.png`
  shows this.

## Recommender design
`src/tourism/recommender.py`.

**Collaborative route** (needs a known user id), all on the mean-centred user x place matrix:
- user-based cosine kNN (the notebook's approach turned into a rating predictor), k=10 positive neighbours,
  shrinkage 3 on the weight sum so a prediction backed by one neighbour stays near the user's mean;
- item-based cosine kNN, same settings;
- TruncatedSVD matrix factorisation, rank 2 (the default predictor; higher ranks only fit noise here).

**Content route** (works for a known user or a cold interests query): place vectors = one-hot category
(weight 1.0) + one-hot city (0.7) + scaled log price as two columns (0.3) + TF-IDF of the Indonesian
description with a small stop-word list (1.0), L2-normalised. For a user, the profile is the places they rated
4–5 weighted by how far above their mean; for interests, a query vector from category, city and budget with
hard filters on those fields and a 30% weight on the weighted popularity so ties break toward well-rated places.

**Hybrid** (default for a known user): reciprocal-rank fusion of the collaborative and content top-30 lists;
the reason shown comes from the route that ranked the place higher, and "hybrid" marks places on both lists.
Rated places are always excluded.

### Metrics (`models/metrics.json`, 20% random hold-out of the 9,597 pairs, seed 42)
| Predictor | RMSE |
|---|---|
| global mean (baseline) | **1.363** |
| user mean (baseline) | 1.383 |
| item mean (baseline) | 1.403 |
| user kNN | 1.384 |
| item kNN | 1.382 |
| SVD rank 2 | 1.384 |

| Ranker (280 test users with a held-out rating >= 4) | hit-rate@10 | precision@5 | recall@10 |
|---|---|---|---|
| collaborative (SVD) | 0.054 | 0.0071 | 0.019 |
| content | 0.050 | 0.0029 | 0.021 |
| hybrid | 0.054 | 0.0029 | 0.020 |
| random top-10, expected | 0.065 | 0.0067 | – |

Read this as: nothing beats guessing the global mean, and every ranker is at the random level. A sweep over
k in {5..100}, SVD rank in {2..40} and shrinkage in {1..30}, on four seeds, never changed that. It is a
property of the data (see the findings), not something more tuning can fix, so the numbers are left as they
are with the baselines beside them. The content route is the one that gives sensible, explainable
suggestions because it uses the catalogue (category, city, price, description) rather than the ratings.

## How to run
```
just setup          # uv sync, copy the shared stylesheet into static/
just train          # clean data, 7 charts, models/eda.json, models/metrics.json, models/recommender.pkl (about 2 s)
just test           # pytest, offline, 9 tests
just up             # background server on 8103 (pid in .run.pid, log in .run.log)
just demo           # scripted calls: health, metrics, user 1, interests query
just down
just run            # foreground server
uv run ruff check . # lint
```
`just train --full` is accepted and identical to `--quick`: the whole pipeline takes seconds.

API: `GET /health`, `GET /metrics`, `GET /api/options`, `GET /api/users/{id}`, `POST /recommend` with
`{"user_id": 1, "n": 10, "route": "hybrid|collaborative|content"}` or
`{"category": "Cagar alam", "city": "Bandung", "budget": 20000}`. Swagger at `/docs`.

## Two-minute interviewer walkthrough
1. Open `http://127.0.0.1:8103/`. User 1 is preselected and ten recommendations are already on screen, each
   tagged collaborative, content or hybrid with a one-line reason. Expand "Places they already rated" to see
   the 29 places that are excluded.
2. Switch Route to "Content only" and press Recommend: every reason now names the rated place it is similar to
   and whether they share category or city.
3. Click "Nature, Bandung, ≤20k" under the interests form: ten nature preserves in Bandung at or under
   Rp 20,000, ordered by similarity and weighted rating.
4. Open "Findings and charts": the six notebook findings in plain words, then the rating-signal chart that
   explains why the metric tiles on the first page sit at the baseline.

## Limits
- The ratings are synthetic, so the collaborative route cannot be judged on this file; on real ratings the
  same code (kNN, SVD, hold-out RMSE, hit-rate) would be the evaluation harness.
- TF-IDF uses a 60-word Indonesian stop list and no stemming; descriptions still dominate similarity for
  long texts, which is why category and city get explicit weights and hard filters.
- No time or geography in the model: `Lat`/`Long` are kept in the data but unused; a next step is a
  "near my current place" route using the haversine distance, which is what the problem statement's
  "current tourist location" hints at.
- One process, in-memory model, no auth; fine for a demo, not a deployment.

## Standalone repo

Exported from Max's `caltech-capstone` workspace. `common/` is vendored (LLM door, UI tokens). Run: `just setup && just test && just up`.
LLM-backed features read `ANTHROPIC_API_KEY` (or `LLM_PROVIDER` + `OPENAI_API_KEY` / `GOOGLE_API_KEY`); with no key they run in a deterministic offline mode.
Dataset: included under `data/`; run with `DATA_ROOT=data` (the justfile sets it when the folder exists).
Film: `demo/film.mp4` (90 s, what it does, how it is built, the proof, the answer) and `demo/film-60s.mp4` (60 s cut); `demo/demo.mp4` is the plain walkthrough.
