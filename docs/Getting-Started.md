# Getting started

## What you need
- [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just). On a Mac: `brew install uv just`.
- Python 3.11 is installed by uv.
- No API key is needed anywhere in this project.

## Three commands
```bash
git clone https://github.com/Keshigami/capstone2-tourism-recommender.git && cd capstone2-tourism-recommender
just setup
just up           # http://127.0.0.1:8103/  (just down to stop)
```
The dataset (`data/`) and the trained models (`models/`) are in the repo, so the app runs as is; `just train` retrains.

## Commands
`just setup`, `just train` where a model is trained, `just up` / `just down`, `just test`, `just demo`.

## API
API: `GET /health`, `GET /metrics`, `GET /api/options`, `GET /api/users/{id}`, `POST /recommend` with
`{"user_id": 1, "n": 10, "route": "hybrid|collaborative|content"}` or
`{"category": "Cagar alam", "city": "Bandung", "budget": 20000}`. Swagger at `/docs`.

## What you see
- Run: `just setup && just train && just up`, open `http://localhost:8103/`.
- Two-minute walkthrough: (1) the page opens on user 1 with ten picks already shown; open "Places they
  already rated" to see what is excluded; (2) set Route to "Content only", press Recommend, read the reasons
  ("Similar to Museum Bahari Jakarta which you rated 5/5; same category and city"); (3) press "Nature, Bandung,
  ≤20k" for an interests query; (4) open "Findings and charts" and scroll to the rating-signal chart.
- Video clip: `demo/clips/capstone2-tourism-recommender.webm` (recorded by `just demo-video` at the root).
- Sample inputs: user ids 1, 42, 286 (buttons on the page); interests "Cagar alam + Bandung + 20000" returns
  Bukit Jamur, Sanghyang Heuleut, Rainbow Garden, Curug Batu Templek, Tafso Barn. `just demo` prints the
  same calls from the shell.

## Try this
Try it: Nature, Bandung, under 20k.
