"""FastAPI service for the tourism recommender (port 8103)."""

from __future__ import annotations

import pickle
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from tourism import data, eda, evaluate
from tourism.recommender import Recommender
from tourism.train import MODEL_PATH

SERVICE = "capstone2-tourism-recommender"
PORT = 8103
HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))
state: dict = {"rec": None, "metrics": None, "eda": None, "model_loaded": False, "source": None}


def load_model() -> Recommender:
    if MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as fh:
            state["source"] = "models/recommender.pkl"
            return pickle.load(fh)
    state["source"] = "fitted at startup (run `just train` for metrics and charts)"
    return Recommender().fit(data.load())


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["rec"] = load_model()
    state["model_loaded"] = True
    state["metrics"] = evaluate.load_metrics()
    state["eda"] = eda.load_findings()
    yield


app = FastAPI(title="Tourism recommender", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


class RecommendRequest(BaseModel):
    user_id: int | None = Field(None, description="Known user id (1-300). Omit for an interests query.")
    category: str | None = Field(None, description="Indonesian category label, e.g. 'Cagar alam'")
    city: str | None = None
    budget: float | None = Field(None, ge=0, description="Max ticket price in IDR")
    n: int = Field(10, ge=1, le=50)
    route: str = Field("hybrid", pattern="^(hybrid|collaborative|content|interests)$")


def _rec() -> Recommender:
    if state["rec"] is None:
        raise HTTPException(503, "model not loaded")
    return state["rec"]


def _base_ctx(request: Request) -> dict:
    m = state["metrics"] or {}
    return {"request": request, "service": SERVICE, "port": PORT, "metrics": m, "model_source": state["source"],
            "options": _rec().options()}


@app.get("/health")
def health():
    return {"ok": True, "service": SERVICE, "model_loaded": bool(state["model_loaded"])}


@app.get("/metrics")
def metrics():
    if not state["metrics"]:
        raise HTTPException(404, "models/metrics.json missing; run `just train`")
    return state["metrics"]


@app.get("/api/options")
def options():
    return _rec().options()


@app.get("/api/users/{user_id}")
def user(user_id: int):
    rec = _rec()
    if not rec.known_user(user_id):
        raise HTTPException(404, f"unknown user {user_id}")
    return {"profile": rec.user_profile(user_id), "rated": rec.user_history(user_id).to_dict(orient="records")}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    rec = _rec()
    if req.user_id is None and not (req.category or req.city or req.budget is not None):
        raise HTTPException(422, "give a user_id or at least one of category, city, budget")
    if req.user_id is not None and not rec.known_user(req.user_id):
        raise HTTPException(404, f"unknown user {req.user_id}; valid ids are 1-{max(rec.user_ids())}")
    recs = rec.recommend(req.user_id, req.category, req.city, req.budget, req.n, req.route)
    mode = "user" if req.user_id is not None and req.route != "interests" else "interests"
    return {"mode": mode, "route": req.route if mode == "user" else "content",
            "query": req.model_dump(), "recommendations": [r.as_dict() for r in recs]}


@app.get("/", response_class=HTMLResponse)
def index(request: Request, user_id: int | None = None, category: str | None = None, city: str | None = None,
          budget: str | None = None, route: str = "hybrid", mode: str | None = None):
    rec = _rec()
    ctx = _base_ctx(request)
    ctx.update({"mode": None, "error": None, "recs": [], "profile": None, "rated": [], "route": route,
                "form": {"user_id": user_id, "category": category or "", "city": city or "", "budget": budget or ""}})
    budget_val = None
    if budget not in (None, ""):
        try:
            budget_val = float(budget)
        except ValueError:
            ctx["error"] = f"Budget '{budget}' is not a number. Enter an amount in rupiah, e.g. 20000."
            return templates.TemplateResponse(request, "index.html", ctx)
    if mode == "interests" or (user_id is None and (category or city or budget_val is not None)):
        ctx["mode"] = "interests"
        ctx["recs"] = [r.as_dict() for r in rec.by_interests(category or None, city or None, budget_val)]
        if not ctx["recs"]:
            ctx["error"] = "No place matches those interests. Loosen the budget or drop the city."
        return templates.TemplateResponse(request, "index.html", ctx)
    if user_id is None:
        user_id = 1  # the notebook's picked user, so the page shows a real result on first load
        ctx["form"]["user_id"] = user_id
    ctx["mode"] = "user"
    if not rec.known_user(user_id):
        ctx["error"] = f"User {user_id} is not in the ratings file. Pick an id between 1 and {max(rec.user_ids())}."
        return templates.TemplateResponse(request, "index.html", ctx)
    if route not in ("hybrid", "collaborative", "content"):
        route = "hybrid"
    ctx["route"] = route
    ctx["profile"] = rec.user_profile(user_id)
    ctx["rated"] = rec.user_history(user_id).to_dict(orient="records")
    ctx["recs"] = [r.as_dict() for r in rec.recommend(user_id=user_id, route=route)]
    return templates.TemplateResponse(request, "index.html", ctx)


@app.get("/eda", response_class=HTMLResponse)
def eda_page(request: Request):
    ctx = _base_ctx(request)
    ctx["f"] = state["eda"]
    return templates.TemplateResponse(request, "eda.html", ctx)


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"ok": False, "error": exc.detail}, status_code=exc.status_code)
