"""Hold out 20% of ratings; report RMSE per predictor and ranking metrics for the recommender."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PROJECT_DIR, Datasets
from .recommender import CollaborativeFilter, Recommender

METRICS_JSON = PROJECT_DIR / "models" / "metrics.json"
RELEVANT_AT = 4.0  # a held-out rating >= 4 counts as a relevant place


def split(ratings: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ratings))
    n_test = int(round(test_frac * len(ratings)))
    test = ratings.iloc[idx[:n_test]].reset_index(drop=True)
    train = ratings.iloc[idx[n_test:]].reset_index(drop=True)
    return train, test


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def rating_metrics(cf: CollaborativeFilter, test: pd.DataFrame) -> dict:
    m = cf.m
    rows = np.array([m.u_index.get(u, -1) for u in test["User_Id"]])
    cols = np.array([m.p_index.get(p, -1) for p in test["Place_Id"]])
    ok = (rows >= 0) & (cols >= 0)
    truth = test["Place_Ratings"].to_numpy()[ok]
    r, c = rows[ok], cols[ok]
    out = {
        "baseline_global_mean": rmse(np.full(len(truth), cf.global_mean), truth),
        "baseline_user_mean": rmse(m.user_mean[r], truth),
        "baseline_item_mean": rmse(cf.item_mean[c], truth),
    }
    for method in ("user_knn", "item_knn", "svd"):
        out[method] = rmse(cf.predict(method)[r, c], truth)
    return {k: round(v, 4) for k, v in out.items()}


def ranking_metrics(rec: Recommender, test: pd.DataFrame, routes=("collaborative", "content", "hybrid"),
                    k_hit: int = 10, k_prec: int = 5) -> dict:
    """For every test user with >= 1 relevant held-out place: hit-rate@10, precision@5, recall@10."""
    relevant = test[test["Place_Ratings"] >= RELEVANT_AT].groupby("User_Id")["Place_Id"].apply(set)
    relevant = {u: s for u, s in relevant.items() if rec.known_user(u)}
    held = test.groupby("User_Id")["Place_Id"].apply(set).to_dict()
    results = {}
    for route in routes:
        hits = precs = recs = 0.0
        for u, rel in relevant.items():
            top = [r.place_id for r in rec.recommend(user_id=u, n=k_hit, route=route)]
            hits += float(any(p in rel for p in top))
            precs += sum(p in rel for p in top[:k_prec]) / k_prec
            recs += sum(p in rel for p in top) / len(rel)
        n = max(len(relevant), 1)
        results[route] = {f"hit_rate@{k_hit}": round(hits / n, 4), f"precision@{k_prec}": round(precs / n, 4),
                          f"recall@{k_hit}": round(recs / n, 4)}
    # what a random top-10 would score, for scale
    n_places = len(rec.table)
    exp_prec = float(np.mean([len(rel) / (n_places - len(held[u])) for u, rel in relevant.items()]))
    results["random_expected"] = {f"precision@{k_prec}": round(exp_prec, 4),
                                  f"hit_rate@{k_hit}": round(1 - (1 - exp_prec) ** k_hit, 4)}
    results["n_eval_users"] = len(relevant)
    return results


def run(ds: Datasets, out: Path = METRICS_JSON, seed: int = 42) -> dict:
    t0 = time.time()
    train, test = split(ds.ratings, seed=seed)
    train_ds = Datasets(ds.places, ds.users, train, ds.raw_rating_rows, ds.duplicate_pairs)
    rec = Recommender().fit(train_ds)
    metrics = {
        "split": {"train_pairs": len(train), "test_pairs": len(test), "test_frac": 0.2, "seed": seed},
        "rmse": rating_metrics(rec.cf, test),
        "ranking": ranking_metrics(rec, test),
        "relevant_threshold": RELEVANT_AT,
        "model": {"cf_method": rec.cf_method, "k_neighbours": rec.cf.k, "svd_components": rec.cf.svd.n_components,
                  "tfidf_vocab": len(rec.content.tfidf.vocabulary_)},
        "data": {"users": len(ds.users), "places": len(ds.places), "rating_pairs": len(ds.ratings)},
        "eval_seconds": round(time.time() - t0, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_metrics(path: Path = METRICS_JSON) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
