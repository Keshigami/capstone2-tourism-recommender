"""Build every artifact: cleaned data, EDA charts + findings, the fitted recommender, metrics.

    uv run python -m tourism.train [--quick | --full]

Both modes finish in seconds on this dataset (437 places x 300 users); the flags exist so the
suite-wide `just train --quick` works the same way in every project.
"""

from __future__ import annotations

import argparse
import pickle
import time

from . import data, eda, evaluate
from .recommender import Recommender

MODEL_PATH = data.PROJECT_DIR / "models" / "recommender.pkl"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="same as --full here; kept for suite parity")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    t0 = time.time()

    ds = data.load(use_cache=False)
    data.save_clean(ds)
    print(f"data: {len(ds.places)} places, {len(ds.users)} users, {len(ds.ratings)} rating pairs "
          f"({ds.duplicate_pairs} duplicate pairs averaged)")

    findings = eda.run(ds)
    print(f"eda: {len(findings['charts'])} charts -> static/, findings -> models/eda.json")

    metrics = evaluate.run(ds, seed=args.seed)
    print("rmse:", metrics["rmse"])
    print("ranking:", {k: v for k, v in metrics["ranking"].items()})

    rec = Recommender().fit(ds)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as fh:
        pickle.dump(rec, fh)
    print(f"model -> {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
