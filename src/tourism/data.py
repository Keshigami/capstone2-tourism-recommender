"""Load and clean the three Part 2 datasets.

Source files are read from DATA_ROOT (default: the Caltech Datasets folder) and never
written. Cleaned copies go to ./data/clean so the app and tests do not depend on openpyxl
at request time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path.home() / "Caltech CTME/PGC AIML - Capstone/Datasets/Capstone 2/Part 2"
CLEAN_DIR = PROJECT_DIR / "data" / "clean"

# Indonesian category labels as they appear in the sheet, with an English gloss for the UI.
CATEGORY_EN = {
    "Budaya": "Culture",
    "Taman hiburan": "Amusement park",
    "Cagar alam": "Nature preserve",
    "Bahari": "Marine / beach",
    "Pusat perbelanjaan": "Shopping centre",
    "Tempat ibadah": "Place of worship",
}


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser()


@dataclass
class Datasets:
    places: pd.DataFrame  # one row per place, cleaned
    users: pd.DataFrame  # one row per user, with a `city` column
    ratings: pd.DataFrame  # one row per (User_Id, Place_Id), duplicates averaged
    raw_rating_rows: int  # before de-duplication
    duplicate_pairs: int  # (user, place) pairs that appeared more than once


def _clean_places(raw: pd.DataFrame) -> pd.DataFrame:
    places = raw.copy()
    places.columns = places.columns.str.strip()
    # Unnamed: 11 is empty, Unnamed: 12 is a serial number, Coordinate duplicates Lat/Long,
    # Time_Minutes is 53% missing. All four are dropped, as in the solution notebook.
    drop = [c for c in places.columns if c.startswith("Unnamed")] + ["Coordinate", "Time_Minutes"]
    places = places.drop(columns=[c for c in drop if c in places.columns])
    places["Category"] = places["Category"].str.strip().str.capitalize()
    places["City"] = places["City"].str.strip()
    places["Place_Name"] = places["Place_Name"].str.strip()
    places["Category_en"] = places["Category"].map(CATEGORY_EN).fillna(places["Category"])
    places = places.drop_duplicates("Place_Id").sort_values("Place_Id").reset_index(drop=True)
    return places


def _clean_users(raw: pd.DataFrame) -> pd.DataFrame:
    users = raw.drop_duplicates("User_Id").copy()
    users["city"] = users["Location"].str.split(",").str[0].str.strip()
    users["province"] = users["Location"].str.split(",").str[-1].str.strip()
    return users.sort_values("User_Id").reset_index(drop=True)


def load_raw(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = root or data_root()
    places = pd.read_excel(root / "tourism_with_id.xlsx")
    ratings = pd.read_csv(root / "tourism_rating.csv")
    users = pd.read_csv(root / "user.csv")
    return places, users, ratings


def load(root: Path | None = None, use_cache: bool = True) -> Datasets:
    """Return cleaned datasets. Uses ./data/clean when present, else reads DATA_ROOT."""
    if use_cache and (CLEAN_DIR / "ratings.csv").exists():
        places = pd.read_csv(CLEAN_DIR / "places.csv")
        users = pd.read_csv(CLEAN_DIR / "users.csv")
        ratings = pd.read_csv(CLEAN_DIR / "ratings.csv")
        meta = pd.read_json(CLEAN_DIR / "meta.json", typ="series")
        return Datasets(places, users, ratings, int(meta["raw_rating_rows"]), int(meta["duplicate_pairs"]))

    raw_places, raw_users, raw_ratings = load_raw(root)
    places = _clean_places(raw_places)
    users = _clean_users(raw_users)
    dup_pairs = int(raw_ratings.duplicated(["User_Id", "Place_Id"]).sum())
    ratings = (
        raw_ratings.groupby(["User_Id", "Place_Id"], as_index=False)["Place_Ratings"]
        .mean()
        .astype({"User_Id": int, "Place_Id": int})
    )
    ratings = ratings[ratings["Place_Id"].isin(places["Place_Id"])].reset_index(drop=True)
    return Datasets(places, users, ratings, len(raw_ratings), dup_pairs)


def save_clean(ds: Datasets, out: Path = CLEAN_DIR) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ds.places.to_csv(out / "places.csv", index=False)
    ds.users.to_csv(out / "users.csv", index=False)
    ds.ratings.to_csv(out / "ratings.csv", index=False)
    pd.Series({"raw_rating_rows": ds.raw_rating_rows, "duplicate_pairs": ds.duplicate_pairs}).to_json(
        out / "meta.json"
    )


def place_ratings_table(ds: Datasets, min_count_quantile: float = 0.5) -> pd.DataFrame:
    """Places joined with their user-rating statistics.

    `mean_rating` is the plain average used in the notebook. `weighted_rating` is the
    IMDb-style Bayesian average  WR = v/(v+m) * R + m/(v+m) * C  where C is the global mean
    and m the median number of ratings per place, so a place with two 5-star votes does not
    outrank one with thirty 4.3-star votes.
    """
    stats = ds.ratings.groupby("Place_Id")["Place_Ratings"].agg(["mean", "count"]).reset_index()
    stats.columns = ["Place_Id", "mean_rating", "n_ratings"]
    c = ds.ratings["Place_Ratings"].mean()
    m = stats["n_ratings"].quantile(min_count_quantile)
    v, r = stats["n_ratings"], stats["mean_rating"]
    stats["weighted_rating"] = (v / (v + m)) * r + (m / (v + m)) * c
    table = ds.places.merge(stats, on="Place_Id", how="left")
    table["n_ratings"] = table["n_ratings"].fillna(0).astype(int)
    table["mean_rating"] = table["mean_rating"].round(2)
    table["weighted_rating"] = table["weighted_rating"].round(3)
    return table
