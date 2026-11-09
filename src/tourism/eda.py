"""Exploratory analysis: the notebook's tables as JSON plus six PNG charts under static/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import PROJECT_DIR, Datasets, place_ratings_table  # noqa: E402

STATIC_DIR = PROJECT_DIR / "static"
EDA_JSON = PROJECT_DIR / "models" / "eda.json"

# Reference palette from the dataviz skill (light mode). One hue for magnitude charts,
# fixed categorical slots when identity matters.
# Charts sit on the shared dark panel (common/ui.css): accent teal for magnitude, the
# dataviz dark-mode categorical slots when identity matters.
SURFACE = "#182033"
BLUE = "#4fd1c5"
CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]
INK = "#e8ecf6"
INK2 = "#aab3c8"
GRID = "#2a3347"
LOVED_THRESHOLD = 3.5  # the notebook's cut for "most loved"


def _style(ax: plt.Axes, horizontal: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x" if horizontal else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, name: str, tight: bool = True) -> str:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(STATIC_DIR / name, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return name


def _barh(ax: plt.Axes, labels: list[str], values: list[float], color: str = BLUE, fmt: str = "{:.0f}") -> None:
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=color, height=0.62)
    ax.set_yticks(list(y), labels)
    for yi, v in zip(y, values, strict=True):
        ax.text(v, yi, " " + fmt.format(v), va="center", ha="left", fontsize=9, color=INK)
    ax.set_xlim(0, max(values) * 1.15)
    _style(ax, horizontal=True)


def chart_users_age(users: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    bins = pd.cut(users["Age"], bins=5)
    vc = bins.value_counts().sort_index()
    labels = [f"{int(i.left)+1 if i.left > 17 else 18}-{int(i.right)}" for i in vc.index]
    axes[0].bar(labels, vc.values, color=BLUE, width=0.62)
    for i, v in enumerate(vc.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9, color=INK)
    axes[0].set_title("Users by age band (n=300)", loc="left", color=INK, fontsize=11)
    _style(axes[0])
    axes[1].boxplot(users["Age"], vert=False, widths=0.5, patch_artist=True,
                    boxprops={"facecolor": "#1f4a4a", "edgecolor": BLUE}, medianprops={"color": BLUE})
    axes[1].set_yticks([])
    axes[1].set_title(f"Age: median {users['Age'].median():.0f}, range {users['Age'].min()}-{users['Age'].max()}",
                      loc="left", color=INK, fontsize=11)
    _style(axes[1], horizontal=True)
    return _save(fig, "users_age.png")


def chart_users_city(users: pd.DataFrame, top: int = 15) -> str:
    vc = users["city"].value_counts().head(top)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _barh(ax, list(vc.index), list(vc.values))
    ax.set_title(f"Where the raters come from (top {top} of {users['city'].nunique()} home cities)",
                 loc="left", color=INK, fontsize=11)
    return _save(fig, "users_city.png")


def chart_places_category(places: pd.DataFrame) -> str:
    vc = places["Category"].value_counts()
    labels = [f"{c} ({places.loc[places['Category'] == c, 'Category_en'].iloc[0]})" for c in vc.index]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    _barh(ax, labels, list(vc.values))
    ax.set_title("Places per category (437 places)", loc="left", color=INK, fontsize=11)
    return _save(fig, "places_category.png")


def chart_places_city_category(places: pd.DataFrame) -> str:
    cats = list(places["Category"].value_counts().index)
    cities = list(places["City"].value_counts().index)
    ct = pd.crosstab(places["City"], places["Category"]).reindex(index=cities, columns=cats).fillna(0)
    fig, axes = plt.subplots(1, len(cities), figsize=(14, 3.8), sharey=True)
    for ax, city in zip(axes, cities, strict=True):
        vals = ct.loc[city].values
        ax.bar(range(len(cats)), vals, color=CATEGORICAL[: len(cats)], width=0.7)
        for i, v in enumerate(vals):
            if v:
                ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_title(f"{city} ({int(vals.sum())})", loc="left", color=INK, fontsize=11)
        ax.set_xticks([])
        _style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORICAL[i]) for i in range(len(cats))]
    fig.legend(handles, cats, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(cats), frameon=False,
               fontsize=9, labelcolor=INK)
    fig.suptitle("What each city offers: places by category", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    return _save(fig, "places_city_category.png", tight=False)


def chart_weighted_ratings(table: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    axes[0].hist(table["mean_rating"], bins=20, color=BLUE)
    axes[0].axvline(LOVED_THRESHOLD, color=INK2, linestyle="--", linewidth=1)
    axes[0].text(LOVED_THRESHOLD, axes[0].get_ylim()[1] * 0.95, " 'loved' cut 3.5", color=INK2, fontsize=9)
    axes[0].set_title("Per-place mean user rating (437 places)", loc="left", color=INK, fontsize=11)
    _style(axes[0])
    top = table.sort_values("weighted_rating", ascending=False).head(10)
    labels = [f"{n} · {c}" for n, c in zip(top["Place_Name"], top["City"], strict=True)]
    _barh(axes[1], labels, list(top["weighted_rating"]), fmt="{:.2f}")
    axes[1].set_xlim(0, 5)
    axes[1].set_title("Top 10 by Bayesian weighted rating", loc="left", color=INK, fontsize=11)
    return _save(fig, "ratings_weighted.png")


def chart_top_loved(table: pd.DataFrame) -> str:
    loved = table[table["mean_rating"] > LOVED_THRESHOLD]
    city = loved["City"].value_counts()
    cat = loved["Category"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    _barh(axes[0], list(city.index), list(city.values))
    axes[0].set_title(f"Cities with the most loved spots (mean > {LOVED_THRESHOLD}, n={len(loved)})",
                      loc="left", color=INK, fontsize=11)
    _barh(axes[1], list(cat.index), list(cat.values))
    axes[1].set_title("Categories users love most", loc="left", color=INK, fontsize=11)
    return _save(fig, "top_loved.png")


def signal_check(ratings: pd.DataFrame, n_shuffles: int = 50, seed: int = 0) -> dict:
    """How much structure is in the ratings? Compare the spread of per-place and per-user
    means with the spread you get after shuffling ratings across (user, place) pairs."""
    rng = np.random.default_rng(seed)
    r = ratings["Place_Ratings"].to_numpy()
    obs_place = ratings.groupby("Place_Id")["Place_Ratings"].mean().std()
    obs_user = ratings.groupby("User_Id")["Place_Ratings"].mean().std()
    null_place, null_user = [], []
    for _ in range(n_shuffles):
        sh = ratings.assign(Place_Ratings=rng.permutation(r))
        null_place.append(sh.groupby("Place_Id")["Place_Ratings"].mean().std())
        null_user.append(sh.groupby("User_Id")["Place_Ratings"].mean().std())
    return {
        "place_mean_std_observed": round(float(obs_place), 4),
        "place_mean_std_shuffled": round(float(np.mean(null_place)), 4),
        "user_mean_std_observed": round(float(obs_user), 4),
        "user_mean_std_shuffled": round(float(np.mean(null_user)), 4),
        "rating_entropy_bits": round(float(-(ratings["Place_Ratings"].round().value_counts(normalize=True)
                                              .apply(lambda q: q * np.log2(q))).sum()), 3),
        "max_entropy_bits": round(float(np.log2(5)), 3),
        "n_shuffles": n_shuffles,
    }


def chart_rating_signal(ratings: pd.DataFrame, sig: dict, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    shuffled = ratings.assign(Place_Ratings=rng.permutation(ratings["Place_Ratings"].to_numpy()))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    vc = ratings["Place_Ratings"].round().value_counts().sort_index()
    axes[0].bar([str(int(k)) for k in vc.index], vc.values, color=BLUE, width=0.62)
    for i, v in enumerate(vc.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9, color=INK)
    axes[0].set_title("Ratings 1-5 are almost flat", loc="left", color=INK, fontsize=11)
    _style(axes[0])
    for ax, key, title in ((axes[1], "Place_Id", "Per-place mean"), (axes[2], "User_Id", "Per-user mean")):
        obs = ratings.groupby(key)["Place_Ratings"].mean()
        nul = shuffled.groupby(key)["Place_Ratings"].mean()
        bins = np.linspace(min(obs.min(), nul.min()), max(obs.max(), nul.max()), 24)
        ax.hist(nul, bins=bins, color=INK2, alpha=0.55, label="ratings shuffled at random")
        ax.hist(obs, bins=bins, color=BLUE, alpha=0.75, label="observed")
        k = "place" if key == "Place_Id" else "user"
        ax.set_title(f"{title}: std {sig[f'{k}_mean_std_observed']:.3f} observed vs "
                     f"{sig[f'{k}_mean_std_shuffled']:.3f} shuffled", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        _style(ax)
    fig.suptitle("Is there signal in the ratings? Observed spread vs a shuffled null",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "rating_signal.png", tight=False)


def run(ds: Datasets) -> dict:
    """Build every chart and the findings dict, write models/eda.json, return the dict."""
    places, users = ds.places, ds.users
    table = place_ratings_table(ds)
    loved = table[table["mean_rating"] > LOVED_THRESHOLD]
    nature = places[places["Category"] == "Cagar alam"]
    age_bands = pd.cut(users["Age"], bins=5).value_counts().sort_index()
    sig = signal_check(ds.ratings)
    top_mean = table.sort_values(["mean_rating", "n_ratings"], ascending=False).head(10)
    top_weighted = table.sort_values("weighted_rating", ascending=False).head(10)
    cols = ["Place_Id", "Place_Name", "City", "Category", "Price", "mean_rating", "n_ratings", "weighted_rating"]

    findings = {
        "counts": {
            "places": int(len(places)),
            "users": int(len(users)),
            "rating_rows_raw": ds.raw_rating_rows,
            "rating_pairs_unique": int(len(ds.ratings)),
            "duplicate_user_place_pairs": ds.duplicate_pairs,
            "cities": int(places["City"].nunique()),
            "categories": int(places["Category"].nunique()),
            "ratings_per_user_mean": round(float(ds.ratings.groupby("User_Id").size().mean()), 1),
            "ratings_per_place_mean": round(float(table["n_ratings"].mean()), 1),
            "matrix_density_pct": round(100 * len(ds.ratings) / (len(users) * len(places)), 2),
        },
        "cleaning": [
            "Dropped 'Unnamed: 11' (empty) and 'Unnamed: 12' (serial number) from the places sheet.",
            "Dropped 'Coordinate' (duplicates Lat/Long) and 'Time_Minutes' (232 of 437 missing, 53%).",
            "Trimmed and capitalised Category so 'Budaya' and 'budaya ' are one class.",
            f"Averaged {ds.duplicate_pairs} duplicate (user, place) rating pairs: "
            f"{ds.raw_rating_rows} rows -> {len(ds.ratings)} pairs.",
            "No missing values in users or ratings; no duplicate users or places.",
        ],
        "users": {
            "age_min": int(users["Age"].min()),
            "age_max": int(users["Age"].max()),
            "age_mean": round(float(users["Age"].mean()), 1),
            "age_median": float(users["Age"].median()),
            "age_bands": {str(k): int(v) for k, v in age_bands.items()},
            "home_cities": int(users["city"].nunique()),
            "top_home_cities": users["city"].value_counts().head(10).to_dict(),
            "top_provinces": users["province"].value_counts().head(5).to_dict(),
        },
        "places": {
            "by_category": places["Category"].value_counts().to_dict(),
            "category_en": places.drop_duplicates("Category").set_index("Category")["Category_en"].to_dict(),
            "by_city": places["City"].value_counts().to_dict(),
            "city_by_category": {c: g["Category"].value_counts().to_dict() for c, g in places.groupby("City")},
            "top_category_per_city": {c: g["Category"].value_counts().idxmax() for c, g in places.groupby("City")},
            "nature_preserve_by_city": nature["City"].value_counts().to_dict(),
            "nature_best_city": nature["City"].value_counts().idxmax(),
            "price": {
                "free_places": int((places["Price"] == 0).sum()),
                "median": float(places["Price"].median()),
                "mean": round(float(places["Price"].mean()), 0),
                "max": int(places["Price"].max()),
                "max_place": places.loc[places["Price"].idxmax(), "Place_Name"],
            },
        },
        "ratings": {
            "rating_distribution": {int(k): int(v) for k, v in
                                    ds.ratings["Place_Ratings"].round().value_counts().sort_index().items()},
            "global_mean": round(float(ds.ratings["Place_Ratings"].mean()), 3),
            "place_mean_min": float(table["mean_rating"].min()),
            "place_mean_max": float(table["mean_rating"].max()),
            "sheet_rating_vs_user_rating_corr": round(float(table["Rating"].corr(table["mean_rating"])), 3),
            "signal": sig,
            "top10_by_mean": top_mean[cols].to_dict(orient="records"),
            "top10_by_weighted": top_weighted[cols].to_dict(orient="records"),
            "highest_rated_place": top_mean.iloc[0]["Place_Name"],
            "loved_threshold": LOVED_THRESHOLD,
            "loved_count": int(len(loved)),
            "loved_by_city": loved["City"].value_counts().to_dict(),
            "loved_by_category": loved["Category"].value_counts().to_dict(),
            "most_loved_city": loved["City"].value_counts().idxmax(),
            "most_loved_category": loved["Category"].value_counts().idxmax(),
            "loved_share_by_category_pct": {
                c: round(100 * float(loved["Category"].eq(c).sum()) / float(n), 1)
                for c, n in places["Category"].value_counts().items()
            },
        },
        "charts": [
            chart_users_age(users),
            chart_users_city(users),
            chart_places_category(places),
            chart_places_city_category(places),
            chart_weighted_ratings(table),
            chart_top_loved(table),
            chart_rating_signal(ds.ratings, sig),
        ],
    }
    EDA_JSON.parent.mkdir(parents=True, exist_ok=True)
    EDA_JSON.write_text(json.dumps(findings, indent=2, default=_json_default))
    return findings


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def load_findings(path: Path = EDA_JSON) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
