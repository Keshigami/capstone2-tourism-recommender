"""Collaborative and content-based recommenders for the tourism ratings.

Collaborative route (needs a known user):
  * user-based cosine kNN on the mean-centred user x place matrix (the notebook's approach,
    made into a rating predictor);
  * item-based cosine kNN on the same matrix, transposed;
  * TruncatedSVD matrix factorisation of the centred matrix.
Content route (works for a known user or a cold "interests" query):
  * place vectors = one-hot category + one-hot city + scaled log price + TF-IDF of the
    Indonesian description; cosine similarity between places, and between a user's
    liked places (or an interest query) and all places.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .data import CATEGORY_EN, Datasets, place_ratings_table

# Frequent Indonesian function words so TF-IDF keys on content words.
ID_STOPWORDS = """yang dan di ke dari ini itu dengan untuk pada adalah atau juga tidak ada akan
sebagai oleh dalam bisa dapat karena telah sudah seperti para lebih tersebut serta hingga saat
merupakan menjadi sebuah salah satu nya berada terletak memiliki mulai sampai antara bagi agar
kota tempat wisata kawasan""".split()


@dataclass
class RatingMatrix:
    users: np.ndarray  # user ids in row order
    places: np.ndarray  # place ids in column order
    R: np.ndarray  # raw ratings, nan where unrated
    mask: np.ndarray  # bool, rated
    user_mean: np.ndarray
    centred: np.ndarray  # R - user_mean, 0 where unrated
    u_index: dict = field(default_factory=dict)
    p_index: dict = field(default_factory=dict)

    @classmethod
    def from_ratings(cls, ratings: pd.DataFrame, place_ids: np.ndarray) -> RatingMatrix:
        users = np.sort(ratings["User_Id"].unique())
        u_index = {u: i for i, u in enumerate(users)}
        p_index = {p: j for j, p in enumerate(place_ids)}
        R = np.full((len(users), len(place_ids)), np.nan)
        for u, p, r in ratings[["User_Id", "Place_Id", "Place_Ratings"]].itertuples(index=False):
            if p in p_index:
                R[u_index[u], p_index[p]] = r
        mask = ~np.isnan(R)
        user_mean = np.where(mask.sum(1) > 0, np.nanmean(np.where(mask, R, np.nan), axis=1), 0.0)
        centred = np.where(mask, R - user_mean[:, None], 0.0)
        return cls(users, place_ids, R, mask, user_mean, centred, u_index, p_index)


class CollaborativeFilter:
    """User-kNN, item-kNN and SVD rating predictors sharing one rating matrix."""

    def __init__(self, k: int = 10, n_components: int = 2, shrink: float = 3.0, random_state: int = 42):
        # shrink: pseudo-count added to the neighbour weight sum so a prediction backed by
        # one or two neighbours stays close to the user's own mean (ratings here are noisy).
        self.k, self.n_components, self.shrink, self.random_state = k, n_components, shrink, random_state

    def fit(self, ratings: pd.DataFrame, place_ids: np.ndarray) -> CollaborativeFilter:
        self.m = RatingMatrix.from_ratings(ratings, place_ids)
        self.global_mean = float(np.nanmean(self.m.R))
        self.item_mean = np.where(self.m.mask.sum(0) > 0, np.nanmean(self.m.R, axis=0), self.global_mean)
        self.user_sim = cosine_similarity(self.m.centred)
        np.fill_diagonal(self.user_sim, 0.0)
        self.item_sim = cosine_similarity(self.m.centred.T)
        np.fill_diagonal(self.item_sim, 0.0)
        n_comp = min(self.n_components, min(self.m.centred.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state).fit(self.m.centred)
        self.svd_recon = self.svd.transform(self.m.centred) @ self.svd.components_
        return self

    # --- full prediction matrices (users x places), clipped to the 1..5 scale -----------
    def predict_user_knn(self) -> np.ndarray:
        sim = self.user_sim.copy()
        if self.k < sim.shape[0]:
            thresh = -np.sort(-sim, axis=1)[:, self.k - 1][:, None]
            sim[sim < thresh] = 0.0
        sim = np.clip(sim, 0, None)  # only positively similar neighbours vote
        num = sim @ self.m.centred
        den = np.abs(sim) @ self.m.mask.astype(float) + self.shrink
        adj = num / den
        return np.clip(self.m.user_mean[:, None] + adj, 1, 5)

    def predict_item_knn(self) -> np.ndarray:
        sim = self.item_sim.copy()
        if self.k < sim.shape[0]:
            thresh = -np.sort(-sim, axis=1)[:, self.k - 1][:, None]
            sim[sim < thresh] = 0.0
        sim = np.clip(sim, 0, None)
        num = self.m.centred @ sim.T
        den = self.m.mask.astype(float) @ np.abs(sim).T + self.shrink
        adj = num / den
        return np.clip(self.m.user_mean[:, None] + adj, 1, 5)

    def predict_svd(self) -> np.ndarray:
        return np.clip(self.m.user_mean[:, None] + self.svd_recon, 1, 5)

    def predict(self, method: str = "svd") -> np.ndarray:
        return {"user_knn": self.predict_user_knn, "item_knn": self.predict_item_knn, "svd": self.predict_svd}[method]()

    def neighbours(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        i = self.m.u_index[user_id]
        order = np.argsort(-self.user_sim[i])[:n]
        return [(int(self.m.users[j]), float(self.user_sim[i, j])) for j in order if self.user_sim[i, j] > 0]


class ContentModel:
    """Place feature vectors and similarity; also answers cold interest queries."""

    def __init__(self, w_category: float = 1.0, w_city: float = 0.7, w_price: float = 0.3, w_text: float = 1.0):
        self.w = dict(category=w_category, city=w_city, price=w_price, text=w_text)

    def fit(self, places: pd.DataFrame) -> ContentModel:
        self.places = places.reset_index(drop=True)
        self.place_ids = self.places["Place_Id"].to_numpy()
        self.categories = sorted(self.places["Category"].unique())
        self.cities = sorted(self.places["City"].unique())
        self.cat_index = {c: i for i, c in enumerate(self.categories)}
        self.city_index = {c: i for i, c in enumerate(self.cities)}
        self.log_price_max = float(np.log1p(self.places["Price"]).max())
        self.tfidf = TfidfVectorizer(stop_words=ID_STOPWORDS, min_df=2, max_features=3000, sublinear_tf=True)
        text = self.tfidf.fit_transform(self.places["Description"].fillna(""))
        cat = csr_matrix(np.eye(len(self.categories))[self.places["Category"].map(self.cat_index)])
        city = csr_matrix(np.eye(len(self.cities))[self.places["City"].map(self.city_index)])
        price = csr_matrix(self._price_feat(self.places["Price"].to_numpy()))
        self.X = normalize(hstack([cat * self.w["category"], city * self.w["city"], price * self.w["price"],
                                   text * self.w["text"]]).tocsr())
        self.sim = cosine_similarity(self.X)
        np.fill_diagonal(self.sim, 0.0)
        return self

    def _price_feat(self, price: np.ndarray) -> np.ndarray:
        """Two columns: scaled log price and its complement, so free and cheap places are close."""
        p = np.log1p(np.asarray(price, dtype=float)) / self.log_price_max
        return np.column_stack([p, 1 - p])

    def query_vector(self, category: str | None, city: str | None, budget: float | None) -> np.ndarray:
        cat = np.zeros(len(self.categories))
        if category and category in self.cat_index:
            cat[self.cat_index[category]] = 1.0
        cty = np.zeros(len(self.cities))
        if city and city in self.city_index:
            cty[self.city_index[city]] = 1.0
        price = self._price_feat(np.array([budget if budget is not None else self.places["Price"].median()]))[0]
        text = np.zeros(len(self.tfidf.vocabulary_))
        v = np.concatenate([cat * self.w["category"], cty * self.w["city"], price * self.w["price"],
                            text * self.w["text"]])
        n = np.linalg.norm(v)
        return v / n if n else v

    def similar_places(self, place_id: int, n: int = 10) -> list[tuple[int, float]]:
        j = int(np.where(self.place_ids == place_id)[0][0])
        order = np.argsort(-self.sim[j])[:n]
        return [(int(self.place_ids[i]), float(self.sim[j, i])) for i in order]


@dataclass
class Recommendation:
    place_id: int
    name: str
    city: str
    category: str
    category_en: str
    price: int
    score: float
    source: str  # "collaborative" | "content" | "hybrid"
    reason: str
    mean_rating: float
    n_ratings: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Recommender:
    """Facade used by the API: fits both routes and explains each recommendation."""

    def __init__(self, cf_method: str = "svd", k: int = 10, n_components: int = 2, shrink: float = 3.0):
        self.cf_method = cf_method
        self.cf = CollaborativeFilter(k=k, n_components=n_components, shrink=shrink)
        self.content = ContentModel()

    def fit(self, ds: Datasets) -> Recommender:
        self.table = place_ratings_table(ds).reset_index(drop=True)
        self.content.fit(self.table)
        self.cf.fit(ds.ratings, self.content.place_ids)
        self.pred = {m: self.cf.predict(m) for m in ("user_knn", "item_knn", "svd")}
        self.ratings = ds.ratings
        self.users = ds.users
        self.pop = self.table["weighted_rating"].to_numpy()
        self.pop_norm = (self.pop - self.pop.min()) / (np.ptp(self.pop) or 1)
        return self

    # --- helpers ---------------------------------------------------------------------
    def user_ids(self) -> list[int]:
        return [int(u) for u in self.cf.m.users]

    def known_user(self, user_id: int) -> bool:
        return user_id in self.cf.m.u_index

    def user_history(self, user_id: int) -> pd.DataFrame:
        h = self.ratings[self.ratings["User_Id"] == user_id].merge(
            self.table[["Place_Id", "Place_Name", "City", "Category", "Category_en", "Price"]], on="Place_Id"
        )
        return h.sort_values(["Place_Ratings", "Place_Name"], ascending=[False, True]).reset_index(drop=True)

    def user_profile(self, user_id: int) -> dict:
        h = self.user_history(user_id)
        liked = h[h["Place_Ratings"] >= 4]
        row = self.users[self.users["User_Id"] == user_id]
        return {
            "user_id": int(user_id),
            "age": int(row["Age"].iloc[0]) if len(row) else None,
            "location": row["Location"].iloc[0] if len(row) else None,
            "n_rated": int(len(h)),
            "mean_rating": round(float(h["Place_Ratings"].mean()), 2) if len(h) else None,
            "favourite_category": liked["Category"].mode().iloc[0] if len(liked) else None,
            "favourite_city": liked["City"].mode().iloc[0] if len(liked) else None,
        }

    def _row(self, j: int) -> pd.Series:
        return self.table.iloc[j]

    def _make(self, j: int, score: float, source: str, reason: str) -> Recommendation:
        r = self._row(j)
        return Recommendation(int(r["Place_Id"]), r["Place_Name"], r["City"], r["Category"], r["Category_en"],
                              int(r["Price"]), round(float(score), 4), source, reason,
                              float(r["mean_rating"]) if pd.notna(r["mean_rating"]) else 0.0, int(r["n_ratings"]))

    # --- collaborative route ---------------------------------------------------------
    def collaborative(self, user_id: int, n: int = 10, method: str | None = None) -> list[Recommendation]:
        method = method or self.cf_method
        i = self.cf.m.u_index[user_id]
        scores = self.pred[method][i].copy()
        scores[self.cf.m.mask[i]] = -np.inf
        order = np.argsort(-scores)[:n]
        out = []
        for j in order:
            raters = int(self.cf.m.mask[:, j].sum())
            reason = (f"Predicted {scores[j]:.1f}/5 for you by {self._label(method)}; "
                      f"{raters} users rated it, average {self._row(j)['mean_rating']:.2f}.")
            out.append(self._make(j, scores[j], "collaborative", reason))
        return out

    @staticmethod
    def _label(method: str) -> str:
        return {"svd": "matrix factorisation (SVD)", "user_knn": "users with similar taste",
                "item_knn": "places rated like yours"}[method]

    # --- content route ---------------------------------------------------------------
    def content_for_user(self, user_id: int, n: int = 10) -> list[Recommendation]:
        i = self.cf.m.u_index[user_id]
        rated = self.cf.m.mask[i]
        weights = self.cf.m.centred[i] * rated  # above-mean ratings pull, below-mean push
        liked = np.where((self.cf.m.R[i] >= 4) & rated)[0]
        if len(liked) == 0:
            liked = np.where(rated)[0]
        scores = (self.content.sim[liked].T @ np.clip(weights[liked], 0.1, None)) / max(len(liked), 1)
        scores = scores + 0.05 * self.pop_norm
        scores[rated] = -np.inf
        order = np.argsort(-scores)[:n]
        out = []
        for j in order:
            anchor = liked[np.argmax(self.content.sim[liked, j])]
            a = self._row(anchor)
            reason = (f"Similar to {a['Place_Name']} ({a['Category_en']}, {a['City']}) which you rated "
                      f"{self.cf.m.R[i, anchor]:.0f}/5; same {self._shared(a, self._row(j))}.")
            out.append(self._make(j, scores[j], "content", reason))
        return out

    @staticmethod
    def _shared(a: pd.Series, b: pd.Series) -> str:
        parts = []
        if a["Category"] == b["Category"]:
            parts.append("category")
        if a["City"] == b["City"]:
            parts.append("city")
        return " and ".join(parts) if parts else "kind of description"

    def by_interests(self, category: str | None = None, city: str | None = None, budget: float | None = None,
                     n: int = 10) -> list[Recommendation]:
        q = self.content.query_vector(category, city, budget)
        sim = np.asarray(self.content.X @ q).ravel()
        scores = 0.7 * sim + 0.3 * self.pop_norm
        keep = np.ones(len(scores), dtype=bool)
        if category:
            keep &= (self.table["Category"] == category).to_numpy()
        if city:
            keep &= (self.table["City"] == city).to_numpy()
        if budget is not None:
            keep &= (self.table["Price"] <= budget).to_numpy()
        if keep.sum() == 0:  # relax the hard filters, keep the soft similarity ranking
            keep[:] = True
        scores[~keep] = -np.inf
        order = np.argsort(-scores)[:n]
        out = []
        for j in order:
            r = self._row(j)
            bits = []
            if category:
                bits.append(f"{r['Category_en']}")
            if city:
                bits.append(f"in {r['City']}")
            if budget is not None:
                bits.append("free" if r["Price"] == 0 else f"Rp {int(r['Price']):,} within budget")
            bits.append(f"rated {r['mean_rating']:.2f} by {int(r['n_ratings'])} users")
            out.append(self._make(j, scores[j], "content", "Matches your interests: " + ", ".join(bits) + "."))
        return out

    # --- hybrid for a known user ------------------------------------------------------
    def hybrid(self, user_id: int, n: int = 10, method: str | None = None) -> list[Recommendation]:
        """Reciprocal-rank fusion of the collaborative and content lists; reason from the stronger."""
        cf = self.collaborative(user_id, n=3 * n, method=method)
        cb = self.content_for_user(user_id, n=3 * n)
        fused: dict[int, float] = {}
        best: dict[int, Recommendation] = {}
        contrib: dict[int, dict[str, float]] = {}
        for lst in (cf, cb):
            for rank, rec in enumerate(lst):
                s = 1.0 / (10 + rank)
                fused[rec.place_id] = fused.get(rec.place_id, 0.0) + s
                contrib.setdefault(rec.place_id, {})[rec.source] = s
                if rec.place_id not in best or s > max(contrib[rec.place_id].values()) - 1e-12:
                    best[rec.place_id] = rec
        top = sorted(fused, key=lambda p: -fused[p])[:n]
        out = []
        for pid in top:
            c = contrib[pid]
            src = max(c, key=c.get)
            rec = next(r for r in (cf if src == "collaborative" else cb) if r.place_id == pid)
            both = len(c) == 2
            reason = rec.reason + (" Also on the other route's list." if both else "")
            out.append(Recommendation(**{**rec.as_dict(), "score": round(fused[pid], 4),
                                         "source": "hybrid" if both else src, "reason": reason}))
        return out

    def recommend(self, user_id: int | None = None, category: str | None = None, city: str | None = None,
                  budget: float | None = None, n: int = 10, route: str = "hybrid") -> list[Recommendation]:
        if user_id is not None and self.known_user(user_id) and route != "interests":
            if route == "collaborative":
                return self.collaborative(user_id, n)
            if route == "content":
                return self.content_for_user(user_id, n)
            return self.hybrid(user_id, n)
        return self.by_interests(category, city, budget, n)

    def options(self) -> dict:
        return {
            "categories": [{"id": c, "label": f"{c} ({CATEGORY_EN.get(c, c)})"} for c in self.content.categories],
            "cities": self.content.cities,
            "user_ids": self.user_ids(),
            "price_quantiles": {q: int(self.table["Price"].quantile(q)) for q in (0.5, 0.75, 0.9)},
        }
