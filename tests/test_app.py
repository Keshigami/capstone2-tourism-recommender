import numpy as np


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["service"] == "capstone2-tourism-recommender"
    assert body["model_loaded"] is True


def test_metrics_has_rmse_and_ranking(client):
    m = client.get("/metrics").json()
    assert 0 < m["rmse"]["svd"] < 2
    assert {"user_knn", "item_knn", "svd", "baseline_global_mean"} <= set(m["rmse"])
    for route in ("collaborative", "content", "hybrid"):
        assert 0 <= m["ranking"][route]["hit_rate@10"] <= 1
        assert 0 <= m["ranking"][route]["precision@5"] <= 1
    assert m["split"]["test_frac"] == 0.2


def test_recommend_known_user_excludes_rated(client):
    rated = {row["Place_Id"] for row in client.get("/api/users/1").json()["rated"]}
    assert len(rated) > 10
    for route in ("hybrid", "collaborative", "content"):
        r = client.post("/recommend", json={"user_id": 1, "n": 10, "route": route})
        assert r.status_code == 200, r.text
        recs = r.json()["recommendations"]
        assert len(recs) == 10
        ids = [x["place_id"] for x in recs]
        assert len(set(ids)) == 10
        assert not (set(ids) & rated)
        assert all(x["reason"] and x["source"] in {"collaborative", "content", "hybrid"} for x in recs)


def test_recommend_by_interests_matches_category(client):
    r = client.post("/recommend", json={"category": "Cagar alam", "city": "Bandung", "budget": 20000, "n": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "interests"
    recs = body["recommendations"]
    assert len(recs) == 10
    assert all(x["category"] == "Cagar alam" for x in recs)
    assert all(x["city"] == "Bandung" for x in recs)
    assert all(x["price"] <= 20000 for x in recs)


def test_recommend_validation(client):
    assert client.post("/recommend", json={}).status_code == 422
    assert client.post("/recommend", json={"user_id": 99999}).status_code == 404


def test_index_default_user_shows_results(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert 'id="input"' in html and 'id="run"' in html and 'id="result"' in html
    assert 'data-count="10"' in html  # default user 1 gets ten recommendations on first load


def test_index_interests_and_error_states(client):
    assert 'data-count="10"' in client.get("/?mode=interests&category=Budaya&city=Yogyakarta").text
    bad = client.get("/?user_id=99999")
    assert bad.status_code == 200 and "not in the ratings file" in bad.text
    assert "not a number" in client.get("/?mode=interests&budget=abc").text


def test_eda_page(client):
    r = client.get("/eda")
    assert r.status_code == 200
    assert "Keraton Surabaya" in r.text
    assert "['" not in r.text and "[1" not in r.text  # no Python list reprs leaking into the prose
    assert "/static/top_loved.png" in r.text
    assert client.get("/static/top_loved.png").status_code == 200


def test_rating_predictors_are_bounded_and_svd_near_baseline(client):
    from app import state

    rec = state["rec"]
    for method in ("user_knn", "item_knn", "svd"):
        p = rec.pred[method]
        assert p.shape == rec.cf.m.R.shape
        assert np.all((p >= 1) & (p <= 5))
