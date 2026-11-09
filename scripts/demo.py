"""Scripted end-to-end call against a running server: health, metrics, user recs, interest recs."""

import json
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8103"
c = httpx.Client(base_url=BASE, timeout=10)

print("== health:", c.get("/health").json())
m = c.get("/metrics").json()
print("== rmse:", json.dumps(m["rmse"]))
print("== ranking (hybrid):", json.dumps(m["ranking"]["hybrid"]))
print("== ranking (random expected):", json.dumps(m["ranking"]["random_expected"]))

print("== user 1, hybrid top 5")
for r in c.post("/recommend", json={"user_id": 1, "n": 5}).json()["recommendations"]:
    print(f"  {r['name'][:38]:38s} {r['city']:11s} {r['category_en']:17s} {r['source']:13s} {r['reason'][:72]}")

print("== interests: nature preserve in Bandung under Rp 20,000, top 5")
q = {"category": "Cagar alam", "city": "Bandung", "budget": 20000, "n": 5}
for r in c.post("/recommend", json=q).json()["recommendations"]:
    print(f"  {r['name'][:38]:38s} Rp {r['price']:>7,}  rated {r['mean_rating']} by {r['n_ratings']} users")
