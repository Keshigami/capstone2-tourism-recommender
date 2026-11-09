export DATA_ROOT := if path_exists("data") == "true" { "data" } else { env_var_or_default("DATA_ROOT", "") }
# Capstone 2 Part 2: tourism EDA + recommender. Port 8103.
set shell := ["bash", "-cu"]
port := "8103"
pidfile := ".run.pid"

setup:
    uv sync
    mkdir -p static models data
    cp common/ui.css static/ui.css

# Build cleaned data, EDA charts, metrics.json and models/recommender.pkl (seconds either way).
train mode="--quick":
    PYTHONPATH=src uv run python -m tourism.train {{mode}}

run:
    PYTHONPATH=src uv run uvicorn app:app --host 127.0.0.1 --port {{port}}

up:
    @if [ -f {{pidfile}} ] && kill -0 $(cat {{pidfile}}) 2>/dev/null; then echo "already running pid $(cat {{pidfile}})"; exit 0; fi
    PYTHONPATH=src nohup uv run uvicorn app:app --host 127.0.0.1 --port {{port}} > .run.log 2>&1 & echo $! > {{pidfile}}
    @for i in $(seq 1 30); do curl -sf http://127.0.0.1:{{port}}/health >/dev/null && break; sleep 0.5; done
    @echo "up on http://127.0.0.1:{{port}} pid $(cat {{pidfile}})"

down:
    @if [ -f {{pidfile}} ]; then kill $(cat {{pidfile}}) 2>/dev/null || true; rm -f {{pidfile}}; echo "stopped"; else echo "not running"; fi

test:
    PYTHONPATH=src uv run pytest -q

lint:
    uv run ruff check .

# Scripted end-to-end call against a running server (just up first).
demo:
    uv run python scripts/demo.py http://127.0.0.1:{{port}}
