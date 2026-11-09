import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def artifacts():
    """Build models/metrics.json and the recommender once if `just train` has not run."""
    if not (ROOT / "models" / "metrics.json").exists() or not (ROOT / "models" / "recommender.pkl").exists():
        from tourism.train import main

        main(["--quick"])


@pytest.fixture(scope="session")
def client(artifacts):
    from app import app

    with TestClient(app) as c:
        yield c
