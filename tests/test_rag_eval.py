import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_api import app

client = TestClient(app)

DOC = (
    "SynapseBug is a Flask web app that summarizes text. It uses spaCy for sentence "
    "splitting, RAKE for keyword extraction, and NLTK for stopword removal. Passwords "
    "are hashed with bcrypt, and the app enforces a password policy of at least 8 "
    "characters with uppercase, lowercase, a digit and a special character. Users "
    "register and log in, and admins can view and delete users. Data is stored in "
    "SQLite through SQLAlchemy. The summary is built by scoring each sentence by how "
    "many keywords it contains and keeping the top third of sentences."
)

CASES = json.loads(Path(__file__).with_name("golden_set.json").read_text(encoding="utf-8"))

REFUSAL_MARKERS = [
    "cannot find", "can't find", "could not find", "not mentioned",
    "no information", "not in the", "does not", "doesn't", "not provided",
]


@pytest.fixture(scope="module", autouse=True)
def indexed_document():
    resp = client.post("/documents", json={"text": DOC})
    assert resp.status_code == 200


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_golden_case(case):
    resp = client.post("/ask", json={"question": case["question"]})
    assert resp.status_code == 200
    body = resp.json()
    answer = body["answer"].lower()

    if case["type"] == "answerable":
        assert any(k.lower() in answer for k in case["expected_any"]), f"Answer was: {answer}"
        assert body["sources"], "No source passages returned"
    else:
        assert any(m in answer for m in REFUSAL_MARKERS), f"Should have declined, got: {answer}"


def test_ask_before_indexing_is_rejected():
    import rag_api
    rag_api.store = None
    resp = client.post("/ask", json={"question": "Anything?"})
    assert resp.status_code == 400