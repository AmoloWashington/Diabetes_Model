import json

import pytest
from fastapi.testclient import TestClient

from app import feedback
from app.main import _hits, app

GOOD = {"role": "clinician", "rating": 4, "would_use": "yes", "pages": ["Glucose data (CGM)"],
        "accuracy_concerns": "The GMI label could say it is an estimate.", "suggestions": "Add mmol/l by default."}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GLUCOLAB_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("FEEDBACK_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("FEEDBACK_ADMIN_TOKEN", raising=False)
    _hits.clear()
    return TestClient(app)


def test_submit_stores_feedback(client):
    r = client.post("/api/feedback", json=GOOD)
    assert r.status_code == 200 and r.json()["ok"] and r.json()["filed_on_github"] is False
    items = feedback.list_all()
    assert len(items) == 1 and items[0]["rating"] == 4 and items[0]["role"] == "clinician"


def test_honeypot_is_accepted_but_not_stored(client):
    r = client.post("/api/feedback", json={**GOOD, "website": "http://spam.example"})
    assert r.status_code == 200
    assert feedback.list_all() == []


def test_validation_rejects_bad_input(client):
    assert client.post("/api/feedback", json={**GOOD, "rating": 6}).status_code == 422
    assert client.post("/api/feedback", json={**GOOD, "role": "hacker"}).status_code == 422
    assert client.post("/api/feedback", json={**GOOD, "email": "x@y.z"}).status_code == 422  # extra fields forbidden


def test_rate_limit(client):
    codes = [client.post("/api/feedback", json=GOOD).status_code for _ in range(6)]
    assert codes[:5] == [200] * 5 and codes[5] == 429


def test_admin_read_requires_token(client, monkeypatch):
    client.post("/api/feedback", json=GOOD)
    assert client.get("/api/feedback").status_code == 404  # no token configured
    monkeypatch.setenv("FEEDBACK_ADMIN_TOKEN", "s3cret")
    assert client.get("/api/feedback", headers={"x-admin-token": "wrong"}).status_code == 404
    r = client.get("/api/feedback", headers={"x-admin-token": "s3cret"})
    assert r.status_code == 200 and len(r.json()["items"]) == 1


def test_github_issue_filed_when_configured(client, monkeypatch):
    sent = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"html_url": "https://github.com/o/r/issues/1"}).encode()

    def fake_urlopen(req, timeout):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data)
        sent["auth"] = req.get_header("Authorization")
        return Resp()

    monkeypatch.setenv("FEEDBACK_GITHUB_TOKEN", "tok")
    monkeypatch.setenv("FEEDBACK_GITHUB_REPO", "o/r")
    monkeypatch.setattr(feedback.urllib.request, "urlopen", fake_urlopen)
    r = client.post("/api/feedback", json=GOOD)
    assert r.json()["filed_on_github"] is True
    assert sent["url"] == "https://api.github.com/repos/o/r/issues"
    assert sent["auth"] == "Bearer tok"
    assert sent["body"]["title"] == "[Feedback] Doctor or clinician: 4/5"
    assert "The GMI label could say it is an estimate." in sent["body"]["body"]
