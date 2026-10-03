"""HTTP API and AI-assistant loop (with a fake Anthropic client; no network)."""

import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from app.ai import assistant
from app.config import settings
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


PATIENT = {"age": 50, "male": True, **{k: False for k in [
    "polyuria", "polydipsia", "sudden_weight_loss", "weakness", "polyphagia", "genital_thrush",
    "visual_blurring", "itching", "irritability", "delayed_healing", "partial_paresis",
    "muscle_stiffness", "alopecia", "obesity"]}}


def test_health_and_frontend(client):
    from app.main import FRONTEND_DIR

    assert client.get("/api/health").json()["status"] == "ok"
    r = client.get("/")
    assert r.status_code == 200 and "GlucoLab" in r.text
    assert r.headers["x-content-type-options"] == "nosniff"
    if FRONTEND_DIR.is_dir():  # built UI: client-side routes fall back to index.html
        assert client.get("/rna/sequences").text == r.text
    assert client.get("/api/does-not-exist").status_code == 404


def test_meal_endpoint(client):
    r = client.post("/api/physiology/meal", json={"meals": [{"time_min": 0, "carbs_g": 75}], "duration_min": 300})
    assert r.status_code == 200
    body = r.json()
    assert len(body["series"]["t_min"]) == 301
    assert body["phenotype"]["published_parameter_set"] is True


@pytest.mark.parametrize("payload", [
    {"meals": [{"time_min": 0, "carbs_g": -5}]},
    {"meals": [{"time_min": 400, "carbs_g": 50}], "duration_min": 300},
    {"phenotype": "type1"},
    {"unexpected": 1},
])
def test_meal_validation(client, payload):
    assert client.post("/api/physiology/meal", json=payload).status_code == 422


def test_other_endpoints(client):
    assert client.post("/api/physiology/beta-cell", json={"years": 2}).status_code == 200
    assert client.post("/api/physiology/beta-cell", json={"years": 2, "si_decline_years": 5}).status_code == 422
    assert client.get("/api/physiology/beta-cell/fixed-points?si_fraction=0.5").status_code == 200
    assert client.get("/api/physiology/beta-cell/fixed-points?si_fraction=9").status_code == 422
    assert client.post("/api/physiology/ivgtt", json={}).status_code == 200
    r = client.post("/api/clinical/indices", json={"fasting_glucose_mg_dl": 100, "fasting_insulin_uU_ml": 10})
    assert r.status_code == 200 and "homa1" in r.json()
    assert client.post("/api/dna/analyze", json={"sequence": "ATGAAATAG"}).status_code == 200
    assert client.post("/api/dna/analyze", json={"sequence": "XYZ"}).status_code == 422
    assert client.post("/api/risk/predict", json={"patient": PATIENT}).status_code == 200
    assert client.get("/api/references").json()["references"]


def test_chat_validation(client):
    bad = {"messages": [{"role": "assistant", "content": "hi"}]}
    assert client.post("/api/ai/chat", json=bad).status_code == 422
    bad2 = {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}
    assert client.post("/api/ai/chat", json=bad2).status_code == 422


def test_chat_unconfigured_returns_503(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    r = client.post("/api/ai/chat", json={"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 503


# ------------------------------------------------------------ fake Anthropic client

def _msg(content, stop_reason, model="claude-opus-5"):
    return NS(content=content, stop_reason=stop_reason, model=model, usage=NS(input_tokens=10, output_tokens=5))


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kw):
        # snapshot the messages list as sent
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return self.responses.pop(0)


def fake_client(responses):
    msgs = FakeMessages(responses)
    return NS(beta=NS(messages=msgs)), msgs


def test_agent_loop_runs_tools_and_returns_grounded_answer():
    tool_use = NS(type="tool_use", id="toolu_1", name="compute_clinical_indices",
                  input={"fasting_glucose_mg_dl": 90, "fasting_insulin_uU_ml": 10})
    c, msgs = fake_client([
        _msg([NS(type="thinking", thinking=""), tool_use], "tool_use"),
        _msg([NS(type="text", text="HOMA-IR is 2.22.")], "end_turn"),
    ])
    reply = assistant.chat([{"role": "user", "content": "HOMA?"}], client=c)
    assert reply.text == "HOMA-IR is 2.22."
    assert [t.name for t in reply.tool_calls] == ["compute_clinical_indices"]
    assert not reply.tool_calls[0].is_error
    first, second = msgs.calls
    assert first["model"] == settings.claude_model
    assert first["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert first["output_config"] == {"effort": settings.claude_effort}
    if settings.claude_fallbacks:
        assert first["fallbacks"] == "default"
        assert first["betas"] == ["server-side-fallback-2026-07-01"]
    # the assistant turn is replayed verbatim, followed by one user turn of tool results
    assert second["messages"][1]["content"][1] is tool_use
    result = second["messages"][2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_1"
    homa = json.loads(result["content"])["homa1"]["homa_ir"]
    assert homa == pytest.approx(10 * (90 / 18.016) / 22.5)
    assert reply.usage == {"input_tokens": 20, "output_tokens": 10}


def test_invalid_tool_input_is_reported_as_tool_error():
    tu = NS(type="tool_use", id="t1", name="simulate_meal", input={"meals": [{"time_min": 0, "carbs_g": 9999}]})
    c, msgs = fake_client([_msg([tu], "tool_use"), _msg([NS(type="text", text="Invalid.")], "end_turn")])
    reply = assistant.chat([{"role": "user", "content": "x"}], client=c)
    assert reply.tool_calls[0].is_error
    assert msgs.calls[1]["messages"][2]["content"][0]["is_error"] is True


def test_unknown_tool_and_refusal():
    tu = NS(type="tool_use", id="t1", name="rm_rf", input={})
    c, _ = fake_client([_msg([tu], "tool_use"), _msg([], "refusal")])
    reply = assistant.chat([{"role": "user", "content": "x"}], client=c)
    assert reply.tool_calls[0].is_error
    assert reply.stop_reason == "refusal"


def test_max_tool_rounds_is_enforced():
    tu = NS(type="tool_use", id="t", name="get_references", input={})
    c, msgs = fake_client([_msg([tu], "tool_use") for _ in range(settings.ai_max_tool_rounds + 1)])
    reply = assistant.chat([{"role": "user", "content": "loop"}], client=c)
    assert reply.stop_reason == "max_tool_rounds"
    assert len(msgs.calls) == settings.ai_max_tool_rounds + 1


def test_every_tool_runs_with_minimal_valid_input(tmp_path, monkeypatch):
    from app.ai.tools import TOOLS, run_tool
    from app.rna import datasets, store

    store.set_store(store.Store(tmp_path / "t.db"))
    monkeypatch.setattr(datasets, "DATA_DIR", tmp_path / "datasets")
    samples = {
        "fold_rna": {"sequence": "GGGAAAUCCCGCGCAAAGCGC"},
        "predict_rna_3d": {"sequence": "GGGCGCAAGCCUAUGCGCUUCGGCGCAUAGGCUUGCGCCC", "method": "denovo"},
        "list_rna_sequences": {},
        "list_datasets": {},
        "membrane_biophysics": {"K": {"in": 140, "out": 5}},
        "diffusion_time": {"radius_nm": 2.0, "distance_um": 10},
        "simulate_meal": {"meals": [{"time_min": 0, "carbs_g": 50}], "duration_min": 240},
        "simulate_beta_cell_progression": {"years": 2},
        "simulate_ivgtt": {},
        "compute_clinical_indices": {"hba1c_pct": 6.0},
        "predict_symptom_risk": {"patient": PATIENT},
        "get_risk_model_card": {},
        "get_references": {},
    }
    assert set(samples) == {t["name"] for t in TOOLS}
    for name, args in samples.items():
        out, err = run_tool(name, args)
        assert not err, (name, out)
        json.loads(out)


def test_chat_endpoint_with_fake_client(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    c, _ = fake_client([_msg([NS(type="text", text="Hello from the model.")], "end_turn")])
    monkeypatch.setattr(assistant, "_client", c)
    r = client.post("/api/ai/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["text"] == "Hello from the model."


# ------------------------------------------------------------ streaming


class FakeStream:
    def __init__(self, events, final):
        self.events, self.final = events, final

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.final


class FakeStreamingMessages:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def stream(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        events, final = self.turns.pop(0)
        return FakeStream(events, final)


def test_stream_events_text_thinking_and_tools():
    tu = NS(type="tool_use", id="t1", name="membrane_biophysics", input={})
    turns = [
        ([NS(type="thinking", thinking="Need the GHK potential."), NS(type="text", text="Computing… ")],
         _msg([NS(type="text", text="Computing… "), tu], "tool_use")),
        ([NS(type="text", text="V_m is "), NS(type="text", text="-67 mV.")],
         _msg([NS(type="text", text="V_m is -67 mV.")], "end_turn")),
    ]
    msgs = FakeStreamingMessages(turns)
    client = NS(beta=NS(messages=msgs))
    evs = list(assistant.stream_events([{"role": "user", "content": "Resting potential?"}], client=client,
                                       context={"page": "Biophysics", "summary": "V_m = -67.3 mV"}))
    kinds = [e["type"] for e in evs]
    assert kinds[0] == "thinking"
    assert "tool_start" in kinds and "tool_result" in kinds
    assert kinds[-1] == "done"
    text = "".join(e["text"] for e in evs if e["type"] == "text")
    assert "V_m is -67 mV." in text
    assert evs[-1]["text"].endswith("-67 mV.")
    # page context is attached to the user turn as data
    first_user = msgs.calls[0]["messages"][0]["content"]
    assert "Biophysics" in first_user and first_user.endswith("Resting potential?")
    assert msgs.calls[0]["max_tokens"] == 32000


def test_stream_endpoint_sse(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    msgs = FakeStreamingMessages([([NS(type="text", text="Hi there.")], _msg([NS(type="text", text="Hi there.")], "end_turn"))])
    monkeypatch.setattr(assistant, "_client", NS(beta=NS(messages=msgs)))
    r = client.post("/api/ai/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[0] == {"type": "text", "text": "Hi there."}
    assert events[-1]["type"] == "done"


def test_stream_endpoint_unconfigured_emits_error(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(assistant, "_client", None)
    r = client.post("/api/ai/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["type"] == "error" and "ANTHROPIC_API_KEY" in events[-1]["message"]
