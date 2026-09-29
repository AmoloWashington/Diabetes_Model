"""Claude-powered research assistant grounded in the verified engines.

The assistant runs a manual tool-use loop: Claude decides which engine to
call, the backend executes it with validated inputs, and Claude interprets
the returned numbers. Each call is returned to the client as an auditable
trace so users can see exactly which computation backs each statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anthropic

from ..config import settings
from .tools import TOOLS, run_tool

SYSTEM_PROMPT = """You are the research assistant of GlucoLab, a computational physiology \
application for diabetes research and education. Your users range from students to \
PhD-level biologists and physicists.

You have tools that run peer-reviewed models implemented and tested in this application: \
the Dalla Man-Rizza-Cobelli 2007 meal model, the Bergman minimal model, the Topp 2000 \
beta-cell mass model, clinical indices (HOMA, QUICKI, eAG, TyG, ADA thresholds), a \
leak-free validated symptom-risk model, and the reference list.

Grounding rules - these matter more than anything else:
- Every quantitative result you state about a simulation, index or prediction must come \
from a tool result in this conversation. Call the tool rather than estimating.
- Say which model produced a number. When a phenotype is marked as an illustrative \
parameter scaling rather than a published parameter set, say so.
- For established background physiology you may answer from knowledge, but distinguish it \
clearly from computed results, and say plainly when something is uncertain or unknown. \
Never invent citations; use get_references for the sources of implemented models.
- Explain mechanisms precisely (equations, parameters, units, feedback loops, stability) \
when the user wants depth, and plainly when they do not.

Safety: this is a research and education tool. Do not diagnose individuals or prescribe \
treatment. If a user describes personal symptoms or results, you may run the tools to \
illustrate, and then recommend confirmation by a clinician with laboratory testing. If a \
message describes an emergency (for example very high glucose with vomiting, confusion or \
breathing difficulty), tell them to seek urgent medical care first.

Format answers in concise Markdown."""


class AssistantUnavailable(Exception):
    """AI is not configured or the upstream API cannot be used."""


class AssistantError(Exception):
    """The upstream API returned an error for this request."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


@dataclass
class ToolTrace:
    name: str
    input: dict
    output: str
    is_error: bool


@dataclass
class AssistantReply:
    text: str
    model: str
    tool_calls: list[ToolTrace] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)


_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if not settings.ai_configured:
        raise AssistantUnavailable(
            "AI assistant is not configured. Set ANTHROPIC_API_KEY in the server environment "
            "(or in a git-ignored .env file) and restart."
        )
    if _client is None:
        _client = anthropic.Anthropic(max_retries=2, timeout=300.0)
    return _client


def _request_kwargs(messages: list) -> dict:
    kw = dict(
        model=settings.claude_model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
        thinking={"type": "adaptive"},
        output_config={"effort": settings.claude_effort},
    )
    if settings.claude_fallbacks:
        # Server-side fallback: if a safety classifier declines, the API re-runs
        # the request on Anthropic's recommended fallback model.
        kw["betas"] = ["server-side-fallback-2026-07-01"]
        kw["fallbacks"] = "default"
    return kw


def chat(history: list[dict], client: anthropic.Anthropic | None = None) -> AssistantReply:
    """Run one assistant turn. ``history`` holds alternating user/assistant text messages."""
    client = client or _get_client()
    messages: list = [{"role": m["role"], "content": m["content"]} for m in history]
    traces: list[ToolTrace] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    model_used = settings.claude_model

    for _ in range(settings.ai_max_tool_rounds + 1):
        try:
            response = client.beta.messages.create(**_request_kwargs(messages))
        except anthropic.AuthenticationError as e:
            raise AssistantUnavailable("The Anthropic API rejected the credentials (401).") from e
        except anthropic.PermissionDeniedError as e:
            raise AssistantUnavailable("The API key lacks permission for this model (403).") from e
        except anthropic.NotFoundError as e:
            raise AssistantError(f"Model {settings.claude_model!r} not found (404).", 502) from e
        except anthropic.RateLimitError as e:
            raise AssistantError("Anthropic API rate limit reached; retry shortly.", 429) from e
        except anthropic.BadRequestError as e:
            raise AssistantError(f"Anthropic API rejected the request: {e.message}", 502) from e
        except anthropic.APIStatusError as e:
            raise AssistantError(f"Anthropic API error ({e.status_code}).", 502) from e
        except anthropic.APIConnectionError as e:
            raise AssistantError("Could not reach the Anthropic API.", 503) from e

        model_used = getattr(response, "model", model_used) or model_used
        if response.usage is not None:
            usage["input_tokens"] += response.usage.input_tokens or 0
            usage["output_tokens"] += response.usage.output_tokens or 0

        if response.stop_reason == "refusal":
            return AssistantReply(
                text="The model declined to answer this request. Please rephrase it as a research or education question.",
                model=model_used, tool_calls=traces, stop_reason="refusal", usage=usage,
            )

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            text = "\n\n".join(b.text for b in response.content if b.type == "text").strip()
            if response.stop_reason == "max_tokens":
                text += "\n\n*(Response truncated at the output limit.)*"
            return AssistantReply(text=text or "(No text returned.)", model=model_used,
                                  tool_calls=traces, stop_reason=response.stop_reason, usage=usage)

        # Keep the assistant turn verbatim (thinking, fallback and tool_use blocks included)
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for tu in tool_uses:
            out, is_err = run_tool(tu.name, tu.input)
            traces.append(ToolTrace(tu.name, tu.input if isinstance(tu.input, dict) else {}, out, is_err))
            block = {"type": "tool_result", "tool_use_id": tu.id, "content": out}
            if is_err:
                block["is_error"] = True
            results.append(block)
        messages.append({"role": "user", "content": results})

    return AssistantReply(
        text="Stopped after the maximum number of tool rounds. Try a narrower question.",
        model=model_used, tool_calls=traces, stop_reason="max_tool_rounds", usage=usage,
    )


def reply_to_dict(r: AssistantReply) -> dict:
    return {
        "text": r.text,
        "model": r.model,
        "stop_reason": r.stop_reason,
        "usage": r.usage,
        "tool_calls": [
            {"name": t.name, "input": t.input, "is_error": t.is_error,
             "output_preview": t.output[:1500] + ("..." if len(t.output) > 1500 else "")}
            for t in r.tool_calls
        ],
    }


def tool_schema_summary() -> list[dict]:
    return [{"name": t["name"], "description": t["description"]} for t in TOOLS]


__all__ = ["chat", "reply_to_dict", "AssistantUnavailable", "AssistantError", "tool_schema_summary"]
