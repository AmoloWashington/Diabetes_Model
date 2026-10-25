"""GlucoLab API: computational physiology of glucose regulation, with an AI assistant."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import secrets
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import feedback, services
from .ai import assistant
from .config import settings
from .molecular import dna
from .physiology import uva_padova
from .rna.api import router as rna_router
from .schemas import (
    BetaCellRequest, CgmAnalyzeRequest, ChatRequest, CrisprRequest, DiffusionRequest, FeedbackRequest, DNARequest, GlucoseForecastRequest, IndicesRequest, IVGTTFitRequest, IVGTTRequest,
    MealSimRequest, MembraneRequest, RiskRequest,
)

log = logging.getLogger("glucolab")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "web" / "dist"
mimetypes.add_type("application/manifest+json", ".webmanifest")


def _warm_risk_model(delay_s: float = 10.0) -> None:
    time.sleep(delay_s)
    from .ml import glucose_forecast, risk_model

    risk_model.get_bundle()
    try:
        glucose_forecast.get_benchmark()
    except Exception:
        log.exception("Glucose forecast benchmark warm-up failed")


def _risk_model_ready() -> bool:
    # The warm-up thread may be importing the module right now; a partially initialised
    # module has no is_ready yet, which simply means "not ready".
    is_ready = getattr(sys.modules.get("app.ml.risk_model"), "is_ready", None)
    try:
        return bool(is_ready and is_ready())
    except Exception:
        return False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the risk model (and scikit-learn) in the background after the server is up, so a
    # cold start on a small instance answers its first page request without waiting for it.
    threading.Thread(target=_warm_risk_model, name="risk-model-warmup", daemon=True).start()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="GlucoLab API",
    version="2.0.0",
    description=(
        "Peer-reviewed models of glucose-insulin physiology (Dalla Man 2007, Bergman 1979, "
        "Topp 2000), clinical indices, a leak-free validated symptom risk model, DNA sequence "
        "analysis and an AI research assistant grounded in those engines. "
        "For research and education only; not a medical device."
    ),
    docs_url="/docs" if settings.api_docs else None,
    redoc_url="/redoc" if settings.api_docs else None,
    openapi_url="/openapi.json" if settings.api_docs else None,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.include_router(rna_router)
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    return response


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_request: Request, exc: RuntimeError):
    log.exception("Computation failed")
    return JSONResponse(status_code=500, content={"detail": f"Computation failed: {exc}"})


# ---------------------------------------------------------------- rate limiting

_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _rate_limit(request: Request, limit: int | None = None, bucket: str = "ai",
                detail: str = "Too many AI requests; please wait a minute.") -> None:
    limit = settings.ai_rate_limit_per_min if limit is None else limit
    if limit <= 0:
        return
    key = f"{bucket}:{request.client.host if request.client else 'unknown'}"
    now = time.monotonic()
    with _hits_lock:
        q = _hits[key]
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(status_code=429, detail=detail)
        q.append(now)


# ---------------------------------------------------------------- meta

@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "risk_model_ready": _risk_model_ready(),
        "ai_configured": settings.ai_configured,
    }


@app.get("/api/references")
def references() -> dict:
    return {"references": services.REFERENCES}


# ---------------------------------------------------------------- physiology

@app.get("/api/physiology/phenotypes")
def phenotypes() -> dict:
    return {"phenotypes": [
        {"key": p.key, "label": p.label, "description": p.description, "published": p.published,
         "basal_glucose_mg_dl": p.basal_glucose, "basal_insulin_pmol_l": p.basal_insulin,
         "insulin_sensitivity_scale": p.insulin_sensitivity, "beta_cell_function_scale": p.beta_cell_function}
        for p in uva_padova.PHENOTYPES.values()
    ]}


@app.post("/api/physiology/meal")
async def meal(req: MealSimRequest) -> dict:
    return await run_in_threadpool(services.meal_simulation, req)


@app.post("/api/physiology/beta-cell")
async def beta_cell_sim(req: BetaCellRequest) -> dict:
    return await run_in_threadpool(services.beta_cell_simulation, req)


@app.get("/api/physiology/beta-cell/fixed-points")
def beta_cell_fp(si_fraction: float = 1.0) -> dict:
    if not (0.02 <= si_fraction <= 2.0):
        raise HTTPException(422, "si_fraction must be within 0.02-2")
    return services.beta_cell_fixed_points(si_fraction)


@app.post("/api/physiology/ivgtt")
async def ivgtt(req: IVGTTRequest) -> dict:
    return await run_in_threadpool(services.ivgtt_simulation, req)


@app.post("/api/physiology/ivgtt/fit")
async def ivgtt_fit(req: IVGTTFitRequest) -> dict:
    return await run_in_threadpool(services.ivgtt_fit, req)


@app.post("/api/biophysics/membrane")
def biophysics_membrane(req: MembraneRequest) -> dict:
    return services.membrane(req)


@app.post("/api/biophysics/diffusion")
def biophysics_diffusion(req: DiffusionRequest) -> dict:
    return services.diffusion(req)


@app.post("/api/cgm/analyze")
async def cgm_analyze(req: CgmAnalyzeRequest) -> dict:
    return await run_in_threadpool(services.cgm_analyze, req)


@app.get("/api/cgm/sample")
async def cgm_sample(phenotype: Literal["normal", "insulin_resistant", "type2"] = "type2",
                     days: int = Query(14, ge=1, le=28)) -> dict:
    return await run_in_threadpool(services.cgm_sample, phenotype, days)


@app.post("/api/clinical/indices")
def clinical(req: IndicesRequest) -> dict:
    return services.clinical_indices(req)


# ---------------------------------------------------------------- ML

@app.post("/api/risk/predict")
async def risk_predict(req: RiskRequest) -> dict:
    return await run_in_threadpool(services.symptom_risk, req)


@app.get("/api/risk/model-card")
async def risk_card() -> dict:
    return await run_in_threadpool(services.risk_model_card)


# ---------------------------------------------------------------- molecular

@app.get("/api/dna/demo")
def dna_demo() -> dict:
    return {"sequence": dna.DEMO_SEQUENCE, "note": dna.DEMO_SEQUENCE_NOTE,
            "insulin_b_chain": dna.INSULIN_B_CHAIN, "insulin_a_chain": dna.INSULIN_A_CHAIN}


@app.post("/api/dna/analyze")
def dna_analyze(req: DNARequest) -> dict:
    return services.dna_analysis(req.sequence)


@app.post("/api/crispr/guides")
async def crispr_guides(req: CrisprRequest) -> dict:
    return await run_in_threadpool(services.crispr_guides, req)


@app.get("/api/forecast/benchmark")
async def forecast_benchmark() -> dict:
    return await run_in_threadpool(services.glucose_forecast_benchmark)


@app.post("/api/forecast/evaluate")
async def forecast_evaluate(req: GlucoseForecastRequest) -> dict:
    return await run_in_threadpool(services.glucose_forecast_series, req)


# ---------------------------------------------------------------- feedback

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest, request: Request) -> dict:
    _rate_limit(request, limit=5, bucket="feedback", detail="Too many submissions; please wait a minute.")
    if req.website:  # honeypot filled in: almost certainly a bot; accept silently, store nothing
        return {"ok": True, "id": None, "filed_on_github": False}
    return await run_in_threadpool(feedback.submit, req.model_dump(exclude={"website"}))


@app.get("/api/feedback", include_in_schema=False)
def read_feedback(request: Request) -> dict:
    token = os.environ.get("FEEDBACK_ADMIN_TOKEN", "")
    given = request.headers.get("x-admin-token", "")
    if not token or not secrets.compare_digest(given.encode(), token.encode()):
        raise HTTPException(status_code=404, detail="Not found")
    return {"items": feedback.list_all()}


# ---------------------------------------------------------------- AI

@app.get("/api/ai/status")
def ai_status() -> dict:
    # The model name is server configuration and is not exposed to visitors.
    return {"configured": settings.ai_configured, "tools": assistant.tool_schema_summary()}


@app.post("/api/ai/chat")
async def ai_chat(req: ChatRequest, request: Request) -> dict:
    _rate_limit(request)
    try:
        reply = await run_in_threadpool(assistant.chat, [m.model_dump() for m in req.messages], None,
                                        req.context.model_dump() if req.context else None)
    except assistant.AssistantUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except assistant.AssistantError as e:
        raise HTTPException(status_code=e.status, detail=str(e)) from e
    return assistant.reply_to_dict(reply)


@app.post("/api/ai/chat/stream")
def ai_chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """Server-Sent Events: text deltas, reasoning summaries and tool calls as they happen."""
    _rate_limit(request)
    history = [m.model_dump() for m in req.messages]
    ctx = req.context.model_dump() if req.context else None

    def gen():
        for ev in assistant.stream_events(history, context=ctx):
            yield f"data: {json.dumps(ev, default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- frontend (React build)

if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        f = (FRONTEND_DIR / path).resolve()
        if path and f.is_file() and FRONTEND_DIR.resolve() in f.parents:
            # The service worker and manifest must be re-checked on every load so updates reach phones.
            headers = {"Cache-Control": "no-cache"} if path in ("sw.js", "manifest.webmanifest") else None
            return FileResponse(f, headers=headers)
        return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-cache"})
else:

    @app.get("/", include_in_schema=False)
    def no_frontend() -> JSONResponse:
        return JSONResponse({
            "message": "GlucoLab API is running. The web UI has not been built yet: "
                       "run `npm install && npm run build` in web/ (or use scripts/dev.sh). API docs: /docs",
        })
