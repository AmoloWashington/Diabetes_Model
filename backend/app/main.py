"""GlucoLab API: computational physiology of glucose regulation, with an AI assistant."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import services
from .ai import assistant
from .config import settings
from .ml import risk_model
from .molecular import dna
from .physiology import uva_padova
from .rna.api import router as rna_router
from .schemas import (
    BetaCellRequest, ChatRequest, DNARequest, IndicesRequest, IVGTTFitRequest, IVGTTRequest,
    MealSimRequest, RiskRequest,
)

log = logging.getLogger("glucolab")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "web" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Train or load the risk model in the background so the first request is fast.
    threading.Thread(target=risk_model.get_bundle, name="risk-model-warmup", daemon=True).start()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="GlucoLab API",
    version="2.0.0",
    description=(
        "Peer-reviewed models of glucose-insulin physiology (Dalla Man 2007, Bergman 1979, "
        "Topp 2000), clinical indices, a leak-free validated symptom risk model, DNA sequence "
        "analysis and a Claude-powered research assistant grounded in those engines. "
        "For research and education only; not a medical device."
    ),
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


def _rate_limit(request: Request) -> None:
    limit = settings.ai_rate_limit_per_min
    if limit <= 0:
        return
    key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _hits_lock:
        q = _hits[key]
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(status_code=429, detail="Too many AI requests; please wait a minute.")
        q.append(now)


# ---------------------------------------------------------------- meta

@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "risk_model_ready": risk_model.is_ready(),
        "ai_configured": settings.ai_configured,
        "ai_model": settings.claude_model if settings.ai_configured else None,
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


# ---------------------------------------------------------------- AI

@app.get("/api/ai/status")
def ai_status() -> dict:
    return {"configured": settings.ai_configured, "model": settings.claude_model,
            "effort": settings.claude_effort, "tools": assistant.tool_schema_summary()}


@app.post("/api/ai/chat")
async def ai_chat(req: ChatRequest, request: Request) -> dict:
    _rate_limit(request)
    try:
        reply = await run_in_threadpool(assistant.chat, [m.model_dump() for m in req.messages])
    except assistant.AssistantUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except assistant.AssistantError as e:
        raise HTTPException(status_code=e.status, detail=str(e)) from e
    return assistant.reply_to_dict(reply)


# ---------------------------------------------------------------- frontend (React build)

if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        f = (FRONTEND_DIR / path).resolve()
        if path and f.is_file() and FRONTEND_DIR.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(FRONTEND_DIR / "index.html")
else:

    @app.get("/", include_in_schema=False)
    def no_frontend() -> JSONResponse:
        return JSONResponse({
            "message": "GlucoLab API is running. The web UI has not been built yet: "
                       "run `npm install && npm run build` in web/ (or use scripts/dev.sh). API docs: /docs",
        })
