"""
FastAPI — ADHD Handwriting Analysis Backend
============================================
POST /api/analyze         — PCA anomaly detection on stroke data
POST /api/analyze-reading — Reading error analysis (pre-transcribed)
POST /api/analyze-image   — Reading error analysis (audio → Whisper → analyze)
GET  /api/health          — Liveness probe
GET  /api/model/info      — Model metadata

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import pathlib, asyncio, json, logging, time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from reading_routes import router as reading_router  # ← separate file

# ── Model state ────────────────────────────────────────────────────────────────
MODEL = None
THRESHOLD: float = 0.0
MODEL_META: dict = {}

INPUT_TYPE_ADJUSTMENTS: dict[str, float] = {
    "pen": 1.0,
    "touch": 1.0,
    "unknown": 1.0,
    "mouse": 1.0,  # retrained model handles mouse natively
}

logger = logging.getLogger("adhd.api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")


# ── Pydantic schemas ───────────────────────────────────────────────────────────


class StrokePoint(BaseModel):
    x: float
    y: float
    t: float = 0.0
    pressure: float = Field(default=1.0, ge=0.0, le=1.0)


class StrokePayload(BaseModel):
    strokes: list[list[StrokePoint]] = Field(..., min_length=1)
    canvas_width: float = Field(default=600.0, gt=0)
    canvas_height: float = Field(default=176.0, gt=0)
    child_id: str | None = None
    input_type: str = Field(default="unknown", pattern="^(mouse|touch|pen|unknown)$")

    @field_validator("strokes")
    @classmethod
    def at_least_one_point(cls, v):
        filtered = [s for s in v if len(s) >= 1]
        if not filtered:
            raise ValueError("No valid strokes — each stroke needs at least 1 point.")
        return filtered


class FlagDetail(BaseModel):
    speed_variance: float
    acceleration_peaks: int
    long_pauses: int
    high_curvature_pct: float
    stroke_count: int


class AnalysisResult(BaseModel):
    anomaly_score: float
    threshold: float
    is_anomalous: bool
    confidence: float
    risk_level: str
    window_scores: list[float]
    flags: FlagDetail
    total_points: int
    processing_ms: float
    note: str
    input_type: str
    threshold_used: float


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, THRESHOLD, MODEL_META
    logger.info("Loading model …")
    try:
        from model.lstm_anomaly import load_model_and_meta

        MODEL, THRESHOLD = await asyncio.to_thread(load_model_and_meta)
        MODEL_META = {"threshold": THRESHOLD, "status": "loaded"}
        logger.info(f"Model ready. Threshold = {THRESHOLD:.6f}")
    except FileNotFoundError as exc:
        logger.warning(f"Weights not found — demo mode. ({exc})")
        MODEL_META = {"threshold": 0.05, "status": "demo"}
        THRESHOLD = 0.05
    yield
    logger.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DyslexAI Analysis API",
    version="1.1.0",
    description="Handwriting + reading error analysis for dyslexia/ADHD screening.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(reading_router)  # mounts /api/analyze-reading and /api/analyze-image


# ── Helpers ────────────────────────────────────────────────────────────────────


def _normalise_payload(payload: StrokePayload) -> list[list[dict]]:
    result = []
    for stroke in payload.strokes:
        pts = []
        for p in stroke:
            pts.append(
                {
                    "x": p.x / payload.canvas_width,
                    "y": p.y / payload.canvas_height,
                    "t": p.t,
                    "pressure": p.pressure,
                }
            )
        result.append(pts)
    return result


def _risk_level(confidence: float) -> str:
    if confidence < 0.35:
        return "low"
    if confidence < 0.65:
        return "moderate"
    return "high"


def _note(result: dict) -> str:
    f = result["flags"]
    parts = []
    if result["is_anomalous"]:
        parts.append(
            f"Anomalous handwriting pattern detected (confidence {result['confidence']:.0%})."
        )
    else:
        parts.append("Handwriting pattern within expected range.")
    if f["long_pauses"] > 5:
        parts.append(f"Frequent hesitation pauses detected ({f['long_pauses']}).")
    if f["acceleration_peaks"] > 20:
        parts.append(
            "Irregular stroke acceleration — possible motor control variation."
        )
    if f["high_curvature_pct"] > 0.3:
        parts.append("High curvature rate — potential letter formation difficulty.")
    return " ".join(parts)


def _demo_score(strokes: list[list[dict]]) -> dict:
    from model.lstm_anomaly import extract_features, _heuristic_flags

    raw = extract_features(strokes)
    flags = _heuristic_flags(raw)
    score = min(float(flags["speed_variance"] * 0.5 + flags["long_pauses"] * 0.01), 1.0)
    conf = float(np.clip((score - THRESHOLD) / (THRESHOLD + 0.01), 0, 1))
    return {
        "anomaly_score": round(score, 6),
        "threshold": THRESHOLD,
        "is_anomalous": score > THRESHOLD,
        "confidence": round(conf, 4),
        "window_scores": [round(score, 6)],
        "flags": flags,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/api/health", tags=["Meta"])
async def health():
    return {"status": "ok", "model": MODEL_META.get("status", "unknown")}


@app.get("/api/model/info", tags=["Meta"])
async def model_info():
    return MODEL_META


@app.post(
    "/api/analyze",
    response_model=AnalysisResult,
    status_code=status.HTTP_200_OK,
    tags=["Handwriting"],
    summary="Analyse handwriting strokes for ADHD/dyslexia markers",
)
async def analyze(payload: StrokePayload, request: Request):
    t0 = time.perf_counter()
    total_pts = sum(len(s) for s in payload.strokes)

    if total_pts < 10:
        raise HTTPException(422, "Too few stroke points. Minimum: 10 events.")

    multiplier = INPUT_TYPE_ADJUSTMENTS.get(payload.input_type, 1.0)
    adjusted_thresh = THRESHOLD * multiplier

    logger.info(
        f"input_type={payload.input_type}  threshold={THRESHOLD:.4f}"
        f"  adjusted={adjusted_thresh:.4f}  multiplier={multiplier}"
    )

    strokes_norm = _normalise_payload(payload)

    try:
        if MODEL is not None:
            from model.lstm_anomaly import score_strokes

            raw_result = await asyncio.to_thread(
                score_strokes, strokes_norm, MODEL, adjusted_thresh
            )
        else:
            raw_result = _demo_score(strokes_norm)
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(500, f"Inference failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Save mouse sessions for future retraining
    if payload.input_type == "mouse":
        import time as _time

        _DATA = pathlib.Path("data/sessions")
        _DATA.mkdir(parents=True, exist_ok=True)
        (_DATA / f"mouse_{int(_time.time()*1000)}.json").write_text(
            json.dumps(
                {
                    "label": "normal",
                    "source": "browser_mouse",
                    "input_type": payload.input_type,
                    "strokes": [[p.dict() for p in s] for s in payload.strokes],
                    "canvas_width": payload.canvas_width,
                    "canvas_height": payload.canvas_height,
                }
            )
        )

    return AnalysisResult(
        anomaly_score=raw_result["anomaly_score"],
        threshold=raw_result["threshold"],
        is_anomalous=raw_result["is_anomalous"],
        confidence=raw_result["confidence"],
        risk_level=_risk_level(raw_result["confidence"]),
        window_scores=raw_result["window_scores"],
        flags=FlagDetail(**raw_result["flags"]),
        total_points=total_pts,
        processing_ms=round(elapsed_ms, 2),
        note=_note(raw_result),
        input_type=payload.input_type,
        threshold_used=round(adjusted_thresh, 6),
    )


@app.exception_handler(Exception)
async def global_exc_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
