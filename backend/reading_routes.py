"""
Reading Error Analysis — FastAPI router
========================================
Mounted by main.py via app.include_router(reading_router).

POST /api/analyze-reading  — pre-transcribed text → error analysis
POST /api/analyze-image    — audio file → Whisper → error analysis

Dependencies:
    pip install openai-whisper spacy
    python -m spacy download en_core_web_sm
"""

import asyncio, logging, tempfile, time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from model.reading_error_analyzer import analyze_reading_errors

logger = logging.getLogger("adhd.reading")
router = APIRouter(tags=["Reading Analysis"])

# ── Lazy Whisper load ──────────────────────────────────────────────────────────
_whisper_model = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper

        logger.info("Loading Whisper model (base.en) …")
        _whisper_model = whisper.load_model("base.en")
        logger.info("Whisper ready.")
    return _whisper_model


# ── Schemas ────────────────────────────────────────────────────────────────────


class ReadingAnalysisRequest(BaseModel):
    original_text: str = Field(..., min_length=1)
    spoken_text: str = Field(..., min_length=1)
    duration_seconds: float = Field(..., gt=0)
    child_age: int = Field(..., ge=4, le=18)
    child_id: Optional[str] = None


class ReadingAnalysisResult(BaseModel):
    reversal_count: int
    reversal_rate: float
    reversal_examples: list[list[str]]
    number_reversal_count: int
    number_reversal_rate: float
    number_reversal_examples: list[list[str]]
    substitution_count: int
    substitution_rate: float
    omission_count: int
    omission_rate: float
    addition_count: int
    omitted_words: list[str]
    actual_wpm: int
    expected_wpm: int
    wpm_deficit: float
    reading_risk_score: float
    risk_level: str
    dyslexia_indicators: bool
    adhd_indicators: bool
    dyscalculia_indicators: bool
    transcription_source: str
    spoken_text_used: str
    processing_ms: float
    note: str


# ── Helpers ────────────────────────────────────────────────────────────────────


def _risk_level(score: float) -> str:
    if score < 0.35:
        return "low"
    if score < 0.60:
        return "moderate"
    return "high"


def _note(raw: dict) -> str:
    parts = []
    rl = _risk_level(raw["reading_risk_score"])

    if rl == "low":
        parts.append("Reading pattern within expected range for this age group.")
    elif rl == "moderate":
        parts.append(
            f"Some reading difficulties detected (score {raw['reading_risk_score']:.0%})."
        )
    else:
        parts.append(
            f"Significant reading difficulties detected (score {raw['reading_risk_score']:.0%})."
        )

    if raw["dyslexia_indicators"]:
        parts.append(
            f"Dyslexia markers present: {raw['reversal_count']} reversal(s), "
            f"WPM {raw['actual_wpm']} vs expected {raw['expected_wpm']}."
        )
    if raw["adhd_indicators"]:
        parts.append(f"ADHD markers: {raw['omission_count']} word omission(s).")
    if raw["dyscalculia_indicators"]:
        parts.append(
            f"Dyscalculia markers: {raw['number_reversal_count']} number confusion(s)."
        )

    return " ".join(parts)


def _build_response(
    raw: dict, source: str, spoken: str, elapsed: float
) -> ReadingAnalysisResult:
    return ReadingAnalysisResult(
        reversal_count=raw["reversal_count"],
        reversal_rate=raw["reversal_rate"],
        reversal_examples=[[a, b] for a, b in raw["reversal_examples"]],
        number_reversal_count=raw["number_reversal_count"],
        number_reversal_rate=raw["number_reversal_rate"],
        number_reversal_examples=[[a, b] for a, b in raw["number_reversal_examples"]],
        substitution_count=raw["substitution_count"],
        substitution_rate=raw["substitution_rate"],
        omission_count=raw["omission_count"],
        omission_rate=raw["omission_rate"],
        addition_count=raw["addition_count"],
        omitted_words=raw["omitted_words"],
        actual_wpm=raw["actual_wpm"],
        expected_wpm=raw["expected_wpm"],
        wpm_deficit=raw["wpm_deficit"],
        reading_risk_score=raw["reading_risk_score"],
        risk_level=_risk_level(raw["reading_risk_score"]),
        dyslexia_indicators=raw["dyslexia_indicators"],
        adhd_indicators=raw["adhd_indicators"],
        dyscalculia_indicators=raw["dyscalculia_indicators"],
        transcription_source=source,
        spoken_text_used=spoken,
        processing_ms=round(elapsed * 1000, 2),
        note=_note(raw),
    )


# ── Endpoint 1: pre-transcribed ────────────────────────────────────────────────


@router.post(
    "/api/analyze-reading",
    response_model=ReadingAnalysisResult,
    summary="Analyze reading errors from pre-transcribed text",
)
async def analyze_reading(payload: ReadingAnalysisRequest):
    t0 = time.perf_counter()
    try:
        raw = await asyncio.to_thread(
            analyze_reading_errors,
            payload.original_text,
            payload.spoken_text,
            payload.duration_seconds,
            payload.child_age,
        )
    except Exception as exc:
        logger.exception("Reading analysis failed")
        raise HTTPException(500, f"Analysis failed: {exc}") from exc

    return _build_response(
        raw, "provided", payload.spoken_text, time.perf_counter() - t0
    )


# ── Endpoint 2: audio → Whisper → analyze ─────────────────────────────────────


@router.post(
    "/api/analyze-image",
    response_model=ReadingAnalysisResult,
    summary="Transcribe audio with Whisper then analyze reading errors",
)
async def analyze_image(
    file: UploadFile = File(...),
    original_text: str = Form(...),
    duration_seconds: float = Form(...),
    child_age: int = Form(...),
    child_id: str = Form(default=""),
):
    t0 = time.perf_counter()

    allowed = {
        "audio/webm",
        "audio/wav",
        "audio/mpeg",
        "audio/mp4",
        "audio/x-m4a",
        "audio/ogg",
        "video/webm",
    }
    ct = (file.content_type or "").lower()
    if ct not in allowed and not (file.filename or "").endswith(
        (".webm", ".wav", ".mp3", ".m4a", ".ogg")
    ):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio format: {ct}. Use webm, wav, mp3, or m4a.",
        )

    audio_bytes = await file.read()
    if len(audio_bytes) < 1024:
        raise HTTPException(422, "Audio file too small — minimum 1 KB.")

    def _transcribe(audio: bytes, suffix: str) -> str:
        model = get_whisper()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio)
            tmp_path = tmp.name
        try:
            result = model.transcribe(
                tmp_path, language="en", fp16=False, verbose=False
            )
            return result["text"].strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    suffix = Path(file.filename or "audio.webm").suffix or ".webm"

    try:
        spoken_text = await asyncio.to_thread(_transcribe, audio_bytes, suffix)
    except Exception as exc:
        logger.exception("Whisper transcription failed")
        raise HTTPException(500, f"Transcription failed: {exc}") from exc

    if not spoken_text:
        raise HTTPException(
            422, "Whisper returned empty transcription. Check audio quality."
        )

    logger.info(f"Whisper ({len(spoken_text)} chars): {spoken_text[:80]}…")

    try:
        raw = await asyncio.to_thread(
            analyze_reading_errors,
            original_text,
            spoken_text,
            duration_seconds,
            child_age,
        )
    except Exception as exc:
        logger.exception("Reading analysis failed")
        raise HTTPException(500, f"Analysis failed: {exc}") from exc

    return _build_response(raw, "whisper", spoken_text, time.perf_counter() - t0)
