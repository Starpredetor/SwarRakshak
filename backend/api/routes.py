"""REST surface. Small on purpose -- the live work happens over the socket."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.ws import CLIPS_DIR, SESSION_SUMMARIES
from backend.config import get_config, get_settings
from backend.detector import loaded_detectors, primary_detector
from backend.models import ClipInfo
from backend.stream.replay import list_clips

router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict:
    """Liveness plus which detector is actually loaded and on what device.

    Check this before recording. It is the difference between demoing the GPU
    model and unknowingly demoing the fallback.
    """
    primary = primary_detector()
    config = get_config()

    return {
        "status": "ok" if primary is not None else "degraded",
        "detector_name": primary.name if primary else "none",
        "detector_is_fallback": bool(primary and primary.is_fallback),
        "model_id": getattr(primary, "model_id", None),
        "device": getattr(primary, "device", "cpu"),
        "layers": [d.name for d in loaded_detectors()],
        "window_seconds": config["audio"]["window_seconds"],
        "hop_seconds": config["audio"]["hop_seconds"],
        "calibration": config["fusion"]["calibration"],
        "force_fallback": get_settings().force_fallback,
    }


@router.get("/clips", response_model=list[ClipInfo])
async def clips() -> list[ClipInfo]:
    """Bundled demo clips available for replay."""
    return list_clips(CLIPS_DIR)


@router.get("/session/{session_id}")
async def session_summary(session_id: str) -> dict:
    """Post-call rollup for a finished session."""
    summary = SESSION_SUMMARIES.get(session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="unknown or still-running session")
    return summary
