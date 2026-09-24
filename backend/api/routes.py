"""REST surface. Small on purpose -- the live work happens over the socket."""
from __future__ import annotations

from fastapi import APIRouter

from backend.models import ClipInfo

router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict:
    """Liveness plus which detector is actually loaded and on what device.

    Check this before recording. It is the difference between demoing the
    GPU model and unknowingly demoing the fallback.
    """
    raise NotImplementedError


@router.get("/clips", response_model=list[ClipInfo])
async def clips() -> list[ClipInfo]:
    """Bundled demo clips available for replay."""
    raise NotImplementedError


@router.get("/session/{session_id}")
async def session_summary(session_id: str) -> dict:
    """Post-call rollup for a finished session."""
    raise NotImplementedError
