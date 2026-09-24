"""FastAPI entrypoint.

Models load once at startup and stay resident. Loading per request would put a
multi-second stall in the middle of a live call.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import routes, ws
from backend.config import get_settings
from backend.detector import build_detectors, loaded_detectors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("swarrakshak")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build detectors on startup, release GPU memory on shutdown."""
    detectors = build_detectors()
    logger.info("detectors ready: %s", [d.name for d in detectors])
    try:
        yield
    finally:
        for detector in loaded_detectors():
            try:
                detector.close()
            except Exception:
                logger.warning("failed to close %s", detector.name, exc_info=True)


app = FastAPI(
    title="SwarRakshak",
    description="Real-time voice-clone detection for live calls.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)
app.include_router(ws.router)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)
