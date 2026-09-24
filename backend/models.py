"""Wire formats shared by the WebSocket and REST APIs.

Kept in one file so the TypeScript in frontend/src/types.ts has a single
thing to mirror.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Band(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class LayerScore(BaseModel):
    """One detection layer's opinion about one window.

    A layer that cannot produce a trustworthy answer sets `abstain=True` rather
    than guessing -- fusion then drops it and renormalises the weights. This is
    the graceful-degradation rule: a broken layer must never look confident.
    """

    layer: str
    score: float = Field(ge=0.0, le=1.0, description="1.0 = maximally synthetic")
    confidence: float = Field(ge=0.0, le=1.0)
    abstain: bool = False
    detail: dict[str, float] = Field(default_factory=dict)


class RunInfo(BaseModel):
    """A new run started: the client should clear its timeline.

    A "run" is one continuous audio source -- one replayed clip, or one
    microphone session. It exists because a WebSocket connection can carry
    several in sequence, and scores from the previous one must not bleed into
    the next.
    """

    type: Literal["run"] = "run"
    session_id: str
    source: str
    source_label: str | None = Field(
        default=None,
        description="ground truth from the clip's folder (real/fake), for "
                    "checking results by hand -- the detector never sees this",
    )


class Verdict(BaseModel):
    """Emitted once per hop (default 1 s) over the WebSocket."""

    type: Literal["verdict"] = "verdict"
    session_id: str
    source: str = "mic"
    source_label: str | None = None
    t: float = Field(description="seconds since session start")
    risk: int = Field(ge=0, le=100)
    band: Band
    band_changed: bool = False
    layers: list[LayerScore]
    latency_ms: float
    detector_name: str


class SessionInfo(BaseModel):
    type: Literal["session"] = "session"
    session_id: str
    sample_rate: int
    window_seconds: float
    hop_seconds: float
    detector_name: str
    detector_is_fallback: bool


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str
    fatal: bool = False


class ReplayRequest(BaseModel):
    """Client asks the server to stream a bundled clip through the pipeline."""

    type: Literal["replay"] = "replay"
    filename: str


class ClipInfo(BaseModel):
    filename: str
    label: str | None = Field(default=None, description="real | fake | unknown")
    duration_seconds: float
