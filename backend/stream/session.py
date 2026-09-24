"""Per-connection session: buffering, windowing, scoring, verdict emission."""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import numpy as np

from backend.detector.base import Detector
from backend.fusion.scorer import FusionState
from backend.models import SessionInfo, Verdict


class Session:
    """Owns one call's state from connect to disconnect.

    Scoring runs in a thread executor so a slow forward pass cannot stall the
    event loop and stop us reading the socket -- if inference falls behind,
    windows are dropped rather than queued, because a stale verdict on a live
    call is worse than a missing one.
    """

    def __init__(self, detectors: list[Detector], config: dict) -> None:
        self.session_id = str(uuid.uuid4())
        self.detectors = detectors
        self.config = config
        self.fusion = FusionState()
        self._samples_seen = 0
        self._samples_since_hop = 0

    def info(self) -> SessionInfo:
        """Handshake payload sent to the client on connect."""
        raise NotImplementedError

    def feed(self, pcm: np.ndarray) -> bool:
        """Add samples. Returns True when a hop boundary was crossed."""
        raise NotImplementedError

    async def score_window(self) -> Verdict:
        """Run every loaded detector over the current window and fuse."""
        raise NotImplementedError

    async def run(self, pcm_stream: AsyncIterator[np.ndarray]) -> AsyncIterator[Verdict]:
        """Consume PCM, yield a verdict per hop. The whole pipeline in one place."""
        raise NotImplementedError

    def summary(self) -> dict:
        """Post-call rollup: peak risk, time in each band, windows scored.

        Feature-derived numbers only -- no audio, no transcript.
        """
        raise NotImplementedError
