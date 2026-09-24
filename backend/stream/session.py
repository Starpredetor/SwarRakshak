"""Per-connection session: buffering, windowing, scoring, verdict emission."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator

import numpy as np

from backend.detector.base import Detector
from backend.detector.dsp_evidence import DSPEvidenceDetector
from backend.fusion.scorer import (
    AllLayersAbstained,
    FusionState,
    band_for,
    calibrate,
    fuse,
    smooth,
    to_risk,
)
from backend.models import Band, LayerScore, SessionInfo, Verdict
from backend.stream.ringbuffer import RingBuffer

logger = logging.getLogger(__name__)


class Session:
    """Owns one call's state from connect to disconnect.

    Scoring runs in a thread executor so a slow forward pass cannot stall the
    event loop and stop us reading the socket. If inference falls behind,
    windows are dropped rather than queued -- a stale verdict on a live call is
    worse than a missing one.
    """

    def __init__(self, detectors: list[Detector], config: dict) -> None:
        self.session_id = str(uuid.uuid4())
        self.detectors = detectors
        self.config = config

        audio = config["audio"]
        self.sample_rate: int = audio["sample_rate"]
        self.window_seconds: float = audio["window_seconds"]
        self.hop_seconds: float = audio["hop_seconds"]
        self.window_samples = int(self.sample_rate * self.window_seconds)
        self.hop_samples = int(self.sample_rate * self.hop_seconds)
        self.silence_rms: float = audio.get("silence_rms", 0.005)

        self.buffer = RingBuffer(self.window_samples)
        self.fusion = FusionState()
        self._samples_seen = 0
        self._samples_since_hop = 0
        self._last_risk = 0
        self._scoring = False
        self._windows_scored = 0
        self._windows_dropped = 0
        self._peak_risk = 0
        self._band_seconds: dict[str, float] = {b.value: 0.0 for b in Band}

    @property
    def primary(self) -> Detector | None:
        """The L1 layer -- whatever the DSP evidence layer is not."""
        return next(
            (d for d in self.detectors if not isinstance(d, DSPEvidenceDetector)), None
        )

    def info(self) -> SessionInfo:
        """Handshake payload sent to the client on connect."""
        primary = self.primary
        return SessionInfo(
            session_id=self.session_id,
            sample_rate=self.sample_rate,
            window_seconds=self.window_seconds,
            hop_seconds=self.hop_seconds,
            detector_name=primary.name if primary else "none",
            detector_is_fallback=bool(primary and primary.is_fallback),
        )

    def feed(self, pcm: np.ndarray) -> bool:
        """Add samples. Returns True when a hop boundary was crossed."""
        self.buffer.write(pcm)
        self._samples_seen += pcm.size
        self._samples_since_hop += pcm.size

        if not self.buffer.ready or self._samples_since_hop < self.hop_samples:
            return False

        self._samples_since_hop = 0
        return True

    @property
    def elapsed(self) -> float:
        return self._samples_seen / self.sample_rate

    async def score_window(self) -> Verdict | None:
        """Run every loaded detector over the current window and fuse.

        Returns None if a previous window is still being scored -- dropping the
        window keeps the gauge honest about *now* instead of replaying a
        backlog a second late.
        """
        if self._scoring:
            self._windows_dropped += 1
            return None

        wav = self.buffer.read_window()
        if wav is None:
            return None

        self._scoring = True
        started = time.perf_counter()
        try:
            scores = await self._score_layers(wav)
        finally:
            self._scoring = False

        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._build_verdict(scores, latency_ms)

    async def _score_layers(self, wav: np.ndarray) -> list[LayerScore]:
        """Score every layer, or abstain across the board on a silent window.

        The silence gate is applied once here rather than inside each layer, so
        that "nobody is speaking" produces one consistent answer instead of
        each layer improvising its own.
        """
        if float(np.sqrt(np.mean(wav ** 2))) < self.silence_rms:
            return [
                LayerScore(layer=d.name, score=0.0, confidence=0.0, abstain=True)
                for d in self.detectors
            ]

        return list(
            await asyncio.gather(
                *(self._score_one(d, wav) for d in self.detectors)
            )
        )

    async def _score_one(self, detector: Detector, wav: np.ndarray) -> LayerScore:
        """Score one layer, converting any escaped exception into an abstention.

        Layers are supposed to catch their own failures, but this is the last
        line of the graceful-degradation promise: a layer that throws anyway --
        a new one, a third-party one, an OOM on the GPU -- costs its own
        opinion for that window and nothing more. It must never end the call.
        """
        try:
            return await asyncio.to_thread(detector.score, wav, self.sample_rate)
        except Exception:
            logger.exception("%s raised; abstaining for this window", detector.name)
            return LayerScore(
                layer=detector.name, score=0.0, confidence=0.0, abstain=True
            )

    def _build_verdict(self, scores: list[LayerScore], latency_ms: float) -> Verdict:
        cfg = self.config
        cal = cfg["fusion"]["calibration"]

        try:
            raw = fuse(scores, cfg["fusion"]["weights"])
        except AllLayersAbstained:
            # Hold the last risk rather than decaying toward zero. Silence is
            # not evidence of authenticity, and a needle that sags whenever the
            # speaker pauses would read as the system clearing them.
            risk, changed = self._last_risk, False
        else:
            value = smooth(
                self.fusion,
                calibrate(raw, cal["midpoint"], cal["temperature"]),
                cfg["fusion"]["ema_alpha"],
            )
            risk = to_risk(value)
            _, changed = band_for(
                self.fusion,
                risk,
                cfg["risk"]["bands"],
                cfg["risk"]["min_dwell_windows"],
            )

        self._last_risk = risk
        self._peak_risk = max(self._peak_risk, risk)
        self._windows_scored += 1
        self._band_seconds[self.fusion.band.value] += self.hop_seconds

        primary = self.primary
        return Verdict(
            session_id=self.session_id,
            t=round(self.elapsed, 2),
            risk=risk,
            band=self.fusion.band,
            band_changed=changed,
            layers=scores,
            latency_ms=round(latency_ms, 1),
            detector_name=primary.name if primary else "none",
        )

    async def run(self, pcm_stream: AsyncIterator[np.ndarray]) -> AsyncIterator[Verdict]:
        """Consume PCM, yield a verdict per hop. The whole pipeline in one place."""
        async for chunk in pcm_stream:
            if self.feed(chunk):
                verdict = await self.score_window()
                if verdict is not None:
                    yield verdict

    def summary(self) -> dict:
        """Post-call rollup: peak risk, time in each band, windows scored.

        Feature-derived numbers only -- no audio, no transcript.
        """
        return {
            "session_id": self.session_id,
            "duration_seconds": round(self.elapsed, 2),
            "windows_scored": self._windows_scored,
            "windows_dropped": self._windows_dropped,
            "peak_risk": self._peak_risk,
            "final_band": self.fusion.band.value,
            "band_seconds": {k: round(v, 1) for k, v in self._band_seconds.items()},
        }
