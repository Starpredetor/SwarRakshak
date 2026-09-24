"""Detector registry.

Picks the best layer set that actually loaded, and remembers whether it had to
fall back so the API can tell the operator. Nothing here guesses: a layer that
fails to load is dropped, not silently replaced by a stub that returns 0.5.
"""
from __future__ import annotations

from backend.detector.base import Detector
from backend.detector.dsp_evidence import DSPEvidenceDetector
from backend.detector.fallback import FallbackDetector
from backend.detector.hf_spoof import HFSpoofDetector

__all__ = [
    "Detector",
    "DSPEvidenceDetector",
    "FallbackDetector",
    "HFSpoofDetector",
    "build_detectors",
    "loaded_detectors",
]

_LOADED: list[Detector] = []


def build_detectors() -> list[Detector]:
    """Load L1 (or the fallback) plus L1b. Called once at startup.

    Order of attempts:
      1. HFSpoofDetector on the configured model id, unless SWAR_FORCE_FALLBACK.
      2. Each remaining candidate in config.yaml detector.spoof_model_candidates.
      3. FallbackDetector, if its weights exist.
    Raises if nothing at all could be loaded -- better a server that refuses to
    start than a demo that scores noise.
    """
    raise NotImplementedError


def loaded_detectors() -> list[Detector]:
    """The detectors built at startup."""
    return _LOADED
