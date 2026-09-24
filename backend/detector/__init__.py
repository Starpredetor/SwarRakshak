"""Detector registry.

Picks the best layer set that actually loaded, and remembers whether it had to
fall back so the API can tell the operator. Nothing here guesses: a layer that
fails to load is dropped, not silently replaced by a stub returning 0.5.
"""
from __future__ import annotations

import logging

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
    "primary_detector",
]

logger = logging.getLogger(__name__)

_LOADED: list[Detector] = []


def build_detectors() -> list[Detector]:
    """Load L1 (or the fallback) plus L1b. Called once at startup.

    Order of attempts:
      1. HFSpoofDetector on the configured model id, unless SWAR_FORCE_FALLBACK.
      2. Each remaining candidate from config.yaml.
      3. FallbackDetector, if its weights exist.

    Raises if nothing at all loads. A server that refuses to start is better
    than one that scores noise and looks convincing doing it.
    """
    from backend.config import get_config, get_settings, resolve_device

    global _LOADED

    settings = get_settings()
    config = get_config()
    detectors: list[Detector] = []

    if not settings.force_fallback:
        device = resolve_device()
        spoof_label = config["detector"].get("spoof_label", "fake")

        candidates = [settings.spoof_model]
        for c in config["detector"].get("spoof_model_candidates", []):
            if c not in candidates:
                candidates.append(c)

        for model_id in candidates:
            detector = HFSpoofDetector(model_id, device, spoof_label)
            try:
                detector.load()
            except Exception:
                logger.warning("could not load %s; trying next candidate", model_id,
                               exc_info=True)
                continue
            detectors.append(detector)
            break
    else:
        logger.warning("SWAR_FORCE_FALLBACK=1 -- skipping the trained L1 layer")

    if not detectors:
        weights = settings.model_dir / "fallback_lr.joblib"
        fallback = FallbackDetector(weights)
        try:
            fallback.load()
            detectors.append(fallback)
            logger.warning(
                "running on the FALLBACK detector. Accuracy is materially worse "
                "and the dashboard will say so."
            )
        except Exception as exc:
            raise RuntimeError(
                "No detector could be loaded. Run scripts/fetch_models.py for the "
                "trained layer, or scripts/calibrate.py to train the fallback."
            ) from exc

    dsp = DSPEvidenceDetector()
    dsp.load()
    detectors.append(dsp)

    _LOADED = detectors
    return detectors


def loaded_detectors() -> list[Detector]:
    """The detectors built at startup."""
    return _LOADED


def primary_detector() -> Detector | None:
    """The L1 layer -- whatever the DSP evidence layer is not."""
    return next((d for d in _LOADED if not isinstance(d, DSPEvidenceDetector)), None)
