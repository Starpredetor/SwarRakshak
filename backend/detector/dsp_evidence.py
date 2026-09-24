"""L1b: cheap DSP evidence, for explainability rather than for accuracy.

These features are what the dashboard shows the viewer when it claims a window
looks synthetic. They carry a low fusion weight on purpose: they are
heuristics with hand-picked thresholds, not a calibrated model, and saying so
is more defensible than dressing them up.

All of it is CPU work on numpy/librosa and costs a couple of milliseconds.
"""
from __future__ import annotations

import numpy as np

from backend.detector.base import Detector
from backend.models import LayerScore


def spectral_flatness(wav: np.ndarray, sample_rate: int) -> float:
    """Geometric/arithmetic mean ratio of the magnitude spectrum.

    Vocoded speech tends to sit flatter than natural speech, which has sharper
    harmonic peaks.
    """
    raise NotImplementedError


def hf_energy_ratio(wav: np.ndarray, sample_rate: int, cutoff_hz: int = 6000) -> float:
    """Fraction of energy above `cutoff_hz`.

    Many vocoders under-generate the top octave; some over-generate it. Either
    way it deviates from the band a real 16 kHz voice occupies.
    """
    raise NotImplementedError


def jitter_shimmer_proxy(wav: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Cycle-to-cycle F0 and amplitude variation.

    Real phonation is slightly irregular. Synthesised phonation is often too
    regular -- unnaturally low jitter is itself a tell.
    """
    raise NotImplementedError


class DSPEvidenceDetector(Detector):
    name = "l1b_dsp"

    def load(self) -> None:
        """No weights. Present so the registry can treat every layer alike."""
        return None

    def score(self, wav: np.ndarray, sample_rate: int) -> LayerScore:
        """Combine the features above into a weak score plus a `detail` dict.

        `detail` is what the frontend renders in the evidence panel, so keep
        the keys stable: flatness, hf_ratio, jitter, shimmer.
        """
        raise NotImplementedError
