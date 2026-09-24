"""Offline fallback: LFCC + logistic regression.

This exists for one reason -- so that a model download failing the night
before the demo does not end the demo. It is weaker than L1 and the UI says
so explicitly when it is the layer in use.

Trained by scripts/calibrate.py on whatever clips are in data/demo_clips/,
and persisted to data/models/fallback_lr.joblib.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.detector.base import Detector
from backend.models import LayerScore


def lfcc(wav: np.ndarray, sample_rate: int, n_filters: int = 20) -> np.ndarray:
    """Linear-frequency cepstral coefficients.

    LFCC rather than MFCC: mel spacing throws away resolution exactly in the
    upper bands where vocoder artifacts live.
    """
    raise NotImplementedError


class FallbackDetector(Detector):
    name = "l1_fallback"
    is_fallback = True

    def __init__(self, weights_path: Path) -> None:
        self.weights_path = weights_path
        self._clf = None

    def load(self) -> None:
        raise NotImplementedError

    def score(self, wav: np.ndarray, sample_rate: int) -> LayerScore:
        raise NotImplementedError
