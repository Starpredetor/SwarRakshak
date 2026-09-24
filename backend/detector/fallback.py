"""Offline fallback: LFCC + logistic regression.

This exists for one reason -- so that a model download failing the night
before the demo does not end the demo. It is weaker than L1 and the UI says so
explicitly when it is the layer in use.

Trained by scripts/calibrate.py on whatever clips are in data/demo_clips/, and
persisted to data/models/fallback_lr.joblib.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from backend.detector.base import Detector
from backend.models import LayerScore

logger = logging.getLogger(__name__)

N_FILTERS = 20
N_CEPS = 20


def lfcc(wav: np.ndarray, sample_rate: int, n_filters: int = N_FILTERS) -> np.ndarray:
    """Linear-frequency cepstral coefficients, summarised to a fixed vector.

    LFCC rather than MFCC: mel spacing throws away resolution exactly in the
    upper bands where vocoder artifacts live. Mel is tuned to how humans hear,
    and the whole point here is to look where humans do not.

    Returns [means, stds] over time so a clip of any length maps to one
    fixed-length feature vector.
    """
    import librosa
    from scipy.fftpack import dct

    n_fft, hop = 512, 160
    power = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)

    # Triangular filterbank on a linear frequency axis.
    edges = np.linspace(0, sample_rate / 2, n_filters + 2)
    bank = np.zeros((n_filters, freqs.size), dtype=np.float32)
    for i in range(n_filters):
        left, centre, right = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs - left) / (centre - left)
        falling = (right - freqs) / (right - centre)
        bank[i] = np.clip(np.minimum(rising, falling), 0.0, None)

    log_energy = np.log(bank @ power + 1e-10)
    ceps = dct(log_energy, axis=0, norm="ortho")[:N_CEPS]
    return np.concatenate([ceps.mean(axis=1), ceps.std(axis=1)]).astype(np.float32)


class FallbackDetector(Detector):
    name = "l1_fallback"
    is_fallback = True

    def __init__(self, weights_path: Path) -> None:
        self.weights_path = Path(weights_path)
        self._clf = None

    def load(self) -> None:
        import joblib

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"{self.weights_path} not found. Train it with scripts/calibrate.py."
            )
        self._clf = joblib.load(self.weights_path)
        logger.info("fallback detector loaded from %s", self.weights_path)

    def score(self, wav: np.ndarray, sample_rate: int) -> LayerScore:
        try:
            features = lfcc(wav, sample_rate).reshape(1, -1)
            p_spoof = float(self._clf.predict_proba(features)[0, 1])
        except Exception:
            logger.exception("%s failed on a window; abstaining", self.name)
            return LayerScore(layer=self.name, score=0.0, confidence=0.0, abstain=True)

        return LayerScore(
            layer=self.name,
            score=p_spoof,
            # Capped below what the trained layer can claim. This model saw a
            # handful of clips; it should not sound as sure as one that saw a
            # corpus.
            confidence=min(abs(p_spoof - 0.5) * 2.0, 0.6),
            detail={"p_spoof": p_spoof},
        )
