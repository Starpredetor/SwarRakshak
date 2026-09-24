import numpy as np
import pytest

from backend.detector.base import Detector
from backend.models import LayerScore


class FakeDetector(Detector):
    """A detector that returns whatever you tell it to.

    Lets the session pipeline be tested without loading a 300 MB checkpoint,
    and lets failure modes (abstention, exceptions) be triggered on demand
    rather than waited for.
    """

    def __init__(self, name="l1_spoof", score=0.9, abstain=False, raises=False):
        self.name = name
        self._score = score
        self._abstain = abstain
        self._raises = raises
        self.calls = 0

    def load(self):
        return None

    def score(self, wav: np.ndarray, sample_rate: int) -> LayerScore:
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return LayerScore(
            layer=self.name,
            score=self._score,
            confidence=1.0,
            abstain=self._abstain,
        )


@pytest.fixture
def config():
    return {
        "audio": {
            "sample_rate": 16000,
            "window_seconds": 2.0,
            "hop_seconds": 1.0,
            "silence_rms": 0.005,
        },
        "fusion": {
            "weights": {"l1_spoof": 0.85, "l1b_dsp": 0.15},
            "ema_alpha": 0.4,
            "calibration": {"midpoint": 0.5, "temperature": 0.15},
        },
        "risk": {
            "bands": {
                "LOW": {"enter": 0, "exit": 44},
                "MEDIUM": {"enter": 40, "exit": 79},
                "HIGH": {"enter": 75, "exit": 100},
            },
            "min_dwell_windows": 2,
        },
    }


@pytest.fixture
def speech():
    """2 s of tone loud enough to clear the silence gate."""
    t = np.arange(32000) / 16000
    return (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


@pytest.fixture
def silence():
    return np.zeros(32000, dtype=np.float32)
