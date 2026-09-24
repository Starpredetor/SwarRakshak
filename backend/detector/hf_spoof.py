"""L1: the trained anti-spoof layer.

Wraps a HuggingFace audio-classification checkpoint (wav2vec2 / AASIST-style)
that was fine-tuned to separate genuine speech from vocoder output. TTS
vocoders reconstruct the spectrum statistically, which leaves the
vocal-tract resonances over-smoothed and slightly mistimed -- artifacts that
survive translation across languages, which is why this layer carries most of
the fused weight.
"""
from __future__ import annotations

import numpy as np

from backend.detector.base import Detector
from backend.models import LayerScore


class HFSpoofDetector(Detector):
    name = "l1_spoof"

    def __init__(self, model_id: str, device: str, spoof_label: str = "fake") -> None:
        self.model_id = model_id
        self.device = device
        self.spoof_label = spoof_label
        self._model = None
        self._extractor = None
        self._spoof_index: int | None = None

    def load(self) -> None:
        """Pull the checkpoint, move it to `device`, and warm up.

        Warmup matters: the first CUDA forward pass costs far more than the
        rest, and without it the very first window of the demo blows the
        300 ms budget on camera.
        """
        raise NotImplementedError

    def _resolve_spoof_index(self) -> int:
        """Map `spoof_label` onto the checkpoint's id2label.

        Checkpoints disagree about label order -- some put 'fake' at index 0,
        some at 1. Getting this backwards inverts the entire demo, so it is
        resolved explicitly at load time instead of being assumed.
        """
        raise NotImplementedError

    def score(self, wav: np.ndarray, sample_rate: int) -> LayerScore:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
