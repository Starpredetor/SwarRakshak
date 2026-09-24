"""Detector interface. Every layer implements exactly this."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from backend.models import LayerScore


class Detector(ABC):
    """Scores one analysis window.

    Contract:
      - `score` receives float32 mono PCM in [-1, 1] at the configured rate.
      - It must not block for longer than the hop interval; the caller runs it
        in a worker thread but a slow layer still starves the next window.
      - On any internal failure it returns `abstain=True` rather than raising.
        The pipeline must survive a broken layer mid-call.
    """

    name: str = "unnamed"
    is_fallback: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load weights and warm up. Raises if the layer cannot be used."""

    @abstractmethod
    def score(self, wav: np.ndarray, sample_rate: int) -> LayerScore:
        """Return this layer's opinion about `wav`."""

    def close(self) -> None:
        """Release GPU memory. Default: nothing to do."""
        return None
