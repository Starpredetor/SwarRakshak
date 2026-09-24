"""Server-side WAV replay, paced to real time.

Replay deliberately goes through the same Session as the microphone rather
than batch-scoring the file. If the demo clip and the live mic took different
code paths, the clip would prove nothing about the live path -- and the live
path is what you are claiming on stage.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from backend.models import ClipInfo


def list_clips(clips_dir: Path) -> list[ClipInfo]:
    """Enumerate data/demo_clips/. Label is inferred from the subdirectory."""
    raise NotImplementedError


def load_wav(path: Path, target_sr: int) -> np.ndarray:
    """Read a WAV as float32 mono, resampled to `target_sr`."""
    raise NotImplementedError


async def stream_clip(path: Path, target_sr: int, chunk_ms: int = 100) -> AsyncIterator[np.ndarray]:
    """Yield chunks with real-time sleeps between them.

    The sleeps are the point: without them the file scores in a fraction of a
    second and the dashboard's timeline is meaningless.
    """
    raise NotImplementedError
