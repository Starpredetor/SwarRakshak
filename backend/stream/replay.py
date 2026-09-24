"""Server-side WAV replay, paced to real time.

Replay deliberately goes through the same Session as the microphone rather
than batch-scoring the file. If the demo clip and the live mic took different
code paths, the clip would prove nothing about the live path -- and the live
path is what you are claiming on stage.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from backend.models import ClipInfo

logger = logging.getLogger(__name__)

_KNOWN_LABELS = {"real", "fake", "bonafide", "spoof"}


def list_clips(clips_dir: Path) -> list[ClipInfo]:
    """Enumerate data/demo_clips/. Label is inferred from the subdirectory."""
    import soundfile as sf

    if not clips_dir.exists():
        return []

    clips: list[ClipInfo] = []
    for path in sorted(clips_dir.rglob("*.wav")):
        parent = path.parent.name.lower()
        try:
            info = sf.info(str(path))
        except Exception:
            logger.warning("skipping unreadable clip %s", path, exc_info=True)
            continue
        clips.append(
            ClipInfo(
                filename=str(path.relative_to(clips_dir)).replace("\\", "/"),
                label=parent if parent in _KNOWN_LABELS else None,
                duration_seconds=round(info.duration, 2),
            )
        )
    return clips


def resolve_clip(clips_dir: Path, filename: str) -> Path:
    """Resolve a client-supplied name inside `clips_dir`.

    Resolved and re-checked against the root: the filename arrives over a
    WebSocket, and "../../.env" is a perfectly valid string.
    """
    root = clips_dir.resolve()
    path = (root / filename).resolve()
    if not path.is_file() or root not in path.parents:
        raise ValueError(f"no such clip: {filename}")
    return path


def load_wav(path: Path, target_sr: int) -> np.ndarray:
    """Read a WAV as float32 mono, resampled to `target_sr`."""
    import librosa
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(mono, dtype=np.float32)


async def stream_clip(
    path: Path, target_sr: int, chunk_ms: int = 100
) -> AsyncIterator[np.ndarray]:
    """Yield chunks with real-time sleeps between them.

    The sleeps are the point: without them the file scores in a fraction of a
    second and the dashboard's timeline is meaningless.

    Paced against a wall-clock deadline rather than sleeping a fixed interval
    per chunk, so scoring time does not make a 30 s clip take 40 s.
    """
    wav = load_wav(path, target_sr)
    chunk = max(1, int(target_sr * chunk_ms / 1000))
    started = time.perf_counter()

    for i in range(0, len(wav), chunk):
        due = started + (i / target_sr)
        delay = due - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        yield wav[i : i + chunk]
