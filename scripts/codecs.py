"""Telephony codec simulation.

Your own risk table lists codec and packet loss as the second-biggest threat
to this approach, and for good reason: the artifacts L1 keys on live in the
fine spectral detail that lossy speech codecs are specifically built to throw
away.

It also fixes an evaluation trap. Genuine clips recorded on a phone mic and
synthetic clips produced at studio quality differ in channel as much as in
provenance -- a detector can score brilliantly by learning the microphone. Put
both through the same codec and that shortcut closes.

mu-law needs nothing but numpy. Opus and AMR-NB need ffmpeg, which comes from
the imageio-ffmpeg wheel rather than a system install -- everything stays
inside the venv. If neither that nor a PATH ffmpeg is present they are skipped
rather than silently faked.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SR = 16000


def ffmpeg_exe() -> str | None:
    """Locate ffmpeg: the bundled wheel first, then PATH.

    Preferring the wheel keeps the codec results reproducible across machines
    -- a system ffmpeg built without libopencore_amrnb would silently drop
    AMR-NB, the harshest and most informative of the three.
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def has_ffmpeg() -> bool:
    return ffmpeg_exe() is not None


def mulaw(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """G.711 mu-law companding round-trip.

    8-bit, 8 kHz -- what a plain PSTN call actually delivers. Pure numpy, so
    this one always runs.
    """
    import librosa

    narrow = librosa.resample(wav, orig_sr=sr, target_sr=8000)

    mu = 255.0
    compressed = np.sign(narrow) * np.log1p(mu * np.abs(narrow)) / np.log1p(mu)
    quantised = np.round((compressed + 1) * 127.5).astype(np.uint8)

    restored = quantised.astype(np.float32) / 127.5 - 1.0
    expanded = np.sign(restored) * ((1 + mu) ** np.abs(restored) - 1) / mu

    return librosa.resample(
        expanded.astype(np.float32), orig_sr=8000, target_sr=sr
    ).astype(np.float32)


def _ffmpeg_roundtrip(wav: np.ndarray, sr: int, args: list[str], suffix: str) -> np.ndarray:
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        mid = Path(tmp) / f"mid{suffix}"
        dst = Path(tmp) / "out.wav"

        exe = ffmpeg_exe()
        if exe is None:
            raise RuntimeError("ffmpeg not available")

        sf.write(str(src), wav, sr)
        subprocess.run([exe, "-y", "-i", str(src), *args, str(mid)],
                       check=True, capture_output=True)
        subprocess.run([exe, "-y", "-i", str(mid), "-ar", str(sr), "-ac", "1", str(dst)],
                       check=True, capture_output=True)

        out, _ = sf.read(str(dst), dtype="float32")
        return out


def opus(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """Opus at 24 kbps -- what most VoIP and conferencing actually carries."""
    return _ffmpeg_roundtrip(wav, sr, ["-c:a", "libopus", "-b:a", "24k"], ".ogg")


def amr_nb(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """AMR-NB at 12.2 kbps -- 2G/3G mobile. The harshest of the three."""
    return _ffmpeg_roundtrip(
        wav, sr, ["-c:a", "libopencore_amrnb", "-ar", "8000", "-ab", "12.2k"], ".amr"
    )


def mp3(wav: np.ndarray, sr: int = SR, bitrate: str = "128k") -> np.ndarray:
    """MP3 encode/decode round-trip.

    Not a telephony codec -- this one exists to equalise an evaluation set.
    If the genuine recordings arrived as mp3 and the clones are generated
    clean, the two sets differ by lossy compression in exactly the upper bands
    L1 reads, and the detector can score well by learning "mp3 or not" instead
    of "synthetic or not". Putting the clones through the same encoder at the
    same bitrate closes that gap.
    """
    return _ffmpeg_roundtrip(wav, sr, ["-c:a", "libmp3lame", "-b:a", bitrate], ".mp3")


def clean(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    return wav


CODECS = {
    "clean": clean,
    "mp3": mp3,
    "mulaw": mulaw,
    "opus": opus,
    "amr_nb": amr_nb,
}
ALWAYS_AVAILABLE = {"clean", "mulaw"}  # the rest need ffmpeg


def available() -> list[str]:
    if has_ffmpeg():
        return list(CODECS)
    return sorted(ALWAYS_AVAILABLE)
