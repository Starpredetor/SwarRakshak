"""Report what is actually in a folder of audio, and whether it can be used.

Run this before generating anything. A reference clip that is too short, too
noisy, clipped, or stereo-with-a-dead-channel produces a bad clone, and a bad
clone makes the detector look better than it is -- which is the one direction
of error you cannot afford, because it survives the demo and fails in the
question round.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm"}

# XTTS-v2 and F5-TTS both want a clean reference of roughly this length.
# Shorter and the timbre is under-determined; much longer adds nothing and
# raises the chance of including a cough or a second speaker.
MIN_REF_SECONDS = 6.0
IDEAL_REF_SECONDS = (6.0, 20.0)


def analyse(path: Path) -> dict | None:
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        return {"path": path, "error": str(exc)}

    mono = data.mean(axis=1)
    duration = len(mono) / sr
    rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0

    # Fraction of samples sitting at full scale -- a proxy for clipping, which
    # a cloner will happily reproduce as distortion.
    clipped = float(np.mean(np.abs(mono) > 0.99)) if mono.size else 0.0

    # Crude noise-floor estimate: the quietest 10% of 20 ms frames.
    frame = max(1, int(0.02 * sr))
    n = len(mono) // frame
    if n >= 10:
        frames = np.abs(mono[: n * frame].reshape(n, frame)).mean(axis=1)
        floor = float(np.quantile(frames, 0.1))
        snr = 20 * np.log10((rms + 1e-9) / (floor + 1e-9))
    else:
        snr = float("nan")

    return {
        "path": path,
        "sr": sr,
        "channels": data.shape[1],
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "clipped": clipped,
        "snr_db": snr,
    }


def verdict(info: dict) -> tuple[str, list[str]]:
    """Usable / marginal / unusable, with the reasons."""
    notes: list[str] = []

    if info.get("error"):
        return "UNREADABLE", [info["error"]]

    if info["duration"] < MIN_REF_SECONDS:
        notes.append(f"only {info['duration']:.1f}s — want {MIN_REF_SECONDS:.0f}s+")
    elif info["duration"] > IDEAL_REF_SECONDS[1]:
        notes.append(f"{info['duration']:.0f}s — trim to ~15s of clean speech")

    if info["sr"] < 16000:
        notes.append(f"{info['sr']} Hz — below the 16 kHz the pipeline runs at")

    if info["rms"] < 0.01:
        notes.append(f"very quiet (rms {info['rms']:.4f})")

    if info["clipped"] > 0.001:
        notes.append(f"clipped ({info['clipped']:.1%} of samples at full scale)")

    if not np.isnan(info["snr_db"]) and info["snr_db"] < 15:
        notes.append(f"noisy (~{info['snr_db']:.0f} dB above floor)")

    if not notes:
        return "USABLE", []
    hard = any("only" in n or "below the" in n or "UNREADABLE" in n for n in notes)
    return ("UNUSABLE" if hard else "MARGINAL"), notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="folder containing the clips")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"not a directory: {args.folder}")

    pattern = "**/*" if args.recursive else "*"
    files = sorted(
        p for p in args.folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )

    if not files:
        sys.exit(f"no audio files found in {args.folder}")

    print(f"{len(files)} file(s) in {args.folder}\n")

    counts = {"USABLE": 0, "MARGINAL": 0, "UNUSABLE": 0, "UNREADABLE": 0}

    for path in files:
        info = analyse(path)
        status, notes = verdict(info)
        counts[status] += 1

        if info.get("error"):
            print(f"[{status:<10}] {path.name}")
            print(f"             {info['error']}")
            continue

        print(
            f"[{status:<10}] {path.name}\n"
            f"             {info['sr']} Hz · {info['channels']}ch · "
            f"{info['duration']:.1f}s · rms {info['rms']:.3f} · "
            f"snr ~{info['snr_db']:.0f} dB"
        )
        for note in notes:
            print(f"             ! {note}")

    print(
        f"\nusable {counts['USABLE']} · marginal {counts['MARGINAL']} · "
        f"unusable {counts['UNUSABLE']} · unreadable {counts['UNREADABLE']}"
    )
    print(
        "\nOne good reference per speaker is enough for zero-shot cloning.\n"
        "Pick the cleanest 6-15s of continuous speech from a single voice."
    )


if __name__ == "__main__":
    main()
