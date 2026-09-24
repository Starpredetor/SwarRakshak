"""Normalise arbitrary recordings into clean 16 kHz mono WAV references.

The engines will happily decode mp3, m4a or flac themselves, but each one
does it through its own path. Decoding once here means every engine clones
from bit-identical input, so a difference between engines is a difference
between engines rather than between their decoders.

It also does the boring things that decide clone quality: mono downmix, peak
normalisation, and trimming leading and trailing silence.

    .venv/Scripts/python.exe scripts/prepare_refs.py <in> --out data/refs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TARGET_SR = 16000
SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm"}

# Long enough to pin down timbre, short enough to stay clean. More reference
# audio past ~20 s buys almost nothing and raises the odds of catching a cough
# or a second voice, either of which the cloner will faithfully reproduce.
MAX_SECONDS = 20.0


def pick_best_segment(wav: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """Return the densest `seconds` of speech, not simply the first.

    A 95-second recording is mostly not the part you want: it opens with room
    tone and someone deciding to start, and ends with a pause before the stop
    button. Taking the head of it hands the cloner the worst material in the
    file. This slides a window and keeps the one containing the most speech
    frames, which is a decent proxy for continuous talking without gaps.
    """
    target = int(seconds * sr)
    if len(wav) <= target:
        return wav

    frame = max(1, int(0.02 * sr))
    n_frames = len(wav) // frame
    energy = np.abs(wav[: n_frames * frame].reshape(n_frames, frame)).mean(axis=1)

    # Half the median frame energy separates speech from room tone well enough
    # here; a real VAD would be better and is not worth the dependency.
    speech = (energy > np.median(energy) * 0.5).astype(np.float32)

    per_window = target // frame
    cumulative = np.concatenate([[0.0], np.cumsum(speech)])

    step = max(1, per_window // 20)
    best_start, best_score = 0, -1.0
    for i in range(0, n_frames - per_window + 1, step):
        score = float(cumulative[i + per_window] - cumulative[i])
        if score > best_score:
            best_score, best_start = score, i

    start = best_start * frame
    return wav[start : start + target]


def prepare(path: Path, out_dir: Path, max_seconds: float, peak: float) -> Path | None:
    import librosa
    import soundfile as sf

    try:
        wav, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    except Exception as exc:
        print(f"  FAILED {path.name}: {exc}")
        return None

    if wav.size == 0:
        print(f"  FAILED {path.name}: empty")
        return None

    before = len(wav) / TARGET_SR

    # 30 dB below peak: aggressive enough to cut room tone, gentle enough to
    # keep quiet consonants that carry the speaker's identity.
    trimmed, _ = librosa.effects.trim(wav, top_db=30)
    if trimmed.size > TARGET_SR:
        wav = trimmed

    wav = pick_best_segment(wav, TARGET_SR, max_seconds)

    peak_now = float(np.max(np.abs(wav)))
    if peak_now > 0:
        wav = wav * (peak / peak_now)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}.wav"
    sf.write(str(out), wav.astype(np.float32), TARGET_SR)

    after = len(wav) / TARGET_SR
    print(f"  {path.name:<36} {before:5.1f}s -> {after:5.1f}s   {out.name}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="a recording, or a folder of them")
    parser.add_argument("--out", type=Path, default=Path("data/refs"))
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--peak", type=float, default=0.95)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if args.source.is_file():
        if args.source.suffix.lower() not in SUFFIXES:
            sys.exit(f"unsupported file type: {args.source.suffix}")
        files = [args.source]
    elif args.source.is_dir():
        pattern = "**/*" if args.recursive else "*"
        files = sorted(
            p for p in args.source.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUFFIXES
        )
        if not files:
            sys.exit(f"no audio files in {args.source}")
    else:
        sys.exit(f"no such file or directory: {args.source}")

    print(f"preparing {len(files)} file(s) -> {args.out}\n")
    written = [p for p in (prepare(f, args.out, args.max_seconds, args.peak)
                           for f in files) if p]

    print(f"\n{len(written)} reference(s) written.")
    print("\nOne reference per speaker is what you want. If several files are the")
    print("same person, keep the cleanest and delete the rest -- the filename")
    print("becomes the speaker label in the evaluation table.")


if __name__ == "__main__":
    main()
