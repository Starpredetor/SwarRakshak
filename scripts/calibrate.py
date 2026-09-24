"""Fit the decision threshold on your own demo clips.

Why this is needed: a pretrained spoof detector that never saw your cloning
tool will often still *separate* your real and fake clips while placing the
boundary nowhere near 0.5. Fitting the midpoint recovers the separation the
model already has.

Why this is not cheating, stated plainly so you can say it on camera: you are
choosing an operating point on your own channel, which is what any deployed
detector does. What it is *not* is a benchmark result -- do not report an EER
from these numbers. For that, run ASVspoof properly.

Writes fusion.calibration.{midpoint,temperature} back into config.yaml and,
with --train-fallback, fits the offline LFCC classifier too.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import ROOT, get_config  # noqa: E402
from backend.detector import build_detectors  # noqa: E402
from backend.detector.dsp_evidence import DSPEvidenceDetector  # noqa: E402
from backend.stream.replay import load_wav  # noqa: E402


def _windows(wav: np.ndarray, sr: int, window: float, hop: float):
    size, step = int(sr * window), int(sr * hop)
    for start in range(0, max(1, len(wav) - size + 1), step):
        chunk = wav[start : start + size]
        if len(chunk) == size:
            yield chunk


def score_directory(path: Path, detector, config: dict) -> list[float]:
    """Run every clip in `path` through the detector, window by window."""
    sr = config["audio"]["sample_rate"]
    scores: list[float] = []

    clips = sorted(path.glob("*.wav"))
    if not clips:
        print(f"  WARNING: no .wav files in {path}")

    for clip in clips:
        wav = load_wav(clip, sr)
        clip_scores = [
            detector.score(chunk, sr).score
            for chunk in _windows(
                wav, sr, config["audio"]["window_seconds"], config["audio"]["hop_seconds"]
            )
        ]
        if clip_scores:
            print(f"  {clip.name:<40} {np.mean(clip_scores):.3f} "
                  f"({len(clip_scores)} windows)")
            scores.extend(clip_scores)

    return scores


def pick_midpoint(real: list[float], fake: list[float]) -> tuple[float, float]:
    """Choose the threshold maximising separation; return (midpoint, temperature).

    Also reports the achieved separation. If the distributions overlap badly,
    it says so loudly -- that means the detector does not transfer to your
    cloning tool, and no threshold will rescue it.
    """
    real_arr, fake_arr = np.array(real), np.array(fake)

    candidates = np.linspace(
        min(real_arr.min(), fake_arr.min()),
        max(real_arr.max(), fake_arr.max()),
        400,
    )
    accuracies = [
        ((real_arr < t).sum() + (fake_arr >= t).sum()) / (real_arr.size + fake_arr.size)
        for t in candidates
    ]
    best = int(np.argmax(accuracies))
    midpoint = float(candidates[best])

    # Temperature sets how sharply the sigmoid turns. Scaling it to the spread
    # of the data keeps the gauge from pinning to 0 or 100 the moment the score
    # crosses the line.
    spread = float(np.std(np.concatenate([real_arr, fake_arr])))
    temperature = max(0.02, spread / 2)

    print(f"\nreal:  n={real_arr.size:<5} mean {real_arr.mean():.3f}  sd {real_arr.std():.3f}")
    print(f"fake:  n={fake_arr.size:<5} mean {fake_arr.mean():.3f}  sd {fake_arr.std():.3f}")
    print(f"midpoint {midpoint:.3f}  temperature {temperature:.3f}")
    print(f"window accuracy at that threshold: {accuracies[best]:.1%}")

    if accuracies[best] < 0.75:
        print("\n  WARNING: the two distributions overlap badly.")
        print("  The detector does not transfer to your cloning tool. No threshold")
        print("  will fix that -- try another checkpoint (scripts/fetch_models.py")
        print("  --all) rather than nudging this number until the demo passes.")

    return midpoint, temperature


def train_fallback(real_dir: Path, fake_dir: Path, out: Path, config: dict) -> None:
    """Fit LFCC + logistic regression for backend/detector/fallback.py."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from backend.detector.fallback import lfcc

    sr = config["audio"]["sample_rate"]
    features, labels = [], []

    for directory, label in ((real_dir, 0), (fake_dir, 1)):
        for clip in sorted(directory.glob("*.wav")):
            wav = load_wav(clip, sr)
            for chunk in _windows(
                wav, sr, config["audio"]["window_seconds"], config["audio"]["hop_seconds"]
            ):
                features.append(lfcc(chunk, sr))
                labels.append(label)

    if len(set(labels)) < 2:
        print("  skipping fallback: need clips in BOTH real/ and fake/")
        return

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(np.array(features), np.array(labels))

    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"  fallback trained on {len(labels)} windows -> {out}")


def write_config(midpoint: float, temperature: float) -> None:
    """Rewrite the two calibration values in place, preserving comments.

    A line-level edit rather than a YAML round-trip: dumping the parsed tree
    would strip every comment in the file, and those comments are load-bearing
    for anyone reading the config at 2 a.m.
    """
    path = ROOT / "config.yaml"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    for i, line in enumerate(lines):
        if line.strip().startswith("midpoint:"):
            lines[i] = f"    midpoint: {midpoint:.4f}\n"
        elif line.strip().startswith("temperature:"):
            lines[i] = f"    temperature: {temperature:.4f}\n"

    path.write_text("".join(lines), encoding="utf-8")
    print(f"\nwrote calibration to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--fake", type=Path, required=True)
    parser.add_argument("--train-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the numbers without touching config.yaml")
    args = parser.parse_args()

    for directory in (args.real, args.fake):
        if not directory.is_dir():
            sys.exit(f"not a directory: {directory}")

    config = get_config()
    detector = next(
        d for d in build_detectors() if not isinstance(d, DSPEvidenceDetector)
    )
    print(f"calibrating {detector.name}\n")

    print("real clips:")
    real = score_directory(args.real, detector, config)
    print("\nfake clips:")
    fake = score_directory(args.fake, detector, config)

    if not real or not fake:
        sys.exit("\nneed scored windows from BOTH directories")

    midpoint, temperature = pick_midpoint(real, fake)

    if args.dry_run:
        print("\n--dry-run: config.yaml unchanged")
    else:
        write_config(midpoint, temperature)

    if args.train_fallback:
        print("\ntraining the offline fallback:")
        train_fallback(
            args.real, args.fake, ROOT / "data" / "models" / "fallback_lr.joblib", config
        )


if __name__ == "__main__":
    main()
