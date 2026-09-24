"""Score every cloning engine through the real detector and print the table.

Runs in the detector venv (.venv), using the same L1 layer the live pipeline
uses, so the numbers here describe the thing you are actually demonstrating.

What to expect, and how to read it: some engines will be caught cleanly and
some will not. That is the finding, not the failure. A pretrained detector
generalises poorly to vocoders it has never seen -- that is precisely the
In-the-Wild / WaveFake gap your evidence base cites. A table showing four
families tested, two caught, with a stated plan for the other two, is a
stronger position in a judging room than an unblemished table that invites
the question of what you left out.

    .venv/Scripts/python.exe scripts/evaluate.py --real data/refs
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_config  # noqa: E402
from backend.detector import build_detectors  # noqa: E402
from backend.detector.dsp_evidence import DSPEvidenceDetector  # noqa: E402
from backend.stream.replay import load_wav  # noqa: E402
from scripts.codecs import CODECS, available  # noqa: E402

EVAL_ROOT = Path("data/eval")


def windows(wav: np.ndarray, sr: int, size_s: float, hop_s: float):
    size, hop = int(sr * size_s), int(sr * hop_s)
    for start in range(0, max(1, len(wav) - size + 1), hop):
        chunk = wav[start : start + size]
        if len(chunk) == size:
            yield chunk


def score_folder(folder: Path, detector, config: dict, codec: str) -> dict:
    """Score every clip in `folder`, returning window and clip-level results."""
    sr = config["audio"]["sample_rate"]
    transform = CODECS[codec]

    window_scores: list[float] = []
    clip_medians: list[float] = []

    for clip in sorted(folder.glob("*.wav")):
        wav = load_wav(clip, sr)
        try:
            wav = transform(wav, sr)
        except Exception as exc:
            print(f"    codec {codec} failed on {clip.name}: {exc}")
            continue

        scores = [
            detector.score(chunk, sr).score
            for chunk in windows(
                wav, sr, config["audio"]["window_seconds"], config["audio"]["hop_seconds"]
            )
        ]
        if scores:
            window_scores.extend(scores)
            clip_medians.append(float(np.median(scores)))

    return {
        "clips": len(clip_medians),
        "windows": len(window_scores),
        "window_scores": np.array(window_scores),
        "clip_medians": np.array(clip_medians),
    }


def summarise(result: dict, threshold: float, genuine: bool) -> dict:
    if result["windows"] == 0:
        return {"empty": True}

    windows_flagged = float(np.mean(result["window_scores"] >= threshold))
    clips_flagged = float(np.mean(result["clip_medians"] >= threshold))

    return {
        "empty": False,
        "clips": result["clips"],
        "windows": result["windows"],
        "mean": float(result["window_scores"].mean()),
        "window_rate": windows_flagged,
        "clip_rate": clips_flagged,
        # For genuine audio a flag is a false positive; for spoofs it is a
        # catch. Same number, opposite meaning -- labelled so nobody reads the
        # table the wrong way round at 1 a.m.
        "genuine": genuine,
    }


def print_table(rows: list[tuple[str, dict]], threshold: float, codec: str) -> None:
    print(f"\n--- codec: {codec}   threshold: {threshold:.3f} ---")
    print(f"{'source':<22}{'clips':>6}{'windows':>9}{'mean':>8}{'clip-level':>17}   verdict")
    print("-" * 80)

    for name, s in rows:
        if s["empty"]:
            print(f"{name:<22}{'—':>6}{'—':>9}{'—':>8}{'—':>17}   no clips")
            continue

        rate = s["clip_rate"]
        if s["genuine"]:
            label = f"{rate:.0%} false pos"
            verdict = "good" if rate <= 0.1 else "TOO MANY FALSE POSITIVES"
        else:
            label = f"{int(rate * s['clips'])}/{s['clips']} caught"
            verdict = "caught" if rate >= 0.8 else ("partial" if rate >= 0.4 else "MISSED")

        print(
            f"{name:<22}{s['clips']:>6}{s['windows']:>9}{s['mean']:>8.3f}"
            f"{label:>17}   {verdict}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True,
                        help="folder of genuine clips (the reference recordings)")
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--codecs", nargs="+", default=None,
                        help=f"default: everything available ({available()})")
    parser.add_argument("--threshold", type=float, default=None,
                        help="default: fusion.calibration.midpoint from config.yaml")
    parser.add_argument("--model", default=None,
                        help="override SWAR_SPOOF_MODEL for this run, so two "
                             "checkpoints can be compared on identical clips")
    args = parser.parse_args()

    # Must be set before get_settings() is first called -- it is lru_cached,
    # so a later assignment would be silently ignored and you would compare a
    # checkpoint against itself.
    if args.model:
        os.environ["SWAR_SPOOF_MODEL"] = args.model

    config = get_config()
    threshold = args.threshold or config["fusion"]["calibration"]["midpoint"]
    codecs = args.codecs or available()

    if not args.real.is_dir():
        sys.exit(f"not a directory: {args.real}")

    engine_dirs = sorted(
        d for d in args.eval_root.glob("*") if d.is_dir() and any(d.glob("*.wav"))
    )
    if not engine_dirs:
        sys.exit(f"no engine folders with clips under {args.eval_root} "
                 f"(run scripts/clone_voices.py first)")

    detector = next(
        d for d in build_detectors() if not isinstance(d, DSPEvidenceDetector)
    )
    print(f"detector: {detector.name}")
    print(f"codecs:   {codecs}")
    if "opus" not in codecs:
        print("          (install ffmpeg for Opus and AMR-NB)")

    for codec in codecs:
        rows: list[tuple[str, dict]] = []

        genuine = score_folder(args.real, detector, config, codec)
        rows.append(("real (genuine)", summarise(genuine, threshold, genuine=True)))

        for folder in engine_dirs:
            result = score_folder(folder, detector, config, codec)
            rows.append((folder.name, summarise(result, threshold, genuine=False)))

        print_table(rows, threshold, codec)

    print(
        "\nReading this:\n"
        "  MISSED on an engine means the detector does not transfer to that\n"
        "  vocoder family. Say so in the video and name the mitigation --\n"
        "  adversarial retraining on that family, or a one-class objective.\n"
        "  Do not tune the threshold until the miss disappears: that trades a\n"
        "  real miss for false positives on genuine callers, which is the\n"
        "  error that actually costs a customer their transaction."
    )


if __name__ == "__main__":
    main()
