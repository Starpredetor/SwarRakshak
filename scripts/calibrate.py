"""Fit the decision threshold on your own demo clips.

Why this is needed: a pretrained spoof detector that never saw your cloning
tool will often still *separate* your real and fake clips while placing the
boundary nowhere near 0.5. Fitting the midpoint recovers the separation the
model already has.

Why this is not cheating, stated plainly so you can say it on camera: you are
calibrating an operating point on held-out material, which is what any
deployed detector does per-channel. What it is *not* is a benchmark result --
do not report an EER from these numbers. For that, run ASVspoof properly.

Writes fusion.calibration.{midpoint,temperature} back into config.yaml and,
if sklearn is available, trains the offline fallback classifier too.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def score_directory(path: Path) -> list[float]:
    """Run every clip in `path` through the loaded detector, window by window."""
    raise NotImplementedError


def pick_midpoint(real: list[float], fake: list[float]) -> tuple[float, float]:
    """Choose the threshold maximising separation; return (midpoint, temperature).

    Also prints the achieved separation. If the two distributions overlap
    badly, say so loudly -- that means the detector does not transfer to your
    cloning tool and no threshold will rescue it.
    """
    raise NotImplementedError


def train_fallback(real: list[Path], fake: list[Path], out: Path) -> None:
    """Fit LFCC + logistic regression for backend/detector/fallback.py."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--fake", type=Path, required=True)
    parser.add_argument("--write-config", action="store_true", default=True)
    raise NotImplementedError


if __name__ == "__main__":
    main()
