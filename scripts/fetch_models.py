"""Download and verify the L1 checkpoint ahead of time.

Run this once, on the machine you will demo from, while you still have
bandwidth and patience. It also prints the checkpoint's id2label so you can
confirm which index means "synthetic" -- get that backwards and the demo
inverts while still looking like a working detector.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from backend.config import get_config, get_settings, resolve_device  # noqa: E402


def verify(model_id: str) -> bool:
    """Load the checkpoint, resolve its labels, run one timed forward pass."""
    from backend.detector.hf_spoof import HFSpoofDetector

    config = get_config()
    device = resolve_device()
    label = config["detector"].get("spoof_label", "fake")

    print(f"\n=== {model_id} ===")
    detector = HFSpoofDetector(model_id, device, label)

    try:
        started = time.perf_counter()
        detector.load()
        print(f"loaded in {time.perf_counter() - started:.1f}s on {device}")
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False

    id2label = detector._model.config.id2label
    print(f"labels:      {id2label}")
    print(f"spoof index: {detector._spoof_index}  -> '{id2label[detector._spoof_index]}'")
    print("  ^ confirm this is the SYNTHETIC class. If it is not, set")
    print("    detector.spoof_label in config.yaml and rerun.")

    sr = config["audio"]["sample_rate"]
    window = np.random.default_rng(0).standard_normal(
        int(sr * config["audio"]["window_seconds"])
    ).astype(np.float32) * 0.1

    started = time.perf_counter()
    score = detector.score(window, sr)
    elapsed = (time.perf_counter() - started) * 1000

    print(f"forward:     {elapsed:.0f} ms  (target < 300 ms)")
    print(f"score:       {score.score:.3f} on noise")
    detector.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="override config.yaml candidates")
    parser.add_argument("--all", action="store_true", help="try every candidate")
    args = parser.parse_args()

    if args.model:
        candidates = [args.model]
    else:
        settings, config = get_settings(), get_config()
        candidates = [settings.spoof_model]
        for c in config["detector"].get("spoof_model_candidates", []):
            if c not in candidates:
                candidates.append(c)

    working = []
    for model_id in candidates:
        if verify(model_id):
            working.append(model_id)
            if not args.all:
                break

    print("\n" + "=" * 60)
    if working:
        print(f"usable checkpoints: {working}")
        print(f"set SWAR_SPOOF_MODEL={working[0]} in .env to pin it")
        sys.exit(0)

    print("NO checkpoint loaded. Options:")
    print("  - check your network / HuggingFace availability")
    print("  - pass --model with a different id")
    print("  - train the offline fallback: scripts/calibrate.py")
    sys.exit(1)


if __name__ == "__main__":
    main()
