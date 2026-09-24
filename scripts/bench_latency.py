"""Measure per-window latency against the 300 ms target.

Reports p50 and p95 separately: the mean hides exactly the occasional slow
window that will stall the gauge while you are recording.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_config  # noqa: E402
from backend.detector import build_detectors  # noqa: E402

TARGET_MS = 300


def bench(iterations: int = 100, warmup: int = 5) -> dict:
    """Time `iterations` scoring passes over a 2 s window of speech-like noise."""
    config = get_config()
    sr = config["audio"]["sample_rate"]
    window = int(sr * config["audio"]["window_seconds"])

    detectors = build_detectors()
    print(f"layers: {[d.name for d in detectors]}\n")

    rng = np.random.default_rng(0)
    t = np.arange(window) / sr
    wav = (0.2 * np.sin(2 * np.pi * 180 * t)
           + 0.02 * rng.standard_normal(window)).astype(np.float32)

    # Warmup passes are excluded. The first forward on CUDA pays for kernel
    # autotuning and would drag the p50 into telling you a lie.
    for _ in range(warmup):
        for detector in detectors:
            detector.score(wav, sr)

    per_layer: dict[str, list[float]] = {d.name: [] for d in detectors}
    totals: list[float] = []

    for _ in range(iterations):
        window_started = time.perf_counter()
        for detector in detectors:
            started = time.perf_counter()
            detector.score(wav, sr)
            per_layer[detector.name].append((time.perf_counter() - started) * 1000)
        totals.append((time.perf_counter() - window_started) * 1000)

    return {"per_layer": per_layer, "totals": totals}


def report(samples: list[float], label: str) -> float:
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    p95 = ordered[int(len(ordered) * 0.95) - 1]
    print(f"{label:<16} p50 {p50:7.1f} ms   p95 {p95:7.1f} ms   max {ordered[-1]:7.1f} ms")
    return p95


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    results = bench(args.iterations)

    for name, samples in results["per_layer"].items():
        report(samples, name)
    print()
    p95 = report(results["totals"], "TOTAL")

    print()
    if p95 <= TARGET_MS:
        print(f"PASS -- p95 {p95:.0f} ms is within the {TARGET_MS} ms budget.")
        sys.exit(0)

    print(f"FAIL -- p95 {p95:.0f} ms exceeds the {TARGET_MS} ms budget.")
    print("Options: confirm you are on CUDA (/api/health), shorten")
    print("audio.window_seconds, or drop the DSP layer's heavier features.")
    sys.exit(1)


if __name__ == "__main__":
    main()
