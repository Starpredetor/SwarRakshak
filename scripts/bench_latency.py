"""Measure per-window latency against the 300 ms target.

Reports p50 and p95 separately: the mean hides exactly the occasional slow
window that will stall the gauge while you are recording.
"""
from __future__ import annotations

import argparse


def bench(iterations: int = 100, warmup: int = 5) -> dict[str, float]:
    """Time `iterations` forward passes over a 2 s window of noise."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default=None)
    raise NotImplementedError


if __name__ == "__main__":
    main()
