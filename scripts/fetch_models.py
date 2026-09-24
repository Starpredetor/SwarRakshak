"""Download and verify the L1 checkpoint ahead of time.

Run this once, on the machine you will demo from, while you still have
bandwidth and patience. It also prints the checkpoint's id2label so you can
confirm which index means "synthetic" -- get that backwards and the demo
inverts.
"""
from __future__ import annotations

import argparse


def fetch(model_id: str, target_dir: str) -> None:
    """Pull weights into `target_dir` and report label mapping and size."""
    raise NotImplementedError


def verify(model_id: str) -> bool:
    """Load the checkpoint, run one dummy forward, report device and latency."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="override config.yaml candidates")
    parser.add_argument("--all", action="store_true", help="fetch every candidate")
    raise NotImplementedError


if __name__ == "__main__":
    main()
