"""End-to-end check of the WebSocket path using a synthetic tone.

Run this before a microphone is anywhere near the problem. If it passes, any
later failure is in audio capture, not in the pipeline -- which halves the
search space when something breaks twenty minutes before you record.

Exits non-zero on failure so it can gate a pre-demo checklist.
"""
from __future__ import annotations

import argparse


def synth_tone(seconds: float, sample_rate: int) -> bytes:
    """Generate int16 PCM: a tone plus light noise, shaped like speech levels."""
    raise NotImplementedError


async def run(url: str, seconds: float) -> int:
    """Connect, stream the tone in real time, assert verdicts arrive per hop.

    Checks: handshake received, verdict count matches elapsed hops, every
    verdict carries a band and a latency figure.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://localhost:8000/ws/stream")
    parser.add_argument("--seconds", type=float, default=10.0)
    raise NotImplementedError


if __name__ == "__main__":
    main()
