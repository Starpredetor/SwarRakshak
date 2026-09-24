"""End-to-end check of the WebSocket path using a synthetic tone.

Run this before a microphone is anywhere near the problem. If it passes, any
later failure is in audio capture rather than in the pipeline -- and that
distinction is worth a lot when something breaks twenty minutes before you
record.

Exits non-zero on failure so it can gate a pre-demo checklist.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import numpy as np

SAMPLE_RATE = 16000
CHUNK_MS = 100


def synth_tone(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A tone plus light noise at speech-like levels.

    Amplitude matters: it has to clear the silence gate in config.yaml, or the
    smoke test would "pass" while every layer abstained and nothing was ever
    actually scored.
    """
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 180 * t) + 0.1 * np.sin(2 * np.pi * 420 * t)
    noise = 0.01 * np.random.default_rng(0).standard_normal(t.size)
    return (tone + noise).astype(np.float32)


def to_pcm16(wav: np.ndarray) -> bytes:
    return (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2").tobytes()


async def run(url: str, seconds: float) -> int:
    import websockets

    wav = synth_tone(seconds)
    chunk = int(SAMPLE_RATE * CHUNK_MS / 1000)

    verdicts: list[dict] = []
    session: dict | None = None
    errors: list[str] = []

    async with websockets.connect(url, max_size=None) as ws:

        async def reader() -> None:
            nonlocal session
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "session":
                    session = msg
                elif msg["type"] == "verdict":
                    verdicts.append(msg)
                elif msg["type"] == "error":
                    errors.append(msg["message"])

        reader_task = asyncio.create_task(reader())

        for i in range(0, len(wav), chunk):
            await ws.send(to_pcm16(wav[i : i + chunk]))
            await asyncio.sleep(CHUNK_MS / 1000)

        # Let the last window finish scoring before tearing the socket down.
        await asyncio.sleep(1.5)
        reader_task.cancel()

    print(f"session:  {session}")
    print(f"verdicts: {len(verdicts)}")
    if verdicts:
        latencies = [v["latency_ms"] for v in verdicts]
        print(f"latency:  min {min(latencies):.0f} ms  max {max(latencies):.0f} ms")
        print(f"risk:     {[v['risk'] for v in verdicts]}")

    failures: list[str] = []
    if session is None:
        failures.append("no session handshake received")
    if errors:
        failures.append(f"server reported errors: {errors}")

    # Expect roughly one verdict per hop after the first window fills. Allow
    # slack for dropped windows -- those are by design under load, not a bug.
    expected = max(1, int((seconds - 2.0) / 1.0))
    if len(verdicts) < expected * 0.5:
        failures.append(f"expected ~{expected} verdicts, got {len(verdicts)}")

    if any(v["layers"] and all(lyr["abstain"] for lyr in v["layers"]) for v in verdicts):
        failures.append(
            "some windows had every layer abstain -- check the silence gate "
            "and that a detector actually loaded"
        )

    if session and session.get("detector_is_fallback"):
        failures.append("running the FALLBACK detector, not the trained model")

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)

    if not failures:
        print("\nPASS -- the socket path works end to end.")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://localhost:8000/ws/stream")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.url, args.seconds)))


if __name__ == "__main__":
    main()
