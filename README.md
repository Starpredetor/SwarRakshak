# SwarRakshak

Real-time voice-clone detection for live calls. SIH 2026 · PS ID SIH26104 · Team PCR (120414).

Existing deepfake-audio tools are forensic: you hand them a recording after the
money is gone. SwarRakshak sits inside the call and answers the question while
the call is still happening. Raw audio is never persisted -- only derived
features.

## MVP scope

This first cut ships **one trained detection layer**, not three:

| Layer | Status |
|---|---|
| L1 anti-spoof (wav2vec2 / AASIST-style synthetic-speech classifier) | **live** |
| L1b DSP evidence (spectral flatness, HF-energy ratio, jitter/shimmer proxy) | **live**, low weight, explainability only |
| L2 speaker verification (ECAPA-TDNN vs enrolled voiceprint) | not in MVP |
| L3 diarisation / speaker-change | not in MVP |

Both audio sources converge on one pipeline: browser microphone (AudioWorklet
-> 16 kHz mono PCM -> WebSocket) and server-side real-time WAV replay. One code
path, one thing to debug.

## Quick start

Backend:

```bash
python -m venv .venv && .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cp .env.example .env
python scripts/fetch_models.py
uvicorn backend.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173.

## Before you record

Run these in order. Each one fails loudly rather than silently degrading.

```bash
python scripts/smoke_test.py
```

Drives the whole WebSocket path with a synthetic tone. Proves ingest works
before a microphone is anywhere near the problem.

```bash
python scripts/bench_latency.py
```

Prints p50/p95 per-window latency against the 300 ms target.

```bash
python scripts/calibrate.py --real data/demo_clips/real --fake data/demo_clips/fake
```

Pretrained spoof detectors generalise poorly to vocoders they have not seen --
that is the In-the-Wild / WaveFake gap. This picks a threshold that separates
*your* clips and writes it into `config.yaml`.

**Do not quote an EER figure in the video** unless you have actually run
ASVspoof. Calibrating on your own demo clips is honest; presenting it as a
benchmark result is not.

## Layout

```
backend/
  api/        FastAPI routes + WebSocket ingest
  detector/   L1 model, DSP evidence, offline fallback, registry
  fusion/     weighted fuse, EMA smoothing, band hysteresis
  stream/     ring buffer, per-session state, WAV replay pacing
frontend/     Vite + React + TS dashboard
scripts/      fetch / calibrate / bench / smoke
data/         demo clips and fetched weights (gitignored)
```

## Deliberately not here

PostgreSQL/TimescaleDB, Redis, Celery/RabbitMQ, Triton, Kubernetes, SIP tap,
voiceprint enrolment. All of it is setup cost with no on-screen payoff for a
single demo session. They belong in the roadmap slide, not in the MVP.
