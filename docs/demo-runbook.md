# Demo runbook

Written for the version of you that is twenty minutes from recording and has
stopped thinking clearly.

## T-60 min — prove the pipeline

```bash
curl http://localhost:8000/api/health
```

Read the response. It names the detector that actually loaded and the device
it is on. If `detector_is_fallback` is true, you are about to record the weak
model — stop and fix it.

```bash
python scripts/smoke_test.py
python scripts/bench_latency.py
```

Smoke test first. It exercises the whole socket path with a synthetic tone, so
if it passes, any later failure is in audio capture rather than in the
pipeline. That distinction is worth a lot when something breaks.

## T-30 min — calibrate on your own clips

```bash
python scripts/calibrate.py --real data/demo_clips/real --fake data/demo_clips/fake
```

If it reports heavy overlap between the two distributions, the detector does
not transfer to your cloning tool and no threshold will fix it. Fall back to
a different checkpoint from the candidate list rather than nudging the
midpoint until the demo passes.

## T-10 min — capture setup

- Disable OS-level noise suppression. It rewrites exactly the bands the
  detector reads.
- Use the same input device you tested with. A different mic is a different
  channel and the calibration does not carry over.
- Have the replay clip queued. If the live take goes wrong twice, cut to it.

## Recording order

1. Speak naturally. Gauge sits LOW. Let it hold for ten seconds — steadiness
   is the claim.
2. Play the cloned sample. Gauge climbs into MEDIUM, then HIGH.
3. Show the evidence panel while it is HIGH: flat spectrum, pitch too steady.
4. Say the honest bit — MEDIUM advises MFA, it does not block the call.

## What not to claim

- No EER number unless you actually ran ASVspoof. Calibration on your own
  clips is an operating point, not a benchmark.
- One trained layer is running, not three. The greyed-out cards are the
  roadmap and saying so costs nothing — being caught overstating it costs the
  round.
