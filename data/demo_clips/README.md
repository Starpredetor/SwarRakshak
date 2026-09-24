# Demo clips

`real/` and `fake/` are read by the replay control; the subdirectory name
becomes the label.

## The placeholder clips are not demo material

`placeholder_natural.wav` and `placeholder_synthetic.wav` are generated
signals -- a jittered harmonic stack and an over-regular one. They exist to
prove the replay path works and that the model's label mapping is the right
way round. They score 6 and 94 respectively, which demonstrates plumbing, not
accuracy.

**Delete them and drop in real clips before recording anything:**

- `real/` -- your own voice, 16 kHz mono WAV, 10-30 s
- `fake/` -- a clone of that same voice from whatever TTS tool you are
  demonstrating against

Then calibrate on them:

```bash
python scripts/calibrate.py --real data/demo_clips/real --fake data/demo_clips/fake
```

If calibration reports heavy overlap, the detector does not transfer to your
cloning tool. Try another checkpoint (`scripts/fetch_models.py --all`) rather
than nudging the threshold until the demo passes.
