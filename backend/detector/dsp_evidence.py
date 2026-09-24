"""L1b: cheap DSP evidence, for explainability rather than for accuracy.

These features are what the dashboard shows the viewer when it claims a window
looks synthetic. They carry a low fusion weight on purpose: they are
heuristics with hand-picked thresholds, not a calibrated model, and saying so
is more defensible than dressing them up.

All of it is CPU work on numpy/librosa and costs a couple of milliseconds.
"""
from __future__ import annotations

import logging

import numpy as np

from backend.detector.base import Detector
from backend.models import LayerScore

logger = logging.getLogger(__name__)

# Rough natural-speech ranges, used only to normalise each feature onto [0, 1]
# for display and for the weak score. Hand-picked from typical 16 kHz
# telephone-band speech -- not fitted, and labelled as such in the UI.
_FLATNESS_RANGE = (0.02, 0.35)
_HF_RATIO_RANGE = (0.02, 0.30)
_JITTER_RANGE = (0.005, 0.04)


def spectral_flatness(wav: np.ndarray, sample_rate: int) -> float:
    """Geometric/arithmetic mean ratio of the magnitude spectrum.

    Vocoded speech tends to sit flatter than natural speech, which has sharper
    harmonic peaks.
    """
    import librosa

    return float(np.mean(librosa.feature.spectral_flatness(y=wav)))


def hf_energy_ratio(wav: np.ndarray, sample_rate: int, cutoff_hz: int = 6000) -> float:
    """Fraction of energy above `cutoff_hz`.

    Many vocoders under-generate the top octave; some over-generate it. Either
    way it deviates from the band a real 16 kHz voice occupies.
    """
    spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
    freqs = np.fft.rfftfreq(len(wav), 1.0 / sample_rate)
    total = float(np.sum(spec ** 2))
    if total <= 0:
        return 0.0
    return float(np.sum(spec[freqs >= cutoff_hz] ** 2) / total)


def jitter_shimmer_proxy(wav: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Cycle-to-cycle F0 and amplitude variation.

    Real phonation is slightly irregular. Synthesised phonation is often too
    regular -- unnaturally *low* jitter is itself a tell, which is why the
    score below penalises both extremes rather than just the high end.

    Returns (jitter, shimmer), both as relative variation. Unvoiced or silent
    windows return (0, 0) and the caller treats that as no evidence.
    """
    import librosa

    f0 = librosa.yin(wav, fmin=60, fmax=400, sr=sample_rate)
    voiced = f0[np.isfinite(f0) & (f0 > 0)]
    if voiced.size < 3:
        return 0.0, 0.0

    jitter = float(np.mean(np.abs(np.diff(voiced))) / np.mean(voiced))

    frame = max(1, int(0.02 * sample_rate))
    n = len(wav) // frame
    if n < 3:
        return jitter, 0.0
    amps = np.abs(wav[: n * frame].reshape(n, frame)).max(axis=1)
    amps = amps[amps > 0]
    if amps.size < 3:
        return jitter, 0.0
    shimmer = float(np.mean(np.abs(np.diff(amps))) / np.mean(amps))

    return jitter, shimmer


def _normalise(value: float, lo: float, hi: float) -> float:
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


class DSPEvidenceDetector(Detector):
    name = "l1b_dsp"

    def load(self) -> None:
        """No weights -- but librosa still has to be imported and warmed.

        Importing librosa costs seconds, and its first call JITs a numba
        kernel. Left lazy, that whole bill lands on the first window of the
        call and stalls the gauge for three seconds at exactly the moment
        somebody starts recording. Paying it at startup is the entire point of
        this method existing.
        """
        import librosa  # noqa: F401

        warm = np.zeros(16000, dtype=np.float32)
        warm[::100] = 0.5  # enough structure for yin to do real work
        self.score(warm, 16000)

    def score(self, wav: np.ndarray, sample_rate: int) -> LayerScore:
        """Combine the features above into a weak score plus a `detail` dict.

        `detail` is what the frontend renders in the evidence panel, so the
        keys are stable: flatness, hf_ratio, jitter, shimmer.
        """
        try:
            flatness = spectral_flatness(wav, sample_rate)
            hf_ratio = hf_energy_ratio(wav, sample_rate)
            jitter, shimmer = jitter_shimmer_proxy(wav, sample_rate)
        except Exception:
            logger.exception("%s failed on a window; abstaining", self.name)
            return LayerScore(layer=self.name, score=0.0, confidence=0.0, abstain=True)

        detail = {
            "flatness": flatness,
            "hf_ratio": hf_ratio,
            "jitter": jitter,
            "shimmer": shimmer,
        }

        if jitter == 0.0:
            # Unvoiced window: the pitch features say nothing, so this layer
            # declines rather than scoring on spectrum alone.
            return LayerScore(
                layer=self.name, score=0.0, confidence=0.0, abstain=True, detail=detail
            )

        # Flatter spectrum -> more suspicious. Pitch regularity is suspicious
        # at both ends, so the jitter term is folded about the middle.
        flat_term = _normalise(flatness, *_FLATNESS_RANGE)
        hf_term = abs(_normalise(hf_ratio, *_HF_RATIO_RANGE) - 0.5) * 2.0
        jitter_term = abs(_normalise(jitter, *_JITTER_RANGE) - 0.5) * 2.0
        score = float(np.clip(0.5 * flat_term + 0.25 * hf_term + 0.25 * jitter_term, 0, 1))

        return LayerScore(
            layer=self.name,
            score=score,
            # Deliberately capped. This layer is a heuristic and should never
            # be able to out-shout a trained model in the fused result.
            confidence=0.4,
            detail=detail,
        )
