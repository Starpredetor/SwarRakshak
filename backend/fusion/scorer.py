"""Fusion, smoothing and banding.

The needle on screen is as much a product of this file as of the model. A
detector that is right 90% of the time but flickers between bands every second
looks broken to a viewer; a slightly duller needle that moves decisively and
stays put reads as confident. Hence EMA plus hysteresis plus a dwell
requirement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from backend.models import Band, LayerScore

_BAND_ORDER: tuple[Band, ...] = (Band.LOW, Band.MEDIUM, Band.HIGH)

_ACTIONS: dict[Band, str] = {
    Band.LOW: "Pass",
    # Advisory only. A false positive on a bad line should cost the customer a
    # verification step, never their transaction.
    Band.MEDIUM: "Advise MFA",
    Band.HIGH: "Hold & escalate",
}


class AllLayersAbstained(RuntimeError):
    """Every layer declined to score. The caller must emit no verdict."""


@dataclass
class FusionState:
    """Per-session smoothing state. One instance per call."""

    ema: float | None = None
    band: Band = Band.LOW
    candidate_band: Band | None = None
    candidate_windows: int = 0
    history: list[float] = field(default_factory=list)


def fuse(scores: list[LayerScore], weights: dict[str, float]) -> float:
    """Weighted mean of the non-abstaining layers, renormalised.

    Abstentions are dropped and the remaining weights rescaled to sum to 1, so
    losing a layer mid-call shifts what the score means rather than dragging it
    toward zero. A layer with no configured weight is ignored rather than
    defaulted -- silently inventing a weight would make a config typo look like
    a working fusion.
    """
    usable = [(s, weights[s.layer]) for s in scores
              if not s.abstain and weights.get(s.layer, 0.0) > 0.0]
    if not usable:
        raise AllLayersAbstained("no layer produced a usable score")

    total = sum(w for _, w in usable)
    return sum(s.score * w for s, w in usable) / total


def calibrate(raw: float, midpoint: float, temperature: float) -> float:
    """Squash the fused score through a sigmoid centred on `midpoint`.

    `midpoint` comes from scripts/calibrate.py. Pretrained spoof detectors are
    often confidently wrong in absolute terms while still separating the
    classes cleanly, so the threshold is fitted rather than assumed to be 0.5.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # Clamped because exp() overflows well before the score does anything
    # interesting.
    z = max(-60.0, min(60.0, (raw - midpoint) / temperature))
    return 1.0 / (1.0 + math.exp(-z))


def smooth(state: FusionState, value: float, alpha: float) -> float:
    """Exponential moving average; updates `state` in place."""
    state.ema = value if state.ema is None else alpha * value + (1 - alpha) * state.ema
    state.history.append(state.ema)
    return state.ema


def to_risk(calibrated: float) -> int:
    """Map [0, 1] onto the 0-100 risk score the UI and the pitch both use."""
    return int(round(max(0.0, min(1.0, calibrated)) * 100))


def band_for(
    state: FusionState,
    risk: int,
    bands: dict,
    min_dwell: int,
) -> tuple[Band, bool]:
    """Resolve the band with hysteresis and a dwell requirement.

    Two mechanisms, doing different jobs:

    - Hysteresis: the current band holds while the score stays inside its
      [enter, exit] range. Those ranges overlap, so a score of 42 keeps a LOW
      call LOW and a MEDIUM call MEDIUM. Without this the gauge oscillates
      wherever the score sits near a boundary -- which, on a marginal line, is
      most of the call.
    - Dwell: a new band must be the candidate for `min_dwell` consecutive
      windows before it is announced, so one freak window cannot flip the
      display.

    Returns the band in force and whether it changed this window.
    """
    current = bands[state.band.value]
    if current["enter"] <= risk <= current["exit"]:
        state.candidate_band = None
        state.candidate_windows = 0
        return state.band, False

    # Outside the current band: the target is the highest band this score has
    # cleared the entry threshold for.
    candidate = state.band
    for band in _BAND_ORDER:
        if risk >= bands[band.value]["enter"]:
            candidate = band

    if candidate is state.band:
        state.candidate_band = None
        state.candidate_windows = 0
        return state.band, False

    if state.candidate_band is candidate:
        state.candidate_windows += 1
    else:
        state.candidate_band = candidate
        state.candidate_windows = 1

    if state.candidate_windows >= min_dwell:
        state.band = candidate
        state.candidate_band = None
        state.candidate_windows = 0
        return state.band, True

    return state.band, False


def action_for(band: Band) -> str:
    """The operator-facing instruction.

    MEDIUM is advisory and never blocks the call -- see _ACTIONS.
    """
    return _ACTIONS[band]
