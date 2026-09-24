import pytest

from backend.fusion.scorer import (
    AllLayersAbstained,
    FusionState,
    action_for,
    band_for,
    calibrate,
    fuse,
    smooth,
    to_risk,
)
from backend.models import Band, LayerScore

WEIGHTS = {"l1_spoof": 0.85, "l1b_dsp": 0.15}

BANDS = {
    "LOW": {"enter": 0, "exit": 44},
    "MEDIUM": {"enter": 40, "exit": 79},
    "HIGH": {"enter": 75, "exit": 100},
}


def layer(name: str, score: float, abstain: bool = False) -> LayerScore:
    return LayerScore(layer=name, score=score, confidence=1.0, abstain=abstain)


# --- fuse ---------------------------------------------------------------


def test_fuse_weights_layers():
    got = fuse([layer("l1_spoof", 1.0), layer("l1b_dsp", 0.0)], WEIGHTS)
    assert got == pytest.approx(0.85)


def test_abstaining_layer_is_dropped_and_weights_renormalise():
    # Losing the DSP layer must not drag the score toward zero -- the L1
    # opinion should stand on its own at full scale.
    got = fuse([layer("l1_spoof", 1.0), layer("l1b_dsp", 0.0, abstain=True)], WEIGHTS)
    assert got == pytest.approx(1.0)


def test_all_abstain_raises_rather_than_inventing_a_score():
    with pytest.raises(AllLayersAbstained):
        fuse([layer("l1_spoof", 0.9, abstain=True)], WEIGHTS)


def test_unweighted_layer_is_ignored():
    got = fuse([layer("l1_spoof", 1.0), layer("mystery", 0.0)], WEIGHTS)
    assert got == pytest.approx(1.0)


# --- calibrate / smooth / to_risk ---------------------------------------


def test_calibrate_is_centred_on_midpoint():
    assert calibrate(0.5, midpoint=0.5, temperature=0.15) == pytest.approx(0.5)


def test_calibrate_monotonic_and_bounded():
    lo, hi = calibrate(0.1, 0.5, 0.15), calibrate(0.9, 0.5, 0.15)
    assert 0.0 < lo < 0.5 < hi < 1.0


def test_calibrate_does_not_overflow_at_extremes():
    assert calibrate(0.0, 0.5, 0.001) == pytest.approx(0.0, abs=1e-9)
    assert calibrate(1.0, 0.5, 0.001) == pytest.approx(1.0, abs=1e-9)


def test_smooth_first_value_seeds_the_ema():
    state = FusionState()
    assert smooth(state, 0.8, alpha=0.4) == pytest.approx(0.8)


def test_smooth_lags_a_step_change():
    state = FusionState()
    smooth(state, 0.0, alpha=0.4)
    assert smooth(state, 1.0, alpha=0.4) == pytest.approx(0.4)


def test_to_risk_clamps():
    assert to_risk(-0.2) == 0
    assert to_risk(1.7) == 100
    assert to_risk(0.824) == 82


# --- banding ------------------------------------------------------------


def test_starts_low():
    assert FusionState().band is Band.LOW


def test_climbs_to_high_after_dwell():
    state = FusionState()
    assert band_for(state, 90, BANDS, min_dwell=2)[0] is Band.LOW  # 1st window
    band, changed = band_for(state, 90, BANDS, min_dwell=2)        # 2nd window
    assert band is Band.HIGH and changed


def test_single_freak_window_does_not_flip_the_band():
    state = FusionState()
    band_for(state, 95, BANDS, min_dwell=2)
    band, changed = band_for(state, 10, BANDS, min_dwell=2)
    assert band is Band.LOW and not changed


def test_hysteresis_holds_low_inside_the_overlap():
    # 42 is above MEDIUM.enter but inside LOW's exit -- a LOW call stays LOW.
    state = FusionState()
    for _ in range(5):
        band, changed = band_for(state, 42, BANDS, min_dwell=2)
        assert band is Band.LOW and not changed


def test_hysteresis_holds_medium_inside_the_same_overlap():
    # The same score, approached from above, stays MEDIUM. This is the whole
    # point: the band depends on where the call came from, not just on 42.
    state = FusionState(band=Band.MEDIUM)
    for _ in range(5):
        band, changed = band_for(state, 42, BANDS, min_dwell=2)
        assert band is Band.MEDIUM and not changed


def test_oscillating_score_does_not_flicker_the_band():
    state = FusionState()
    changes = 0
    for risk in [30, 50, 30, 50, 30, 50, 30, 50]:
        if band_for(state, risk, BANDS, min_dwell=2)[1]:
            changes += 1
    assert changes == 0


def test_sustained_climb_is_announced_once():
    state = FusionState()
    changes = sum(band_for(state, 85, BANDS, min_dwell=2)[1] for _ in range(10))
    assert changes == 1


def test_medium_advises_rather_than_blocks():
    assert action_for(Band.MEDIUM) == "Advise MFA"
    assert action_for(Band.LOW) == "Pass"
    assert action_for(Band.HIGH) == "Hold & escalate"
