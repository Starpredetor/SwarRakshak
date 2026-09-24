import numpy as np
import pytest

from backend.stream.session import Session
from tests.conftest import FakeDetector


def feed_seconds(session, audio, seconds):
    """Feed `seconds` of audio in 100 ms chunks, returning hop crossings."""
    chunk = 1600
    crossings = 0
    for _ in range(int(seconds * 10)):
        offset = (crossings * chunk) % (len(audio) - chunk)
        if session.feed(audio[offset : offset + chunk]):
            crossings += 1
    return crossings


def test_no_verdict_before_the_window_fills(config, speech):
    session = Session([FakeDetector()], config)
    # 1.5 s < the 2 s window
    assert feed_seconds(session, speech, 1.5) == 0


def test_hop_crossing_once_the_window_is_full(config, speech):
    session = Session([FakeDetector()], config)
    assert feed_seconds(session, speech, 5.0) >= 2


@pytest.mark.asyncio
async def test_scores_a_full_window(config, speech):
    detector = FakeDetector(score=0.9)
    session = Session([detector], config)
    session.feed(speech)

    verdict = await session.score_window()

    assert verdict is not None
    assert detector.calls == 1
    assert verdict.risk > 50
    assert verdict.latency_ms >= 0


@pytest.mark.asyncio
async def test_silence_makes_every_layer_abstain(config, silence):
    # A detector fed near-silence produces confident nonsense. The gate must
    # fire before the model does -- so the detector is never even called.
    detector = FakeDetector(score=0.99)
    session = Session([detector], config)
    session.feed(silence)

    verdict = await session.score_window()

    assert detector.calls == 0
    assert all(layer.abstain for layer in verdict.layers)


@pytest.mark.asyncio
async def test_silence_holds_the_last_risk_rather_than_decaying(config, speech, silence):
    # Silence is not evidence of authenticity. A needle that sags when the
    # speaker pauses would read as the system clearing them.
    session = Session([FakeDetector(score=0.95)], config)

    session.feed(speech)
    for _ in range(4):
        high = await session.score_window()

    session.feed(silence)
    quiet = await session.score_window()

    assert quiet.risk == high.risk
    assert quiet.band is high.band


@pytest.mark.asyncio
async def test_a_layer_that_raises_abstains_instead_of_killing_the_session(
    config, speech
):
    # A layer throwing must cost its own opinion for that window and nothing
    # more. With only one layer there is nothing left to fuse, so the session
    # holds the previous risk rather than inventing one.
    session = Session([FakeDetector(raises=True)], config)
    session.feed(speech)

    verdict = await session.score_window()

    assert verdict is not None
    assert all(layer.abstain for layer in verdict.layers)


@pytest.mark.asyncio
async def test_one_layer_failing_leaves_the_others_scoring(config, speech):
    broken = FakeDetector(name="l1b_dsp", raises=True)
    working = FakeDetector(name="l1_spoof", score=0.95)
    session = Session([broken, working], config)
    session.feed(speech)

    verdict = await session.score_window()

    by_layer = {layer.layer: layer for layer in verdict.layers}
    assert by_layer["l1b_dsp"].abstain
    assert not by_layer["l1_spoof"].abstain
    assert verdict.risk > 50


@pytest.mark.asyncio
async def test_abstaining_layer_is_excluded_from_fusion(config, speech):
    loud = FakeDetector(name="l1_spoof", score=0.95)
    dead = FakeDetector(name="l1b_dsp", score=0.0, abstain=True)
    session = Session([loud, dead], config)
    session.feed(speech)

    verdict = await session.score_window()

    # The dead layer's 0.0 must not drag the score down.
    assert verdict.risk > 50


@pytest.mark.asyncio
async def test_summary_contains_no_audio(config, speech):
    session = Session([FakeDetector()], config)
    session.feed(speech)
    await session.score_window()

    summary = session.summary()

    assert set(summary) == {
        "session_id", "source", "source_label", "duration_seconds",
        "windows_scored", "windows_dropped", "peak_risk", "final_band",
        "band_seconds",
    }
    assert summary["windows_scored"] == 1


@pytest.mark.asyncio
async def test_sustained_clone_reaches_high(config, speech):
    session = Session([FakeDetector(score=0.99)], config)
    session.feed(speech)

    bands = []
    for _ in range(8):
        bands.append((await session.score_window()).band.value)

    assert bands[-1] == "HIGH"


def test_info_reports_the_fallback_honestly(config):
    detector = FakeDetector()
    detector.is_fallback = True
    session = Session([detector], config)

    assert session.info().detector_is_fallback is True
