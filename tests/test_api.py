"""Integration tests for the socket and REST surface.

These run the real FastAPI app with a fake detector swapped in, so the whole
ingest -> window -> fuse -> emit path is exercised without loading a
checkpoint. That is what makes them runnable before the model has downloaded,
and fast enough to run on every change afterwards.
"""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.api.ws as ws_module
import backend.detector as registry
import backend.main as main
from tests.conftest import FakeDetector


@pytest.fixture
def fake_detector():
    return FakeDetector(name="l1_spoof", score=0.95)


@pytest.fixture
def client(monkeypatch, fake_detector):
    monkeypatch.setattr(main, "build_detectors", lambda: [fake_detector])
    monkeypatch.setattr(registry, "_LOADED", [fake_detector])
    with TestClient(main.app) as c:
        yield c


def pcm(seconds: float, sample_rate: int = 16000, amplitude: float = 0.2) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    wav = amplitude * np.sin(2 * np.pi * 180 * t)
    return (wav * 32767).astype("<i2").tobytes()


def test_health_reports_the_loaded_detector(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["detector_name"] == "l1_spoof"
    assert body["detector_is_fallback"] is False


def test_clips_endpoint_survives_an_empty_directory(client):
    assert client.get("/api/clips").status_code == 200


def test_unknown_session_is_404(client):
    assert client.get("/api/session/nope").status_code == 404


def test_handshake_precedes_any_verdict(client):
    with client.websocket_connect("/ws/stream") as socket:
        first = json.loads(socket.receive_text())
    assert first["type"] == "session"
    assert first["sample_rate"] == 16000


def test_streaming_speech_produces_verdicts(client):
    with client.websocket_connect("/ws/stream") as socket:
        json.loads(socket.receive_text())  # handshake

        # 100 ms at a time, as the browser worklet sends it.
        for _ in range(40):
            socket.send_bytes(pcm(0.1))

        verdict = json.loads(socket.receive_text())

    assert verdict["type"] == "verdict"
    assert 0 <= verdict["risk"] <= 100
    assert verdict["band"] in {"LOW", "MEDIUM", "HIGH"}
    assert verdict["latency_ms"] >= 0


def test_sustained_synthetic_speech_escalates_to_high(client):
    # The score itself does not ramp -- the EMA seeds on its first value, so a
    # confident layer reads high from the first scored window. What ramps is
    # the *band*, held back by the dwell requirement so that one window cannot
    # flip the display.
    verdicts = []
    with client.websocket_connect("/ws/stream") as socket:
        json.loads(socket.receive_text())
        for _ in range(80):
            socket.send_bytes(pcm(0.1))
        for _ in range(4):
            verdicts.append(json.loads(socket.receive_text()))

    assert verdicts[0]["band"] == "LOW"
    assert verdicts[-1]["band"] == "HIGH"
    assert sum(v["band_changed"] for v in verdicts) == 1
    assert all(v["risk"] > 75 for v in verdicts)


def test_malformed_command_is_reported_not_fatal(client):
    with client.websocket_connect("/ws/stream") as socket:
        json.loads(socket.receive_text())
        socket.send_text("this is not json")
        error = json.loads(socket.receive_text())

        assert error["type"] == "error"
        assert error["fatal"] is False

        # The socket must still work afterwards.
        for _ in range(40):
            socket.send_bytes(pcm(0.1))
        assert json.loads(socket.receive_text())["type"] == "verdict"


def test_unknown_command_is_rejected(client):
    with client.websocket_connect("/ws/stream") as socket:
        json.loads(socket.receive_text())
        socket.send_text(json.dumps({"type": "launch_missiles"}))
        error = json.loads(socket.receive_text())
    assert error["type"] == "error"


def test_replay_of_a_missing_clip_errors_without_dropping_the_socket(client):
    with client.websocket_connect("/ws/stream") as socket:
        json.loads(socket.receive_text())
        socket.send_text(json.dumps({"type": "replay", "filename": "nope.wav"}))
        error = json.loads(socket.receive_text())
    assert error["type"] == "error"


def test_replay_cannot_escape_the_clips_directory(client):
    # The filename arrives over a socket. "../../.env" is a valid string.
    with client.websocket_connect("/ws/stream") as socket:
        json.loads(socket.receive_text())
        socket.send_text(json.dumps({"type": "replay", "filename": "../../.env"}))
        error = json.loads(socket.receive_text())
    assert error["type"] == "error"
    assert "no such clip" in error["message"]


def test_summary_is_recorded_after_disconnect(client):
    with client.websocket_connect("/ws/stream") as socket:
        session_id = json.loads(socket.receive_text())["session_id"]
        for _ in range(40):
            socket.send_bytes(pcm(0.1))
        json.loads(socket.receive_text())

    summary = ws_module.SESSION_SUMMARIES[session_id]
    assert summary["windows_scored"] >= 1
    assert "band_seconds" in summary
