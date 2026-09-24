"""WebSocket ingest -- the one path both the microphone and replay use.

Protocol:
  client -> server   binary frames: little-endian int16 mono PCM @ 16 kHz
  client -> server   text frames:   {"type": "replay", "filename": "..."}
                                    {"type": "stop"}
  server -> client   text frames:   SessionInfo, then Verdict per hop,
                                    ErrorMessage on trouble

int16 rather than float32 halves the bytes on the wire for no audible loss at
these levels, and every browser can produce it from an AudioWorklet.
"""
from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import ROOT, get_config
from backend.detector import loaded_detectors
from backend.models import ErrorMessage, RunInfo
from backend.stream.replay import _KNOWN_LABELS, resolve_clip, stream_clip
from backend.stream.session import Session

logger = logging.getLogger(__name__)
router = APIRouter()

CLIPS_DIR = ROOT / "data" / "demo_clips"

# Finished sessions, kept in memory for /api/session/{id}. Feature-derived
# summaries only -- no audio ever reaches this dict.
SESSION_SUMMARIES: dict[str, dict] = {}
_MAX_SUMMARIES = 50


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Decode a binary frame into float32 in [-1, 1]."""
    if len(data) % 2:
        data = data[:-1]  # a torn frame costs one sample, not the connection
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


async def _send(websocket: WebSocket, payload) -> None:
    await websocket.send_text(payload.model_dump_json())


@router.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket) -> None:
    """Accept a connection, build a Session, pump verdicts until it closes.

    Errors are reported to the client as ErrorMessage and the socket stays open
    where possible. Dropping the connection mid-demo looks like a crash even
    when the cause is a single bad window.
    """
    await websocket.accept()

    config = get_config()
    session = Session(loaded_detectors(), config)
    await _send(websocket, session.info())

    replay: asyncio.Task | None = None

    async def run_replay(filename: str) -> None:
        """Feed a bundled clip through the same session the mic would use."""
        try:
            path = resolve_clip(CLIPS_DIR, filename)
        except ValueError as exc:
            await _send(websocket, ErrorMessage(message=str(exc)))
            return

        # Ground truth from the folder name. Sent to the client for checking
        # results by hand; it never reaches a detector.
        parent = path.parent.name.lower()
        label = parent if parent in _KNOWN_LABELS else None

        session.reset(filename, label)
        await _send(websocket, RunInfo(
            session_id=session.session_id, source=filename, source_label=label
        ))

        try:
            async for chunk in stream_clip(path, session.sample_rate):
                if session.feed(chunk):
                    verdict = await session.score_window()
                    if verdict is not None:
                        await _send(websocket, verdict)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("replay of %s failed", filename)
            await _send(websocket, ErrorMessage(message=f"replay failed: {filename}"))

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes") is not None:
                # While a clip is playing, ignore the mic. Mixing two sources
                # into one window would score a chimera that exists nowhere.
                if replay is not None and not replay.done():
                    continue
                if session.feed(pcm16_to_float32(message["bytes"])):
                    verdict = await session.score_window()
                    if verdict is not None:
                        await _send(websocket, verdict)
                continue

            if message.get("text") is None:
                continue

            try:
                command = json.loads(message["text"])
            except json.JSONDecodeError:
                await _send(websocket, ErrorMessage(message="malformed command"))
                continue

            kind = command.get("type")
            if kind == "replay":
                if replay is not None and not replay.done():
                    replay.cancel()
                replay = asyncio.create_task(run_replay(command.get("filename", "")))
            elif kind == "mic_start":
                if replay is not None and not replay.done():
                    replay.cancel()
                    replay = None
                session.reset("mic", None)
                await _send(websocket, RunInfo(
                    session_id=session.session_id, source="mic", source_label=None
                ))
            elif kind == "stop":
                if replay is not None:
                    replay.cancel()
                    replay = None
            else:
                await _send(websocket, ErrorMessage(message=f"unknown command: {kind}"))

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("session %s failed", session.session_id)
    finally:
        if replay is not None:
            replay.cancel()
        SESSION_SUMMARIES[session.session_id] = session.summary()
        if len(SESSION_SUMMARIES) > _MAX_SUMMARIES:
            SESSION_SUMMARIES.pop(next(iter(SESSION_SUMMARIES)))
        logger.info("session %s ended: %s", session.session_id, session.summary())
