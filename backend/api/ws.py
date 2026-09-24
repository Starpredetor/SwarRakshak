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

import numpy as np
from fastapi import APIRouter, WebSocket

router = APIRouter()


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Decode a binary frame into float32 in [-1, 1]."""
    raise NotImplementedError


@router.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket) -> None:
    """Accept a connection, build a Session, pump verdicts until it closes.

    Errors are reported to the client as ErrorMessage and the socket stays open
    where possible. Dropping the connection mid-demo looks like a crash even
    when the cause is a single bad window.
    """
    raise NotImplementedError
