"""Fixed-capacity float32 ring buffer for incoming PCM."""
from __future__ import annotations

import numpy as np


class RingBuffer:
    """Holds the most recent `capacity` samples.

    Sized to the analysis window, not the whole call: raw audio is never
    retained beyond what the current window needs. That is the DPDP-2023
    posture expressed in code rather than in a policy document.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._pos = 0       # next write index
        self._written = 0   # total samples ever written

    def write(self, samples: np.ndarray) -> None:
        """Append samples, overwriting the oldest."""
        samples = np.asarray(samples, dtype=np.float32).ravel()
        n = samples.size
        if n == 0:
            return

        # A chunk larger than the window: only its tail can survive, so skip
        # the wrap arithmetic entirely.
        if n >= self.capacity:
            self._buf[:] = samples[-self.capacity:]
            self._pos = 0
            self._written += n
            return

        end = self._pos + n
        if end <= self.capacity:
            self._buf[self._pos:end] = samples
        else:
            split = self.capacity - self._pos
            self._buf[self._pos:] = samples[:split]
            self._buf[:n - split] = samples[split:]

        self._pos = end % self.capacity
        self._written += n

    def read_window(self) -> np.ndarray | None:
        """Return the newest `capacity` samples in order, or None if not filled.

        Returns a copy: the caller hands this to a model that may hold it
        across an await, while writes keep landing in the underlying buffer.
        """
        if not self.ready:
            return None
        if self._pos == 0:
            return self._buf.copy()
        return np.concatenate((self._buf[self._pos:], self._buf[:self._pos]))

    @property
    def ready(self) -> bool:
        return self._written >= self.capacity
