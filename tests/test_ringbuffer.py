import numpy as np
import pytest

from backend.stream.ringbuffer import RingBuffer


def test_not_ready_until_filled():
    rb = RingBuffer(10)
    rb.write(np.arange(9, dtype=np.float32))
    assert not rb.ready
    assert rb.read_window() is None


def test_returns_newest_samples_in_order():
    rb = RingBuffer(4)
    rb.write(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
    np.testing.assert_array_equal(rb.read_window(), [3, 4, 5, 6])


def test_wraps_across_writes():
    rb = RingBuffer(4)
    rb.write(np.array([1, 2, 3], dtype=np.float32))
    rb.write(np.array([4, 5], dtype=np.float32))
    np.testing.assert_array_equal(rb.read_window(), [2, 3, 4, 5])


def test_chunk_larger_than_capacity_keeps_tail():
    rb = RingBuffer(3)
    rb.write(np.arange(100, dtype=np.float32))
    np.testing.assert_array_equal(rb.read_window(), [97, 98, 99])


def test_read_window_is_a_copy():
    # The model may hold this across an await while writes keep landing.
    rb = RingBuffer(4)
    rb.write(np.array([1, 2, 3, 4], dtype=np.float32))
    w = rb.read_window()
    rb.write(np.array([9, 9, 9, 9], dtype=np.float32))
    np.testing.assert_array_equal(w, [1, 2, 3, 4])


def test_empty_write_is_noop():
    rb = RingBuffer(4)
    rb.write(np.array([], dtype=np.float32))
    assert not rb.ready


def test_rejects_zero_capacity():
    with pytest.raises(ValueError):
        RingBuffer(0)
