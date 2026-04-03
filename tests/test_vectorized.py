"""Tests for vectorized numpy Morton encode/decode functions."""
import numpy as np
import pytest
from libmorton import (
    morton2D_32_encode,
    morton2D_32_encode_array, morton2D_32_decode_array,
    morton2D_64_encode_array, morton2D_64_decode_array,
)


# -- Roundtrip 32-bit arrays --

def test_roundtrip_32():
    x = np.array([0, 1, 0, 1, 100, 65535, 0, 65535], dtype=np.uint16)
    y = np.array([0, 0, 1, 1, 200, 0, 65535, 65535], dtype=np.uint16)
    codes = morton2D_32_encode_array(x, y)
    rx, ry = morton2D_32_decode_array(codes)
    np.testing.assert_array_equal(rx, x)
    np.testing.assert_array_equal(ry, y)


def test_roundtrip_64():
    x = np.array([0, 1, 100000, 2**32 - 1], dtype=np.uint32)
    y = np.array([0, 1, 200000, 2**32 - 1], dtype=np.uint32)
    codes = morton2D_64_encode_array(x, y)
    rx, ry = morton2D_64_decode_array(codes)
    np.testing.assert_array_equal(rx, x)
    np.testing.assert_array_equal(ry, y)


# -- Cross-check: array results match scalar results --

def test_array_matches_scalar_32():
    xs = [0, 1, 0, 1, 42, 255]
    ys = [0, 0, 1, 1, 17, 255]
    expected = np.array([morton2D_32_encode(x, y) for x, y in zip(xs, ys)],
                        dtype=np.uint32)
    x_arr = np.array(xs, dtype=np.uint16)
    y_arr = np.array(ys, dtype=np.uint16)
    codes = morton2D_32_encode_array(x_arr, y_arr)
    np.testing.assert_array_equal(codes, expected)


# -- Shape preservation --

def test_shape_2d_input():
    x = np.array([[0, 1], [2, 3]], dtype=np.uint16)
    y = np.array([[0, 0], [0, 0]], dtype=np.uint16)
    codes = morton2D_32_encode_array(x, y)
    assert codes.shape == (2, 2)
    rx, ry = morton2D_32_decode_array(codes)
    assert rx.shape == (2, 2)
    np.testing.assert_array_equal(rx, x)


# -- Edge cases --

def test_empty_array():
    x = np.array([], dtype=np.uint16)
    y = np.array([], dtype=np.uint16)
    codes = morton2D_32_encode_array(x, y)
    assert len(codes) == 0


def test_single_element():
    x = np.array([42], dtype=np.uint16)
    y = np.array([17], dtype=np.uint16)
    codes = morton2D_32_encode_array(x, y)
    assert codes[0] == morton2D_32_encode(42, 17)


# -- Input validation --

def test_32_overflow_raises():
    x = np.array([65536], dtype=np.uint32)
    y = np.array([0], dtype=np.uint32)
    with pytest.raises(ValueError):
        morton2D_32_encode_array(x, y)
