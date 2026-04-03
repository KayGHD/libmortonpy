"""Tests for scalar C++ Morton encode/decode functions."""
import pytest
from libmorton import (
    morton2D_32_encode, morton2D_32_decode,
    morton2D_64_encode, morton2D_64_decode,
)


# -- Known values (2D 32-bit) --

KNOWN_32 = [
    # (x, y, expected_morton)
    (0, 0, 0),
    (1, 0, 1),
    (0, 1, 2),
    (1, 1, 3),
    (2, 0, 4),
    (0, 2, 8),
    (2, 2, 12),
    (3, 3, 15),
    (255, 255, 65535),
]


@pytest.mark.parametrize("x, y, expected", KNOWN_32)
def test_encode_32_known(x, y, expected):
    assert morton2D_32_encode(x, y) == expected


@pytest.mark.parametrize("x, y, morton", KNOWN_32)
def test_decode_32_known(x, y, morton):
    assert morton2D_32_decode(morton) == (x, y)


# -- Roundtrip 32-bit --

COORDS_32 = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (100, 200),
    (65535, 0),
    (0, 65535),
    (65535, 65535),
]


@pytest.mark.parametrize("x, y", COORDS_32)
def test_roundtrip_32(x, y):
    code = morton2D_32_encode(x, y)
    rx, ry = morton2D_32_decode(code)
    assert (rx, ry) == (x, y)


# -- Roundtrip 64-bit --

COORDS_64 = [
    (0, 0),
    (1, 1),
    (100, 200),
    (65535, 65535),
    (100000, 200000),
    (2**32 - 1, 0),
    (0, 2**32 - 1),
    (2**32 - 1, 2**32 - 1),
]


@pytest.mark.parametrize("x, y", COORDS_64)
def test_roundtrip_64(x, y):
    code = morton2D_64_encode(x, y)
    rx, ry = morton2D_64_decode(code)
    assert (rx, ry) == (x, y)


# -- Cross-check: 32-bit encode should match 64-bit for small values --

@pytest.mark.parametrize("x, y", [(0, 0), (1, 1), (255, 255)])
def test_32_64_agree_small(x, y):
    code32 = morton2D_32_encode(x, y)
    code64 = morton2D_64_encode(x, y)
    assert code32 == code64


# -- Input validation --

def test_encode_32_overflow_x():
    with pytest.raises((ValueError, OverflowError)):
        morton2D_32_encode(65536, 0)


def test_encode_32_overflow_y():
    with pytest.raises((ValueError, OverflowError)):
        morton2D_32_encode(0, 65536)


def test_encode_32_negative():
    with pytest.raises((ValueError, OverflowError, TypeError)):
        morton2D_32_encode(-1, 0)


def test_encode_64_negative():
    with pytest.raises((ValueError, OverflowError, TypeError)):
        morton2D_64_encode(-1, 0)
