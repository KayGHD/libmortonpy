"""
Vectorized Morton 2D encoding/decoding using magic-bits method.

Works with both numpy and cupy arrays. The array module is detected
automatically from the input type, so the same functions handle CPU
(numpy) and GPU (cupy) arrays transparently.
"""

import numpy as np


def _get_xp(*arrays):
    """Return the array module (numpy or cupy) for the given arrays."""
    for arr in arrays:
        mod = type(arr).__module__.split(".")[0]
        if mod == "cupy":
            import cupy
            return cupy
    return np


# ---------- 32-bit (16-bit inputs -> 32-bit Morton code) ----------

def _spread_bits_32(v, xp):
    """Spread bits of v for 2D interleaving (16-bit -> 32-bit)."""
    v = v.astype(xp.uint32)
    v = (v | (v << xp.uint32(8))) & xp.uint32(0x00FF00FF)
    v = (v | (v << xp.uint32(4))) & xp.uint32(0x0F0F0F0F)
    v = (v | (v << xp.uint32(2))) & xp.uint32(0x33333333)
    v = (v | (v << xp.uint32(1))) & xp.uint32(0x55555555)
    return v


def _compact_bits_32(v, xp):
    """Compact bits from 2D Morton code (32-bit -> 16-bit)."""
    v = v & xp.uint32(0x55555555)
    v = (v | (v >> xp.uint32(1))) & xp.uint32(0x33333333)
    v = (v | (v >> xp.uint32(2))) & xp.uint32(0x0F0F0F0F)
    v = (v | (v >> xp.uint32(4))) & xp.uint32(0x00FF00FF)
    v = (v | (v >> xp.uint32(8))) & xp.uint32(0x0000FFFF)
    return v.astype(xp.uint16)


def morton2D_32_encode(x, y):
    """Encode arrays of 2D coordinates into 32-bit Morton codes.

    Args:
        x: Array of x coordinates (uint16 range, 0..65535).
        y: Array of y coordinates (uint16 range, 0..65535).

    Returns:
        Array of uint32 Morton codes.

    Works with numpy and cupy arrays.
    """
    xp = _get_xp(x, y)
    return _spread_bits_32(x, xp) | (_spread_bits_32(y, xp) << xp.uint32(1))


def morton2D_32_decode(morton):
    """Decode arrays of 32-bit Morton codes into 2D coordinates.

    Args:
        morton: Array of uint32 Morton codes.

    Returns:
        Tuple of (x, y) arrays with uint16 dtype.

    Works with numpy and cupy arrays.
    """
    xp = _get_xp(morton)
    m = morton.astype(xp.uint32)
    x = _compact_bits_32(m, xp)
    y = _compact_bits_32(m >> xp.uint32(1), xp)
    return x, y


# ---------- 64-bit (32-bit inputs -> 64-bit Morton code) ----------

def _spread_bits_64(v, xp):
    """Spread bits of v for 2D interleaving (32-bit -> 64-bit)."""
    v = v.astype(xp.uint64)
    v = (v | (v << xp.uint64(16))) & xp.uint64(0x0000FFFF0000FFFF)
    v = (v | (v << xp.uint64(8))) & xp.uint64(0x00FF00FF00FF00FF)
    v = (v | (v << xp.uint64(4))) & xp.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v << xp.uint64(2))) & xp.uint64(0x3333333333333333)
    v = (v | (v << xp.uint64(1))) & xp.uint64(0x5555555555555555)
    return v


def _compact_bits_64(v, xp):
    """Compact bits from 2D Morton code (64-bit -> 32-bit)."""
    v = v & xp.uint64(0x5555555555555555)
    v = (v | (v >> xp.uint64(1))) & xp.uint64(0x3333333333333333)
    v = (v | (v >> xp.uint64(2))) & xp.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v >> xp.uint64(4))) & xp.uint64(0x00FF00FF00FF00FF)
    v = (v | (v >> xp.uint64(8))) & xp.uint64(0x0000FFFF0000FFFF)
    v = (v | (v >> xp.uint64(16))) & xp.uint64(0x00000000FFFFFFFF)
    return v.astype(xp.uint32)


def morton2D_64_encode(x, y):
    """Encode arrays of 2D coordinates into 64-bit Morton codes.

    Args:
        x: Array of x coordinates (uint32 range).
        y: Array of y coordinates (uint32 range).

    Returns:
        Array of uint64 Morton codes.

    Works with numpy and cupy arrays.
    """
    xp = _get_xp(x, y)
    return _spread_bits_64(x, xp) | (_spread_bits_64(y, xp) << xp.uint64(1))


def morton2D_64_decode(morton):
    """Decode arrays of 64-bit Morton codes into 2D coordinates.

    Args:
        morton: Array of uint64 Morton codes.

    Returns:
        Tuple of (x, y) arrays with uint32 dtype.

    Works with numpy and cupy arrays.
    """
    xp = _get_xp(morton)
    m = morton.astype(xp.uint64)
    x = _compact_bits_64(m, xp)
    y = _compact_bits_64(m >> xp.uint64(1), xp)
    return x, y
