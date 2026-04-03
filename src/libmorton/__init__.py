"""
libmorton - Morton 2D encoding/decoding for Python.

Scalar functions (single values, C++ via libmorton):
    morton2D_32_encode(x, y)   -> int
    morton2D_64_encode(x, y)   -> int
    morton2D_32_decode(morton)  -> (x, y)
    morton2D_64_decode(morton)  -> (x, y)

Vectorized functions (numpy/cupy arrays, magic-bits method):
    morton2D_32_encode_array(x, y)   -> array
    morton2D_64_encode_array(x, y)   -> array
    morton2D_32_decode_array(morton)  -> (x_array, y_array)
    morton2D_64_decode_array(morton)  -> (x_array, y_array)
"""

try:
    from libmorton._morton import (
        morton2D_32_encode,
        morton2D_64_encode,
        morton2D_32_decode,
        morton2D_64_decode,
    )
except ImportError:
    raise ImportError(
        "libmorton C++ extension not found. "
        "Reinstall with: pip install libmorton-python"
    )

from libmorton._vectorized import (
    morton2D_32_encode as morton2D_32_encode_array,
    morton2D_64_encode as morton2D_64_encode_array,
    morton2D_32_decode as morton2D_32_decode_array,
    morton2D_64_decode as morton2D_64_decode_array,
)

__all__ = [
    "morton2D_32_encode",
    "morton2D_64_encode",
    "morton2D_32_decode",
    "morton2D_64_decode",
    "morton2D_32_encode_array",
    "morton2D_64_encode_array",
    "morton2D_32_decode_array",
    "morton2D_64_decode_array",
]

__version__ = "0.1.0"
