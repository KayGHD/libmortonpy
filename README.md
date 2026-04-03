# libmorton-python

Python bindings for [libmorton](https://github.com/Forceflow/libmorton) -- Morton code (Z-order curve) encoding/decoding for 2D coordinates.

Two APIs:
- **Scalar** -- C++ bindings via nanobind wrapping libmorton. Single-value encode/decode.
- **Vectorized** -- Pure Python magic-bits method. Works on numpy arrays, auto-detects cupy for GPU.

Both provide 32-bit and 64-bit variants.

## Install

```bash
pip install libmorton-python
```

GPU support (optional): `pip install libmorton-python[cuda]`

Build from source (requires CMake, C++17 compiler):
```bash
pip install . --no-build-isolation
```

## Usage

Scalar (single values, C++ backed):
```python
from libmorton import morton2D_32_encode, morton2D_32_decode

code = morton2D_32_encode(42, 17)
x, y = morton2D_32_decode(code)      # -> (42, 17)
```

64-bit variant for larger coordinate ranges:
```python
from libmorton import morton2D_64_encode, morton2D_64_decode

code = morton2D_64_encode(100000, 200000)
x, y = morton2D_64_decode(code)
```

Vectorized (numpy/cupy arrays):
```python
import numpy as np
from libmorton import morton2D_32_encode_array, morton2D_32_decode_array

x = np.array([0, 1, 2, 3], dtype=np.uint16)
y = np.array([0, 0, 0, 0], dtype=np.uint16)
codes = morton2D_32_encode_array(x, y)   # array([0, 1, 4, 5], dtype=uint32)
x_out, y_out = morton2D_32_decode_array(codes)
```

GPU arrays work transparently -- pass cupy arrays instead of numpy.

## Requirements

- Python >= 3.12
- numpy >= 1.24
- cupy (optional, for GPU arrays)

## Coordinate ranges

- 32-bit: x, y in [0, 65535] (16-bit each)
- 64-bit: x, y in [0, 4294967295] (32-bit each)

Out-of-range inputs raise ValueError.

## License

MIT
