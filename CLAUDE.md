# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python bindings for the [libmorton](https://github.com/Forceflow/libmorton) C++ Morton code (Z-order curve) library. Provides both scalar C++ bindings (via nanobind) and vectorized array operations (numpy/cupy magic-bits method) for 2D Morton encoding/decoding in 32-bit and 64-bit variants.

## Build

```powershell
pip install .                        # standard build
pip install . --no-build-isolation   # if nanobind/scikit-build-core already installed
pip install . --force-reinstall      # rebuild after changes
```

Build requires: `scikit-build-core>=0.9.2`, `nanobind>=2.0.0`. CMake fetches libmorton headers automatically via `FetchContent` (no vendored C++ code).

## Architecture

Two layers, both 2D-only (32-bit and 64-bit):

- **`_morton` (C++ extension)** - `ext/morton_ext.cpp` compiled via nanobind. Scalar functions wrapping `libmorton::morton2D_*`. Uses BMI2 intrinsics if available at compile time, LUT fallback otherwise. Built with `STABLE_ABI` + `NB_STATIC` (single `.pyd`, Python 3.12+).

- **`_vectorized.py`** - Pure Python array operations using magic-bits bit-interleaving. Auto-detects numpy vs cupy via `_get_xp()` module inspection. Same algorithm, works on CPU or GPU transparently.

- **`__init__.py`** - Wires both layers. Scalar C++ functions keep original names (`morton2D_64_encode`). Vectorized functions get `_array` suffix (`morton2D_64_encode_array`).

## Src Layout

Uses `src/` layout to prevent local package shadowing installed package. Python package lives at `src/libmorton/`, C++ extension source at `ext/`.
