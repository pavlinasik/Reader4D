# -*- coding: utf-8 -*-
"""Canonical NumPy dtypes used across timepix3 I/O and models."""

from __future__ import annotations
import numpy as np

# Public API
__all__ = [
    "BIN_DESCRIPTOR_DTYPE",
    "ACQ_DATA_PACKET_DTYPE",
    "DTYPE_MAP",
]

# One row per scan pixel / frame
TP3_BIN_DESCRIPTOR_DTYPE = np.dtype([
    ("offset",       np.uint64),
    ("packet_count", np.uint32),
])

# One entry per nonzero/packet
TP3_ACQ_DATA_PACKET_DTYPE = np.dtype([
    ("address", np.uint32),
    ("count",   np.uint32),
    ("itot",    np.uint32),
])

# String → NumPy dtype resolver (for TOML/JSON configs)
TP3_DTYPE_MAP = {
    "int32":   np.int32,
    "uint32":  np.uint32,
    "int64":   np.int64,
    "uint64":  np.uint64,
    "float32": np.float32,
    "float64": np.float64,
}
