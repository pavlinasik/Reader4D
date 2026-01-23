# -*- coding: utf-8 -*-
"""
Reader4D.dtypes
===============

Canonical NumPy dtypes used across Reader4D for Timepix3 I/O and processing.

The library uses a sparse CSR-like representation for 4D-STEM scans:

- ``descriptors`` contains one row per scan position (frame).
- ``packets`` contains one row per recorded detector hit (nonzero entry).

These dtypes define the expected field names and numeric types across the
Reader4D loaders, converters, and visualizers.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "TP3_BIN_DESCRIPTOR_DTYPE",
    "TP3_ACQ_DATA_PACKET_DTYPE",
    "TP3_DTYPE_MAP",
    # Backwards-compatible aliases:
    "BIN_DESCRIPTOR_DTYPE",
    "ACQ_DATA_PACKET_DTYPE",
    "DTYPE_MAP",
]

# -----------------------------------------------------------------------------
# Timepix3 sparse descriptors
# -----------------------------------------------------------------------------

TP3_BIN_DESCRIPTOR_DTYPE = np.dtype(
    [
        ("offset", np.uint64),
        ("packet_count", np.uint32),
    ]
)
"""
Structured dtype for the per-frame descriptor table.

Each element corresponds to one scan position (one diffractogram).
The fields define where the frame is located in the packet array:

- ``offset``: start index into the packet array (inclusive)
- ``packet_count``: number of packets/hits belonging to the frame
"""

TP3_ACQ_DATA_PACKET_DTYPE = np.dtype(
    [
        ("address", np.uint32),
        ("count", np.uint32),
        ("itot", np.uint32),
    ]
)
"""
Structured dtype for sparse detector packets (one row per recorded hit).

Fields
------
address : uint32
    Linear detector index in the range ``[0, det_width * det_height)``.
count : uint32
    Hit count value for this detector pixel.
itot : uint32
    iToT (integral Time-over-Threshold) value for this detector pixel.
"""

TP3_DTYPE_MAP = {
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "float32": np.float32,
    "float64": np.float64,
}
"""
Mapping of dtype name strings to NumPy dtypes.

This is primarily used to interpret types stored in TOML/JSON metadata.
"""

# -----------------------------------------------------------------------------
# Backwards-compatible aliases (recommended)
# -----------------------------------------------------------------------------

BIN_DESCRIPTOR_DTYPE = TP3_BIN_DESCRIPTOR_DTYPE
ACQ_DATA_PACKET_DTYPE = TP3_ACQ_DATA_PACKET_DTYPE
DTYPE_MAP = TP3_DTYPE_MAP
