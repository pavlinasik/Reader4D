# Reader4D
Fast readers for 4D-STEM datasets (Timepix3, CSR/HDF5/ADVB) with Python.

Reader4D is a lightweight toolkit for loading and slicing large 4D-STEM datasets from common detector/export formats—Timepix3, CSR triplets (indptr/indices/values), HDF5, etc. It focuses on zero-copy I/O (memmap), row-major scan reconstruction, and handy utilities for ROI extraction, detector-space masking, and serpentine scan fixes.
