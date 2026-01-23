"""
Reader4D is a lightweight Python toolkit for loading, converting, and 
visualizing 4D-STEM (scan × detector) data acquired with pixelated detectors.

The library focuses on two common data representations:

1) Dense frame stacks
   - Direct detector frames stored as many raw ``.dat`` files (1 frame/file)
   - Export to a single HDF5 container with dataset ``/data`` shaped as
     ``(n_frames, det_y, det_x)`` (optionally reshaped to scan dimensions)

2) Sparse (packet-based) representation
   - CSR-like storage of non-zero events/packets
   - HDF5 layout compatible with fast random access:
       ``/packets/address``, ``/packets/count``, ``/packets/itot``
       ``/descriptors/offset``, ``/descriptors/packet_count``
   - Efficient reconstruction of individual diffraction patterns on demand

Reader4D provides utilities for:
- converting raw acquisition outputs to structured HDF5 archives
- lazy loading (HDF5 slicing without full materialization)
- reconstructing diffractograms from sparse packets
- quick visualization helpers for dense and sparse datasets
- canonical NumPy dtypes for interoperable I/O

Typical workflow
----------------
Convert a folder of ``.dat`` frames to one HDF5 stack::

    from Reader4D import convertor as r4dConv

    out = r4dConv.dat2hdf5(
        dat_path=r"C:\\path\\to\\DATA",
        output_path=r".\\converted",
        filename="dataset.h5",
        det_dim=(256, 256),
        overwrite=True
    )

Visualize a frame from a HDF5 stack::
    from Reader4D import visualizer as r4dVisu
    r4dVisu.show_dense_frame(
        r".\\converted\\dataset.h5", 
        index=0, 
        percentile=(1, 99)
        )

Load sparse HDF5 lazily and reconstruct a single diffractogram::

    from Reader4D import convertor as conv
    from Reader4D import visualizer as visu

    f, gP, gD, header = r4dConv.load_sparse(
        "sparse_dataset.h5", 
        lazy=True
        )
    try:
        img = r4dVisu.get_diffractogram_lazy(
            gP, gD,
            pattern_index=0,
            detector_dims=header["sig_shape"],
            values_field="count"
        )
    finally:
        f.close()

Modules
-------
- ``Reader4D.detectors``   : detector definitions
- ``Reader4D.dtypes``      : canonical structured dtypes used across I/O
- ``Reader4D.convertor``   : conversion utilities (CSR ↔ HDF5, DAT → HDF5)
- ``Reader4D.visualizer``  : plotting and frame reconstruction utilities
- ``Reader4D.present``     : presentation/report helpers (optional)

Version
-------
This package follows semantic versioning starting from the development series.
"""

__version__ = "0.0.1"


import Reader4D.detectors
import Reader4D.dtypes
import Reader4D.convertor
import Reader4D.visualizer
import Reader4D.present