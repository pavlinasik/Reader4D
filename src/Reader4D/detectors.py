"""
This module provides the :class:`Reader4D.detectors` loaders, high-level 
convenience wrappers for working with 4D-STEM datasets.

The primary supported input format is a CSR triplet written by the acquisition
pipeline:

- ``indptr.raw``   : CSR row pointer (length = n_frames + 1)
- ``indices.raw``  : detector addresses for nonzero events (length = nnz)
- ``values.raw``   : per-event values (counts or iToT; length = nnz)
- ``*.toml``       : sidecar metadata describing shapes, dtypes, acquisition
  parameters

After loading, the data are exposed in a downstream-friendly representation:

- ``packets``      : structured array with dtype
  :data:`~Reader4D.dtypes.TP3_ACQ_DATA_PACKET_DTYPE`
  and fields ``('address', 'count', 'itot')``
- ``descriptors``  : structured array with dtype
  :data:`~Reader4D.dtypes.TP3_BIN_DESCRIPTOR_DTYPE`
  and fields ``('offset', 'packet_count')``
- ``header``       : a flat metadata/statistics dict derived from the TOML and
  from basic CSR statistics

The loader supports optional visualization and micrograph reconstruction
(count and iToT), and can optionally export derived outputs to disk via
:mod:`Reader4D.convertor` and :mod:`Reader4D.visualizer`.

Quick start
-----------
Load a CSR dataset (with TOML sidecar) and preview diffractograms::

    import Reader4D.detectors as r4dDet

    detector = r4dDet.Timepix3(
        in_dir=r"C:\\path\\to\\csr_dataset",
        values_role="count",
        show=True,
        micrographs=True,
        progress=True,
        verbose=1,
    )

    packets = detector.packets
    descriptors = detector.descriptors
    header = detector.header

Reconstruct a single diffractogram for a scan position::

    img, total = detector.get_diffractogram(
        packets=detector.packets,
        descriptors=detector.descriptors,
        pattern_index=0,
        scan_dims=detector.SCAN_DIMS,
        detector_dims=detector.DET_DIMS,
    )

Notes
-----
- This module assumes that ``packets['address']`` is a linear index into the
  detector plane (0..W*H-1). If your acquisition encodes addresses differently,
  you must decode the address prior to accumulation.
- The CSR arrays are memory-mapped for efficient access and may be prefetched
  into the OS cache to reduce random I/O overhead for interactive exploration.
"""

import glob
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import tomlib packages
try:
    import tomllib
    def _read_toml(path):
        with open(path, "rb") as f:
            return tomllib.load(f)
except Exception:
    import toml
    def _read_toml(path):
        return toml.load(path)
            

# Import data types
from .dtypes import TP3_BIN_DESCRIPTOR_DTYPE as BIN_DESCRIPTOR_DTYPE
from .dtypes import TP3_ACQ_DATA_PACKET_DTYPE as ACQ_DATA_PACKET_DTYPE
from .dtypes import TP3_DTYPE_MAP as _DTYPE_MAP

# Import other Reader4D modules:
import Reader4D.convertor as r4dConv
import Reader4D.visualizer as r4dVisu


    
class Timepix3:
    """
    High-level loader for Timepix3 4D-STEM datasets stored as a CSR triplet.

    The loader expects a directory containing a CSR triplet (``indptr.raw``,
    ``indices.raw``, ``values.raw``) and a TOML sidecar (``*.toml``) describing
    dtypes and shapes. Data are exposed in a downstream-friendly form as
    ``packets`` and ``descriptors`` structured arrays, plus an optional
    normalized ``header`` dictionary.

    Attributes
    ----------
    in_dir : str or os.PathLike
        Input directory containing CSR files and a TOML sidecar.

    out_dir : str or os.PathLike
        Output directory used for derived artifacts (plots, exports). If not
        provided, defaults to ``in_dir``.

    indptr_file : str
        Filename of the CSR row-pointer array. Typically ``"indptr.raw"``.
        The row-pointer array has length ``n_frames + 1`` and must be
        non-decreasing.

    indices_file : str
        Filename of the CSR indices array. Typically ``"indices.raw"``.
        Holds detector addresses for each packet/event.

    values_file : str
        Filename of the CSR values array. Typically ``"values.raw"``.
        Holds per-packet values interpreted as either counts or iToT.

    values_role : {"count", "itot"}
        How ``values_file`` should be mapped into the packet structure.
        If ``"count"``, values populate ``packets["count"]`` and
        ``packets["itot"]`` is zeroed. If ``"itot"``, the inverse mapping
        is used.

    scan_dims : tuple[int, int] or None
        Scan grid dimensions stored as ``(width, height)``. If None, inferred
        from the TOML ``nav_shape`` (which is commonly stored as ``(H, W)``).

    det_dims : tuple[int, int] or None
        Detector dimensions stored as ``(width, height)`` (recommended
        convention). If None, inferred from the TOML ``sig_shape``. If your
        TOML stores detector shape as ``(Hdet, Wdet)``, this class normalizes
        it to ``(Wdet, Hdet)`` for consistency.

    packets : numpy.ndarray
        Structured array of dtype
        :data:`~Reader4D.dtypes.TP3_ACQ_DATA_PACKET_DTYPE` with fields
        ``('address', 'count', 'itot')`` and length ``nnz``.

    descriptors : numpy.ndarray
        Structured array of dtype
        :data:`~Reader4D.dtypes.TP3_BIN_DESCRIPTOR_DTYPE` with fields
        ``('offset', 'packet_count')`` and length ``n_frames``.

    header : dict or None
        Normalized metadata/statistics dictionary assembled from TOML and
        derived CSR statistics. Only available when ``return_header=True``.

    output_dir : str
        Resolved output directory used internally for saving figures and other
        derived artifacts. A subfolder may be chosen based on ``values_role``.

    Notes
    -----
    - CSR arrays are loaded using ``numpy.memmap`` to avoid up-front copies.
    - Packet addresses are assumed to be linear indices into the detector plane
      (0..W*H-1) unless your acquisition format specifies otherwise.
    """

    def __init__(
        self,
        in_dir,
        out_dir=None,
        indptr="indptr.raw",
        indices="indices.raw",
        data="values.raw",
        values_role="count",
        data_type="csr",
        scan_dims=None,
        det_dims=None,
        h5file=None,
        cmap="gray",
        show=True,
        progress=True,
        verbose=1,
        return_header=True,
        print_header=True,
        micrographs=False,
    ):
        """
        Initialize and load a Timepix3 CSR dataset.
    
        Notes
        -----
        This initializer intentionally exposes only the *dataset state* 
        as instance attributes (e.g., `packets`, `descriptors`, `scan_dims`). 
        Configuration options controlling the load process (e.g., `verbose`, 
        `progress`) are used during initialization but are not persisted 
        as public attributes to keep the object API minimal and consistent.
        """
    
        # ---- Public attributes (dataset state) — always defined
        self.in_dir = in_dir
        self.out_dir = in_dir if out_dir is None else out_dir
    
        # ---- Canonical dimension names
        self.scan_dims = scan_dims  # (W, H) or None
        self.det_dims = det_dims    # (W, H) or None
    
        self.values_role = str(values_role).lower()
        if self.values_role not in ("count", "itot"):
            raise ValueError("values_role must be 'count' or 'itot'")
    
        # ---- Results (always exist, may remain None until loaded)
        self.packets = None
        self.descriptors = None
        self.header = None
    
        # ---- Derived outputs / previews (always exist)
        self.output_dir = None
        self.example = None
        self.count = None
        self.itot = None
        self.image = None
        
        self.show = show
    
        # Validate supported input type (do not store as attribute)
        if str(data_type).lower() != "csr":
            raise ValueError("Only data_type='csr' is currently supported")
    
        # ---- Load CSR triplet 
        # filenames are constructor args; no need to store them as attributes 
        # unless you explicitly want them as public API
        if verbose:
            print("[INFO] Loading raw data...")
    
        packets, descriptors, header = self.CSRLoader(
            indptr=indptr,
            indices=indices,
            data=data,
            values_role=self.values_role,
            progress=bool(progress),
            return_header=bool(return_header),
        )
    
        self.packets = packets
        self.descriptors = descriptors
        self.header = header
    
        # Infer dimensions (normalize to (W, H)) if missing and header present
        # The TOML convention: nav_shape=(H, W), sig_shape=(Hdet, Wdet)
        if self.header is not None:
            if self.scan_dims is None and "nav_shape" in self.header:
                h, w = map(int, self.header["nav_shape"])
                self.scan_dims = (w, h)
    
            if self.det_dims is None and "sig_shape" in self.header:
                hdet, wdet = map(int, self.header["sig_shape"])
                self.det_dims = (wdet, hdet)
    
        # ---- Output directory conventions (derived artifact location)
        base_output = os.path.join(self.out_dir, "Reader4D_output")
        os.makedirs(base_output, exist_ok=True)
    
        role_folder = "count" if self.values_role == "count" else "itot"
        self.output_dir = os.path.join(base_output, role_folder)
        os.makedirs(self.output_dir, exist_ok=True)
    
        # ---- Optional: print header (do not store print_header flag)
        if bool(print_header) and self.header is not None:
            self.print_csr_header(self.header)
    
        # ---- Optional: export to HDF5 (do not store h5file flag)
        if h5file is not None:
            r4dConv.csr2hdf5(
                csr_path=None,
                packets=self.packets,
                descriptors=self.descriptors,
                header=self.header,
                output_path=self.out_dir,
                filename=h5file,
                overwrite=True,
                progress=bool(progress),
            )
    
        # ---- Optional: preview diffractograms (do not store show/cmap flags)
        
        if bool(show):
            if verbose:
                print("[INFO] Displaying sample diffractograms...")
            self.example, _ = r4dVisu.show_random_diffractograms(
                packets=self.packets,
                descriptors=self.descriptors,
                scan_dims=self.scan_dims if \
                    self.scan_dims is not None else (1024, 768),
                icut=None,
            )
    
        # ---- Optional: micrograph reconstruction
        # always assign `self.image` consistently if computed
        
        if bool(micrographs):
            if verbose:
                print(
                    "[INFO] Reconstructing Hit Count and iToT micrographs..."
                    )
    
            self.count, self.itot = self.get_count_itot(
                packets=self.packets,
                descriptors=self.descriptors,
                scan_dims=self.scan_dims if \
                    self.scan_dims is not None else (1024, 768),
                show=bool(show),
                cmap=cmap,
            )
    
            if self.count is not None:
                plt.imsave(
                    os.path.join(self.output_dir, "hit_count.png"),
                    self.count,
                    cmap=cmap,
                )
            if self.itot is not None:
                plt.imsave(
                    os.path.join(self.output_dir, "itot.png"),
                    self.itot,
                    cmap=cmap,
                )
    
            self.image = self.count if \
                self.values_role == "count" else self.itot
    
        if verbose:
            print("[DONE : Loading data]")


    def CSRLoader(
        self,
        indptr="indptr.raw",
        indices="indices.raw",
        data="values.raw",
        values_role="count",
        progress=True,
        prefetch_mb=256,
        return_header=False,
    ):
        """
        Load a CSR triplet from ``self.in_dir`` using TOML metadata.

        This method reads a TOML sidecar (``*.toml``) from ``self.in_dir`` to
        determine file names, dtypes, and shapes. The CSR triplet is memory-
        mapped (no up-front copy) and converted into:

        - ``descriptors``: structured array with dtype
          :data:`~Reader4D.dtypes.TP3_BIN_DESCRIPTOR_DTYPE` and fields
          ``('offset', 'packet_count')`` (one row per frame/scan pixel)
        - ``packets``: structured array with dtype
          :data:`~Reader4D.dtypes.TP3_ACQ_DATA_PACKET_DTYPE` and fields
          ``('address', 'count', 'itot')`` (one row per nonzero/packet)

        Parameters
        ----------
        indptr : str, optional
            Default filename for the CSR row-pointer file if not specified in
            TOML (default ``"indptr.raw"``).

        indices : str, optional
            Default filename for the CSR indices file if not specified in TOML
            (default ``"indices.raw"``).

        data : str, optional
            Default filename for the CSR values file if not specified in TOML
            (default ``"values.raw"``).

        values_role : {"count", "itot"}, optional
            How to map CSR values into the packet fields. If ``"count"``,
            values populate ``packets["count"]`` and ``packets["itot"]`` is set
            to zero. If ``"itot"``, values populate ``packets["itot"]`` and
            ``packets["count"]`` is set to zero.

        progress : bool, optional
            If True, prefetches memory maps using tqdm progress bars.

        prefetch_mb : int, optional
            Block size (MiB) used for sequential prefetching. Larger values
            reduce Python overhead but increase I/O burst size.

        return_header : bool, optional
            If True, returns a normalized header/stats dictionary as a third
            result.

        Returns
        -------
        packets : numpy.ndarray
            Structured array of shape ``(nnz,)`` with dtype
            :data:`~Reader4D.dtypes.TP3_ACQ_DATA_PACKET_DTYPE`.

        descriptors : numpy.ndarray
            Structured array of shape ``(n_frames,)`` with dtype
            :data:`~Reader4D.dtypes.TP3_BIN_DESCRIPTOR_DTYPE`.

        header : dict or None
            Normalized header/stats dictionary if ``return_header=True``,
            otherwise None.

        """
        folder = self.in_dir

        def _find_one(path, pattern):
            """ Return the first filesystem entry matching a glob pattern 
            in a directory. The results are sorted lexicographically to make
            the selection deterministic."""
            hits = sorted(glob.glob(os.path.join(path, pattern)))
            return hits[0] if hits else None

        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        toml_file = _find_one(folder, "*.toml")
        if not toml_file:
            raise FileNotFoundError(
                "No TOML file found in the input directory."
                )

        cfg = _read_toml(toml_file)
        raw = cfg.get("raw_csr", {})
        params = cfg.get("params", {})

        # Resolve file names (TOML overrides defaults)
        indptr_name = raw.get("indptr_file", indptr)
        indices_name = raw.get("indices_file", indices)
        data_name = raw.get("data_file", data)

        # ---- Resolve dtypes
        indptr_dtype = _DTYPE_MAP.get(
            raw.get("indptr_dtype", "int64"), 
            np.int64
            )
        indices_dtype = _DTYPE_MAP.get(
            raw.get("indices_dtype", "int32"), 
            np.int32
            )
        data_dtype = _DTYPE_MAP.get(
            raw.get("data_dtype", "int32"), 
            np.int32
            )

        nav_shape = params.get("nav_shape", None)
        sig_shape = params.get("sig_shape", None)

        indptr_path = os.path.join(folder, indptr_name)
        indices_path = os.path.join(folder, indices_name)
        data_path = os.path.join(folder, data_name)

        for p in (indptr_path, indices_path, data_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing CSR file: {p}")

        # ---- Memory map (no copy)
        indptr_mm = np.memmap(
            indptr_path, 
            mode="r", 
            dtype=indptr_dtype
            )
        indices_mm = np.memmap(
            indices_path, 
            mode="r", 
            dtype=indices_dtype
            )
        values_mm = np.memmap(
            data_path, 
            mode="r", 
            dtype=data_dtype
            )

        if progress:
            self._prefetch_memmap(
                indptr_mm, 
                "Prefetch indptr", 
                prefetch_mb, 
                position=0
                )
            self._prefetch_memmap(
                indices_mm, 
                "Prefetch indices", 
                prefetch_mb, 
                position=1
                )
            self._prefetch_memmap(
                values_mm, 
                "Prefetch values", 
                prefetch_mb, 
                position=2
                )

        # ---- CSR sanity
        if indptr_mm.shape[0] < 2:
            raise ValueError("indptr must have length >= 2 (n_frames + 1).")

        nnz = int(indptr_mm[-1])
        if nnz != indices_mm.size or nnz != values_mm.size:
            raise ValueError(
                f"CSR mismatch: indptr[-1]={nnz}, ",
                " len(indices)={indices_mm.size}, len(values)={values_mm.size}"
            )

        n_frames = int(indptr_mm.shape[0] - 1)

        # ---- Build descriptors
        descriptors = np.empty(n_frames, dtype=BIN_DESCRIPTOR_DTYPE)
        descriptors["offset"] = indptr_mm[:-1].astype(np.uint64, copy=False)

        row_counts = (indptr_mm[1:]-indptr_mm[:-1]).astype(np.int64, copy=False)
        
        if np.any(row_counts < 0):
            raise ValueError("indptr must be non-decreasing.")
            
        if np.any(row_counts > np.iinfo(np.uint32).max):
            raise ValueError(
                "A row has >2^32-1 packets; cannot fit into uint32.")

        descriptors["packet_count"] = row_counts.astype(np.uint32, copy=False)

        # ---- Build packets
        packets = np.empty(nnz, dtype=ACQ_DATA_PACKET_DTYPE)
        packets["address"] = indices_mm.astype(np.uint32, copy=False)

        values_role = str(values_role).lower()
        if values_role == "count":
            packets["count"] = values_mm.astype(np.uint32, copy=False)
            packets["itot"] = 0
        elif values_role == "itot":
            packets["itot"] = values_mm.astype(np.uint32, copy=False)
            packets["count"] = 0
        else:
            raise ValueError("values_role must be 'count' or 'itot'")

        # ---- Optional: validate against detector bounds (if provided)
        if sig_shape:
            hdet, wdet = int(sig_shape[0]), int(sig_shape[1])
            det_size = hdet * wdet
            if det_size > 0 and packets["address"].max(initial=0) >= det_size:
                raise ValueError(
                    f"Some packet addresses exceed detector size {sig_shape}: "
                    f"max={int(packets['address'].max())}, limit={det_size-1}"
                )

        # ---- Optional: validate nav shape
        if nav_shape:
            h, w = int(nav_shape[0]), int(nav_shape[1])
            if h * w != n_frames:
                raise ValueError(
                    f"nav_shape {nav_shape} implies {h*w} frames,",
                    f" but indptr gives {n_frames} frames."
                )

        header = self._normalize_header(cfg, indptr_mm, indices_mm, values_mm)
        return (packets, descriptors, header) if return_header else (packets, descriptors, None)

        
        
    def _prefetch_memmap(self,
            mm: np.memmap, desc: str, block_mb: int = 256, position: int = 0):
        """
        Sequentially prefetch a memory-mapped array into the OS page cache.
    
        The function walks the memmap in large contiguous blocks and performs a
        temporary copy of each slice to *force* disk I/O. This warms the cache 
        so that subsequent random access is much faster. No data is modified 
        and the temporary slices are immediately discarded.
    
        Parameters
        ----------
        mm : np.memmap
            Memory-mapped NumPy array to prefetch. Assumed 1-D; for multi-dim 
            arrays, the first axis is iterated. Works with structured dtypes.
        desc : str
            Label shown next to the tqdm progress bar.
        block_mb : int, optional
            Target block size in megabytes for each read chunk (converted to a 
            count of elements using ``mm.dtype.itemsize``). Larger values 
            reduce Python overhead but increase each I/O burst. Default is 256.
        position : int, optional
            Row index for the tqdm bar (useful when displaying multiple bars 
            at once). Default is 0.
    
        Returns
        -------
        None
            The function has side effects only (it populates the OS cache).
        """
        itemsize = mm.dtype.itemsize
        total = int(mm.shape[0])
        
        # Convert the desired block size from megabytes to "number of elements"
        # Each element may be multi-byte (mm.dtype.itemsize).
        block_elems = max(1, (block_mb * 1024 * 1024) // itemsize)    
        
        # How many blocks we'll need to cover the entire array.
        steps = (total + block_elems - 1) // block_elems
        
        # Iterate from 0 up to 'total' in steps of 'bsz'
        for start in tqdm(range(0, total, block_elems), total=steps, desc=desc,
                          unit="blk", position=position, 
                          leave=False, dynamic_ncols=True):
            end = min(start + block_elems, total)
            
            # Slice the memmap for this block and copy it to a temporary array.
            # The copy() call forces the OS to actually read the bytes from 
            # disk into the page cache (and briefly into Python memory for this 
            # slice), # "warming" the cache so later random access is fast.
            # We discard the temporary array immediately by not storing it.
            _ = mm[start:end].copy()  # forces IO


    def _normalize_header(self, raw_cfg, indptr, indices, values):
        """
        Build a friendly, self-contained header dict from the parsed CSR TOML
        plus a few derived statistics from the loaded arrays.
    
        Parameters
        ----------
        raw_cfg : dict-like
            Parsed TOML config (e.g. the result of tomli/tomllib.load). 
            Expected keys:
            - raw_cfg["params"]: may contain "filetype", "nav_shape", 
              "sig_shape", and optionally "threshold", "bias", "dwell_time_ns", 
              "values_role".
            - raw_cfg["raw_csr"]: file/dtype fields for indptr/indices/data.
            
        indptr : np.ndarray or np.memmap, shape (n_frames+1,)
            CSR row pointer; must be non-decreasing. The last element equals 
            nnz.
            
        indices : np.ndarray or np.memmap, shape (nnz,)
            CSR column indices (detector linear addresses).
            
        values : np.ndarray or np.memmap, shape (nnz,)
            CSR values (counts or itot).
    
        Returns
        -------
        stats : dict
            A flat dict suitable for pretty-printing/logging. Contains:
            - "filetype", "nav_shape", "sig_shape"
            - "n_frames", "nnz"
            - file names & dtypes: "indptr_file", "indices_file", "data_file",
              "indptr_dtype", "indices_dtype", "data_dtype"
            - size info: "bytes_indptr", "bytes_indices", "bytes_values"
            - row stats: "packets_per_frame_min/max/mean/median", 
                         "nonempty_frames", "occupancy" (fraction of frames
                                                         with ≥1 packet)
            - optional passthroughs if present: "threshold", "bias",
              "dwell_time_ns", "values_role"
        """
        params = dict(raw_cfg.get("params", {}))
        rcfg   = dict(raw_cfg.get("raw_csr", {}))
    
        # Basic fields from TOML (fallbacks if missing)
        nav_shape = tuple(params.get("nav_shape", (None, None)))
        sig_shape = tuple(params.get("sig_shape", (None, None)))
        filetype  = params.get("filetype", "raw_csr")
    
        # Derived sizes
        n_frames = int(indptr.shape[0] - 1)
        nnz      = int(indptr[-1])
    
        # Per-frame packet counts and simple stats
        row_counts = np.diff(indptr).astype(np.int64, copy=False)
        nonempty   = int((row_counts > 0).sum())
        occupancy  = float(nonempty / n_frames) if n_frames > 0 else 0.0
    
        # Assemble the header dict
        stats = {
            "filetype": filetype,
            "nav_shape": nav_shape,           # (H, W) scan
            "sig_shape": sig_shape,           # (Hdet, Wdet)
            "n_frames": n_frames,
            "nnz": nnz,
    
            # File fields from TOML (with sensible defaults)
            "indptr_file":  rcfg.get("indptr_file",  "indptr.raw"),
            "indices_file": rcfg.get("indices_file", "indices.raw"),
            "data_file":    rcfg.get("data_file",    "values.raw"),
    
            # Dtype strings (prefer TOML if present; otherwise from arrays)
            "indptr_dtype":  rcfg.get("indptr_dtype",  
                                      str(
                                          getattr(
                                              indptr,  "dtype", "unknown"))),
            "indices_dtype": rcfg.get("indices_dtype",
                                      str(
                                          getattr(
                                              indices, "dtype", "unknown"))),
            "data_dtype":    rcfg.get("data_dtype",    
                                      str(
                                          getattr(
                                              values,  "dtype", "unknown"))),
    
            # Byte sizes
            "bytes_indptr":  int(getattr(indptr,  "nbytes", 0)),
            "bytes_indices": int(getattr(indices, "nbytes", 0)),
            "bytes_values":  int(getattr(values,  "nbytes", 0)),
    
            # Row-count stats
            "packets_per_frame_min":   int(row_counts.min(initial=0)),
            "packets_per_frame_max":   int(row_counts.max(initial=0)),
            "packets_per_frame_mean":  float(row_counts.mean() \
                                             if row_counts.size else 0.0),
            "packets_per_frame_median":float(np.median(row_counts) \
                                             if row_counts.size else 0.0),
            "nonempty_frames": nonempty,
            "occupancy":       occupancy,
        }
    
        # Optional passthroughs (only if present in TOML)
        for opt in ("threshold", "bias", "dwell_time_ns", "values_role"):
            if opt in params:
                stats[opt] = params[opt]
    
        return stats
        
    
    def print_csr_header(self, h):
        """ Print a CSR header/stats dict built by `_normalize_header`. """
        
        # Safe getters with defaults
        g = lambda k, d=None: h.get(k, d)

        print("[INFO] CSR Header: ")
        print(" * Basic information: ")
        print(f"      filetype      : {g('filetype')}")
        print(f"      nav_shape     : {g('nav_shape')}   (H, W)")
        print(f"      sig_shape     : {g('sig_shape')}   (Hdet, Wdet)")
        print(f"      frames        : {g('n_frames')}")
        print(f"      nonzeros (nnz): {g('nnz')}")

        # Optional acquisition params
        if g("dwell_time_ns") is not None \
            or g("threshold") is not None \
                or g("bias") is not None:
                    
            print(" * Acquisition information: ")
            if g("dwell_time_ns") is not None:
                print(f"      dwell_time_ns : {g('dwell_time_ns')}")
            if g("threshold") is not None:
                print(f"      threshold     : {g('threshold')}")
            if g("bias") is not None:
                print(f"      bias          : {g('bias')}")
            if g("values_role") is not None:
                print(f"      values_role   : {g('values_role')}  ")

        print(" * Files information : (filename) (data type)")
        print(
            f"      indptr_file   : {g('indptr_file')}   dtype={g('indptr_dtype')}")
        print(
            f"      indices_file  : {g('indices_file')}  dtype={g('indices_dtype')}")
        print(
            f"      data_file     : {g('data_file')}   dtype={g('data_dtype')}")

        # Sizes and simple stats
        print(" * Size information:")
        print(f"      bytes_indptr  : {g('bytes_indptr')}")
        print(f"      bytes_indices : {g('bytes_indices')}")
        print(f"      bytes_values  : {g('bytes_values')}")

        print(" * Per-frame packets:")
        print(f"      min/median/mean/max : {g('packets_per_frame_min')}/"
              f"{g('packets_per_frame_median'):.1f}/"
              f"{g('packets_per_frame_mean'):.1f}/"
              f"{g('packets_per_frame_max')}")
        print(f"      nonempty_frames     : {g('nonempty_frames')} "
              f"({100.0 * g('occupancy', 0.0):.1f}% occupancy)")


    def get_diffractogram(self, 
                      packets, descriptors, pattern_index,
                      scan_dims=(1024, 768),
                      detector_dims=(256, 256),
                      dtype=np.uint32):
        """
        Recovers the 2D diffraction pattern for a single pixel of the scan 
        image.
    
        Parameters
        ----------
        packets : numpy.ndarray
            The raw packet data from the .advb file.
            
        descriptors : numpy.ndarray
            The descriptor data from the .advb.desc file.
       
        pattern_index : integer
            Index of the diffractogram to be reconstructed.
        
        scan_dims : tuple, optional
            The (width, height) of the overall scan grid.
        
        detector_dims : tuple, optional
            The (width, height) of the detector.
    
        Returns
        -------
        numpy.ndarray
            A 2D array (256x256) representing the diffraction pattern.
            
        """
        # sanity on dims
        if (not isinstance(scan_dims, (tuple, list))) or len(scan_dims) != 2:
            raise TypeError(
                f"scan_dims must be (width,height), got {scan_dims}")
            
        if (not isinstance(detector_dims, (tuple, list))) or \
            len(detector_dims) != 2:
            raise TypeError(
                f"detector_dims must be (width,height), got {detector_dims}")
            
        if not isinstance(pattern_index, int):
            raise TypeError("pattern_index must be an integer.")
    
        det_width, det_height = map(int, detector_dims)   # (Wdet, Hdet)
        n_pkts = int(packets.shape[0])
    
        # normalize descriptors to 1D arrays pc (packet_count) and off (offset) 
        desc = np.asarray(descriptors)
        if "packet_count" not in getattr(desc.dtype, "names", ()):
            raise TypeError("descriptors must have field 'packet_count'.")
    
        pc = np.asarray(desc["packet_count"], dtype=np.int64).reshape(-1)
        n_frames = pc.size
        if not (0 <= pattern_index < n_frames):
            raise IndexError("pattern_index out of bounds.")
    
        # prefer provided offset if it matches packets length;otherwise rebuild
        use_provided_off = ("offset" in desc.dtype.names)
        if use_provided_off:
            off = np.asarray(desc["offset"], dtype=np.int64).reshape(-1)
            # check consistency only on the last frame to avoid O(N) sums
            if off[-1] + pc[-1] != n_pkts:
                use_provided_off = False
    
        if not use_provided_off:
            off = np.empty_like(pc, dtype=np.int64)
            if pc.size:
                off[0] = 0
                if pc.size > 1:
                    np.cumsum(pc[:-1], out=off[1:])
    
        # slice this frame
        s = int(off[pattern_index])
        e = s + int(pc[pattern_index])
        if s < 0 or e < s or e > n_pkts:
            # out of range → treat as empty
            return np.zeros((det_height, det_width), dtype=dtype), 0
    
        frame_pkts = packets[s:e]
    
        # reconstruct diffractogram
        # NOTE: image shape must be (Hdet, Wdet) for (y,x) indexing
        img = np.zeros((det_height, det_width), dtype=dtype)
        if frame_pkts.size:
            addr = frame_pkts["address"].astype(np.int64, copy=False)
            vals = frame_pkts[
                getattr(
                    self, "values_role", "count")].astype(np.uint64, 
                                                          copy=False)
    
            det_size = det_width * det_height
            valid = (addr >= 0) & (addr < det_size)
            if not np.all(valid):
                addr = addr[valid]; vals = vals[valid]
    
            # accumulate into raveled image for speed, then done
            np.add.at(img.ravel(), addr, vals)
    
        count_sum = int(img.sum())
        return img, count_sum


    def get_count_itot(self,
                       packets, 
                       descriptors, 
                       scan_dims, 
                       show=True, 
                       cmap="gray"):
        """
        Constructs 2D images of total counts and iToT from raw data and 
        descriptors.
    
        Parameters
        ----------
        packets : numpy.ndarray
            The raw packet data from the .advb file.
        descriptors : numpy.ndarray
            The descriptor data from the .advb.desc file.
        scan_dims : tuple (width, height) 
            Scan size specifications (dimensions of the micrograph)
        show : boolean
            Display reconstructed images. Default is True.
        cmap : string
            Colormap of the displayed images, used when show=True.
            Default is gray.
    
        Returns
        --------
        count_image : numpy.ndarray 
            Image of total event counts per pixel.
        itot_image : numpy.ndarray 
            Image of total iToT per pixel.
        """
        # Initialize dimensions
        width, height = scan_dims
        
        # Initialize arrays for inserting values from packets
        count_image = np.zeros((height, width), dtype=np.uint32)
        itot_image = np.zeros((height, width), dtype=np.uint32)
    
        # Initialize counters
        current_offset = 0
        frame_idx = 0
        
        # Iterate through descriptors and packets
        for descriptor in descriptors:
            # Find the starting point (offset in descriptors)
            frame_start = current_offset
            
            # Find the teminating point (offset + packet counts)
            frame_end = frame_start + descriptor["packet_count"]
            
            # Extract all packets for the one pixel
            frame_packets = packets[frame_start:frame_end]
            
            # Find the next starting point
            current_offset += descriptor["packet_count"]
            
            # Get the coordinates of the pixel position within the array
            frame_pos_x = frame_idx % width
            frame_pos_y = frame_idx // width
    
            if frame_pos_y < height:
                # Hit count image
                count_image[frame_pos_y, frame_pos_x] = \
                    np.sum(frame_packets["count"])
                
                # iToT image
                itot_image[frame_pos_y, frame_pos_x] = \
                    np.sum(frame_packets["itot"])
            
            # Move to the next frame
            frame_idx += 1
        
        # Optionally display reconstructed images 
        if show:
            if np.sum(count_image) != 0:
                r4dVisu.show_micrograph(
                    count_image,
                    title="Hit Counts Micrograph",
                    cmap=cmap,
                    save=False,
                    filename=None,
                    output_dir=self.out_dir,
                    show=self.show
                    )
            else:
                print("[INFO] Hit Counts not available.")
                    
            if np.sum(itot_image) != 0:
                r4dVisu.show_micrograph(
                    itot_image,
                    title="iToT Micrograph",
                    cmap=cmap,
                    save=False,
                    filename=None,
                    output_dir=self.out_dir,
                    show=self.show
                    )
            else:
                print("[INFO] iToTs not available.")
                
                
        if np.sum(count_image) == 0:
            return None, itot_image
        
        elif np.sum(itot_image) == 0:
            return count_image, None
        
        else:
            return count_image, itot_image
    
    
class Timepix1:
    """place holder"""
    pass
