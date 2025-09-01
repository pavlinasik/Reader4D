# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 08:15:59 2025

@author: p-sik
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


    
class CSRTriplet:
    """
    Convenience wrapper for loading a 4D-STEM dataset stored as a CSR triplet
    (`indptr.raw`, `indices.raw`, `values.raw`), validating it, inferring scan
    and detector dimensions, and optionally previewing random diffractograms.

    Parameters
    ----------
    in_dir : str or pathlib.Path
        Directory containing the CSR triplet files (and optionally a TOML
        sidecar). Filenames are given by `indptr`, `indices`, and `data`.
    out_dir : str or pathlib.Path, optional
        Output directory for derived artifacts/figures. Defaults to `in_dir`.
    indptr : str, default "indptr.raw"
        Filename of the CSR row pointer array (int64), length = n_frames + 1.
    indices : str, default "indices.raw"
        Filename of the CSR column indices (detector addresses; int32), 
        length = nnz.
    data : str, default "values.raw"
        Filename of the CSR data values (counts or iToT; int32), length = nnz.
    scan_dims : tuple[int, int] or None, optional
        (width, height) of the scan grid. If None, inferred from header.
        Internally stored as `self.SCAN_DIMS`.
    det_dims : tuple[int, int] or None, optional
        (det_width, det_height) of the detector grid. If None, inferred from 
        header. Internally stored as `self.DET_DIMS`.
    h5file : str
        Filename of a .h5 export for MATLAB to the out_dir. If None, no export 
        is done. Default is None.
    cmap : str, default "gray"
        Colormap used for preview figures.
    show : bool, default True
        If True, display a few random diffractograms after loading.
    progress : bool, default True
        If True, show tqdm progress while loading/prefetching raw arrays.
    verbose : int, default 1
        Verbosity level; >0 prints basic status messages.
    return_header : bool, default True
        If True, `CSRLoader` returns a parsed header/stats dict as `self.header`.
    print_header : bool, default True
        If True and a header is available, pretty-prints it after loading.

    Attributes
    ----------
    packets : np.ndarray
        Structured array with fields ``address``, ``count``, ``itot`` (types 
        depend on loader config). One row per nonzero/packet.
    descriptors : np.ndarray
        Structured array with fields ``offset`` (uint64) and ``packet_count`` 
        (uint32), one row per scan pixel (frame).
    header : dict or None
        Parsed header/stats (if `return_header=True`), including shapes and 
        dtypes. When present, `nav_shape` (H, W) and `sig_shape` (Hdet, Wdet) 
        are used to fill `self.SCAN_DIMS` and `self.DET_DIMS` (note the class
        stores scan as (W, H)).
    SCAN_DIMS : tuple[int, int]
        (width, height) of the scan grid, provided or inferred from header.
    DET_DIMS : tuple[int, int]
        (det_width, det_height) of the detector grid, provided or inferred.
    out_dir : pathlib.Path
        Destination directory used to store outputs (created if missing).
    example : dict or any
        Handle/object returned by `show_random_diffractograms` when `show=True`.

    """    
    def __init__(self,
                in_dir, 
                out_dir       = None,
                indptr        = "indptr.raw",
                indices       = "indices.raw",
                data          = "values.raw",
                scan_dims     = None,
                det_dims      = None,
                h5file        = None,
                cmap          = "gray",
                show          = True,
                progress      = True,
                verbose       = 1,
                return_header = True,
                print_header  = True
                ):
        
        ######################################################################
        # PRIVATE FUNCTION: Initialize CSRTriplet object.
        # The parameters are described above in class definition.
        ######################################################################
        
        ## Initialize input attributes ---------------------------------------
        self.in_dir = in_dir
        
        if out_dir is None:
            self.out_dir = self.in_dir
        else:
            self.out_dir = out_dir
            
        self.triplet1 = indptr
        self.triplet2 = indices
        self.triplet3 = data
        self.SCAN_DIMS = scan_dims
        self.DET_DIMS = det_dims
        self.h5file = h5file
        self.cmap = cmap
        self.show = show
        self.progress = progress
        self.verbose = verbose
        self.return_header = return_header
        self.print_header = print_header
        
    
        # Load CSR triplet ----------------------------------------------------
        if verbose:
            print("[INFO] Loading raw data...")
        self.packets, self.descriptors, self.header = self.CSRLoader(
                                            indptr=self.triplet1,
                                            indices=self.triplet2,
                                            data=self.triplet3,
                                            progress=self.progress,
                                            return_header=self.return_header
                                            )
        
        # Print header information optionally ---------------------------------
        if print_header and self.header is not None:
            self.print_csr_header(self.header)
        
        # Set dimensions automatically based on the header information --------
        if self.header is not None:
            # (a) scanning dimensions
            self.SCAN_DIMS = self.header['nav_shape'][::-1]

            # (b) detector dimensions
            self.DET_DIMS = self.header['sig_shape']

        # Define output folder (change if needed) -----------------------------
        self.output_dir = os.path.join(
            os.path.dirname(self.out_dir), "output_TPX3")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Convert to a .h5 file for MATLAB optionally -------------------------
        if self.h5file is not None:
            r4dConv.diff2hdf5(
                self.packets, 
                self.descriptors, 
                filename=self.output_dir+"\\"+self.h5file,
                verbose=self.verbose)
            
        # Show ranfom diffractograns optionally -------------------------------
        if show:
            if verbose:
                print("[INFO] Displaying sample diffractograms...")
                
            self.example, _ = self.show_random_diffractograms(
                packets=self.packets,
                descriptors=self.descriptors, 
                scan_dims=self.SCAN_DIMS,
                )
        
        if verbose:
            print("[DONE]")
            
            
    def CSRLoader(self,
                    indptr = "indptr.raw",
                    indices = "indices.raw",
                    data = "values.raw",
                    values_role="count",        
                    progress=True,
                    prefetch_mb=256,
                    return_header=False):
        """
        Load a CSR triplet described by a sidecar TOML file and convert it to 
        your downstream-friendly representation:
          - descriptors: structured array with fields
            ('offset': uint64, 'packet_count': uint32)
            1 row per scan pixel/frame  
          - packets: structured array with fields
            ('address': uint32, 'count': uint32, 'itot': uint32)
            1 entry per nonzero

        The CSR arrays are memory-mapped (no upfront copy). Optionally, 
        the function sequentially prefetches each memmap in large blocks.

        Parameters
        ----------
        folder : str
            Directory containing the CSR files and a matching *.toml produced 
            by the acquisition script. The TOML is expected to include:

            - [params]: may define nav_shape = [H, W], sig_shape = [Hdet,Wdet],
              optional filetype, and optional acquisition metadata 
              (e.g., dwell_time_ns,threshold, bias).
            - [raw_csr]: file names and dtypes for the CSR triplet, e.g.
              indptr_file, indices_file, data_file,
              indptr_dtype, indices_dtype, data_dtype.

        values_role : {"count", "itot"}, optional
            How to map values.raw into the output packet fields. 
            If "count", values are written to packets['count'] and 
            packets['itot'] is zeroed. 
            If "itot", values go to packets['itot'] and packets['count'] is
            zeroed. 
            Default is "count".
        progress : bool, optional
            If True, prefetch each memmap with a tqdm progress bar. 
            Default True.
        prefetch_mb : int, optional
            Block size (MiB) used for prefetching sequential reads. Larger 
            values reduce Python overhead but increase each I/O burst. 
            Default 256.
        return_header : bool, optional
            If True, also return a friendly header/stats dict assembled from
            the TOML and derived CSR statistics (via _normalize_header). 
            Default False.

        Returns
        -------
        packets : numpy.ndarray
            Structured array of shape (nnz,) with dtype compatible with your
            acq_data_packet_dtype; fields: 'address', 'count', 'itot'.
        descriptors : numpy.ndarray
            Structured array of shape (n_frames,) with dtype compatible with 
            your bin_descriptor_dtype; fields: 'offset', 'packet_count'.
        header : dict, optional
            Only if return_header=True; a flat dict suitable for printing,
            including shapes, byte sizes, and simple stats.
        """
        
        # Initialize variables
        folder = self.in_dir
        
        # Helper functions
        def _find_one(path, pattern):
            """ 
            Return the first filesystem entry matching a glob pattern inside 
            a directory. Results are sorted lexicographically to make the 
            choice deterministic.
            """
            hits = sorted(glob.glob(os.path.join(path, pattern)))
            
            return hits[0] if hits else None
        
        # Raise Errors
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        # Find TOML
        toml_file = _find_one(folder, "*.toml")
        if not toml_file:
            raise FileNotFoundError("No TOML file found in the folder.")
        
        # Read TOML file
        cfg = _read_toml(toml_file)
        raw = cfg.get("raw_csr", {})
        params = cfg.get("params", {})

        # Filenames (fall back to defaults if missing)
        indptr_name  = raw.get("indptr_file", indptr)
        indices_name = raw.get("indices_file", indices)
        data_name    = raw.get("data_file",   data)

        # Dtypes from TOML (fall back to your writer defaults)
        indptr_dtype  = _DTYPE_MAP.get(raw.get("indptr_dtype", "int64"),  
                                       np.int64)
        indices_dtype = _DTYPE_MAP.get(raw.get("indices_dtype", "int32"),
                                       np.int32)
        data_dtype    = _DTYPE_MAP.get(raw.get("data_dtype",   "int32"),
                                       np.int32)

        # Shapes (nav_shape = [H, W], sig_shape = [Hdet, Wdet]) 
        nav_shape = params.get("nav_shape", None)
        sig_shape = params.get("sig_shape", None)

        # Resolve full paths
        indptr_path  = os.path.join(folder, indptr_name)
        indices_path = os.path.join(folder, indices_name)
        data_path    = os.path.join(folder, data_name)

        for p in (indptr_path, indices_path, data_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing CSR file: {p}")

        # Map files (no copy)
        indptr  = np.memmap(indptr_path,  mode="r", dtype=indptr_dtype)
        indices = np.memmap(indices_path, mode="r", dtype=indices_dtype)
        values  = np.memmap(data_path,    mode="r", dtype=data_dtype)

        # Prefetch (optional)
        if progress:
            self._prefetch_memmap(indptr,  "Prefetch indptr",  
                                  prefetch_mb, position=0)
            self._prefetch_memmap(indices, "Prefetch indices", 
                                  prefetch_mb, position=1)
            self._prefetch_memmap(values,  "Prefetch values",  
                                  prefetch_mb, position=2)

        # CSR sanity
        if indptr.shape[0] < 2:
            raise ValueError("indptr must have length >= 2 (n_frames+1).")
        nnz = int(indptr[-1])
        if nnz != indices.size or nnz != values.size:
            raise ValueError(
                f"CSR mismatch: indptr[-1]={nnz}, len(indices)={indices.size}, len(values)={values.size}")

        n_frames = indptr.shape[0] - 1
        
        # Build descriptors (offset, packet_count)
        desc = np.empty(n_frames, dtype=BIN_DESCRIPTOR_DTYPE)
        # offset is the row pointer (start index into indices/values)
        desc["offset"] = indptr[:-1].astype(np.uint64, copy=False)
        # packet_count is the difference of row pointers
        row_counts = (indptr[1:] - indptr[:-1])
        if np.any(row_counts < 0):
            raise ValueError("indptr must be non-decreasing.")
        if np.any(row_counts > np.iinfo(np.uint32).max):
            raise ValueError("A row has >2^32-1 packets; no fit into uint32.")
        desc["packet_count"] = row_counts.astype(np.uint32, copy=False)

        # Build packets (address, count, itot)
        pkt = np.empty(nnz, dtype=ACQ_DATA_PACKET_DTYPE)
        pkt["address"] = indices.astype(np.uint32, copy=False)
        if values_role == "count":
            pkt["count"] = values.astype(np.uint32, copy=False)
            pkt["itot"]  = 0
        elif values_role == "itot":
            pkt["itot"]  = values.astype(np.uint32, copy=False)
            pkt["count"] = 0
        else:
            raise ValueError("values_role must be 'count' or 'itot'")

        # Optional detector bounds check if sig_shape provided
        if sig_shape:
            Hdet, Wdet = int(sig_shape[0]), int(sig_shape[1])
            det_max = Hdet * Wdet
            if det_max > 0 and pkt["address"].max(initial=0) >= det_max:
                raise ValueError(
                    f"Some packet addresses exceed detector size {sig_shape}: "
                    f"max address={int(pkt['address'].max())}, limit={det_max-1}"
                )

        # Optional nav size check (frames = H*W)
        if nav_shape:
            H, W = int(nav_shape[0]), int(nav_shape[1])
            if H * W != n_frames:
                raise ValueError(
                    f"nav_shape {nav_shape} implies {H*W} frames, but indptr gives {n_frames} frames.")

        # Compose a friendly header
        header = self._normalize_header(cfg, indptr, indices, values)

        return (pkt, desc, header) if return_header else (pkt, desc, None)
        
        
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
        
        # Convert the desired block size from megabytes to "number of elements".
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
        """
        Pretty-print a CSR header/stats dict built by `_normalize_header`.

        Parameters
        ----------
        h : dict
            The header/stats dictionary.

        Returns
        -------
        None
        """
        
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
        print(f"      indptr_file   : {g('indptr_file')}   dtype={g('indptr_dtype')}")
        print(f"      indices_file  : {g('indices_file')}  dtype={g('indices_dtype')}")
        print(f"      data_file     : {g('data_file')}   dtype={g('data_dtype')}")

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


    def show_random_diffractograms(self, 
               packets, descriptors, scan_dims=(1024, 768), num_patterns=10, 
               rows=2, cols=5, csquare=256, icut=5):
        """
        Reconstructs and displays a random sample of diffraction patterns directly
        from the raw packet data.
        
        Parameters
        ----------
        packets : numpy.ndarray
            The raw packet data from the .advb file.
        descriptors : numpy.ndarray
            The descriptor data from the .advb.desc file.
        scan_dims : tuple, optional
            The dimensions (width, height) of the original scan grid.
        num_patterns : int, optional
            The total number of patterns to display.
        rows, cols : int, optional
            The dimensions of the subplot grid.
        csquare : int, optional
            Side length of a central square to crop from each pattern for display.
            
        Returns
        -------
        img : numpy.ndarray
            Last displayed image.
        """
        scan_width, scan_height = scan_dims
        
        if rows * cols < num_patterns:
            print("Warning: Subplot grid is too small.")
            num_patterns = rows * cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()
        sums = []
        
        for i in range(num_patterns):
            try:
                # Pick a random valid pattern index
                pattern_index = np.random.randint(0, len(descriptors))
        
                # Reconstruct the pattern on-the-fly
                img, s = self.get_diffractogram(
                        packets, 
                        descriptors, 
                        pattern_index, 
                        scan_dims)
                
                img = np.where(img > icut, icut, img)
        
                sums.append(s)
                
                # Crop the central square of the image
                img_height, img_width = img.shape
                start_x = img_width // 2 - csquare // 2
                start_y = img_height // 2 - csquare // 2
                cimg = img[start_y:start_y + csquare, start_x:start_x + csquare]
        
                # Plot the cropped image on the i-th subplot
                ax = axes[i]
                ax.imshow(cimg, cmap='viridis', origin='lower')
                ax.set_title(f"Pattern ID:({pattern_index})")
                ax.axis("off")
        
            except Exception as e:
                # General error handling
                print(f"An error occurred for pattern ({pattern_index}): {e}")
                axes[i].axis("off")
                axes[i].set_title(f"({pattern_index}) - Error")
                
        # Hide any unused subplots
        for j in range(num_patterns, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return img, sums


    def get_diffractogram(self, 
              packets, descriptors, pattern_index, scan_dims=(1024, 768), 
              detector_dims=(256, 256), dtype=np.uint32):
        """
        Recovers the 2D diffraction pattern for a single pixel of the scan image.
    
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
            A 2D array (256x256) representing the diffraction pattern (hit counts).
            
            
        NOTE:  How It Works
        - Scan Position: When you select a pixel [X,Y], in the (1024x768) scan grid, 
             you are selecting the data acquired when the detector was physically 
             at that position.
        - Descriptor File: The .advb.desc file acts as an index. 
             For scan pixel [X,Y], there is a corresponding entry in the descriptor 
             file that tells you exactly where that detector data begins (offset) 
             and how many packets it contains (packet_count) in the main .advb file.
        - Data Packets: You then read that specific slice of packets. Each packet 
             (address, count, itot) tells you that a detector pixel (identified by 
             address) recorded a certain number of hits (count).
        - Reconstruction: By creating a blank 256x256 array and populating it using 
             the address and count from these packets, you reconstruct 
             the 2D diffraction pattern recorded at that single scan position.
    
        This type of dataset is often called 4D STEM, where for each 2D scan 
             position [X,Y], you record a 2D diffraction pattern (kx, ky).
        """
        
        #  Input Validation (Robustness Check) ------------------------------------
        if not isinstance(scan_dims, (tuple, list)) or len(scan_dims) != 2:
            raise TypeError(
                f"scan_dims must be a tuple of (width, height), but got {scan_dims}")
        if not isinstance(detector_dims, (tuple, list)) or len(detector_dims) != 2:
            raise TypeError(
                f"detector_dims must be a tuple of (width, height), but got {detector_dims}")
        if not isinstance(pattern_index, int):
            raise TypeError("pattern_index must be an integer.")
    
        scan_width, scan_height = scan_dims
        det_width, det_height = detector_dims
        
        # Calculate the linear index for the desired scan pixel -------------------
        if not (0 <= pattern_index < len(descriptors)):
            raise IndexError("Pattern index is out of bounds.")
        
        # Get the specific descriptor for this frame ------------------------------
        descriptor = descriptors[pattern_index]
        frame_offset = descriptor["offset"]
        frame_packet_count = descriptor["packet_count"]
    
        # Extract relevant packets ------------------------------------------------
        # frame_packets contains all the raw measurement data for that single 
        # diffraction pattern at position (scan_x, scan_y).
        frame_packets = packets[frame_offset:frame_offset + frame_packet_count]
    
        # Create a blank canvas for the diffraction pattern -----------------------
        diffraction_image = np.zeros(detector_dims, dtype=dtype)
    
        # Reconstruct the diffraction pattern -------------------------------------
        # populate the canvas with diffractions using the address and count
        # the 'address' is a flattened 1D index for the 256x256 detector grid
        detector_addresses = frame_packets["address"]
        detector_counts = frame_packets["count"]
    
        # Convert the 1D address to 2D (y, x) coordinates on the detector ---------
        det_y = detector_addresses // det_width
        det_x = detector_addresses % det_width
    
        # Place the counts at the correct locations--------------------------------
        np.add.at(diffraction_image, (det_y, det_x), detector_counts)
        
        # Sum diffractogram -------------------------------------------------------
        count_sum = np.sum(np.sum(diffraction_image))
        
        # Return diffractogram ----------------------------------------------------
        return diffraction_image, count_sum


class ADVBdataset:
    # place holder 
    pass    
