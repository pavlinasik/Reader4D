# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 14:33:50 2025

@author: p-sik
"""
import os
import numpy as np
import h5py
from tqdm import tqdm
import Reader4D.detectors as det
import json
import glob 
import matplotlib.pyplot as plt

# Import data types
from .dtypes import TP3_BIN_DESCRIPTOR_DTYPE as BIN_DESCRIPTOR_DTYPE
from .dtypes import TP3_ACQ_DATA_PACKET_DTYPE as ACQ_DATA_PACKET_DTYPE
from .dtypes import TP3_DTYPE_MAP as _DTYPE_MAP


def dat2hdf5(
    dat_path,
    header=None,
    det_dim=(256, 256),
    dtype=np.uint16,
    output_path=r"./converted",
    filename="data.h5",
    overwrite=False,
    progress=True,
):
    """Convert a directory of TimePix .dat frames (one frame per file)
    into a single HDF5 stack.
    """

    if not os.path.isdir(dat_path):
        raise NotADirectoryError(f"Not a directory: {dat_path}")

    # Collect all .dat files
    files = sorted(glob.glob(os.path.join(dat_path, "*.dat")))
    if not files:
        raise FileNotFoundError(f"No .dat files found in {dat_path}")

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Ensure filename ends with .h5
    if not filename.lower().endswith((".h5", ".hdf5")):
        filename = filename + ".h5"

    out_file = os.path.join(output_path, filename)

    # Handle overwrite (on the output FILE)
    if os.path.exists(out_file) and not overwrite:
        raise FileExistsError(
            f"{out_file} exists. Set overwrite=True to replace it."
            )
        
    if os.path.exists(out_file) and overwrite:
        os.remove(out_file)

    det_y, det_x = det_dim
    expected = det_y * det_x
    n_frames = len(files)

    # HDF5 dataset options
    dset_kwargs = dict(chunks=(1, det_y, det_x))

    # Create HDF5 and stream frames
    with h5py.File(out_file, "w") as f:
        dset = f.create_dataset(
            "data",
            shape=(n_frames, det_y, det_x),
            dtype=dtype,
            **dset_kwargs,
        )

        # File-level metadata
        f.attrs["source_dir"] = os.path.abspath(dat_path)
        f.attrs["n_frames"] = np.int64(n_frames)
        f.attrs["detector_dims"] = np.asarray(det_dim, dtype=np.int64)

        # Optional header metadata
        if header is not None:
            g = f.require_group("metadata")
            g.create_dataset("header_json", data=json.dumps(header))

        # Stream frames into dataset (tqdm progress bar)
        iterable = tqdm(
            files, 
            disable=not progress, 
            desc="[INFO] Writing frames", 
            unit="frame"
            )
        
        for i, fp in enumerate(iterable):
            arr = np.fromfile(fp, dtype=dtype)

            if arr.size != expected:
                raise ValueError(
                    f"{fp} has {arr.size} elements, expected {expected}"
                    )

            dset[i] = arr.reshape(det_y, det_x)

    if progress:
        print(f"\n[INFO] Done: {out_file}")

    return out_file


def csr2hdf5(
        csr_path=None,
        packets=None, 
        descriptors=None, 
        header=None, 
        output_path=r"./converted",
        filename="data.h5",
        overwrite=False,
        progress=True,                
    ):
    
    # output path handling
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, filename)

    if os.path.exists(out_file) and not overwrite:
        raise FileExistsError(f"[INFO] {out_file} already exists. ",
                              "Set overwrite=True to replace it.")

    # resolve input mode
    if packets is None or descriptors is None:
        if csr_path is None:
            raise ValueError(
                "Provide either (packets, descriptors) or csr_path."
                )

        data = det.Timepix3(
            in_dir=csr_path,
            show=False,
            print_header=False,
            progress=progress,
        )
        packets = data.pkt
        descriptors = data.desc
        
        # prefer explicit header if provided
        header = data.header if header is None else header  

    # extract fields
    address = packets["address"]
    count = packets["count"]
    itot = packets["itot"]

    offset = descriptors["offset"]
    packet_count = descriptors["packet_count"]

    # validations
    nnz = int(address.shape[0])
    pc_sum = int(np.asarray(packet_count, dtype=np.uint64).sum())
    if pc_sum != nnz:
        raise ValueError(
            f"Inconsistent data: sum(packet_count)={pc_sum} ",
            f"but len(address)={nnz}")

    off = np.asarray(offset, dtype=np.uint64)
    if off.size and np.any(off[1:] < off[:-1]):
        raise ValueError("descriptors['offset'] must be non-decreasing")

    if off.size:
        last_end = int(off[-1] + np.asarray(packet_count, dtype=np.uint64)[-1])
        if last_end != nnz:
            raise ValueError(
                 "Inconsistent offsets: last offset + last packet_count = ",
                f"{last_end}, expected {nnz}")

    # Write to hdf5
    with h5py.File(out_file, "w") as f:
        # header (safe)
        if header is not None:
            f.create_dataset("header_json", data=json.dumps(header))

        gP = f.create_group("packets")
        gP.create_dataset("address", data=address, chunks=True)
        gP.create_dataset("count", data=count, chunks=True)
        gP.create_dataset("itot", data=itot, chunks=True)

        gD = f.create_group("descriptors")
        gD.create_dataset("offset", data=offset, chunks=True)
        gD.create_dataset("packet_count", data=packet_count, chunks=True)

    if progress:
        print(f"[INFO] Data converted to HDF5: {out_file}")

    return out_file
    
                
def load_sparse(path, lazy=False, progress=True):
    """
    Load a sparse Timepix3 HDF5 file produced by ``sparse_hdf5()``.

    The HDF5 file is expected to have the following layout:

    - ``/packets/address``       (uint32)
    - ``/packets/count``         (uint32)
    - ``/packets/itot``          (uint32)
    - ``/descriptors/offset``    (uint64)
    - ``/descriptors/packet_count`` (uint32)
    - optional ``/header_json``  (UTF-8 JSON string)

    When ``lazy=False`` (default), the function materializes the split fields
    into structured NumPy arrays using the canonical dtypes.

    When ``lazy=True``, the function returns HDF5 handles instead of reading
    the arrays into memory. This is useful when datasets are very large and
    you want to slice them on-demand.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the HDF5 file.
        
    lazy : bool, optional
        If False, load datasets into memory and return structured arrays.
        If True, return HDF5 file and group handles for lazy access.
        Default is False.
        
    progress : bool, optional
        If True, info messages will be printed.

    Returns (lazy=False)
    --------------------
    packets : numpy.ndarray
        Only returned if ``lazy=False``. Structured array of shape (nnz,)
        with dtype ``ACQ_DATA_PACKET_DTYPE`` containing fields:
        ``('address', 'count', 'itot')``.
        
    descriptors : numpy.ndarray
        Only returned if ``lazy=False``. Structured array of shape (n_frames,)
        with dtype ``BIN_DESCRIPTOR_DTYPE`` containing fields:
        ``('offset', 'packet_count')``.
        
    header : dict or None
        Parsed JSON header if ``/header_json`` exists, otherwise None.

    Returns (lazy=True)
    --------------------
    f : h5py.File
        Open HDF5 file handle (caller MUST close it).
        
    gP : h5py.Group
        Group handle for ``/packets`` (lazy datasets accessible via
        ``gP['address']``, etc.).
        
    gD : h5py.Group
        Group handle for ``/descriptors`` (lazy datasets accessible via
        ``gD['offset']``, etc.).
        
    header : dict or None
        Parsed JSON header if present, otherwise None.

    """
    # Open file in read-only mode.
    # if lazy=True : intentionally keeping this file open to return it.
    f = h5py.File(path, "r")

    # Optional header
    # header is stored as JSON for maximum compatibility with HDF5 tooling.
    header = None
    if "header_json" in f:
        raw = f["header_json"][()]
        
        # h5py may return bytes or str depending on how it was written.
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        header = json.loads(raw)
    
    # dataset group handles (always available)
    gP = f["packets"]
    gD = f["descriptors"]
    
    # If lazy access is requested, return handles immediately.
    # caller must close the file handle after use.
    if lazy:
        if progress:
            print("[INFO] Sparse data loaded as handles (lazy access).")
            print("[INFO] The file handle must be closed after use.")
        return f, gP, gD, header

    # Materialize into memory
    # read split packet fields into NumPy arrays.
    address = gP["address"][...]
    count   = gP["count"][...]
    itot    = gP["itot"][...]
    
    # read split descriptor fields into NumPy arrays.
    offset       = gD["offset"][...]
    packet_count = gD["packet_count"][...]
    
    # Reconstruct canonical structured arrays.
    # assumes ACQ_DATA_PACKET_DTYPE and BIN_DESCRIPTOR_DTYPE are available
    packets = np.empty(address.shape[0], dtype=ACQ_DATA_PACKET_DTYPE)
    packets["address"] = address
    packets["count"]   = count
    packets["itot"]    = itot

    descriptors = np.empty(offset.shape[0], dtype=BIN_DESCRIPTOR_DTYPE)
    descriptors["offset"]       = offset
    descriptors["packet_count"] = packet_count
    
    # Close the file now that all data is materialized.
    f.close()
    
    if progress:
        print("[INFO] Sparse data loaded as arrays into memory.")

    return packets, descriptors, header
    
    
def print_h5_tree(path):
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"[DSET] {name}  shape={obj.shape}  dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"[GRP ] {name}/")
        f.visititems(visitor)


def save_all_diffractograms(packets, descriptors, output_path,
                            scan_dims=(1024, 768)):
    """
    Reconstructs and saves all diffraction patterns to individual `.dat` files.

    This function iterates through all entries in the descriptor array, 
    reconstructs each diffraction pattern from raw packet data, and saves 
    the resulting 2D arrays as flattened binary `.dat` files. Files are saved 
    using a structured filename format based on the pattern index.

    Parameters
    ----------
    packets : np.ndarray
        Structured array containing raw packet data with fields such as 
        'address' and 'count'.

    descriptors : np.ndarray
        Structured array of descriptors, where each entry contains metadata 
        (e.g., offset and packet_count) for reconstructing one diffraction 
        pattern.

    output_path : str
        Directory path where the output `.dat` files will be saved. 
        If the directory does not exist, it will be created.

    scan_dims : tuple of int, optional
        The scanning dimensions (width, height) of the scan. 
        Default is (1024, 768).

    Returns
    -------
    None
        The function performs disk I/O and does not return a value.
    """
    

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    num_patterns = len(descriptors)

    # Loop through every pattern index
    for i in tqdm(range(num_patterns), desc="Saving Diffractograms"):
        # 1. Reconstruct the diffractogram for the current index
        diff_pattern = get_diffractogram(
            packets, descriptors, i, scan_dims)
        
        # 2. Save the resulting array to a .dat file
        diff2dat(diff_pattern, output_path, i)
        

def get_diffractogram(
    packets,
    descriptors,
    pattern_index,
    scan_dims=(1024, 768),
    detector_dims=(256, 256),
    values_field="count",   # <-- add this
    dtype=np.uint32,
):
    # --- sanity checks ---
    if values_field not in ("count", "itot"):
        raise ValueError("values_field must be 'count' or 'itot'")

    det_width, det_height = map(int, detector_dims)
    n_pkts = int(packets.shape[0])

    desc = np.asarray(descriptors)
    pc = np.asarray(desc["packet_count"], dtype=np.int64).reshape(-1)
    n_frames = pc.size
    if not (0 <= pattern_index < n_frames):
        raise IndexError("pattern_index out of bounds.")

    # offsets
    use_provided_off = ("offset" in desc.dtype.names)
    if use_provided_off:
        off = np.asarray(desc["offset"], dtype=np.int64).reshape(-1)
        if off.size and (off[-1] + pc[-1] != n_pkts):
            use_provided_off = False

    if not use_provided_off:
        off = np.empty_like(pc, dtype=np.int64)
        if pc.size:
            off[0] = 0
            if pc.size > 1:
                np.cumsum(pc[:-1], out=off[1:])

    s = int(off[pattern_index])
    e = s + int(pc[pattern_index])
    if s < 0 or e < s or e > n_pkts:
        return np.zeros((det_height, det_width), dtype=dtype)

    frame_pkts = packets[s:e]

    img = np.zeros((det_height, det_width), dtype=dtype)
    if frame_pkts.size:
        addr = frame_pkts["address"].astype(np.int64, copy=False)
        vals = frame_pkts[values_field].astype(dtype, copy=False)

        det_size = det_width * det_height
        valid = (addr >= 0) & (addr < det_size)
        if not np.all(valid):
            addr = addr[valid]
            vals = vals[valid]

        np.add.at(img.ravel(), addr, vals)

    return img