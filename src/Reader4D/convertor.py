"""
Conversion and I/O utilities for 4D STEM datasets.

This module focuses on two storage styles:

1) **Dense frame stacks** (one full detector frame per scan position)
   stored as a single HDF5 dataset ``/data`` with shape
   ``(n_frames, det_y, det_x)``. See :func:`dat2hdf5`.

2) **Sparse CSR-like packet storage** (variable number of detector hits per
   scan position), stored as split HDF5 datasets:

- ``/packets/address``      : linear detector addresses (uint32)
- ``/packets/count``        : event counts (uint32)
- ``/packets/itot``         : iToT values (uint32)
- ``/descriptors/offset``   : start index into packets for each frame (uint64)
- ``/descriptors/packet_count`` : number of packets per frame (uint32)
- optional ``/header_json`` : UTF-8 JSON metadata

See :func:`csr2hdf5` and :func:`load_sparse`.

The sparse representation is suitable for very large scans because it avoids
materializing dense ``(n_frames, det_y, det_x)`` arrays on disk.

Notes
-----
- Detector addresses are assumed to be linear indices into the detector plane:
  ``0 .. det_width * det_height - 1``.
- All JSON metadata are stored as UTF-8 strings for interoperability with
  non-Python tooling.
"""
import os
import numpy as np
import h5py
from tqdm import tqdm
import Reader4D.detectors as det
import json
import glob 

# Import data types
from .dtypes import TP3_BIN_DESCRIPTOR_DTYPE as BIN_DESCRIPTOR_DTYPE
from .dtypes import TP3_ACQ_DATA_PACKET_DTYPE as ACQ_DATA_PACKET_DTYPE

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
    """
    Convert a directory of Timepix ``.dat`` frames into a dense HDF5 stack.

    This function expects one detector frame per file. Each ``.dat`` file is
    read as a 1D array and reshaped to ``det_dim`` before being written into
    a single HDF5 dataset named ``/data`` of shape
    ``(n_frames, det_y, det_x)``.

    Parameters
    ----------
    dat_path : str or os.PathLike
        Directory containing ``*.dat`` files (one frame per file).
    
    header : dict or None, optional
        Optional metadata to embed as JSON at ``/metadata/header_json``.
    
    det_dim : tuple[int, int], optional
        Detector dimensions as ``(det_y, det_x)``. Default is ``(256, 256)``.
    
    dtype : numpy.dtype, optional
        Data type used to interpret the raw ``.dat`` files and store the HDF5
        dataset. Default is ``numpy.uint16``.
    
    output_path : str or os.PathLike, optional
        Directory where the output HDF5 file will be written.
    
    filename : str, optional
        Output filename. If no ``.h5``/``.hdf5`` suffix is provided, ``.h5`` 
        is appended.
    
    overwrite : bool, optional
        If True, an existing output file is removed and recreated.
    
    progress : bool, optional
        If True, displays a tqdm progress bar.

    Returns
    -------
    str
        Absolute or relative path to the written HDF5 file.
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
    """
    Write a sparse CSR-like Timepix3 dataset to HDF5.

    The sparse representation stores variable-length per-frame packet lists
    in two components:

    - ``packets``: structured array with fields ``address``, ``count``, 
      ``itot`` and shape ``(nnz,)``.
    - ``descriptors``: structured array with fields ``offset``,
      ``packet_count`` and shape ``(n_frames,)``. 
      
      For frame ``i``, the corresponding packets are::

          s = descriptors["offset"][i]
          n = descriptors["packet_count"][i]
          frame_packets = packets[s : s+n]

    The HDF5 layout produced is:

    - ``/packets/address`` (uint32)
    - ``/packets/count`` (uint32)
    - ``/packets/itot`` (uint32)
    - ``/descriptors/offset`` (uint64)
    - ``/descriptors/packet_count`` (uint32)
    - optional ``/header_json`` (UTF-8 JSON)

    Parameters
    ----------
    csr_path : str or os.PathLike, optional
        If ``packets`` and ``descriptors`` are not provided, ``csr_path`` is
        used to instantiate :class:`Reader4D.detectors.Timepix3` and load them.
   
    packets : numpy.ndarray or None, optional
        Structured array with dtype compatible with
        :data:`~Reader4D.dtypes.TP3_ACQ_DATA_PACKET_DTYPE`.
    
    descriptors : numpy.ndarray or None, optional
        Structured array with dtype compatible with
        :data:`~Reader4D.dtypes.TP3_BIN_DESCRIPTOR_DTYPE`.
    
    header : dict or None, optional
        Optional metadata to store as JSON in ``/header_json``.
    
    output_path : str or os.PathLike, optional
        Output directory.
    
    filename : str, optional
        Output filename (usually ``.h5``).
    
    overwrite : bool, optional
        If False and the output file exists, raise :class:`FileExistsError`.
        If True, the file will be overwritten.
    
    progress : bool, optional
        If True, prints a status line when done.

    Returns
    -------
    str
        Path to the written HDF5 file.
    """
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
    Load a sparse Timepix3 HDF5 file produced by :func:`csr2hdf5`.

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
    """
    Print a simple tree view of an HDF5 file (groups and datasets).

    Parameters
    ----------
    path : str or os.PathLike
        Path to an HDF5 file.
    """

    with h5py.File(path, "r") as f:
        
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"[DSET] {name}  shape={obj.shape}  dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"[GRP ] {name}/")
        f.visititems(visitor)
        

def get_diffractogram(
    packets,
    descriptors,
    pattern_index,
    scan_dims=(1024, 1024),
    detector_dims=(256, 256),
    values_field="count",   # <-- add this
    dtype=np.uint32,
):
    """
    Reconstruct a single dense detector-frame (diffractogram) from sparse 
    packets.

    This function uses the CSR-like ``descriptors`` to slice the packet array
    for a given frame and accumulates either ``count`` or ``itot`` values into
    a dense ``(det_height, det_width)`` image.

    Parameters
    ----------
    packets : numpy.ndarray
        Structured array with at least the fields ``address`` and the selected
        ``values_field`` (``count`` or ``itot``).
    
    descriptors : numpy.ndarray
        Structured array with fields ``packet_count`` and optionally ``offset``.
        If ``offset`` is missing or inconsistent, offsets are reconstructed via
        cumulative sum of ``packet_count``.
    
    pattern_index : int
        Frame index to reconstruct (0-based).
   
    detector_dims : tuple[int, int], optional
        Detector dimensions as ``(det_width, det_height)``.
    
    values_field : {"count", "itot"}, optional
        Packet field to accumulate into the image.
    
    dtype : numpy.dtype, optional
        Output dtype of the reconstructed image.

    Returns
    -------
    img : numpy.ndarray
        Reconstructed detector image of shape ``(det_height, det_width)``.

    """
    # sanity checks
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


def iter_events(packets, descriptors, nav_shape, sig_shape):
    Hnav, Wnav = map(int, nav_shape)
    Hdet, Wdet = map(int, sig_shape)

    offsets = descriptors["offset"].astype(np.int64)
    counts  = descriptors["packet_count"].astype(np.int64)
    for i in range(counts.size):
        s = offsets[i]
        n = counts[i]
        if n == 0:
            continue

        # probe coords
        Y = i // Wnav
        X = i %  Wnav

        pkt = packets[s:s+n]
        addr = pkt["address"].astype(np.int64, copy=False)

        # detector coords
        y = addr // Wdet
        x = addr %  Wdet

        # values
        c = pkt["count"]
        t = pkt["itot"]

        # yield per-event arrays for this frame
        # X and Y are scalars; x,y,c,t are length n
        yield X, Y, x, y, c, t
        
        