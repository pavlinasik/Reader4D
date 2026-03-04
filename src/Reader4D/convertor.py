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
import tifffile
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


def _print_hdf5_structure(path):
    """
    Print the hierarchical structure of an HDF5 file.

    The function is intended primarily for debugging and quick inspection
    of the file schema rather than reading actual data values.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the HDF5 file.

    Notes
    -----
    Groups are printed with a trailing "/" while datasets show their
    shape and dtype.
    """
    def _print(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}  {obj.shape}  {obj.dtype}")
        else:
            print(f"{indent}{name}/")

    print("\n[HDF5 STRUCTURE]")
    with h5py.File(path, "r") as f:
        f.visititems(_print)
        

def _tiff_header_to_dict(tiff_path):
    """
    Extract the header metadata of a TIFF file and convert it to a
    JSON-serializable dictionary.
    
    The function reads the first TIFF page and collects all tag values
    from its header. Because TIFF tag values may contain NumPy types,
    byte strings, or arrays that are not directly JSON-serializable,
    the values are converted into standard Python types.
    
    Conversion rules applied:
    - byte strings → decoded UTF-8 strings
    - NumPy arrays → Python lists
    - NumPy scalars → native Python scalars
    - tuples → lists
    
    In addition to standard TIFF tags, the function explicitly extracts
    the ``ImageDescription`` tag (commonly used by microscope software
    to store JSON/XML metadata) and stores it under the key
    ``"_ImageDescription"``.
    
    Parameters
    ----------
    tiff_path : str or os.PathLike
        Path to the TIFF file.
    
    Notes
    -----
    Only the first TIFF page is inspected. Multi-page TIFF files may
    contain additional metadata in other pages which are not extracted
    by this function.
    
    This helper is typically used to store experiment acquisition
    metadata inside an HDF5 file (e.g. under ``/entry/experiment``).
    """

    with tifffile.TiffFile(tiff_path) as t:
        page = t.pages[0]
        tags = {}

        for tag in page.tags.values():
            v = tag.value

            # JSON-serializable conversion
            if isinstance(v, (bytes, bytearray, np.bytes_)):
                try:
                    v = v.decode("utf-8", errors="replace")
                except Exception:
                    v = repr(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                v = v.item()
            elif isinstance(v, tuple):
                v = list(v)

            tags[tag.name] = v

        # common container for microscope metadata
        img_desc = page.tags.get("ImageDescription")
        if img_desc is not None:
            v = img_desc.value
            if isinstance(v, (bytes, bytearray, np.bytes_)):
                v = v.decode("utf-8", errors="replace")
            tags["_ImageDescription"] = v

        return {"file": str(tiff_path), "page_index": 0, "tags": tags}

        
def csr2hdf5(
    csr_path=None,
    packets=None,
    descriptors=None,
    header=None,
    output_path=r"./converted",
    filename="data.h5",
    overwrite=False,
    progress=True,
    chunk_events=2_000_000,
    print_structure=False,
    tiff=None,
    *,
    
    # optional scan metadata
    x_positions=None,           # (Nx,) float32
    y_positions=None,           # (Ny,) float32
    
    #optional detector metadata
    pixel_size=None,            # (2,) float32  [px, py] meters
    distance=None,              # scalar float32 meters
    beam_center=None,           # (2,) float32  [cx, cy] pixels
    
    #optional per-event mask
    write_mask=False,
    mask=None,    
):
    """
    Convert sparse Timepix event data (CSR-style packets + descriptors)
    into a structured HDF5 file using the /entry layout.
    
    The function writes detector events, scan metadata, detector metadata,
    and experiment metadata into a single HDF5 file suitable for later
    reconstruction or analysis of 4D-STEM / diffraction datasets.
    
    The output file follows this structure::

        /entry
            /events
                x           (nnz,) int16      detector x-coordinate per event
                y           (nnz,) int16      detector y-coordinate per event
                address     (nnz,) uint32     flattened detector address
                count       (nnz,) uint32     event count value
                itot        (nnz,) uint32     integrated time-over-threshold
                mask        (nnz,) uint8      per-event mask flag
            /scan
                shape       (2,) uint32       scan dimensions (Nx, Ny)
                event_ptr   (n_frames+1,)     CSR index pointer
                x_positions (Nx,) float32     probe x positions (optional)
                y_positions (Ny,) float32     probe y positions (optional)
            /detector
                shape       (2,) uint16       detector dimensions (nx, ny)
                pixel_size  (2,) float32      detector pixel size [m]
                distance    () float32        sample-detector distance [m]
                beam_center (2,) float32      beam center position [px]
            /experiment
                json                    experiment header metadata
                tiff_header             optional TIFF metadata

    Events are written in chunks to support very large datasets.
    
    Parameters
    ----------
    csr_path : str or None, optional
        Reserved for future use. Intended path to a CSR file if direct
        loading from disk is implemented.
    
    packets : numpy.ndarray
        Structured array containing per-event information with fields:
        ('address', 'count', 'itot').
    
    descriptors : numpy.ndarray
        Structured array describing packet offsets for each scan position
        with fields:
        ('offset', 'packet_count').
    
    header : dict
        Metadata dictionary describing the dataset. Must contain at least:
        - ``nav_shape`` : tuple(int, int)
            Scan shape (H, W) in navigation space.
        - ``sig_shape`` : tuple(int, int)
            Detector shape (Hdet, Wdet).
    
    output_path : str, optional
        Directory where the HDF5 file will be written.
    
    filename : str, optional
        Name of the output HDF5 file.
    
    overwrite : bool, optional
        If True, overwrite existing files.
    
    progress : bool, optional
        Display a progress bar during event writing.
    
    chunk_events : int, optional
        Number of events written per chunk. Larger values improve
        throughput but increase memory usage.
        
    print_structure : bool, optional
        If True, print the resulting HDF5 file structure after writing.
    
    tiff : str or None, optional
        Path to a TIFF file associated with the experiment. If provided,
        its header metadata will be extracted and stored in the HDF5
        file under ``/entry/experiment/tiff_header``.
        
    x_positions : array-like or None, optional
        Physical x-coordinates of scan probe positions (Nx).
    
    y_positions : array-like or None, optional
        Physical y-coordinates of scan probe positions (Ny).
    
    pixel_size : array-like or None, optional
        Detector pixel size in meters (px, py).
    
    distance : float or None, optional
        Sample-to-detector distance in meters.
    
    beam_center : array-like or None, optional
        Direct beam position on detector in pixel coordinates (cx, cy).
    
    write_mask : bool, optional
        If True, enable writing of a per-event mask dataset.
    
    mask : numpy.ndarray or None, optional
        Array of mask flags per event. Must match the total number of events 
        if provided.

    
    Returns
    -------
    str
        Path to the generated HDF5 file.
    

    Notes
    -----
    The function assumes the detector address encoding follows
    row-major ordering::
    
        address = y * Wdet + x
    
    Detector pixel coordinates are reconstructed during writing using::
    
        x = address % Wdet
        y = address // Wdet
    
    The scan event mapping is stored using a CSR-style index array
    ``event_ptr`` derived from descriptor packet counts.
    """
    
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, filename)


    if os.path.exists(out_file) and not overwrite:
        raise FileExistsError(
            f"{out_file} already exists. Set overwrite=True.")

    if packets is None or descriptors is None:
        raise ValueError("Provide packets and descriptors.")
    
    if header is None:
       raise ValueError("header is required (for nav_shape and sig_shape).")

    # Header conventions in the codebase:
    # (1) header["nav_shape"] = (H, W) (rows, cols)
    # (2) header["sig_shape"] = (Hdet, Wdet)
    nav_shape = tuple(map(int, header["nav_shape"]))  
    sig_shape = tuple(map(int, header["sig_shape"]))  
    Hnav, Wnav = nav_shape
    Hdet, Wdet = sig_shape

    n_frames = int(descriptors.shape[0])
    if Hnav * Wnav != n_frames:
        raise ValueError(
            f"nav_shape {nav_shape} implies {Hnav*Wnav} frames but descriptors has {n_frames}."
        )
        
    # Build CSR indptr (event_ptr) from packet_count
    pc = np.asarray(descriptors["packet_count"], dtype=np.uint64)
    event_ptr = np.empty(n_frames + 1, dtype=np.uint64)
    event_ptr[0] = 0
    if n_frames:
        np.cumsum(pc, out=event_ptr[1:])

    nnz = int(event_ptr[-1])
    if nnz != int(packets.shape[0]):
        raise ValueError(
            f"Mismatch: event_ptr[-1]={nnz} but packets has {packets.shape[0]} rows.")

    # Source arrays
    addr_src  = packets["address"]  # flattened detector index (Timepix address)
    count_src = packets["count"]
    itot_src  = packets["itot"]
    
    # Tiff header
    if tiff:
        tiffheader = _tiff_header_to_dict(tiff)
    else:
        tiffheader = None
    
    # Dataset creation kwargs
    ds_kwargs = dict(chunks=True)
    
    # Write file
    with h5py.File(out_file, "w") as f:
        entry = f.create_group("entry")
        
        # /ENTRY/EVENTS
        gE=entry.create_group("events")
        d_x=gE.create_dataset("x", shape=(nnz,),dtype=np.int16,**ds_kwargs)
        d_y=gE.create_dataset("y", shape=(nnz,),dtype=np.int16,**ds_kwargs)
        d_a=gE.create_dataset("address",shape=(nnz,),dtype=np.uint32,**ds_kwargs)
        d_c=gE.create_dataset("count",shape=(nnz,),dtype=np.uint32,**ds_kwargs)
        d_t=gE.create_dataset("itot",shape=(nnz,),dtype=np.uint32,**ds_kwargs)        
        d_m=gE.create_dataset("mask",shape=(nnz,),dtype=np.uint8,**ds_kwargs)
        
        # /ENTRY/SCAN
        gS = entry.create_group("scan")
        gS.create_dataset("shape",data=np.array([Wnav, Hnav],dtype=np.uint32))
        gS.create_dataset("event_ptr",data=event_ptr,dtype=np.uint64,**ds_kwargs)
        
        # x_positions
        if x_positions is None:
            x_positions = np.full(Wnav, np.nan, dtype=np.float32)
        else:
            x_positions = np.asarray(x_positions, dtype=np.float32)
        
        gS.create_dataset("x_positions", data=x_positions, dtype=np.float32)
        
        # y_positions
        if y_positions is None:
            y_positions = np.full(Hnav, np.nan, dtype=np.float32)
        else:
            y_positions = np.asarray(y_positions, dtype=np.float32)
        
        gS.create_dataset("y_positions", data=y_positions, dtype=np.float32)

        
        # /ENTRY/DETECTOR
        gD = entry.create_group("detector")
        gD.create_dataset("shape", data=np.array([Wdet, Hdet],dtype=np.uint16))
        
        # pixel_size
        if pixel_size is None:
            pixel_size = np.array([np.nan, np.nan], dtype=np.float32)
        else:
            pixel_size = np.asarray(pixel_size, dtype=np.float32)
        
        gD.create_dataset("pixel_size", data=pixel_size)
        
        # distance
        if distance is None:
            distance = np.float32(np.nan)
        else:
            distance = np.float32(distance)
        
        gD.create_dataset("distance", data=distance)
        
        # beam_center
        if beam_center is None:
            beam_center = np.array([np.nan, np.nan], dtype=np.float32)
        else:
            beam_center = np.asarray(beam_center, dtype=np.float32)
        
        gD.create_dataset("beam_center", data=beam_center)

        # /ENTRY/EXPERIMENT/JSON
        gX = entry.create_group("experiment")
        gX.create_dataset("json", data=np.bytes_(json.dumps(header)))

        # store TIFF header as JSON bytes (HDF5-safe)
        if tiffheader is None:
            gX.create_dataset("tiff_header", data=np.bytes_(b""))
        else:
            gX.create_dataset("tiff_header", 
                              data=np.bytes_(json.dumps(tiffheader)))
        
        # Write events
        it = range(0, nnz, chunk_events)
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(it, desc="Writing events", unit="ev")
            except Exception:
                pass

        Wdet_i64 = np.int64(Wdet)

        for start in it:
            stop = min(start + chunk_events, nnz)

            a = np.asarray(addr_src[start:stop], dtype=np.int64)
            
            # bounds check chunk (optional but safe)
            if a.size and (a.max(initial=0)>=Hdet*Wdet or a.min(initial=0)<0):
                raise ValueError("Address bounds check failed in chunk.")
            
            # decode x, y (detector pixel coordinates)
            x = (a % Wdet_i64).astype(np.int16, copy=False)
            y = (a // Wdet_i64).astype(np.int16, copy=False)

            d_a[start:stop] = a.astype(np.uint32, copy=False)
            d_x[start:stop] = x
            d_y[start:stop] = y
            d_c[start:stop] = np.asarray(count_src[start:stop], 
                                         dtype=np.uint32, copy=False)
            d_t[start:stop] = np.asarray(itot_src[start:stop],  
                                         dtype=np.uint32, copy=False)
            
            if mask is not None:
                d_m[start:stop] = mask[start:stop]
            else:
                d_m[start:stop] = 0
    
    if progress:
        print(f"[INFO] Wrote /entry layout HDF5: {out_file}")
    
    if print_structure:
        _print_hdf5_structure(out_file)
    

    return out_file


def _csr2hdf5(
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
    Load /entry/... sparse file.
    If lazy=True: returns (f, gE, gS, gD, header) and caller must close f.
    If lazy=False: materializes to structured packets + descriptors (RAM heavy).
    If lazy=="semi": loads descriptors into RAM but keeps events lazy.
    """

    f = h5py.File(path, "r")

    # ---- header/metadata
    header = None
    if "entry/experiment/json" in f:
        raw = f["entry/experiment/json"][()]
        if isinstance(raw, (bytes, bytearray, np.bytes_)):
            raw = raw.decode("utf-8")
        header = json.loads(raw)

    gE = f["entry/events"]
    gS = f["entry/scan"]
    gD = f["entry/detector"]

    if lazy is True:
        if progress:
            print("[INFO] Entry sparse data loaded as handles (lazy access).")
            print("[INFO] The file handle must be closed after use.")
        return f, gE, gS, gD, header

    # ---- geometry (small)
    det_shape = tuple(map(int, gD["shape"][...]))  # (nx, ny)
    nx, ny = det_shape

    event_ptr = gS["event_ptr"][...].astype(np.uint64, copy=False)
    if event_ptr.ndim != 1 or event_ptr.size < 2:
        f.close()
        raise ValueError("entry/scan/event_ptr must be 1D with length >= 2.")

    n_frames = int(event_ptr.size - 1)
    nnz = int(event_ptr[-1])

    packet_count = np.diff(event_ptr).astype(np.uint64, copy=False)
    if packet_count.max(initial=0) > np.iinfo(np.uint32).max:
        f.close()
        raise ValueError("A frame has >2^32-1 events; cannot fit packet_count into uint32.")

    # ---- "semi-lazy": descriptors in RAM, events stay on disk
    if lazy == "semi":
        descriptors = np.empty(n_frames, dtype=BIN_DESCRIPTOR_DTYPE)
        descriptors["offset"] = event_ptr[:-1].astype(np.uint64, copy=False)
        descriptors["packet_count"] = packet_count.astype(np.uint32, copy=False)

        if progress:
            print("[INFO] Semi-lazy load: descriptors in memory, events remain on disk.")
            print("[INFO] Close the returned file handle after use.")

        # return file + groups + descriptors
        return f, gE, gS, gD, header, descriptors

    # ---- materialize everything (RAM heavy)
    # read event arrays
    x = gE["x"][...].astype(np.int64, copy=False)
    y = gE["y"][...].astype(np.int64, copy=False)
    count = gE["count"][...]
    itot  = gE["itot"][...]

    if x.size != nnz or y.size != nnz:
        f.close()
        raise ValueError("events/x or events/y length does not match event_ptr[-1].")

    # bounds check
    if x.size:
        if x.min() < 0 or x.max() >= nx or y.min() < 0 or y.max() >= ny:
            f.close()
            raise ValueError("Some (x,y) events are outside detector bounds.")

    address = (y * nx + x).astype(np.uint32, copy=False)

    packets = np.empty(nnz, dtype=ACQ_DATA_PACKET_DTYPE)
    packets["address"] = address
    packets["count"]   = count.astype(np.uint32, copy=False)
    packets["itot"]    = itot.astype(np.uint32, copy=False)

    descriptors = np.empty(n_frames, dtype=BIN_DESCRIPTOR_DTYPE)
    descriptors["offset"] = event_ptr[:-1].astype(np.uint64, copy=False)
    descriptors["packet_count"] = packet_count.astype(np.uint32, copy=False)

    f.close()

    if progress:
        print("[INFO] Entry sparse data loaded as arrays into memory.")

    return packets, descriptors, header

           
def _load_sparse(path, lazy=False, progress=True):
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
        
        