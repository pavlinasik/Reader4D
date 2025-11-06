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
        
        
def diff2dat(arr, filepath, idx):
    """
    Save a 2D NumPy array to a binary .dat file using a structured naming 
    convention.

    This function flattens a 2D array and writes it to a `.dat` file in binary
    format. The filename is automatically generated based on the index (`idx`) 
    using a 'dif_XXX_YYY.dat' format, where:
        - XXX is the block number (idx // 1000),
        - YYY is the index within the block (idx % 1000).

    Parameters
    ----------
    arr : np.ndarray
        A 2D NumPy array (e.g., a diffraction pattern) to be saved. It will be 
        flattened and cast to `uint16` before saving.
    filepath : str
        Path to the directory where the .dat file will be saved.
    idx : int
        Index of the current file, used to determine the output filename in the
        'dif_XXX_YYY.dat' format.

    Returns
    -------
    None
        The function performs file I/O and does not return a value.
    """
    # Calculate the block number (XXX) and the file number within the block (YYY)
    block_num = idx // 1000
    file_in_block = idx % 1000

    # Format the filename with zero-padding for both parts
    # e.g., idx=1025 -> block_num=1, file_in_block=25 -> "dif_001_025.dat"
    filename = f"dif_{block_num:03d}_{file_in_block:03d}.dat"
    
    full_path = os.path.join(filepath, filename)

    with open(full_path, 'wb') as fh:
        flat_array = arr.flatten()
        block_array = flat_array.astype(np.uint32)
        block_array.tofile(fh)
    return

   
def diff2hdf5(packets, descriptors, filename, verbose=1):
    """
    Save packet and descriptor data to an HDF5 (.h5) file for MATLAB.

    This function stores the `packets` and `descriptors` arrays into an HDF5 
    file, writing the packet data in chunks to manage memory efficiently.

    Parameters
    ----------
    packets : np.ndarray
        A structured NumPy array containing event data, typically with fields 
        like 'count', 'itot', 'address', or similar. Each element represents
        a detector event.
    descriptors : np.ndarray
        A structured NumPy array describing the data layout, with fields such 
        as 'offset' and 'packet_count'. Each descriptor corresponds to a scan
        position.
    filename : str
        Path to the output HDF5 (.h5) file where the data should be saved.

    Returns
    -------
    None
        This function performs file writing and does not return a value.

    """
    if verbose:
        print("[INFO] Saving data to .h5 for MATLAB...")
    
    try:
        total_packets = len(packets)
        packet_dtype = packets.dtype  # Get dtype from the input array

        with h5py.File(filename, 'w') as f:
            # Save descriptors
            f.create_dataset('descriptors', data=descriptors)

            # Save packets in chunks with progress bar
            chunk_size = 10_000
            dset_packets = f.create_dataset(
                'packets', 
                shape=(total_packets,), 
                dtype=packet_dtype)

            with tqdm(
                    total=total_packets, 
                    unit='packets', 
                    desc="Saving descriptors and packets: ") as pbar:
                for i in range(0, total_packets, chunk_size):
                    end_index = min(i + chunk_size, total_packets)
                    chunk_data = packets[i:end_index]
                    dset_packets[i:end_index] = chunk_data
                    pbar.update(len(chunk_data))
        if verbose:                    
            print(f"\n[INFO] Data successfully saved to {os.path.abspath(filename)}")

    except Exception as e:
        print(f"An error occurred during saving: {e}")


def get_diffractogram( 
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
        # --- sanity on dims ---
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
            field = "itot" or "count"
            vals = frame_pkts[field]
    
            det_size = det_width * det_height
            valid = (addr >= 0) & (addr < det_size)
            if not np.all(valid):
                addr = addr[valid]; vals = vals[valid]
    
            # accumulate into raveled image for speed, then done
            np.add.at(img.ravel(), addr, vals)
    

        return img        