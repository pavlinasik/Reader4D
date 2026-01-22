# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 08:34:05 2026

@author: p-sik
"""
import matplotlib.pyplot as plt
import h5py
import numpy as np
import json
import os

import Reader4D.convertor as r4dConv



def show_random_diffractograms(
    packets,
    descriptors,
    scan_dims=(1024, 768),
    num_patterns=10,
    rows=2,
    cols=5,
    csquare=256,
    icut=None,                     # allow None
    detector_dims=(256, 256),
    values_field="count",
    percentile=(1, 99),            # better default for display
):
    if rows * cols < num_patterns:
        num_patterns = rows * cols

    n_frames = descriptors.shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.asarray(axes).ravel()

    sums = []
    last_img = None

    for i in range(num_patterns):
        pattern_index = int(np.random.randint(0, n_frames))
        try:
            img = r4dConv.get_diffractogram(
                packets,
                descriptors,
                pattern_index,
                scan_dims=scan_dims,
                detector_dims=detector_dims,
                values_field=values_field,
            )

            if icut is not None:
                img = np.clip(img, a_min=None, a_max=icut)

            last_img = img
            sums.append(float(img.sum()))

            # crop
            img_h, img_w = img.shape
            side = min(int(csquare), img_h, img_w)
            start_x = img_w // 2 - side // 2
            start_y = img_h // 2 - side // 2
            cimg = img[start_y:start_y + side, start_x:start_x + side]

            ax = axes[i]

            if percentile is not None:
                vmin, vmax = np.percentile(cimg, percentile)
                ax.imshow(cimg, origin="lower", vmin=vmin, vmax=vmax)
            else:
                ax.imshow(cimg, origin="lower")

            ax.set_title(f"Pattern ID: {pattern_index}")
            ax.axis("off")

        except Exception as e:
            print(f"Error for pattern {pattern_index}: {e}")
            axes[i].axis("off")
            axes[i].set_title(f"{pattern_index} - Error")

    for j in range(num_patterns, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    return last_img, sums

    
    
def show_micrograph(img, 
                    title="Micrograph", 
                    cmap="gray", 
                    save=False, 
                    filename=None, 
                    output_dir=None,
                    show=True):
    """
    Display and optionally save a micrograph.

    Parameters
    ----------
    img : 2D array
        Image to display.
    title : str
        Title for the plot.
    cmap : str
        Colormap to use.
    save : bool
        If True, save the image.
    filename : str
        Filename to save as (e.g. "filtered.png").
    output_dir : str
        Directory where to save the image.
    show : bool
        If True, display the image.
    """
    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap=cmap, origin='lower')
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if save:
        if filename is None:
            raise ValueError(
                "If 'save' is True, you must provide a filename.")
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plt.imsave(path, img, cmap=cmap)
        print(f"[INFO] Saved: {path}")
        
            
def show_dense_frame(
    h5_path,
    index=0,
    *,
    dense_dataset="data",
    values_field="count",          # sparse: "count" or "itot"
    det_dim=(256, 256),            # fallback if header lacks sig_shape
    title=None,
    percentile=None,               # e.g. (1, 99)
):
    """
    Display a frame from an HDF5 file that may contain either:
      - Dense stack under /<dense_dataset>, or
      - Sparse TP3 layout under /packets and /descriptors.

    The function auto-detects the layout based on the HDF5 group/dataset
    structure and reconstructs the requested frame accordingly.

    Expected layouts
    ----------------
    (1) Dense:
            /data  (H, W) or (N, H, W)

    (2) Sparse:
            /packets/address, /packets/count, /packets/itot
            /descriptors/offset, /descriptors/packet_count
            optional /header_json (JSON dict, may include "sig_shape")

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
        
    index : int
        Frame index (0-based).
        
    dense_dataset : str
        Name/path of the dense dataset. Default "data".
        
    values_field : {"count","itot"}
        Packet value field used for sparse reconstruction.
        
    det_dim : tuple[int,int]
        Fallback detector dims (Wdet, Hdet) if not present in header_json.
        
    title : str or None
        Plot title override.
        
    percentile : tuple[int,int] or None
        Percentile scaling for display contrast.

    """
    # Check inputs
    if values_field not in ("count", "itot"):
        raise ValueError("values_field must be 'count' or 'itot'")

    with h5py.File(h5_path, "r") as f:
        # parse header_json if present
        header = None
        
        if "header_json" in f:
            raw = f["header_json"][()]
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            header = json.loads(raw)

        # detect layout
        has_dense = dense_dataset in f
        has_sparse = ("packets" in f) and ("descriptors" in f)

        if has_dense and has_sparse:
            # Ambiguous: prefer dense unless you want the opposite
            kind = "dense"
            
        elif has_dense:
            kind = "dense"
            
        elif has_sparse:
            kind = "sparse"
            
        else:
            raise KeyError(
                f"Unrecognized HDF5 structure: neither '{dense_dataset}' nor "
                "('/packets' and '/descriptors') found."
            )

        # load/reconstruct frame
        if kind == "dense":
            dset = f[dense_dataset]
            
            if dset.ndim == 2:
                frame = dset[...]
                
            elif dset.ndim == 3:
                n_frames = dset.shape[0]
                
                if index < 0 or index >= n_frames:
                    raise IndexError(
                        f"index={index} out of range [0, {n_frames})"
                        )
                frame = dset[index, :, :]
            else:
                raise ValueError(
                    f"Unsupported dense dataset rank: {dset.ndim} (expected 2 or 3).")

        else:
            # Determine detector dims:
            # Prefer header["sig_shape"] if available, else fallback det_dim.
            # Your header seems to store shapes as [H, W] or [W, H] depending 
            # on code. Interpretation of sig_shape:
            #        (Hdet, Wdet) if length==2; then convert to (Wdet, Hdet).
            
            if header is not None and "sig_shape" in header:
                sig = header["sig_shape"]
                
                if not (isinstance(sig, (list, tuple)) and len(sig) == 2):
                    raise ValueError(
                        f"header['sig_shape'] must be length-2, got: {sig}"
                        )
                    
                # common convention: sig_shape = [Hdet, Wdet]
                det_h, det_w = int(sig[0]), int(sig[1])
                
            else:
                det_w, det_h = map(int, det_dim)
            
            # detector size (number of pixels)
            det_size = det_w * det_h
            
            # extract packets and descriptors
            gP = f["packets"]
            gD = f["descriptors"]

            address = gP["address"]
            values = gP[values_field]
            offset = gD["offset"]
            packet_count = gD["packet_count"]

            n_frames = int(packet_count.shape[0])
            
            if index < 0 or index >= n_frames:
                raise IndexError(
                    f"index={index} out of range [0, {n_frames})"
                    )
            
            # reconstruct array
            s = int(offset[index])
            n = int(packet_count[index])
            e = s + n

            n_pkts = int(address.shape[0])
            
            if s < 0 or e < s or e > n_pkts:
                frame = np.zeros((det_h, det_w), dtype=np.uint32)
            else:
                addr = address[s:e].astype(np.int64, copy=False)
                vals = values[s:e].astype(np.uint32, copy=False)

                frame = np.zeros((det_h, det_w), dtype=np.uint32)
                if addr.size:
                    # Assumes address is a linear pixel index 0..(W*H-1)
                    # If your encoding differs, you must decode address here.
                    if addr.max(initial=0) >= det_size:
                        raise ValueError(
                          "Packet 'address' exceeds detector size. "
                          "Your address encoding is not linear; decode it first."
                        )
                    np.add.at(frame.ravel(), addr, vals)

        # Display
        plt.figure()
        
        if percentile is not None:
            pmin, pmax = percentile
            vmin, vmax = np.percentile(frame, [pmin, pmax])
            plt.imshow(frame, origin="lower", vmin=vmin, vmax=vmax)
            
        else:
            
            plt.imshow(frame, origin="lower")
            
        plt.colorbar(label="Intensity")
        plt.title(title or f"{kind} | frame {index}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()


def get_diffractogram_lazy(
    gP,
    gD,
    pattern_index: int,
    *,
    detector_dims=(256, 256),     # (Wdet, Hdet)
    values_field="count",         # "count" or "itot"
    dtype=np.uint32,
):
    """
    Recover one diffractogram using HDF5 dataset handles (lazy access).

    Parameters
    ----------
    gP : h5py.Group
        /packets group handle (expects datasets address, count, itot)
    gD : h5py.Group
        /descriptors group handle (expects datasets offset, packet_count)
    pattern_index : int
        Frame index (0-based).
    detector_dims : tuple[int, int]
        (Wdet, Hdet)
    values_field : {"count","itot"}
        Which values dataset to accumulate.
    dtype : numpy dtype
        Output image dtype.

    Returns
    -------
    img : numpy.ndarray
        2D array shape (Hdet, Wdet).
    """
    if values_field not in ("count", "itot"):
        raise ValueError("values_field must be 'count' or 'itot'")

    det_w, det_h = map(int, detector_dims)
    det_size = det_w * det_h

    offset = gD["offset"]
    packet_count = gD["packet_count"]
    n_frames = int(packet_count.shape[0])
    if not (0 <= pattern_index < n_frames):
        raise IndexError(f"pattern_index out of bounds: {pattern_index} (n_frames={n_frames})")

    s = int(offset[pattern_index])
    n = int(packet_count[pattern_index])
    e = s + n

    address = gP["address"]
    values = gP[values_field]
    n_pkts = int(address.shape[0])
    if s < 0 or e < s or e > n_pkts:
        return np.zeros((det_h, det_w), dtype=dtype)

    # Only read needed slices
    addr = address[s:e].astype(np.int64, copy=False)
    vals = values[s:e].astype(dtype, copy=False)

    img = np.zeros((det_h, det_w), dtype=dtype)
    if addr.size:
        valid = (addr >= 0) & (addr < det_size)
        if not np.all(valid):
            addr = addr[valid]
            vals = vals[valid]
        np.add.at(img.ravel(), addr, vals)

    return img