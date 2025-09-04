# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:11:17 2025

@author: p-sik
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from PIL import Image
import matplotlib.patches as patches
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import Reader4D.convertor as r4dConv
import sys


class Extractor:
    def __init__(self,
            micrograph,
            descriptors,
            packets,
            out_dir,
            header=None,
            scan_dims=(1024,768), 
            det_dims=(256,256),
            reconstruct=False,
            show=False,
            cmap="gray",
            verbose=1,
            save_roi=True,
            name_roi=None,
            ):
        
        self.micrograph = micrograph
        self.descriptors = descriptors
        self.out_dir = out_dir
        self.packets = packets
        self.header = header
        self.reconstruct = reconstruct
        self.show = show
        self.CMAP = cmap
        self.verbose = verbose
        self.save_roi = save_roi
        self.name_roi = name_roi
        self.out_dir = out_dir
        
        if self.header is not None:
            self.SCAN_DIMS = self.header['nav_shape'][::-1]
            self.DET_DIMS = self.header['sig_shape']
        else: 
            self.SCAN_DIMS = scan_dims
            self.DET_DIMS = det_dims
            
        self.coords = self.get_ROI(
            self.micrograph,
            self.reconstruct,
            self.packets,
            self.descriptors,
            self.SCAN_DIMS,
            self.DET_DIMS,
            self.show)
        
        self.roi = None
        self.psubset = None
        self.dsubset = None
        self.d_offset = None
        self.d_packet_count = None
        self.p_address = None
        self.p_count = None
        self.dcoords = None
        
        if self.reconstruct:
            self.reconstruct_ROI()
            
            # Get new dimensions
            self.NEW_DIMS = self.roi.shape
        
        if self.save_roi:
            if self.name_roi:
                r4dConv.diff2hdf5(
                    packets=self.psubset, 
                    descriptors=self.dsubset,
                    filename=\
                        self.out_dir+f"\\dim_{int(self.NEW_DIMS[0])}x{int(self.NEW_DIMS[1])}_"+self.name_roi,
                    verbose=self.verbose)
            
                base = os.path.splitext(self.name_roi)[0]
                dims_tag = f"{int(self.NEW_DIMS[0])}x{int(self.NEW_DIMS[1])}"
                png_path = os.path.join(
                    self.out_dir, f"dim_{dims_tag}_{base}.png")
                roi = np.asarray(self.roi)
                
                if roi.ndim == 2: # grayscale
                    rmin = float(np.nanmin(roi))
                    rmax = float(np.nanmax(roi))
                    if rmax > rmin:
                        roi8 = np.clip(
                            (roi-rmin)/(rmax-rmin)*255,0,255).astype(np.uint8)
                    else:
                        roi8 = np.zeros_like(roi, dtype=np.uint8)
                    Image.fromarray(roi8).save(png_path)
                else:
                    if roi.dtype != np.uint8:
                       roi = np.clip(roi, 0, 255).astype(np.uint8)
                    Image.fromarray(roi).save(png_path) 
                
                if self.verbose:
                    print(f"[INFO] Saved ROI PNG to: {png_path}")


            else:
                print("Argument name_roi is missing. Required for saving.")
                sys.exit()
        
        
    def get_ROI(self,
                full_micrograph, 
                reconstruct=True, 
                packets=None, 
                descriptors=None, 
                scan_dims=(1024,768), 
                det_dims=(256,256), 
                show=True):
        
        """
        Interactively select a rectangular Region of Interest (ROI) 
        on a 2-D image using Matplotlib’s RectangleSelector and return its 
        bounds.

        The function pops up a figure showing `full_micrograph`. Drag with
        the left mouse button to draw a rectangle; press **Enter** to 
        confirm. A second figure with a red rectangle overlay is shown for 
        visual confirmation.

        Parameters
        ----------
        full_micrograph : array-like of shape (H, W)
            2-D numeric image to display. Must be convertible to a float 
            array.
        reconstruct : bool, default True
            If True, after selection the function calls reconstruct_ROI()
            Note that the *return value* of `reconstruct_ROI` is **not** 
            propagated; only the selected coordinates are returned by this 
            function.
        packets : optional
            Packets passed through to `reconstruct_ROI` if 
            `reconstruct=True`.
        descriptors : optional
            Descriptors passed through to `reconstruct_ROI` if
            `reconstruct=True`.
        scan_dims : tuple(int, int), default (1024, 768)
            Full scan dimensions (width, height) passed to 
            `reconstruct_ROI`.
        det_dims : tuple(int, int), default (256, 256)
            Detector grid dimensions (width, height) passed to 
            `reconstruct_ROI`.
        show : bool, default True
            If True, show the confirmation figure (and any plots produced
            by `reconstruct_ROI` when `reconstruct=True`).

        Returns
        -------
        coords : tuple[float, float, float, float] or None
            The selected ROI bounds as `(x_min, x_max, y_min, y_max)` in
            image data coordinates (floats), or `None` if no area was 
            selected.

        """
        # Helper functions
        def on_select(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            
            if None in (x1, y1, x2, y2):
                print("Selection contains None coordinates, ignoring.")
                return
            
            selected_extents["coords"] = \
                (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
            
        def on_key_press(event):
            if event.key == 'enter':
                plt.close()

        selected_extents = {"coords": None}
        
        print("[INFO] Selecting ROI for reconstruction...")
        
        # Initialize interactive window
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(full_micrograph, cmap="gray", origin="lower")
        ax.set_title("Drag to select ROI, then press 'Enter' to confirm.")

        rs = RectangleSelector(ax, on_select,
                               useblit=True,
                               button=[1],
                               minspanx=5, minspany=5,
                               spancoords='data',
                               interactive=True)

        fig.canvas.mpl_connect('key_press_event', on_key_press)
        plt.show(block=True)
        
        if selected_extents["coords"] is None:
            print("No area selected.")
            return None
                    
        # Show ROI selection
        if selected_extents["coords"] is not None:
            x_min, x_max, y_min, y_max = selected_extents["coords"]
            
            fig2, ax2 = plt.subplots()
            ax2.imshow(full_micrograph, cmap="gray", origin="lower")
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='red', facecolor='none', label="ROI"
            )
            ax2.axis("off")
            ax2.add_patch(rect)
            plt.title("Selected Region of Interest (ROI)")
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
                        
        return selected_extents['coords']
    
    
    def reconstruct_ROI(self, show=False):
        """
        Reconstruct a Region of Interest (ROI) from binned descriptors/packets 
        and return a quick “sum of counts” micrograph for that ROI, along with
        the packet/descriptor subsets and per-pixel packet lists.

        The function:
          1) Converts the rectangular ROI (given in image coords) into linear 
             indices using row-major order: idx = y * width + x.
          2) For each ROI pixel, slices the corresponding packets via
             descriptors[idx]['offset'] .. + packet_count and accumulates:
             - per-pixel sum of 'count' values (for a quick 2-D ROI image),
             - per-pixel lists of addresses and counts,
             - detector coordinates (dx, dy) for each packet.
          3) Builds `psubset` (a single contiguous structured array of all packets
             belonging to the ROI) and `dsubset` (descriptors in ROI order).

        Parameters
        ----------
        descriptors : np.ndarray
            Structured array of length H*W with fields:
            - 'offset' (uint64 or uint32): starting index into `packets` 
            - 'packet_count' (uint32): number of packets for this pixel.
        packets : np.ndarray
            Structured array of length N (all packets) with fields at least:
            - 'address' (uint32): detector linear address (y * dwidth + x),
            - 'count' (uint32): hit count for that address.
            Additional fields (e.g., 'itot') are ignored here.
        coords : tuple[float, float, float, float]
            ROI bounds as (x_min, x_max, y_min, y_max) in image coordinates.
            This implementation treats bounds as **inclusive** when building the
            (rheight, rwidth) grid: rwidth = x_end - x_start + 1, etc.
        scan_dims : tuple[int, int]
            (width, height) of the **full** scan/image grid in pixels.
            NOTE: order is (W, H).
        det_dims : tuple[int, int]
            (dwidth, dheight) of the detector grid. NOTE: order is (W, H) for
            address decoding via dy = address // dwidth, dx = address % dwidth.
        show : bool, default True
            If True, display the quick reconstruction (sum of counts) for the ROI.

        Returns
        -------
        roi : np.ndarray, shape (rheight, rwidth), dtype float64
            Quick ROI image where each pixel is the sum of 'count' for the
            corresponding scan position.
        psubset : np.ndarray
            Structured view/array of all packets that fall inside the ROI,
            concatenated in ROI order (contiguous index array into `packets`).
        dsubset : np.ndarray
            Structured array of descriptors for ROI pixels (same fields as input
            `descriptors`), in ROI order.
        d_offset : list[int]
            Per-ROI-pixel list of descriptor offsets (start indices into `packets`)
        d_packet_count : list[int]
            Per-ROI-pixel list of packet counts.
        p_address : list[np.ndarray]
            For each ROI pixel, a 1-D array of detector addresses of its packets.
        p_count : list[np.ndarray]
            For each ROI pixel, a 1-D array of packet counts aligned with p_address
        dcoords : list[list[np.ndarray, np.ndarray]]
            For each ROI pixel, a pair [dx, dy] where dx, dy are 1-D arrays of
            detector x/y coordinates decoded from `p_address`.

        Notes
        -----
        - The ROI bounds are handled inclusively (x_start..x_end, y_start..y_end),
          which is why `+1` appears in the shape and slicing logic.
        - Linear indices are computed with row-major layout using the full-image
          width: idx = y * width + x.
        - A debug `last_frame` is built internally but not returned.
        - For very large ROIs, the per-pixel lists (`p_address`, `p_count`) can be
          memory-heavy. If you only need the ROI image and `psubset`, consider
          dropping those lists.

        """
        if self.verbose:
            print("[INFO] Extracting relevant packets and descriptors...")

        descriptors = self.descriptors
        packets = self.packets
        coords = self.coords
        scan_dims = self.SCAN_DIMS
        det_dims = self.DET_DIMS
        
        x_start = int(coords[0])
        x_end = int(coords[1])
        y_start = int(coords[2])
        y_end = int(coords[3])

        # full micrograph dimensions
        width, height = scan_dims

        # detector grid
        dwidth, dheight = det_dims

        # Correct ROI grid dimensions
        rwidth = int(x_end - x_start + 1)
        rheight = int(y_end - y_start + 1)

        # Get the linear indices for the selected ROI
        # Use np.mgrid to get coordinates and then ravel them to linear indices
        y_coords, x_coords = np.mgrid[y_start:y_end+1, x_start:x_end+1]

        # Create a single array of linear indices
        linear_indices = (y_coords.ravel() * width + x_coords.ravel()).astype(int)

        # Extract data using the linear indices
        d_offset, d_packet_count = [],[]
        p_address, p_count, p_sum = [],[],[]
        dcoords = [] 
        
        # Use tqdm for a progress bar
        for i in tqdm(linear_indices, desc="Processing ROI pixels"):
            
            # DESCRIPTORS indexing
            do = descriptors[i]["offset"]
            dpc = descriptors[i]["packet_count"]
            d_offset.append(do)
            d_packet_count.append(dpc)

            # PACKETS indexing
            p = packets[do:do+dpc]
            pa = p["address"]
            pc = p["count"]
            p_address.append(pa)
            p_count.append(pc)
            
            # sum count for count image
            p_sum.append(np.sum(pc))
            
            # conversion to detector grid
            dy = pa//dwidth
            dx = pa%dwidth
            dcoords.append([dx, dy])
        
        
        last_frame = np.zeros((dwidth, dheight))
        last_frame[dy, dx] = pc
        
        dsubset = descriptors[linear_indices]
        idx = np.concatenate([
            np.arange(s, s + l, dtype=np.int64)
            for s, l in zip(d_offset, d_packet_count)
            if l > 0
        ]) if dsubset.size else np.empty(0, dtype=np.int64)
        psubset = packets[idx]  # structured ndarray (not a list)
        
        # Reconstruct the ROI out of p_sum using correct mapping
        roi = np.zeros((rheight, rwidth))

        # The `p_sum` array is ordered by the linear_indices, which
        # is a flattened version of the (y, x) grid from the original image.
        # We can reshape this into the correct ROI dimensions.

        p_sum_reshaped = np.array(p_sum).reshape(rheight, rwidth)
        roi = p_sum_reshaped
        
        if show:
            plt.figure()
            plt.imshow(roi, cmap="gray", origin='lower')
            plt.title("ROI quick reconstruction")
            plt.tight_layout()
            plt.axis("off")
            plt.show(block=False)
        
        self.roi = roi
        self.psubset = psubset
        self.dsubset = dsubset
        self.d_offset = d_offset
        self.d_packet_count = d_packet_count
        self.p_address = p_address
        self.p_count = p_count
        self.dcoords = dcoords
        
        return 
    
    
    def filter_ROI(self, 
                    mask,
                    n_workers=None,                 # e.g. os.cpu_count()
                    chunk_size=4096                 # pixels per worker chunk
                    ):
        """
        Compute a per-pixel, detector-masked sum for a Region of Interest (ROI)
        **without** reconstructing full 2-D diffraction patterns, and do it in
        parallel over pixel chunks.

        This implementation treats each ROI pixel as a sparse list of detector
        events: `address[i]` holds the linear detector indices for pixel *i* 
        and `counts[i]` holds corresponding weights (e.g., hit counts). For a 
        fixed detector-space mask `mask` (2-D), it computes:

            raw_sum[i]  = sum(counts[i])
            filt_sum[i] = sum(counts[i] * mask_flat[address[i]])

        avoiding a 256×256 allocation per pixel. Work is split across threads 
        in chunks to reduce Python overhead.

        Parameters
        ----------
        offset : array-like, unused
            Present for API compatibility with a “flat arrays” style; ignored 
            here (this function expects list-per-pixel inputs in `address`/
                  `counts`).
        packets_count : array-like, unused
            Present for API compatibility; ignored in this list-per-pixel 
            variant.
        address : list of 1D array-like, length Hroi*Wroi
            For each ROI pixel (row-major order), the 1-D array of detector 
            linear
            addresses (0..Hdet*Wdet-1).
        counts : list of 1D array-like, length Hroi*Wroi
            For each ROI pixel, the 1-D array of event weights aligned with 
            address
        im_dims : tuple[int, int]
            `(Hroi, Wroi)` — ROI size in scan pixels.
        det_dims : tuple[int, int]
            `(Hdet, Wdet)` — detector grid shape used to validate addresses 
            and to optionally render an example detector frame.
        mask : ndarray, shape (Hdet, Wdet)
            Detector-space weighting mask (binary or float). The same mask is 
            used
            for all pixels in this function.
        show : bool, default True
            If True, display the unfiltered (`raw_sum`) and filtered 
            (`filt_sum`) ROI images.
        cmap : str, default 'gray'
            Colormap for visualizarion.
        return_example_idx : int or None, default None
            If an integer `k` is provided, also return a `(Hdet, Wdet)`
            detector image for pixel `k` (formed by accumulating `counts[k]` at
            `(address[k] // Wdet, address[k] % Wdet)`).
        n_workers : int or None, default None
            Number of worker threads. Defaults to `min(32, os.cpu_count() or 4)`.
        chunk_size : int, default 4096
            Number of ROI pixels per worker task. Larger chunks amortize Python
            overhead; smaller chunks can improve load balancing.

        Returns
        -------
        froi : ndarray, shape (Hroi, Wroi), dtype float64
            Filtered (masked) sums per ROI pixel.
        rroi : ndarray, shape (Hroi, Wroi), dtype float64
            **Only if** `return_example_idx` is not None: the unfiltered sums.
        example_frame : ndarray, shape (Hdet, Wdet), dtype float64
            **Only if** `return_example_idx` is not None and the index is valid:
            detector image reconstructed for that pixel; otherwise `None`.
        """
        address = self.p_address
        counts = self.p_count
        im_dims = self.NEW_DIMS
        det_dims = self.DET_DIMS
        show = self.show
        cmap = self.CMAP
        
        if self.verbose:
            print("[INFO] Filtering ROI...")

        Hroi, Wroi = im_dims
        Hdet, Wdet = det_dims
        N = len(address)   # list-per-pixel style
        assert Hroi * Wroi == N, f"ROI dims {im_dims} != number of pixels {N}"

        # Precompute flat mask for direct indexing with linear detector addresses
        mask_flat = np.asarray(mask, dtype=np.float64).ravel()
        det_size = Hdet * Wdet

        # Output (1D buffers; we’ll reshape at the end)
        raw_sums  = np.zeros(N, dtype=np.float64)
        filt_sums = np.zeros(N, dtype=np.float64)

        # Worker over a slice of pixel indices
        def _worker(start, end):
            rs = np.zeros(end - start, dtype=np.float64)
            fs = np.zeros(end - start, dtype=np.float64)
            for j, idx in enumerate(range(start, end)):
                a = np.asarray(address[idx], dtype=np.int64)
                c = np.asarray(counts[idx],  dtype=np.float64)
                if c.size == 0:
                    continue
                # guard invalid addresses (just in case)
                valid = (a >= 0) & (a < det_size)
                if not np.all(valid):
                    a = a[valid]; c = c[valid]
                    if c.size == 0:
                        continue
                rs[j] = c.sum()
                fs[j] = np.dot(c, mask_flat[a])
            return start, end, rs, fs

        # Submit chunks
        if n_workers is None:
            n_workers = min(32, (os.cpu_count() or 4))

        starts = list(range(0, N, chunk_size))
        ranges = [(s, min(s + chunk_size, N)) for s in starts]

        with ThreadPoolExecutor(max_workers=n_workers) as ex, tqdm(
                total=N, 
                desc="Reconstructing ROI", 
                unit="px", 
                dynamic_ncols=True) as pbar:
            futures = [ex.submit(_worker, s, e) for (s, e) in ranges]
            for fut in as_completed(futures):
                s, e, rs, fs = fut.result()
                raw_sums[s:e]  = rs
                filt_sums[s:e] = fs
                pbar.update(e - s)

        # Fill images (row-major mapping)
        rroi = raw_sums.reshape(Hroi, Wroi)
        froi = filt_sums.reshape(Hroi, Wroi)

        # optional: enforce the same color scale for both images
        if show:
            # vmin = float(min(rroi.min(), froi.min()))
            # vmax = float(max(rroi.max(), froi.max()))
        
            # --- No filtration ---
            fig, ax = plt.subplots(constrained_layout=True)
            im = ax.imshow(rroi, cmap=cmap, origin="lower")#, vmin=vmin, vmax=vmax)
            ax.set_title("ROI (no filtration)")
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("intensity", rotation=90)
        
            # --- Filtration ---
            fig, ax = plt.subplots(constrained_layout=True)
            im = ax.imshow(froi, cmap=cmap, origin="lower")#, vmin=vmin, vmax=vmax)
            ax.set_title("ROI (filtration)")
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("intensity", rotation=90)
        
            plt.show()
        
        self.froi = froi
        return 
    
    
    def filter_centered_ROI(self,
                    lower=0.01, upper=0.20,
                    sigma=None,                  
                    ema_alpha=None,
                    n_workers=None,
                    chunk_size=4096):
        """
        Per-pixel masked sum using a *per-frame centered annulus* without
        reconstructing full 2-D patterns. Fast path: operate directly on sparse events.
    
        The weight for an event at radius r is:
          - hard ring: 1 if r_in <= r <= r_out else 0
          - soft ring: 0.5*(erf((r-r_in)/(sqrt(2)*sigma)) - erf((r-r_out)/(sqrt(2)*sigma)))
    
        Parameters
        ----------
        lower, upper : float
            Inner/outer radii as fractions of (detector_width/2).
        sigma : float or None
            Soft edge width in *pixels*. None or 0 -> hard (binary) annulus.
        ema_alpha : float or None
            If set (e.g. 0.10–0.20), center-of-mass is smoothed frame-to-frame
            via exponential moving average to reduce jitter in sparse frames.
        n_workers : int or None
            Thread count (default: min(32, os.cpu_count() or 4)).
        chunk_size : int
            Number of ROI pixels per worker task.
    
        Returns
        -------
        froi : (Hroi, Wroi) float64
            Filtered (masked) sums.
        rroi : (Hroi, Wroi) float64
            Unfiltered sums.
        """
        from scipy.special import erf  # vectorized error function

        # Pull inputs from the instance
        address = self.p_address         # list-of-1D arrays per ROI pixel
        counts  = self.p_count           # list-of-1D arrays per ROI pixel
        Hroi, Wroi = self.NEW_DIMS
        Hdet, Wdet = self.DET_DIMS
        cmap = self.CMAP
        show = self.show
    
        if self.verbose:
            print("[INFO] Filtering ROI with per-frame centered annulus...")
    
        # Basic sanity
        N = len(address)
        assert Hroi * Wroi == N, f"ROI dims {Hroi,Wroi} != number of pixels {N}"
    
        # Radii in pixels (use detector width as in your original mask helper)
        r_in  = 0.5 * Wdet * float(lower)
        r_out = 0.5 * Wdet * float(upper)
        soft  = (sigma is not None) and (float(sigma) > 0)
        if soft:
            sigma = float(sigma)
            inv_s = 1.0 / (np.sqrt(2.0) * sigma)
    
        # Output buffers (1-D; reshape at the end)
        raw_sums  = np.zeros(N, dtype=np.float64)
        filt_sums = np.zeros(N, dtype=np.float64)
    
        # Optional EMA center (start from geometric center)
        cx_run = (Wdet - 1) / 2.0
        cy_run = (Hdet - 1) / 2.0
        use_ema = (ema_alpha is not None) and (0.0 < float(ema_alpha) <= 1.0)
        if use_ema:
            ema_alpha = float(ema_alpha)
    
        det_size = Hdet * Wdet
    
        def _worker(start, end):
            rs = np.zeros(end - start, dtype=np.float64)
            fs = np.zeros(end - start, dtype=np.float64)
    
            # Use local copies of running center for each thread
            cx_loc, cy_loc = cx_run, cy_run
    
            for j, idx in enumerate(range(start, end)):
                a = np.asarray(address[idx], dtype=np.int64)
                c = np.asarray(counts[idx],  dtype=np.float64)
                if c.size == 0:
                    continue
    
                # guard invalid addresses
                valid = (a >= 0) & (a < det_size)
                if not np.all(valid):
                    a = a[valid]; c = c[valid]
                    if c.size == 0:
                        continue
    
                # event coords
                y = a // Wdet
                x = a %  Wdet
    
                # center-of-mass (counts-weighted)
                tot = c.sum()
                if tot > 0:
                    cx = (x * c).sum() / tot
                    cy = (y * c).sum() / tot
                    if use_ema:
                        cx_loc = (1.0 - ema_alpha) * cx_loc + ema_alpha * cx
                        cy_loc = (1.0 - ema_alpha) * cy_loc + ema_alpha * cy
                    else:
                        cx_loc, cy_loc = cx, cy
    
                # radial distances from (cx_loc, cy_loc)
                dx = x - cx_loc
                dy = y - cy_loc
                r  = np.sqrt(dx*dx + dy*dy)
    
                if soft:
                    # smooth ring via error function (fast, no 2-D mask needed)
                    # weight ~ 1 in [r_in, r_out], smooth edges of width ~sigma
                    w = 0.5 * (erf((r - r_in) * inv_s) - erf((r - r_out) * inv_s))
                else:
                    # hard binary annulus
                    w = ((r >= r_in) & (r <= r_out)).astype(np.float64)
    
                rs[j] = tot
                fs[j] = np.dot(c, w)
    
            return start, end, rs, fs
    
        # Thread pool over chunks
        if n_workers is None:
            n_workers = min(32, (os.cpu_count() or 4))
        ranges = [(s, min(s + chunk_size, N)) for s in range(0, N, chunk_size)]
    
        with ThreadPoolExecutor(max_workers=n_workers) as ex, tqdm(
                total=N, desc="Reconstructing ROI", unit="px", dynamic_ncols=True) as pbar:
            futures = [ex.submit(_worker, s, e) for (s, e) in ranges]
            for fut in as_completed(futures):
                s, e, rs, fs = fut.result()
                raw_sums[s:e]  = rs
                filt_sums[s:e] = fs
                pbar.update(e - s)
    
        # Images
        rroi = raw_sums.reshape(Hroi, Wroi)
        froi = filt_sums.reshape(Hroi, Wroi)
    
        if show:
            import matplotlib.pyplot as plt
            # No filtration
            fig, ax = plt.subplots(constrained_layout=True)
            im = ax.imshow(rroi, cmap=cmap, origin="lower")
            ax.set_title("ROI (no filtration)"); ax.axis("off")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("intensity", rotation=90)
    
            # Filtration
            fig, ax = plt.subplots(constrained_layout=True)
            im = ax.imshow(froi, cmap=cmap, origin="lower")
            ax.set_title("ROI (filtration, centered annulus)"); ax.axis("off")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("intensity", rotation=90)
            plt.show()
    
        self.froi = froi
        return 