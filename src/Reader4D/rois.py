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
import os, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import Reader4D.convertor as r4dConv
import sys
from scipy.special import erf



class Extractor:
    def __init__(self,
            micrograph,
            descriptors,
            packets,
            values_role,
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
        self.values_role = values_role
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
        
        if verbose:
            print("[DONE : Region of interest extraction]")
        
        
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
        Reconstruct a Region of Interest (ROI) from binned `descriptors` and
        `packets` and build a quick “sum-of-counts” micrograph for that ROI. 
        In addition, gather per-pixel packet subsets and detector coordinates 
        for downstream use.
    
        Overview
        --------
        Given inclusive ROI bounds `self.coords` in image (scan) coordinates 
        and full scan/detector dimensions:
          - `self.SCAN_DIMS = (width, height)`
          - `self.DET_DIMS  = (dwidth, dheight)`
    
        the function:
          1) Generates linear scan indices in row-major order:
             `idx = y * width + x` for all (x, y) in the ROI (inclusive bounds)
          2) For each ROI pixel, slices packets via:
             `start = descriptors[idx]['offset']`,
             `count = descriptors[idx]['packet_count']`,
             and accumulates:
               • per-pixel sum of `packets[self.values_role]` 
                 (for a 2-D ROI image),
               • per-pixel lists of detector `address` and `values_role` 
                 (“counts”),
               • per-packet detector coordinates (dx, dy) decoded as:
                 `dy = address // dwidth`, `dx = address % dwidth`.
          3) Builds:
               • `dsubset`: `descriptors[linear_indices]` (ROI order),
               • `psubset`: a single contiguous slice/view of all ROI packets,
                 constructed by concatenating respective ROI pixels.
    
        Parameters
        ----------
        show : bool, optional (default: False)
            If True, displays the quick reconstruction (sum of counts) for the 
            ROI using `matplotlib` with `origin='lower'`.
    
        Side Effects (attributes set on `self`)
        ---------------------------------------
        roi : np.ndarray, shape (rheight, rwidth), float64
            Sum-of-counts image over the ROI, where
            rwidth  = x_end - x_start + 1,
            rheight = y_end - y_start + 1.
        psubset : np.ndarray (structured)
            Concatenated view/array of all packets belonging to the ROI.
        dsubset : np.ndarray (structured)
            Descriptors for ROI pixels in ROI (scan) order.
        d_offset : list[int]
            Per-ROI-pixel list of packet start offsets into `packets`.
        d_packet_count : list[int]
            Per-ROI-pixel list of packet counts.
        p_address : list[np.ndarray]
            For each ROI pixel, a 1-D array of detector linear addresses.
        p_count : list[np.ndarray]
            For each ROI pixel, a 1-D array of packet values aligned with 
            `p_address`, specifically `packets[self.values_role]`.
        dcoords : list[list[np.ndarray, np.ndarray]]
            For each ROI pixel, `[dx, dy]` where `dx`, `dy` are 1-D arrays of
            detector x/y coordinates decoded from `p_address`.
    
        Notes
        -----
        • ROI bounds in `self.coords` are treated as **inclusive** on both axes
        • Linear indices are computed with row-major layout using the full scan
          width: `idx = y * width + x` (no serpentine/“boustrophedon” handling)
        • The detector width (`dwidth`) is used for address decoding:
          `dy = address // dwidth`, `dx = address % dwidth`.
        • This function assumes `len(descriptors) == width * height` and that
          each descriptor’s (`offset`, `packet_count`) pair correctly indexes
          into `packets`.
    
        Returns
        -------
        None
            Results are stored on `self` (see “Side Effects” above).
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
        linear_indices = \
            (y_coords.ravel() * width + x_coords.ravel()).astype(int)

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
            pc = p[self.values_role]
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
                    chunk_size=4096,                # pixels per worker chunk
                    save_im=True):
        """
        Apply a fixed detector-space mask to a Region of Interest (ROI) and 
        compute per-pixel masked sums without reconstructing full 2D diff.
        frames. Processing is parallelized over pixel chunks to reduce Python 
        overhead.
       
        Overview
        --------
        Each ROI pixel is stored as sparse detector events:
            - `self.p_address[i]`: 1-D array of detector linear addresses for
              pixel *i*
            - `self.p_count[i]`  : corresponding weights (e.g. hit counts)
       
       Given a detector-space mask `mask` (shape = `self.DET_DIMS`), 
       the function computes for each ROI pixel:
           raw_sum[i]  = sum(self.p_count[i])
           filt_sum[i] = sum(self.p_count[i] * mask_flat[self.p_address[i]])
       
       Results are accumulated directly from sparse lists, avoiding the 
       overhead of allocating a `(Hdet, Wdet)` array per pixel.
       
       Parameters
       ----------
       mask : ndarray, shape (Hdet, Wdet)
           Detector-space weighting mask (binary or float). Applied identically 
           to all ROI pixels.
       n_workers : int or None, default None
           Number of worker threads for parallel processing. Defaults to
           `min(32, os.cpu_count() or 4)`.
       chunk_size : int, default 4096
           Number of ROI pixels per worker task. Larger chunks reduce Python
           overhead; smaller chunks improve load balancing.
       save_im : bool, default True
           If True, save the filtered ROI image (`froi`) as an 8-bit PNG into
           `self.out_dir`.
       
       Side Effects (attributes set on `self`)
       ---------------------------------------
       froi : np.ndarray, shape (Hroi, Wroi), float64
           Filtered (masked) ROI image.
       
       Notes
       -----
       • The ROI dimensions are `self.NEW_DIMS = (Hroi, Wroi)`.  
       • The detector grid is `self.DET_DIMS = (Hdet, Wdet)`; linear addresses 
         are validated against `[0, Hdet*Wdet)`.  
       • The `show` and `cmap` attributes of `self` control interactive display
         of the unfiltered vs. filtered ROI images.
       
       Returns
       -------
       None
       
       Results are stored on `self` (see “Side Effects” above).
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

        # Precompute flat mask for direct indexing with linear detector address
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
            im = ax.imshow(rroi, cmap=cmap, origin="lower")
            ax.set_title("ROI (no filtration)")
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("intensity", rotation=90)
        
            # --- Filtration ---
            fig, ax = plt.subplots(constrained_layout=True)
            im = ax.imshow(froi, cmap=cmap, origin="lower")
            ax.set_title("ROI (filtration)")
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("intensity", rotation=90)
        
            plt.show()
        
        self.froi = froi
        
        if save_im:
            base = os.path.splitext(self.name_roi)[0]
            dims_tag = f"{int(self.NEW_DIMS[0])}x{int(self.NEW_DIMS[1])}"
            png_path = os.path.join(
                self.out_dir, f"filt_{dims_tag}_{base}.png")
            roi = np.asarray(self.froi)
            
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
                
        return 
    
    
    def filter_centered_ROI(self,
                        lower=0.01, upper=0.20,
                        sigma=None,
                        ema_alpha=None,
                        n_workers=None,
                        chunk_size=4096,
                        save_im=True,
                        *,
                        normalize=True,         # per-frame dose normalization
                        complement=False,       # use outside-of-mask
                        invert_display=False,   # flip intensities [visualize]
                        trim_pct=0.0            # robust COM; 
                                                # e.g. 0.5 trims top 0.5% 
                        ):
        """
        Apply a **per-frame centered annular/disk mask** to an ROI using sparse
        detector events, computing masked sums without reconstructing full 2-D
        diffraction frames. The annulus is re-centered per frame via center of
        mass (COM), optionally smoothed with an exponential moving average.
    
        Overview
        --------
        For each ROI pixel (sparse events from `self.p_address[i]`,
        `self.p_count[i]`):
          1) Compute a robust center-of-mass from detector coordinates, 
             optionally trimming the top `trim_pct` of intensities to reduce 
             outlier influence.
          2) Form a disk or annulus with inner radius `lower * (Wdet/2)` and 
             outer radius `upper * (Wdet/2)` around the COM (soft edges if 
             `sigma` > 0).
          3) Accumulate:
               • raw_sum  = total intensity
               • filt_sum = inside annulus (or outside, if `complement=True`)
          4) Optionally normalize per frame by dividing `filt_sum` by `raw_sum`
    
        Parameters
        ----------
        lower, upper : float
            Inner and outer radii of the annulus, expressed as fractions of the
            detector half-width. (`lower=0.0, upper=1.0` → full disk).
        sigma : float or None, default None
            If given, apply Gaussian-softened edges to the annulus with width
            `sigma` pixels. If None, a hard-edged annulus is used.
        ema_alpha : float or None, default None
            If in (0,1], apply exponential moving average to COM positions with
            smoothing factor `ema_alpha`.
        n_workers : int or None, default None
            Number of worker threads. 
            Defaults to `min(32, os.cpu_count() or 4)`.
        chunk_size : int, default 4096
            Number of ROI pixels per worker task. Larger chunks amortize 
            overhead.
        save_im : bool, default True
            If True, save the filtered ROI image (`froi`) as an 8-bit PNG.
        normalize : bool, default True
            If True, store masked_fraction = inside_sum / total_sum.
            If False, store absolute intensities.
        complement : bool, default False
            If True, use the outside of the annulus (total − inside).
        invert_display : bool, default False
            If True, invert intensities for display only 
            (saved raw values unaffected).
        trim_pct : float, default 0.0
            Robustify COM calculation by clipping the top `trim_pct` fraction 
            of weights. Example: `0.5` → use values ≤ 99.5th percentile for COM
    
        Side Effects (attributes set on `self`)
        ---------------------------------------
        froi : np.ndarray, shape (Hroi, Wroi), float64
            Filtered ROI image (raw, not inverted).
    
        Notes
        -----
        • Input data are sparse detector events 
          (`self.p_address`, `self.p_count`) stored per pixel in ROI order.  
        • Detector grid is `self.DET_DIMS = (Hdet, Wdet)`; addresses validated
          against `[0, Hdet*Wdet)`.  
        • ROI shape is `self.NEW_DIMS = (Hroi, Wroi)`.  
        • Saved PNG is automatically labeled as `BF`, `DF`, or `HAADF` 
          depending on chosen radii.
    
        Returns
        -------
        None
            Results are stored on `self` (see “Side Effects” above).
        """
    
        address = self.p_address
        counts  = self.p_count
        Hroi, Wroi = self.NEW_DIMS
        Hdet, Wdet = self.DET_DIMS
        cmap = self.CMAP
        show = self.show
    
        if self.verbose:
            print("[INFO] Filtering ROI with per-frame centered annulus...")
    
        N = len(address)
        assert Hroi * Wroi == N, f"ROI dims {Hroi,Wroi} != number of pixels {N}"
    
        # Radii in pixels (based on detector width like your original helper)
        r_in  = 0.5 * Wdet * float(lower)
        r_out = 0.5 * Wdet * float(upper)
        soft  = (sigma is not None) and (float(sigma) > 0)
        if soft:
            sigma = float(sigma)
            inv_s = 1.0 / (np.sqrt(2.0) * sigma)
        
        # unmasked total
        raw_sums  = np.zeros(N, dtype=np.float64) 
        
        # masked (or outside) sum (possibly normalized)
        filt_sums = np.zeros(N, dtype=np.float64) 
    
        # Optional EMA center starting at geometric center
        cx_run = (Wdet - 1) / 2.0
        cy_run = (Hdet - 1) / 2.0
        use_ema = (ema_alpha is not None) and (0.0 < float(ema_alpha) <= 1.0)
        if use_ema:
            ema_alpha = float(ema_alpha)
    
        det_size = Hdet * Wdet
        eps = 1e-12  # to avoid division by zero
    
        def _worker(start, end):
            rs = np.zeros(end - start, dtype=np.float64)
            fs = np.zeros(end - start, dtype=np.float64)
            cx_loc, cy_loc = cx_run, cy_run  # local copies per-thread
    
            for j, idx in enumerate(range(start, end)):
                a = np.asarray(address[idx], dtype=np.int64)
                c = np.asarray(counts[idx],  dtype=np.float64)
                if c.size == 0:
                    continue
    
                # keep valid detector addresses
                valid = (a >= 0) & (a < det_size)
                if not np.all(valid):
                    a = a[valid]; c = c[valid]
                    if c.size == 0:
                        continue
    
                y = a // Wdet
                x = a %  Wdet
    
                # robust center-of-mass (optional trimming)
                c_for_com = c
                if trim_pct and trim_pct > 0:
                    # cap very bright outliers for COM computation
                    hi = np.percentile(c, 100.0 * (1.0 - float(trim_pct)))
                    # use clipped weights only for center location 
                    # (not for intensity)
                    c_for_com = np.minimum(c, hi)
    
                tot = c.sum()
                tot_com = c_for_com.sum()
                if tot_com > 0:
                    cx = (x * c_for_com).sum() / tot_com
                    cy = (y * c_for_com).sum() / tot_com
                    if use_ema:
                        cx_loc = (1.0 - ema_alpha) * cx_loc + ema_alpha * cx
                        cy_loc = (1.0 - ema_alpha) * cy_loc + ema_alpha * cy
                    else:
                        cx_loc, cy_loc = cx, cy
    
                # radial distances
                dx = x - cx_loc
                dy = y - cy_loc
                r  = np.sqrt(dx*dx + dy*dy)
    
                if soft:
                    w_inside = 0.5*(erf((r-r_in)*inv_s)-erf((r-r_out)*inv_s))
                else:
                    w_inside = ((r >= r_in) & (r <= r_out)).astype(np.float64)
    
                inside_sum = np.dot(c, w_inside)
                masked_sum = inside_sum
                if complement:
                    # outside of the ring/disk
                    masked_sum = tot - inside_sum   
    
                # dose-normalization per frame
                if normalize:
                    masked_sum = masked_sum / max(tot, eps)
    
                rs[j] = tot
                fs[j] = masked_sum
    
            return start, end, rs, fs
    
        # Thread pool over chunks
        if n_workers is None:
            n_workers = min(32, (os.cpu_count() or 4))
        ranges = [(s, min(s + chunk_size, N)) for s in range(0, N, chunk_size)]
    
        with ThreadPoolExecutor(max_workers=n_workers) as ex, tqdm(
            total=N, desc="Reconstructing ROI", unit="px", dynamic_ncols=True
        ) as pbar:
            futures = [ex.submit(_worker, s, e) for (s, e) in ranges]
            for fut in as_completed(futures):
                s, e, rs, fs = fut.result()
                raw_sums[s:e]  = rs
                filt_sums[s:e] = fs
                pbar.update(e - s)
    
        rroi = raw_sums.reshape(Hroi, Wroi)
        froi = filt_sums.reshape(Hroi, Wroi)
    
        # optional visual inversion for display only
        froi_vis = (np.max(froi) - froi) if invert_display else froi
    
        if show:
            fig, ax = plt.subplots(constrained_layout=True)
            im = ax.imshow(rroi, cmap=cmap, origin="lower")
            ax.set_title("ROI (no filtration)"); ax.axis("off")
            cb = fig.colorbar(im, 
                              ax=ax, 
                              fraction=0.046, 
                              pad=0.04) 
            cb.set_label("intensity", rotation=90)
    
            fig, ax = plt.subplots(constrained_layout=True)
            im = ax.imshow(froi_vis, cmap=cmap, origin="lower")
            ax.set_title("ROI (filtration, centered annulus)"); ax.axis("off")
            cb = fig.colorbar(im,
                              ax=ax, 
                              fraction=0.046, 
                              pad=0.04)
            cb.set_label("intensity", rotation=90)
            plt.show()
    
        self.froi = froi  # keep the raw (non-inverted) values
    
        if save_im:
            # suffix for filename
            if lower == 0.0:
                suffix = "BF"
            elif upper <= 1.0:
                suffix = "DF"
            else:
                suffix = "HADF"
            base = os.path.splitext(self.name_roi)[0]
            dims_tag = f"{int(self.NEW_DIMS[0])}x{int(self.NEW_DIMS[1])}"
            png_path = os.path.join(
                self.out_dir, f"filtC_{dims_tag}_{base}_{suffix}.png")
    
            # Save an 8-bit preview of froi_vis
            img = froi_vis
            rmin, rmax = float(np.nanmin(img)), float(np.nanmax(img))
            if rmax > rmin:
                img8 = np.clip(
                    (img-rmin)/(rmax-rmin)*255,0,255).astype(np.uint8)
            else:
                img8 = np.zeros_like(img, dtype=np.uint8)
            Image.fromarray(img8).save(png_path)
    
        return



    def _filter_centered_ROI(self,
                    lower=0.01, upper=0.20,
                    sigma=None,                  
                    ema_alpha=None,
                    n_workers=None,
                    chunk_size=4096,
                    save_im=True):
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
        
        if save_im:
            if lower==0.00:
                suffix = "BF"
            elif (lower>0 and upper < 0.5):
                suffix = "DF"
            elif (lower>0 and upper > 0.5):
                suffix = "HADF"
            else: suffix = "XX"
            
            base = os.path.splitext(self.name_roi)[0]
            dims_tag = f"{int(self.NEW_DIMS[0])}x{int(self.NEW_DIMS[1])}"
            png_path = os.path.join(
                self.out_dir, f"filtC_{dims_tag}_{base}_{suffix}.png")
            roi = np.asarray(self.froi)
            
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
            
            
        return 
    
    
    def pick_and_show_diffraction(self,
                              serpentine=False,
                              cmap_diff="viridis",
                              title_prefix="Frame",
                              max_points=None,
                              save_all=True):
        """
        Show micrograph, let the user click multiple pixels (finish with Enter),
        then show the selected points overlaid on the micrograph AND a grid of
        their diffractograms.
    
        Returns
        -------
        idx_list : list[int]
            Linear frame indices (row-major, accounting for serpentine if
                                  enabled).
        xy_list  : list[tuple[int,int]]
            Pixel coordinates (x, y) in display coordinates (0..W-1, 0..H-1).
        frames   : list[np.ndarray]
            Each is (Hdet, Wdet) reconstructed diffractogram.
        """
        packets       = self.psubset
        descriptors   = self.dsubset
        scan_dims     = self.NEW_DIMS    # (H, W)
        det_dims      = self.DET_DIMS    # (Wdet, Hdet)
        values_role   = self.values_role
        cmap_img      = self.CMAP
    
        H, W = map(int, scan_dims)
        Wdet, Hdet = map(int, det_dims)
    
        # packet counts & offsets
        pc = np.asarray(descriptors["packet_count"],dtype=np.int64).reshape(-1)
        n_frames = pc.size
        
        s = np.copy(scan_dims)
        if n_frames != W * H:
            n = np.copy(n_frames)
            raise ValueError(
                f"scan_dims {s} imply {W*H} frames, got {n} descriptors.")
    
        off = np.empty_like(pc, dtype=np.int64)
        if pc.size:
            off[0] = 0
            if pc.size > 1:
                np.cumsum(pc[:-1], out=off[1:])
    
        if off[-1] + pc[-1] > packets.shape[0]:
            raise ValueError(
                "Descriptors reference more packets than available.")
    
        # micrograph (fast)
        vals = np.asarray(packets[values_role], dtype=np.float64)
        
        # Sum per frame safely: only use frames with pc > 0 for reduceat
        mask = pc > 0
        micro_sums = np.zeros(n_frames, dtype=np.float64)
        
        if mask.any():
            # off[mask] are valid start positions < len(vals)
            micro_sums[mask] = np.add.reduceat(vals, off[mask])
        
        # Reshape to 2D micrograph
        micro = micro_sums.reshape(H, W)
    
        # visualization image (optionally serpentine-flipped for display)
        if serpentine:
            micro_vis = micro.copy()
            micro_vis[1::2, :] = micro_vis[1::2, ::-1]
        else:
            micro_vis = micro
    
        # interactive picking (multiple points; finish with Enter)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(micro_vis, origin="lower", cmap=cmap_img)
        ax.set_title("Click multiple pixels; press Enter to finish")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
    
        picked_xy = []          # (x,y) display coordinates
        annots = []
    
        def on_click(event):
            if event.inaxes is not ax:
                return
            if event.button is None:  
                # keyboard event passes here sometimes
                return
            
            # convert to nearest integer pixel
            x = int(np.clip(round(event.xdata), 0, W - 1))
            y = int(np.clip(round(event.ydata), 0, H - 1))
            picked_xy.append((x, y))
            
            # draw a marker+index
            idx_lbl = len(picked_xy)
            m = ax.plot(
                x, y, marker='o', ms=6, mfc='none', mec='r', mew=1.5)[0]
            t = ax.text(x + 2, y + 2, str(idx_lbl), color='yellow', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.15", 
                                  fc="black", 
                                  alpha=0.4))
            annots.append((m, t))
            fig.canvas.draw_idle()
            
            # respect optional max_points
            if (max_points is not None) and (len(picked_xy) >= max_points):
                plt.close(fig)
    
        def on_key(event):
            if event.key in ("enter", "return"):
                plt.close(fig)
    
        cid1 = fig.canvas.mpl_connect('button_press_event', on_click)
        cid2 = fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=True)   # wait until closed by Enter or max_points
    
        # disconnect
        try:
            fig.canvas.mpl_disconnect(cid1)
            fig.canvas.mpl_disconnect(cid2)
        except Exception:
            pass
    
        if not picked_xy:
            raise RuntimeError("No points selected.")
    
        # Map picked (x,y) to linear indices, accounting for serpentine
        idx_list = []
        xy_list  = []
        for (x, y) in picked_xy:
            if serpentine and (y % 2 == 1):   
                # even rows (0-based) scanned R->L
                x_scan = W - 1 - x
            else:
                x_scan = x
            idx = y * W + x_scan
            idx_list.append(int(idx))
            xy_list.append((int(x), int(y)))
    
        # Reconstruct diffractograms for each picked idx
        frames = []
        for i, idx in enumerate(idx_list):
            s = int(off[idx]); e = s + int(pc[idx])
            frame_pkts = packets[s:e]
            img = np.zeros((Hdet, Wdet), dtype=np.float64)
            if e > s:
                addr = frame_pkts["address"].astype(np.int64, copy=False)
                w    = frame_pkts[values_role].astype(np.float64, copy=False)
                det_size = Wdet * Hdet
                valid = (addr >= 0) & (addr < det_size)
                if not np.all(valid):
                    addr = addr[valid]; w = w[valid]
                np.add.at(img.ravel(), addr, w)
            frames.append(img)
    
        # Show: 1) micrograph with picked points (already drawn), 
        #       2) grid of diffractograms
        # re-show micrograph with markers (static)
        fig_m, ax_m = plt.subplots(figsize=(12, 10))
        im_m = ax_m.imshow(micro_vis, origin="lower", cmap=cmap_img)
        ax_m.set_title(f"Selected points (n={len(xy_list)})")
        plt.colorbar(im_m, ax=ax_m, fraction=0.046, pad=0.04)
        for k, (x, y) in enumerate(picked_xy, start=1):
            ax_m.plot(x, y, 'o', ms=6, mfc='none', mec='r', mew=1.5)
            ax_m.text(x + 2, y + 2, str(k), color='yellow', fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.15", 
                                fc="black", 
                                alpha=0.4))
        plt.tight_layout()
        plt.show(block=False)
    
        # grid for diffractograms
        M = len(frames)
        ncols = min(2, M)
        nrows = int(np.ceil(M / ncols))
        fig_d, axes = plt.subplots(nrows, ncols, 
                                   figsize=(3.5*ncols, 3.2*nrows),
                                   squeeze=False)
        for k, ax in enumerate(axes.ravel()):
            if k < M:
                img = frames[k]
                (x, y) = xy_list[k]
                idx = idx_list[k]
                h = ax.imshow(img, origin="lower", cmap=cmap_diff)
                ax.set_title(
                    f"{title_prefix} #{k+1}\n(x={x}, y={y}, idx={idx})", 
                    fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis('off')
        plt.tight_layout()
        plt.show(block=False)
        
        if save_all:
            # decide output folder
            out_dir = getattr(self, "out_dir", None) or os.getcwd()
            os.makedirs(out_dir, exist_ok=True)
    
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            micro_name = f"_micro_{H}x{W}_{ts}.png"
            diffs_name = f"_diffs_{M}_{Wdet}x{Hdet}_{ts}.png"
            micro_path = os.path.join(out_dir, micro_name)
            diffs_path = os.path.join(out_dir, diffs_name)
    
            # save figures
            fig_m.savefig(micro_path, dpi=200, bbox_inches="tight")
            fig_d.savefig(diffs_path, dpi=200, bbox_inches="tight")
    
        return idx_list, xy_list, frames
        