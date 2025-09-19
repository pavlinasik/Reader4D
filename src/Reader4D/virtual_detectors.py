# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:10:50 2025

@author: p-sik
"""
import numpy as np
from scipy.ndimage import gaussian_filter, shift
import matplotlib.pyplot as plt
from skimage.draw import disk
import ediff as ed
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.special import erf
from concurrent.futures import ThreadPoolExecutor, as_completed

import Reader4D.rois as r4dRoi


class Annular:
    def __init__(self,
                pattern,
                packets,
                descriptors,
                values_role,
                out_dir,
                header = None,
                scan_dims = (1024, 768),
                mode = 0,
                rLower = 0.01,
                rUpper = 0.20,
                sigma = None,
                xy = None,
                show = True,
                cmap = "gray",
                verbose = 1,
                save_im = True,
                name_im = None,
                ):
        
        ######################################################################
        # PRIVATE FUNCTION: Initialize CSRTriplet object.
        # The parameters are described above in class definition.
        ######################################################################
        
        ## Initialize input attributes ---------------------------------------
        self.pattern = pattern
        self.packets = packets
        self.descriptors = descriptors
        self.values_role = values_role
        self.out_dir = out_dir
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
    
        self.header = header
        self.DET_DIMS = pattern.shape
        
        if self.header is not None:
            self.SCAN_DIMS = self.header['nav_shape'][::-1]
        else: 
            self.SCAN_DIMS = scan_dims
            
        self.mode = mode
        self.rLOWER = rLower
        self.rUPPER = rUpper
        self.SIGMA = sigma
        self.show = show
        self.CMAP = cmap
        self.verbose = verbose
        self.save_im = save_im
        self.name_im = name_im
        
        if xy is not None:
            self.X = xy[0]
            self.Y = xy[1]
        else:
            if self.verbose:
                print("[INFO] Estimating center of diffractograms...")
                
            self.X, self.Y = self.get_center(
                self.pattern,
                scan_width=self.SCAN_DIMS[1],
                scan_height=self.SCAN_DIMS[0],
                method='curvefit'  
            )
                
        ## Create annular detector --------------------------------------------
        if self.verbose:
            print("[INFO] Creating annular mask...")
        
        if self.mode==0:
            self.mask = self.create_mask(
                dim_x = self.DET_DIMS[0], 
                dim_y = self.DET_DIMS[1],
                lower = self.rLOWER, 
                upper = self.rUPPER,
                coord_x = self.X,
                coord_y = self.Y,
                show = self.show, 
                blur_sigma=self.SIGMA
                )
            
            # Visualize mask application on a single diffractogram
            if self.show:
                if self.verbose:
                    print("[INFO] Showing mask effect on a sample pattern...")
                    
                self.show_filtration(self.pattern, self.mask, "viridis")
    
            # Generate filtered micrograph
            if self.verbose:
                print("[INFO] Applying mask to all diffractograms...")
                
            fimg, fmask = self.get_filtered_im(
                packets=self.packets,
                descriptors=self.descriptors,
                mask=self.mask,
                scan_dims=self.SCAN_DIMS,
                detector_dims=self.DET_DIMS,
                show=True,
                cmap=self.CMAP
            )
        
        elif self.mode==1:
            fimg, fmask = self.get_filtered_im2(
                self.packets, 
                self.descriptors, 
                self.rLOWER, 
                self.rUPPER, 
                self.SIGMA,
                self.SCAN_DIMS, 
                self.DET_DIMS, 
                self.show, 
                self.CMAP
            )
        
        elif self.mode==2:
            
            fimg = self.filter_centered_ROI(
                lower=self.rLOWER, 
                upper=self.rUPPER,
                sigma=self.SIGMA)
        
        if self.save_im:
            name = f"{self.name_im}.png"
            plt.imsave(os.path.join(
                self.out_dir, name), 
                fimg, 
                cmap=self.CMAP
                )
            if self.verbose:
                print(f"[INFO] Saved filtered micrograph to {self.out_dir}")
            

        if verbose:
            print("[DONE : Application of virtual detectors]")
    
    
    def create_mask(self, dim_x=256, dim_y=256, lower=0.01, upper=0.20,
                    coord_x=None, coord_y=None, show=True, blur_sigma=None):
        """
        Create a mask with a soft (blurred) ring (annulus).

        Parameters
        ----------
        dim_x : int, optional
            Width of the output mask (number of columns). Default is 256.
        dim_y : int, optional
            Height of the output mask (number of rows). Default is 256.
        lower : float, optional
            Relative inner radius of the ring, as a fraction of dim_x/2. 
            Default is 0.01.
        upper : float, optional
            Relative outer radius of the ring, as a fraction of dim_x/2. 
            Default is 0.20.
        coord_x : float or int, optional
            X-coordinate (horizontal) of the ring center. 
            If None, defaults to image center.
        coord_y : float or int, optional
            Y-coordinate (vertical) of the ring center. 
            If None, defaults to image center.
        blur_sigma : float, optional
            The standard deviation (sigma) for the Gaussian kernel to control
            the blurriness of the mask edges. If None, a sharp binary mask is
            returned. Default is None.

        Returns
        -------
        mask : ndarray
            2D NumPy array with a soft-edged ring. Values range from 0 to 1.
        """
        # Set center if not provided
        coord_x = dim_x / 2 if coord_x is None else coord_x
        coord_y = dim_y / 2 if coord_y is None else coord_y

        # Preallocate mask
        mask = np.ones((dim_y, dim_x), dtype=np.float32)

        # Fully open mask if lower > upper
        if lower > upper:
            pass  # mask remains all 1s

        # Fully blocked center if upper > 1 (acts as full exclusion zone)
        elif upper > 1.0:
            r_inner = (dim_x / 2) * lower
            mask = self.create_circle(
                mask, 
                (coord_x, coord_y), 
                r_inner, 
                fill=0)
        
        elif lower == 0.0:
            r_outer = (dim_x / 2) * upper
            mask[:] = 0
            mask = self.create_circle(
                mask, 
                (coord_x, coord_y), 
                r_outer, 
                fill=1)

            
        # Create annular region
        else:
            r_outer = (dim_x / 2) * upper
            r_inner = (dim_x / 2) * lower

            mask[:] = 0
            mask = self.create_circle(
                mask, 
                (coord_x, coord_y), 
                r_outer, 
                fill=1)
            mask = self.create_circle(
                mask,
                (coord_x, coord_y), 
                r_inner, 
                fill=0)

        # Apply blur
        if blur_sigma:
            mask = gaussian_filter(mask, sigma=blur_sigma)
            title = f"Blurred Mask (σ={blur_sigma}, {lower*100:.0f}-{upper*100:.0f}%)"
        else:
            title = f"Sharp Mask ({lower*100:.0f}-{upper*100:.0f}%)"

        # Display mask if requested
        if show:
            plt.figure()
            plt.imshow(mask, cmap='viridis', origin="lower")
            plt.title(title)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        
        return mask
    
   
    def create_master_mask(self,
                           detector_dims, lower, upper, blur_sigma=None, 
                           show=False):
        """
        Create a centered annular mask for diffraction 
        data.

        The mask is generated at the center of the detector based on the given 
        inner and outer radius fractions. Special cases handle fully open masks, 
        blocked centers, and filled circles. The mask can optionally be 
        Gaussian blurred for smoother edges.

        Parameters
        ----------
        detector_dims : tuple of int
            Dimensions (width, height) of the detector in pixels.
        lower : float
            Inner radius as a fraction of half the detector width.
            - 0.0 means no inner hole.
        upper : float
            Outer radius as a fraction of half the detector width.
            - >1.0 means the mask is open beyond the detector size.
        blur_sigma : float or None, optional
            Standard deviation for Gaussian blurring. None or 0 means no blur.
        show : bool, default=False
            If True, display the generated mask using matplotlib.

        Returns
        -------
        master_mask : numpy.ndarray
            2D float array of the generated mask, centered at the detector 
            center.

        Notes
        -----
        Mask types:
        - lower > upper: Fully open mask (all True).
        - upper > 1.0: Blocked center mask (exclude pixels inside r_inner).
        - lower == 0.0: Filled circle mask.
        - Otherwise: Annular mask (ring between r_inner and r_outer).

        The mask is returned as a float array so it can be directly multiplied
        with diffraction intensity arrays, and optionally blurred.
        """
        det_width, det_height = detector_dims
        binary_mask = np.zeros(detector_dims, dtype=bool)
        title = "Master Mask"

        if lower > upper:
            # 1. Fully open mask
            binary_mask[:] = True
            title = "Fully Open Mask"
        else:
            # Create coordinate grids and distance array for the other cases
            center_x = det_width / 2
            center_y = det_height / 2
            y, x = np.ogrid[-center_y:det_height-center_y, 
                            -center_x:det_width-center_x]
            dist_from_center = np.sqrt(x*x + y*y)
            
            # Calculate absolute radii
            r_outer = (det_width / 2) * upper
            r_inner = (det_width / 2) * lower

            if upper > 1.0:
                # 2. Blocked center (acts as full exclusion zone)
                binary_mask = dist_from_center >= r_inner
                title = f"Blocked Center Mask (r_inner={lower*100:.0f}%)"
            elif lower == 0.0:
                # 3. Filled circle
                binary_mask = dist_from_center <= r_outer
                title = f"Filled Circle Mask (r_outer={upper*100:.0f}%)"
            else:
                # 4. Standard annulus (ring)
                binary_mask = \
                    (dist_from_center>=r_inner) & (dist_from_center<=r_outer)
                title = f"Annular Mask ({lower*100:.0f}-{upper*100:.0f}%)"

        # Convert boolean mask to float for blurring
        master_mask = binary_mask.astype(np.float32)

        # Apply blur if requested
        if blur_sigma and blur_sigma > 0:
            master_mask = gaussian_filter(
                master_mask, 
                sigma=blur_sigma
                )
            title += f" (Blurred σ={blur_sigma})"

        # Display master mask if requested
        if show:
            plt.figure()
            plt.imshow(master_mask, cmap=self.cmap, origin="lower")
            plt.title(title)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        return master_mask
    
    
    def create_circle(self, blank, center, radius, fill):
        """
        Draw a filled circle on a zero background array.

        Parameters
        ----------
        blank : np.ndarray
            2D array where the circle will be drawn
        center : tuple
            (x, y) coordinates of the circle center
        radius : float or int
            Radius of the circle in pixels
        fill : float or int
            Value to fill inside the circle (1 for foreground, 0 to erase)

        Returns
        -------
        blank : np.ndarray
            Modified input array with the filled circle
        """
        # Create circular pattern
        rr, cc = disk(center[::-1], radius, shape=blank.shape)
        
        # Insert the circular pattern
        blank[rr, cc] = 1.0 * fill
        return blank
    
    
    def get_center(self, pattern, scan_width, scan_height, method="curvefit"):
        """
        Estimates the center of a diffraction pattern using a specified method.

        This function utilizes the CenterLocator class from the ed.center
        module to estimate the center coordinates (x, y) of the input 
        diffraction pattern.

        Parameters
        ----------
        pattern : np.ndarray
            2D array representing the diffraction pattern (typically an image).
        scan_width : int
            The width of the scan grid or image field, used for context or 
            scaling.
        scan_height : int
            The height of the scan grid or image field.
        method : str, optional
            Method used to estimate the center. Default is "curvefit".

        Returns
        -------
        x_center : float
            Estimated x-coordinate of the center in the image.
        y_center : float
            Estimated y-coordinate of the center in the image.

        """
        # Get center coordinates   
        center = ed.center.CenterLocator(
            input_image=pattern,
            determination=method,
            refinement=None,
            rtype=1,
            final_print=False,
            )
        
        return center.x1, center.y1 
    
    
    def show_filtration(self, pattern, mask, cmap): 
        """
        Displays the original and masked diffraction pattern side by side.

        This function applies a binary/weighted mask to a diffraction pattern 
        and visualizes both the original and the resulting filtered image.

        Parameters
        ----------
        pattern : np.ndarray
            2D array representing the original diffraction pattern.
        mask : np.ndarray
            2D array of the same shape as `pattern`, used to filter or weight 
            specific regions of the pattern.
        """      
        filt = pattern * mask

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original image
        axs[0].imshow(pattern, cmap=cmap) 
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        
        # Filtered image
        axs[1].imshow(filt, cmap=cmap)  
        axs[1].set_title("Filtered Image")
        axs[1].axis("off")
        
        plt.tight_layout()
        plt.show()
        
    
    def get_filtered_im(self, packets, descriptors, mask,
                        scan_dims=(1024, 768), detector_dims=(256, 256),
                        show=True, cmap="gray"):
        """
        Reconstruct a filtered micrograph by applying a detector-space mask to
        each diffraction pattern and summing masked counts per scan pixel.
        """
    
        # dims (note: NumPy arrays are (rows, cols) = (H, W))
        width, height = scan_dims                   # (W, H) as in your API
        det_w, det_h = detector_dims                # (Wdet, Hdet)
        det_size = det_w * det_h
    
        # ensure plain ndarrays (not h5py datasets)
        # ensure plain ndarrays (not h5py datasets)
        desc = np.asarray(descriptors)
        
        # pull fields and collapse any singleton dimensions to 1-D
        off_raw = np.asarray(desc["offset"])
        pc_raw  = np.asarray(desc["packet_count"])
        
        # squeeze/reshape to (N,), then cast to integers once
        off = off_raw.squeeze()
        pc  = pc_raw.squeeze()
        if off.ndim != 1:
            off = off.reshape(-1)
        if pc.ndim != 1:
            pc  = pc.reshape(-1)
        
        off = off.astype(np.int64, copy=False)
        pc  = pc.astype(np.int64,  copy=False)
        
        N = off.shape[0]
    
        # mask prep (vectorized) 
        mask = np.asarray(mask)
        assert mask.shape == \
            (det_h, det_w), f"mask shape {mask.shape} != {(det_h, det_w)}"
        mask_flat = mask.ravel()
    
        # output image 
        fimg = np.zeros((height, width), dtype=np.uint32)
    
        # main loop (index by k; pull scalar off/pc) 
        for k in tqdm(range(N), desc="Reconstructing Image"):
            s = int(off[k])
            e = s + int(pc[k])
            frame = packets[s:e]
            if frame.size:
                addr = frame["address"].astype(np.int64, copy=False)
                cnt  = frame[self.values_role].astype(np.float64, copy=False)
    
                # guard addresses
                valid = (addr >= 0) & (addr < det_size)
                if not np.all(valid):
                    addr = addr[valid]; cnt = cnt[valid]
                    if cnt.size == 0:
                        val = 0
                    else:
                        val = np.dot(cnt, mask_flat[addr])
                else:
                    val = np.dot(cnt, mask_flat[addr])
            else:
                val = 0
    
            y = k // width
            x = k %  width
            if y < height:
                fimg[y, x] = np.uint32(val)
    
        if show:
            self.show_micrograph(
                fimg, title="Filtered Micrograph", cmap=cmap,
                save=self.save_im, filename=self.name_im,
                output_dir=self.out_dir, show=self.show
            )
    
        return fimg, mask

    
    def get_filtered_im2(self, packets, descriptors,
                         lower_radius, upper_radius, sigma,
                         scan_dims=(1024, 768),
                         detector_dims=(256, 256),
                         show=False, cmap="gray",
                         batch_size=1000):
    
        """
        Reconstruct a filtered micrograph using a precomputed annular mask
        and batched parallel processing with per-frame centering.
        """
    
        # helpers 
        def get_center_fast(image):
            """Center of intensity (center of mass)."""
            total = image.sum()
            if total == 0:
                h, w = image.shape
                return w / 2.0, h / 2.0
            yy, xx = np.indices(image.shape)
            cy = (yy * image).sum() / total
            cx = (xx * image).sum() / total
            return float(cx), float(cy)
    
        def process_frame(args):
            """
            args = (frame_idx, packets_subset, master_mask, det_h, det_w)
            Builds pattern, centers, shifts mask, returns (frame_idx, sum).
            """
            frame_idx, packets_subset, master_mask, det_h, det_w = args
    
            # 1) reconstruct pattern (Hdet, Wdet)
            pattern = np.zeros((det_h, det_w), dtype=np.float32)
            if packets_subset.size > 0:
                addr = packets_subset["address"].astype(np.int64, copy=False)
                cnt  = packets_subset[self.values_role].astype(np.float32, 
                                                               copy=False)
                # guard bad addresses
                det_size = det_h * det_w
                valid = (addr >= 0) & (addr < det_size)
                if not np.all(valid):
                    addr = addr[valid]; cnt = cnt[valid]
                # linear add into raveled image
                np.add.at(pattern.ravel(), addr, cnt)
    
            # 2) find center and shift master mask
            cx, cy = get_center_fast(pattern)
            # master_mask is centered at (det_h/2, det_w/2)
            shift_y = cy - det_h / 2.0
            shift_x = cx - det_w / 2.0
            sh_mask = shift(master_mask, (shift_y, shift_x),
                            order=1, mode="nearest", prefilter=False)
    
            # 3) masked sum
            val = float((pattern * sh_mask).sum())
            return frame_idx, val
    
        # dims & outputs 
        W, H = map(int, scan_dims)                # (width, height)
        det_w, det_h = map(int, detector_dims)    # (Wdet, Hdet)
        fimg = np.zeros((H, W), dtype=np.float64) # floats (mask is fractional)
    
        # master mask 
        master_mask = self.create_master_mask(
            detector_dims=(det_h, det_w),  # mask uses (Hdet, Wdet)
            lower=lower_radius, upper=upper_radius,
            blur_sigma=sigma, show=False
        )
        if master_mask.shape != (det_h, det_w):
            raise ValueError(
                f"Mask shape {master_mask.shape} != {(det_h, det_w)}")
    
        # normalize descriptors to 1-D scalars
        desc = np.asarray(descriptors)
        names = getattr(desc.dtype, "names", None)
        if not names or ("packet_count" not in names):
            raise TypeError(
                f"Expected structured array with 'packet_count' field. "
                f"Got type={type(descriptors)}, dtype={getattr(desc, 'dtype', None)}"
            )
    
        # packet_count -> (N,)
        pc = np.asarray(desc["packet_count"]).squeeze()
        if pc.ndim != 1:
            pc = pc.reshape(-1)
        pc = pc.astype(np.int64, copy=False)
    
        # offset -> (N,), or reconstruct if absent
        if "offset" in names:
            off = np.asarray(desc["offset"]).squeeze()
            if off.ndim != 1:
                off = off.reshape(-1)
            off = off.astype(np.int64, copy=False)
        else:
            off = np.empty_like(pc, dtype=np.int64)
            off[0] = 0
            if pc.size > 1:
                np.cumsum(pc[:-1], out=off[1:])
    
        N = off.shape[0]
        if N != pc.shape[0]:
            raise ValueError("offset and packet_count length mismatch.")
    
        # batching
        if self.verbose:
            print("[INFO] Preparing batch processing...")
        tasks = []
        batch = []
        for i in range(N):
            s = int(off[i]); e = s + int(pc[i])
            pkt_sub = packets[s:e]
            batch.append((i, pkt_sub, master_mask, det_h, det_w))
            if len(batch) >= batch_size:
                tasks.append(batch); batch = []
        if batch:
            tasks.append(batch)
    
        # Parallel execution (threads) 
        # Threads avoid pickling large arrays (like the mask) for every task.    
        results = Parallel(n_jobs=-1, prefer="threads", batch_size=1)(
            delayed(lambda b: [process_frame(t) for t in b])(b)
            for b in tqdm(tasks, desc="Reconstructing Image", unit="batch")
        )
    
        # Assemble final image
        for batch_results in results:
            for frame_idx, val in batch_results:
                y = frame_idx // W
                x = frame_idx %  W
                if y < H:
                    fimg[y, x] = val
    
        # Optional display
        if show:
            self.show_micrograph(
                fimg, title="Filtered Micrograph", cmap=cmap,
                save=self.save_im, filename=self.name_im,
                output_dir=self.out_dir, show=self.show
            )
    
        return fimg, master_mask


    def filter_centered_ROI(self,
                        lower=0.01, upper=0.20,
                        sigma=None,
                        ema_alpha=None,
                        n_workers=None,
                        chunk_size=4096,
                        save_im=True,
                        *,
                        normalize=True,         # <-- NEW: per-frame dose normalization
                        complement=False,       # <-- NEW: use outside-of-mask (tot - masked)
                        invert_display=False,   # <-- NEW: flip intensities for visualization
                        trim_pct=0.0            # <-- NEW: robust COM; e.g. 0.5 trims top 0.5% counts
                        ):
        """
        Per-frame centered annulus/disk. Works directly on sparse events.
    
        NEW OPTIONS
        -----------
        normalize      : If True, use masked_fraction = masked_sum / total_sum.
                         This usually makes BF/DF/HAADF resemble conventional STEM.
        complement     : If True, use 'outside' of the mask: masked_sum = total_sum - inside_sum.
                         Useful if your mask is defined as an exclusion zone.
        invert_display : Visually invert the final image (no effect on saved raw values).
        trim_pct       : Robustify center-of-mass (COM) by trimming brightest events when computing COM.
                         Example: 0.5 keeps everything <= 99.5th percentile for COM calc.
        """
    
        descriptors = np.asarray(self.descriptors)
        packets     = np.asarray(self.packets)
        flatN       = descriptors.shape[0]

        # which packet value field to use
        values_role = getattr(self, "values_role", "count")
        
        # normalize descriptor fields to flat int arrays ----
        names = getattr(descriptors.dtype, "names", None)
        if not names or ("packet_count" not in names):
            raise TypeError(
                "descriptors must be a structured array with 'packet_count' field")
    
        pc = np.asarray(
            descriptors["packet_count"]
            ).reshape(-1).astype(np.int64, copy=False)
    
        if "offset" in names:
            off = np.asarray(
                descriptors["offset"]
                ).reshape(-1).astype(np.int64, copy=False)
        else:
            off = np.empty_like(pc, dtype=np.int64)
            off[0] = 0
            if pc.size > 1:
                np.cumsum(pc[:-1], out=off[1:])
    
        if off.shape[0] != pc.shape[0]:
            raise ValueError("offset and packet_count length mismatch")
    
        # prepare output containers ----
        address = [None] * flatN
        counts  = [None] * flatN
    
    
        # main loop (fast enough; packets are already contiguous per pixel) ----
        for i in range(flatN):
            s = int(off[i]); l = int(pc[i]); e = s + l
            if l <= 0:
                address[i] = np.empty(0, dtype=np.int64)
                counts[i]  = np.empty(0, dtype=np.float64)
                continue
    
            p  = packets[s:e]
            pa = p["address"].astype(np.int64, copy=False)
            vc = p[values_role].astype(np.float64, copy=False)
    
  
            address[i] = pa
            counts[i]  = vc
    
        #  make available on self ----
        self.p_address = address
        self.p_count   = counts
    
        
    
        Wroi, Hroi = self.SCAN_DIMS
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
    
    
        return froi
         
    
    def show_micrograph(self, 
                        img, 
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
                raise ValueError("If save=True, you must provide a filename.")
            if output_dir is None:
                output_dir = os.getcwd()
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename)
            plt.imsave(path, img, cmap=cmap)
            
            if self.verbose:
                print(f"[INFO] Saved: {path}")