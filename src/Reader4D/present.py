# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 14:48:24 2025

@author: p-sik
"""
import os, glob
import numpy as np
import cv2
import pandas as pd
import imageio
from PIL import Image, ImageDraw, ImageFont
import Virtual4D.virtDets as v4dVDet

class Creator:
    def __init__(self,
                 pattern,
                 packets,
                 descriptors,
                 values_role,
                 out_dir,
                 virtdet=None,
                 scan_dims = (1024, 768),
                 rLower = 0.01,
                 rUpper = 0.20,
                 sigma = None,
                 show = True,
                 cmap = "gray",
                 verbose = 1,
                 save_im = True,
                 name_im = None,
                 ):
        
        self.pattern = pattern
        self.packets = packets
        self.descriptors = descriptors
        self.values_role = values_role

        if isinstance(out_dir, tuple):
            if len(out_dir) == 1:
                out_dir = out_dir[0]
            else:
                raise TypeError(f"out_dir should be str, not tuple: {out_dir}")
        self.OUT_DIR = str(out_dir)

        self.SCAN_DIMS = scan_dims
        self.LOWER = rLower
        self.UPPER = rUpper
        self.SIGMA = sigma
        self.SHOW = show
        self.CMAP = cmap
        self.VERBOSE = verbose
        self.SAVE_IM = save_im
        self.NAME_IM = name_im
        
        
        # Initialize virtual detector
        if virtdet is not None:
            self.VIRTDET=virtdet
        else:
            self.VIRTDET = v4dVDet.Annular(
                pattern,
                self.packets,
                self.descriptors,
                self.values_role,
                self.OUT_DIR,
                scan_dims=self.SCAN_DIMS,
                mode=2,
                rLower=self.LOWER,
                rUpper=self.UPPER,
                sigma=self.SIGMA,
                show=self.SHOW,
                cmap=self.CMAP,
                verbose=self.VERBOSE,
                save_im=False,
                name_im=self.NAME_IM
                )
    
    def Excel(self,
              xlsx_path,
              sheet_name = None,
              columns = ['idx', 'inner', 'outer']):
        
        # Load EXCEL data
        df = pd.read_excel(xlsx_path, 
                           sheet_name)

        # Select the desired columns
        subset = df.loc[:, columns]
        counter = 0
        index = subset[columns[0]][0]
        

            
            
        for i in range(0,len(subset)):
            s = subset[columns[0]][i]
            LOWER = subset[columns[1]][i]
            UPPER = subset[columns[2]][i]
            
            if s == index:
                counter = counter+1
            else:
                counter = 0
                index = s
        
            if counter<10:
                prefix = f"{s}0{counter}_"
            else:
                prefix = f"{s}{counter}_"
        
            FILENAME = prefix+f"lower{LOWER}_upper{UPPER}.png"
            
            # Application of virtual detectors
            self.VIRTDET.filter_centered_ROI(
                lower=LOWER,
                upper=UPPER,
                name_im=FILENAME,
                
                sigma=self.SIGMA)
            
        return
        
    
    def ImageSeries(self, path, filename, xlsx=None, sheet=None, cols=None, ids=None):
        files = sorted(glob.glob(os.path.join(path, "*.png")))
        if not files:
            raise FileNotFoundError(f"No .png files in {path}")
    
        # Normalize ids (optional filter)
        ids_set = None
        if ids is not None:
            ids_set = {str(x).strip().upper() for x in ids}
    
        # Optionally filter files by leading letter in basename
        if ids_set:
            def file_idx_letter(p):
                base = os.path.basename(p)
                return base[0].upper() if base else ""
            files = [f for f in files if file_idx_letter(f) in ids_set]
            if not files:
                raise FileNotFoundError(f"No .png files in {path} match ids={sorted(ids_set)}")
    
        # Load table once
        texts = []
        if xlsx is not None:
            df = pd.read_excel(xlsx, sheet_name=sheet)
            # Require 'idx' column if we're filtering by ids
            if ids_set and "idx" not in df.columns:
                raise KeyError("Excel sheet is missing required 'idx' column for ids filtering.")
            if ids_set:
                df = df[df["idx"].astype(str).str.upper().isin(ids_set)].copy()
        
            sub = df.loc[:, cols]
            for i in range(len(sub)):
                l = sub[cols[0]].iloc[i]
                u = sub[cols[1]].iloc[i]
                texts.append(f"Range: {l:.0f}-{u:.0f} mrad")
        else:
            texts = [""] * len(files)
    
        # If counts don't match, align to min length (warn)
        if len(texts) != len(files):
            m = min(len(texts), len(files))
            print(f"[WARN] Mismatch: {len(files)} image(s) vs {len(texts)} text row(s). Using first {m}.")
            files = files[:m]
            texts = texts[:m]
    
        # Basic image size (assume all the same; enforce it)
        h0, w0 = cv2.imread(files[0]).shape[:2]
    
        # Font helpers
        font_paths = [
            r"C:\Windows\Fonts\times.ttf",
            r"C:\Windows\Fonts\timesbd.ttf",
            r"C:\Windows\Fonts\timesi.ttf",
            "times.ttf",
        ]
        def make_font(sz):
            for p in font_paths:
                try:
                    return ImageFont.truetype(p, sz)
                except OSError:
                    continue
            return ImageFont.load_default()
    
        # text size using Pillow (bbox for Pillow>=10; fallback otherwise)
        def text_size(text, font):
            probe = Image.new("RGB", (1, 1))
            draw = ImageDraw.Draw(probe)
            try:
                l, t, r, b = draw.textbbox((0, 0), text, font=font)
                return (r - l), (b - t)
            except AttributeError:
                return draw.textsize(text, font=font)
    
        # Choose ONE font size that fits the longest line in the given width
        pad_x, pad_y = 20, 10
        target_width = w0 - 2 * pad_x
        base_size = 48
        font = make_font(base_size)
    
        widest = max(texts, key=lambda s: len(s)) if texts else ""
        tw, th = text_size(widest, font)
        if tw > target_width:
            new_size = max(10, int(base_size * target_width / max(tw, 1)))
            font = make_font(new_size)
            tw, th = text_size(widest, font)
            while tw > target_width and new_size > 10:
                new_size -= 1
                font = make_font(new_size)
                tw, th = text_size(widest, font)
    
        strip_h = th + 2 * pad_y            # constant across all frames
        canvas_h, canvas_w = h0 + strip_h, w0
    
        # Build frames (all canvases same shape)
        frames = []
        for i, fpath in enumerate(files):
            img = cv2.imread(fpath)
            h, w = img.shape[:2]
    
            # Enforce same base size as first frame (resize if needed)
            if (h, w) != (h0, w0):
                img = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)
    
            canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
            canvas[:h0, :w0] = img
    
            # Draw centered text
            text = texts[i] if i < len(texts) else ""
            pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            tw_i, th_i = text_size(text, font)
            x = max(pad_x, (canvas_w - tw_i) // 2)
            y = h0 + (strip_h - th_i) // 2
            draw.text((x, y), text, font=font, fill=(0, 0, 0))
    
            canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            
    
        # Save GIF (all frames have identical shape now)
        out_path = os.path.join(path, filename)
        imageio.mimsave(out_path, frames, fps=2)
        print(f"[INFO] Movie saved to: {out_path}")
        
        
    def ImageSeries2(self, path, filename,
                xlsx=None, sheet=None, cols=None, ids=None,
                *,
                fps=10,
                base_font_px=40,
                pad_x=20, pad_y=14,
                vis_lo_pct=1.0, vis_hi_pct=99.5,
                clip_pct=None):
        """
        Build an animated GIF/MP4 from PNGs in `path`, overlaying a text line
        from Excel. Supports filtering by Excel column 'idx' (e.g., 'A').
        Display-only brightness scaling; source images unchanged.
        """
        import os, glob, re
        import numpy as np
        import cv2
        import pandas as pd
        import imageio
        from PIL import Image, ImageDraw, ImageFont
    
        # ---- gather files
        files_all = sorted(glob.glob(os.path.join(path, "*.png")))
        if not files_all:
            raise FileNotFoundError(f"No .png files in {path}")
    
        # normalize ids input -> set[str] (uppercase)
        if ids is None:
            want_ids = None
        elif isinstance(ids, str):
            want_ids = {ids.upper()}
        else:
            want_ids = {str(x).upper() for x in ids}
    
        # helper: get the letter id from a filename (first A–Z at start)
        def file_idx_letter(f):
            m = re.match(r"([A-Za-z])", os.path.basename(f))
            return m.group(1).upper() if m else None
    
        # ---- load Excel (optional) and filter rows by ids
        texts_by_stem = None
        ordered_stems = None
        df = None
        if xlsx is not None:
            df = pd.read_excel(xlsx, sheet_name=sheet)
    
            # Filter DF by ids if provided and 'idx' exists
            if want_ids is not None and "idx" in df.columns:
                df = df[df["idx"].astype(str).str.upper().isin(want_ids)]
    
            # columns to read text from
            if cols is None or len(cols) != 2:
                raise ValueError("`cols` must be a list of two column names, e.g. ['inner [mrad]','outer [mrad]'].")
    
            # If DF has a filename-like column, build mapping {stem: text} and keep table order
            fname_col = next((c for c in ("file","filename","name","stem","basename") if c in df.columns), None)
            if fname_col is not None:
                # normalize stems from table
                stem_series = df[fname_col].astype(str).str.replace(r"\.png$", "", regex=True).str.strip()
                ordered_stems = stem_series.tolist()
    
                texts_by_stem = {}
                for _, row in df.iterrows():
                    stem = str(row[fname_col]).strip()
                    stem = re.sub(r"\.png$", "", stem, flags=re.IGNORECASE)
                    l = float(row[cols[0]])
                    u = float(row[cols[1]])
                    texts_by_stem[stem] = f"Range: {l:.0f}-{u:.0f} mrad"
    
        # ---- select files to process + text per file
        proc_files = []
        proc_texts = []
    
        if texts_by_stem is not None:
            # Use stems from table (already filtered by ids if applicable)
            # Match to actual files; skip stems that don't exist on disk.
            files_by_stem = {os.path.splitext(os.path.basename(f))[0]: f for f in files_all}
            for stem in ordered_stems:
                f = files_by_stem.get(stem)
                if f is None:
                    # try a loose match if the table stem is a prefix
                    match = next((ff for st, ff in files_by_stem.items() if st.startswith(stem)), None)
                    if match is None:
                        continue
                    f = match
                # If ids requested, enforce by filename too
                if want_ids is not None and file_idx_letter(f) not in want_ids:
                    continue
                proc_files.append(f)
                proc_texts.append(texts_by_stem.get(os.path.splitext(os.path.basename(f))[0], ""))
        else:
            # No filename column in Excel.
            # Strategy: filter files by ids (if any), then assign texts per-id in row order.
            files = files_all
            if want_ids is not None:
                files = [f for f in files_all if file_idx_letter(f) in want_ids]
    
            if df is None:
                # No Excel at all -> no text
                proc_files = files
                proc_texts = [""] * len(files)
            else:
                # Build per-id queues of text in DF order
                if "idx" in df.columns:
                    df["_IDX_UP"] = df["idx"].astype(str).str.upper()
                else:
                    # If there’s no 'idx' column, we can’t align by id; just emit in DF order vs files order length
                    df["_IDX_UP"] = ""
    
                # queues: { 'A': [text1, text2, ...], ... }
                queues = {}
                for _, row in df.iterrows():
                    l = float(row[cols[0]])
                    u = float(row[cols[1]])
                    txt = f"Range: {l:.0f}-{u:.0f} mrad"
                    key = row["_IDX_UP"]
                    queues.setdefault(key, []).append(txt)
    
                for f in files:
                    key = file_idx_letter(f) or ""
                    txt = queues.get(key, [""]).pop(0) if queues.get(key) else ""
                    proc_files.append(f)
                    proc_texts.append(txt)
    
        if not proc_files:
            raise RuntimeError("No frames matched the requested `ids`/Excel mapping.")
    
        # ---- robust display scaling
        def robust_scale(img_bgr, lo=1.0, hi=99.5, cap_pct=None):
            arr = img_bgr.astype(np.float32)
            if cap_pct is not None:
                for ch in range(arr.shape[2]):
                    hi_val = np.percentile(arr[..., ch], cap_pct)
                    if np.isfinite(hi_val):
                        arr[..., ch] = np.minimum(arr[..., ch], hi_val)
            out = np.empty_like(arr, dtype=np.uint8)
            for ch in range(arr.shape[2]):
                lo_v = np.percentile(arr[..., ch], lo)
                hi_v = np.percentile(arr[..., ch], hi)
                if hi_v <= lo_v:
                    out[..., ch] = 0
                else:
                    tmp = np.clip((arr[..., ch] - lo_v) / (hi_v - lo_v), 0, 1)
                    out[..., ch] = (tmp * 255).astype(np.uint8)
            return out
    
        # ---- font helpers
        font_paths = [
            r"C:\Windows\Fonts\times.ttf",
            r"C:\Windows\Fonts\timesbd.ttf",
            "times.ttf",
        ]
        def make_font(sz):
            from PIL import ImageFont
            for p in font_paths:
                try:
                    return ImageFont.truetype(p, sz)
                except OSError:
                    continue
            return ImageFont.load_default()
    
        def text_size(text, font):
            probe = Image.new("RGB", (1, 1))
            draw = ImageDraw.Draw(probe)
            try:
                l, t, r, b = draw.textbbox((0, 0), text, font=font)
                return (r - l), (b - t)
            except AttributeError:
                return draw.textsize(text, font=font)
    
        # ---- frame geometry + one font size that fits the widest line
        base = cv2.imread(proc_files[0])
        if base is None:
            raise RuntimeError(f"Cannot read first image: {proc_files[0]}")
        h0, w0 = base.shape[:2]
    
        widest = max(proc_texts, key=len, default="")
        font = make_font(base_font_px)
        target_width = w0 - 2 * pad_x
        tw, th = text_size(widest, font) if widest else (0, base_font_px)
        if widest and tw > target_width:
            new_sz = max(10, int(base_font_px * target_width / max(tw, 1)))
            font = make_font(new_sz)
            tw, th = text_size(widest, font)
            while widest and tw > target_width and new_sz > 10:
                new_sz -= 1
                font = make_font(new_sz)
                tw, th = text_size(widest, font)
    
        strip_h = th + 2 * pad_y
        canvas_h, canvas_w = h0 + strip_h, w0
    
        # ---- build frames
        frames = []
        for fpath, text in zip(proc_files, proc_texts):
            img = cv2.imread(fpath)
            if img is None:
                continue
            h, w = img.shape[:2]
            if (h, w) != (h0, w0):
                img = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)
    
            img_disp = robust_scale(img, lo=vis_lo_pct, hi=vis_hi_pct, cap_pct=clip_pct)
    
            canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
            canvas[:h0, :w0] = img_disp
    
            pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            tw_i, th_i = text_size(text, font)
            x = max(pad_x, (canvas_w - tw_i) // 2)
            y = h0 + (strip_h - th_i) // 2
            draw.text((x, y), text, font=font, fill=(0, 0, 0))
            canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
            frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    
        print(f"[INFO] Frames in movie: {len(frames)}")
    
        # ---- save
        out_path = os.path.join(path, filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".gif":
            imageio.mimsave(out_path, frames, fps=2)
        elif ext in (".mp4", ".mov", ".mkv"):
            try:
                imageio.mimsave(out_path, frames, fps, codec="libx264", quality=8)
            except Exception:
                imageio.mimsave(out_path, frames, fps=2)
        else:
            imageio.mimsave(out_path, frames, fps=2)
    
        print(f"[INFO] Movie saved to: {out_path}")
