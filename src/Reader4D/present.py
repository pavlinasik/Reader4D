# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 14:48:24 2025

@author: p-sik
"""
import pandas as pd
import cv2
import numpy as np
import imageio
import glob
import os

from PIL import Image, ImageDraw, ImageFont
import Reader4D.virtual_detectors as r4dVDet

class Creator:
    def __init__(self,
                 pattern,
                 packets,
                 descriptors,
                 values_role,
                 out_dir,
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
        self.VIRTDET = r4dVDet.Annular(
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
        
    def ImageSeries(self, path, filename, xlsx=None, sheet=None, cols=None):
        files = sorted(glob.glob(os.path.join(path, "*.png")))
        if not files:
            raise FileNotFoundError(f"No .png files in {path}")
    
        # --- Load table once
        texts = []
        if xlsx is not None:
            df = pd.read_excel(xlsx, sheet_name=sheet)
            sub = df.loc[:, cols]
            for i in range(len(sub)):
                l = float(sub[cols[0]].iloc[i])
                u = float(sub[cols[1]].iloc[i])
                texts.append(f"Range: {l:.3f}-{u:.3f} mrad")
        else:
            texts = [""] * len(files)
    
        # --- Basic image size (assume all the same; we’ll enforce it)
        h0, w0 = cv2.imread(files[0]).shape[:2]
    
        # --- Font helpers
        font_paths = [
            r"C:\Windows\Fonts\times.ttf",
            r"C:\Windows\Fonts\timesbd.ttf",
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
    
        # --- Choose ONE font size that fits the longest line in the given width
        pad_x, pad_y = 20, 10
        target_width = w0 - 2 * pad_x
        base_size = 32
        font = make_font(base_size)
    
        # Find widest text
        widest = max(texts, key=lambda s: len(s)) if texts else ""
        tw, th = text_size(widest, font)
        if tw > target_width:
            # scale down in one shot, then fine-tune
            new_size = max(10, int(base_size * target_width / max(tw, 1)))
            font = make_font(new_size)
            tw, th = text_size(widest, font)
            while tw > target_width and new_size > 10:
                new_size -= 1
                font = make_font(new_size)
                tw, th = text_size(widest, font)
    
        strip_h = th + 2 * pad_y            # <-- constant across all frames
        canvas_h, canvas_w = h0 + strip_h, w0
    
        # --- Build frames (all canvases same shape)
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
            x = (canvas_w - tw_i) // 2
            y = h0 + (strip_h - th_i) // 2
            draw.text((x, y), text, font=font, fill=(0, 0, 0))
    
            canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    
        # --- Save GIF (all frames have identical shape now)
        imageio.mimsave(os.path.join(path, filename), frames, fps=2)