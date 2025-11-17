#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build comparison stacks for segments 0..9 with labels.

For each segment i:
    - ORIGINAL_DIR/smile_curve_segment_i.png  (top)
    - One row per filter method under METHODS
Each row gets a white label area with method name and parameters.

Output:
    ./output/labeled_compare_stacks/compare_labeled_segment_i.png
"""

from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

# ---------------- Config ----------------
ORIGINAL_DIR = r"E:\chrome-downloads\20250926_plots\20250926\plots"
# 滤波结果在单方法脚本输出的目录中，例如 ./output/plots
FILTER_BASE = Path(__file__).resolve().parent / "output"

# 只展示以下算法
METHODS: List[Tuple[str, str]] = [
    ("savgol",       "Savitzky–Golay (window=9, poly=2)"),
    ("gaussian",     "Gaussian (σ=1.2, radius=4)"),
    ("moving_average", "Moving Average (window=7)"),
    ("median",       "Median (window=7)"),
    ("butterworth",  "Butterworth (order=3, cutoff=0.05)"),
    ("kalman",       "Kalman (Q=0.02, R=0.15)"),
    ("loess",        "LOESS (frac=0.02)"),
    ("bilateral1d",  "Bilateral (radius=4, σ_s=2.0, σ_v=1.0)"),
]

FILE_PREFIX = "smile_curve_segment_"
FILE_SUFFIX = ".png"
START_IDX = 0
END_IDX   = 9

OUT_DIR = FILTER_BASE / "labeled_compare_stacks"

# --------------- Helpers ----------------
def safe_open(path: Path) -> Optional[Image.Image]:
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open {path}: {e}")
        return None

def add_label(im: Image.Image, label: str) -> Image.Image:
    """Add a white band above image with centered label text."""
    band_height = 50
    w = im.width
    band = Image.new("RGB", (w, band_height), (255, 255, 255))
    draw = ImageDraw.Draw(band)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()
    try:
        # Pillow <10
        tw, th = draw.textsize(label, font=font)
    except AttributeError:
        # Pillow >=10
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    draw.text(((w - tw) / 2, (band_height - th) / 2), label, fill=(0, 0, 0), font=font)
    out = Image.new("RGB", (w, band_height + im.height), (255, 255, 255))
    out.paste(band, (0, 0))
    out.paste(im, (0, band_height))
    return out

def vstack_list(images: List[Image.Image]) -> Image.Image:
    """Stack all images vertically, center-aligned and white-padded."""
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    max_w = max(widths)
    total_h = sum(heights)
    out = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for im in images:
        x = (max_w - im.width) // 2
        out.paste(im, (x, y))
        y += im.height
    return out

# --------------- Main -------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(START_IDX, END_IDX + 1):
        imgs = []
        # 原始图
        orig_path = Path(ORIGINAL_DIR) / f"{FILE_PREFIX}{i}{FILE_SUFFIX}"
        orig = safe_open(orig_path)
        if orig is None:
            print(f"[SKIP] Segment {i}: no original image.")
            continue
        imgs.append(add_label(orig, "Original (Unfiltered)"))

        # 各算法图
        for method, desc in METHODS:
            method_plot = FILTER_BASE / method / "plots" / f"{FILE_PREFIX}{i}{FILE_SUFFIX}"
            im = safe_open(method_plot)
            if im is None:
                print(f"[WARN] Missing {method} for segment {i}")
                continue
            imgs.append(add_label(im, f"{desc}"))

        if len(imgs) < 2:
            print(f"[SKIP] Segment {i}: insufficient images.")
            continue
        stacked = vstack_list(imgs)
        out_path = OUT_DIR / f"compare_labeled_segment_{i}.png"
        stacked.save(out_path)
        print(f"[INFO] Saved: {out_path}  size={stacked.size}")

if __name__ == "__main__":
    main()
