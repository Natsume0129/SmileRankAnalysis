
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Concatenate smile rank CSVs, smooth, and plot per 900 frames.

- Input folder contains files: smile_data_segment_0.csv ... smile_data_segment_84.csv
- Columns expected: frame, rank_original_0based_or_nan, rank_interpolated, is_long_gap, filename
- Smoothing is applied to "rank_interpolated" after stitching all segments
- Plot style: blue solid for smoothed curve; red dashed for long gaps
- Plots are split every 900 frames
- Outputs under ./output relative to this script:
    - output/data/smile_data_smoothed_all.csv
    - output/plots/smile_curve_segment_{i}.png
"""

import os
import sys
import math
import csv
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------- Config -------------------------
# Change this to your local input folder
INPUT_DIR = "E:\\chrome-downloads\\20250926_plots\\20250926\\csv_data"

# File pattern and index range
FILE_PREFIX = "smile_data_segment_"
FILE_SUFFIX = ".csv"
START_IDX = 0
END_IDX = 84   # inclusive

# Plot segmentation
FRAMES_PER_FIG = 900

# Output dirs (relative to this script's directory)
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "output"
PLOT_DIR = OUT_DIR / "plots"
DATA_DIR = OUT_DIR / "data"

# Smoothing options
SMOOTH_METHOD: Literal["savgol", "gaussian", "moving_average"] = "savgol"
# Savitzky–Golay parameters
SAVGOL_WINDOW = 9     # must be odd and >= polyorder+2
SAVGOL_POLY = 2
# Gaussian parameters
GAUSS_SIGMA = 1.0
GAUSS_KERNEL_RADIUS = 3  # kernel size = 2*radius+1
# Moving average parameters
MA_WINDOW = 5


# ------------------------- Utils -------------------------
def ensure_dirs():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def read_all_segments(input_dir: str, start_idx: int, end_idx: int) -> pd.DataFrame:
    """Read and concatenate segment CSVs, return one DataFrame sorted by frame."""
    frames = []
    for i in range(start_idx, end_idx + 1):
        path = Path(input_dir) / f"{FILE_PREFIX}{i}{FILE_SUFFIX}"
        if not path.exists():
            print(f"[WARN] Missing file: {path}")
            continue
        try:
            df = pd.read_csv(path)
            # Basic validation
            required = {"frame", "rank_interpolated", "is_long_gap"}
            if not required.issubset(df.columns):
                raise ValueError(f"File {path} missing required columns: {required - set(df.columns)}")
            df["segment_index"] = i
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {path}: {e}")

    if not frames:
        raise RuntimeError("No CSVs loaded. Check INPUT_DIR and file range.")
    all_df = pd.concat(frames, ignore_index=True)
    # Drop duplicates by frame, keep the first occurrence (lower segment index)
    all_df = all_df.sort_values(["frame", "segment_index"]).drop_duplicates(subset=["frame"], keep="first")
    all_df = all_df.sort_values("frame").reset_index(drop=True)
    return all_df


def gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    """Create a normalized 1D Gaussian kernel with given sigma and radius."""
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def smooth_series(values: np.ndarray,
                  method: str = "savgol",
                  savgol_window: int = 9,
                  savgol_poly: int = 2,
                  gauss_sigma: float = 1.0,
                  gauss_radius: int = 3,
                  ma_window: int = 5) -> np.ndarray:
    """Smooth 1D series with chosen method, handling NaNs by linear interpolation first."""
    x = values.astype(float).copy()

    # Fill NaNs via linear interpolation then nearest fill at ends
    s = pd.Series(x)
    s_interp = s.interpolate(method="linear", limit_direction="both")
    v = s_interp.to_numpy()

    if method == "savgol":
        try:
            from scipy.signal import savgol_filter
            # Adjust window to be odd and <= len(v)
            w = min(max(savgol_window, savgol_poly + 3), len(v) - (1 - len(v) % 2))
            if w % 2 == 0:
                w = max(3, w - 1)
            if w <= savgol_poly:
                w = savgol_poly + 3 if (savgol_poly + 3) % 2 == 1 else savgol_poly + 4
            w = min(w, len(v) if len(v) % 2 == 1 else len(v) - 1)
            return savgol_filter(v, window_length=w, polyorder=savgol_poly, mode="interp")
        except Exception as e:
            print(f"[WARN] Savitzky–Golay unavailable or failed ({e}). Falling back to Gaussian.")
            method = "gaussian"

    if method == "gaussian":
        kernel = gaussian_kernel1d(gauss_sigma, gauss_radius)
        # Reflect padding for edges
        padded = np.pad(v, (gauss_radius, gauss_radius), mode="reflect")
        sm = np.convolve(padded, kernel, mode="valid")
        return sm

    if method == "moving_average":
        w = max(1, int(ma_window))
        kernel = np.ones(w) / w
        padded = np.pad(v, (w // 2, w - 1 - w // 2), mode="edge")
        sm = np.convolve(padded, kernel, mode="valid")
        return sm

    raise ValueError(f"Unknown smoothing method: {method}")


def plot_segments(df: pd.DataFrame, frames_per_fig: int, y_min: Optional[float], y_max: Optional[float]):
    """Plot smoothed curve in blue solid, with red dashed overlay where is_long_gap is True."""
    total_frames = int(df["frame"].max()) + 1
    num_figs = math.ceil(total_frames / frames_per_fig)

    # Establish y-axis
    if y_min is None:
        y_min = float(np.nanmin([df["rank_smoothed"].min(), df["rank_interpolated"].min()]))
    if y_max is None:
        y_max = float(np.nanmax([df["rank_smoothed"].max(), df["rank_interpolated"].max()]))
    # For SmileRank 0=strongest at top like previous code. So invert axis.
    y_min_plot = y_min
    y_max_plot = y_max

    for i in range(num_figs):
        start = i * frames_per_fig
        end = min((i + 1) * frames_per_fig, total_frames)
        seg = df[(df["frame"] >= start) & (df["frame"] < end)].copy()
        if seg.empty:
            continue

        
        fig, ax = plt.subplots(figsize=(18, 5))

        # Masked series:
        long_mask = seg["is_long_gap"].astype(bool)
        blue_series = seg["rank_smoothed"].where(~long_mask, other=np.nan)   # blue only on non-long-gap
        red_series  = seg["rank_smoothed"].where(long_mask,  other=np.nan)   # red only on long-gap

        # Blue solid for non-long-gap segments
        ax.plot(seg["frame"], blue_series, linestyle='-', marker='', color='blue', label='Smoothed')

        # Red dashed for long-gap segments
        ax.plot(seg["frame"], red_series, linestyle='--', marker='', color='red', label='Long Gap (smoothed)', zorder=3)

        # Styling similar to previous
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Smile Intensity Rank (0=Strongest, higher=Weaker)')
        ax.set_title(f'Smile Intensity - Segment {i+1} (Frames {start}-{end-1})')
        ax.grid(True, linestyle=':')
        ax.set_xlim(start - 10, end + 10)
        ax.set_ylim(y_max_plot + 1, y_min_plot - 1)  # invert y-axis by reversing limits
        ax.set_yticks(range(int(math.floor(y_min_plot)), int(math.ceil(y_max_plot)) + 1))
        ax.legend()
        plt.tight_layout()

        out_png = PLOT_DIR / f"smile_curve_segment_{i}.png"
        try:
            fig.savefig(out_png)
            print(f"[INFO] Saved plot: {out_png}")
        finally:
            plt.close(fig)


def main():
    ensure_dirs()
    print(f"[INFO] Reading segments {START_IDX}..{END_IDX} from: {INPUT_DIR}")
    all_df = read_all_segments(INPUT_DIR, START_IDX, END_IDX)

    # Choose base series for smoothing
    base = all_df["rank_interpolated"].to_numpy()
    smoothed = smooth_series(
        base,
        method=SMOOTH_METHOD,
        savgol_window=SAVGOL_WINDOW,
        savgol_poly=SAVGOL_POLY,
        gauss_sigma=GAUSS_SIGMA,
        gauss_radius=GAUSS_KERNEL_RADIUS,
        ma_window=MA_WINDOW,
    )
    all_df["rank_smoothed"] = smoothed

    # Save combined smoothed data
    out_csv = DATA_DIR / "smile_data_smoothed_all.csv"
    keep_cols = ["frame", "rank_original_0based_or_nan", "rank_interpolated", "rank_smoothed", "is_long_gap", "filename"]
    missing = [c for c in keep_cols if c not in all_df.columns]
    for c in missing:
        all_df[c] = np.nan
    all_df[keep_cols].to_csv(out_csv, index=False, float_format="%.6f")
    print(f"[INFO] Saved combined CSV: {out_csv}")

    # Plot per 900 frames
    unique_vals = pd.concat([all_df["rank_interpolated"], all_df["rank_smoothed"]], axis=0)
    y_min, y_max = float(unique_vals.min()), float(unique_vals.max())
    plot_segments(all_df, FRAMES_PER_FIG, y_min=y_min, y_max=y_max)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
