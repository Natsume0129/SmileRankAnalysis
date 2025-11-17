#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare multiple smoothing filters on stitched SmileRank data, then plot per 900 frames.

Input:
- Folder contains: smile_data_segment_0.csv ... smile_data_segment_84.csv
- Columns: frame, rank_original_0based_or_nan, rank_interpolated, is_long_gap, filename

Process:
1) Stitch all CSVs by frame.
2) Apply selected smoothing methods to the single, stitched series.
3) For each method, output:
   - output/{method}/data/smile_data_smoothed_all.csv
   - output/{method}/plots/smile_curve_segment_{i}.png   (900 frames per figure)
   Plot style per method:
     Blue solid = smoothed curve
     Red dashed = long-gap overlay

You can enable/disable methods in METHODS_TO_RUN below.
If SciPy/statsmodels are unavailable, affected methods are skipped automatically.
"""

import os
import math
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------- Config -------------------------
# Change this to your local input folder
INPUT_DIR = r"E:\chrome-downloads\20250926_plots\20250926\csv_data"

FILE_PREFIX = "smile_data_segment_"
FILE_SUFFIX = ".csv"
START_IDX = 0
END_IDX = 84   # inclusive

FRAMES_PER_FIG = 900

# Methods to run. Possible keys:
# "savgol", "gaussian", "moving_average", "median", "butterworth", "kalman", "loess", "bilateral1d"
METHODS_TO_RUN: List[str] = [
    "savgol",
    "gaussian",
    "moving_average",
    "median",
    "butterworth",
    "kalman",
    "loess",
    "bilateral1d",
]

# Parameters
PARAMS = dict(
    # Savitzky–Golay
    savgol_window=9,
    savgol_poly=2,
    # Gaussian
    gauss_sigma=1.2,
    gauss_radius=4,   # kernel size = 2*radius+1
    # Moving average
    ma_window=7,
    # Median filter
    med_window=7,
    # Butterworth low-pass
    butter_order=3,
    butter_cutoff=0.05,   # normalized (0..1). ~0.05 means strong smoothing
    # Kalman 1D
    kalman_Q=0.02,   # process noise variance
    kalman_R=0.15,   # measurement noise variance
    # LOESS (LOWESS)
    loess_frac=0.02,  # fraction of data per local fit
    # Bilateral 1D
    bilateral_radius=4,
    bilateral_sigma_space=2.0,
    bilateral_sigma_value=1.0,
)


# ------------------------- I/O helpers -------------------------
def ensure_dirs(base_out: Path, method: str) -> Dict[str, Path]:
    out_dir = base_out / method
    d = {
        "root": out_dir,
        "plots": out_dir / "plots",
        "data": out_dir / "data",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def read_all_segments(input_dir: str, start_idx: int, end_idx: int) -> pd.DataFrame:
    frames = []
    for i in range(start_idx, end_idx + 1):
        path = Path(input_dir) / f"{FILE_PREFIX}{i}{FILE_SUFFIX}"
        if not path.exists():
            print(f"[WARN] Missing file: {path}")
            continue
        try:
            df = pd.read_csv(path)
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
    all_df = all_df.sort_values(["frame", "segment_index"]).drop_duplicates(subset=["frame"], keep="first")
    all_df = all_df.sort_values("frame").reset_index(drop=True)
    return all_df


# ------------------------- Filters -------------------------
def interp_fill(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values.astype(float))
    s = s.interpolate(method="linear", limit_direction="both")
    return s.to_numpy()


def filt_savgol(v: np.ndarray, window: int, poly: int) -> Optional[np.ndarray]:
    try:
        from scipy.signal import savgol_filter
        n = len(v)
        w = window
        if w > n:
            w = n if n % 2 == 1 else n - 1
        if w < poly + 3:
            w = poly + 3 if (poly + 3) % 2 == 1 else poly + 4
        if w % 2 == 0:
            w = max(3, w - 1)
        return savgol_filter(v, window_length=w, polyorder=poly, mode="interp")
    except Exception as e:
        print(f"[SKIP] Savitzky–Golay unavailable: {e}")
        return None


def gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    return k


def filt_gaussian(v: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    k = gaussian_kernel1d(sigma, radius)
    padded = np.pad(v, (radius, radius), mode="reflect")
    return np.convolve(padded, k, mode="valid")


def filt_moving_average(v: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    k = np.ones(w) / w
    padded = np.pad(v, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(padded, k, mode="valid")


def filt_median(v: np.ndarray, window: int) -> np.ndarray:
    w = max(3, int(window) | 1)  # odd >=3
    return pd.Series(v).rolling(window=w, center=True, min_periods=1).median().to_numpy()


def filt_butterworth(v: np.ndarray, order: int, cutoff: float) -> Optional[np.ndarray]:
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(order, cutoff, btype="low", analog=False)
        return filtfilt(b, a, v, method="pad", padlen=min(3*max(len(a), len(b)), len(v)-1))
    except Exception as e:
        print(f"[SKIP] Butterworth unavailable: {e}")
        return None


def filt_kalman_1d(v: np.ndarray, Q: float, R: float) -> np.ndarray:
    x_est = np.zeros_like(v)
    P = 1.0
    x = v[0]
    for i, z in enumerate(v):
        x_pred = x
        P_pred = P + Q
        K = P_pred / (P_pred + R)
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        x_est[i] = x
    return x_est


def filt_loess(v: np.ndarray, frac: float) -> Optional[np.ndarray]:
    try:
        import statsmodels.api as sm
        x = np.arange(len(v))
        lo = sm.nonparametric.lowess(v, x, frac=frac, return_sorted=False)
        return lo
    except Exception as e:
        print(f"[SKIP] LOESS unavailable: {e}")
        return None


def filt_bilateral1d(v: np.ndarray, radius: int, sigma_space: float, sigma_value: float) -> np.ndarray:
    n = len(v)
    out = np.zeros_like(v, dtype=float)
    idx = np.arange(n)
    for i in range(n):
        left = max(0, i - radius)
        right = min(n - 1, i + radius)
        j = idx[left:right+1]
        spatial = np.exp(-0.5 * ((j - i) / sigma_space) ** 2)
        range_w = np.exp(-0.5 * ((v[j] - v[i]) / sigma_value) ** 2)
        w = spatial * range_w
        s = w.sum()
        out[i] = (w * v[j]).sum() / s if s > 0 else v[i]
    return out


# ------------------------- Plotting -------------------------
def plot_segments(df: pd.DataFrame, frames_per_fig: int, out_plot_dir: Path):
    total_frames = int(df["frame"].max()) + 1
    num_figs = math.ceil(total_frames / frames_per_fig)

    y_min = float(np.nanmin([df["rank_smoothed"].min(), df["rank_interpolated"].min()]))
    y_max = float(np.nanmax([df["rank_smoothed"].max(), df["rank_interpolated"].max()]))
    for i in range(num_figs):
        start = i * frames_per_fig
        end = min((i + 1) * frames_per_fig, total_frames)
        seg = df[(df["frame"] >= start) & (df["frame"] < end)].copy()
        if seg.empty:
            continue

        fig, ax = plt.subplots(figsize=(18, 5))
        ax.plot(seg["frame"], seg["rank_smoothed"], linestyle='-', marker='', color='blue', label='Smoothed')
        long_mask = seg["is_long_gap"].astype(bool)
        long_series = seg["rank_smoothed"].where(long_mask, other=np.nan)
        ax.plot(seg["frame"], long_series, linestyle='--', marker='', color='red', label='Long Gap (smoothed)', zorder=3)

        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Smile Intensity Rank (0=Strongest, higher=Weaker)')
        ax.set_title(f'Smile Intensity - Segment {i+1} (Frames {start}-{end-1})')
        ax.grid(True, linestyle=':')
        ax.set_xlim(start - 10, end + 10)
        ax.set_ylim(y_max + 1, y_min - 1)  # invert
        ax.set_yticks(range(int(math.floor(y_min)), int(math.ceil(y_max)) + 1))
        ax.legend()
        plt.tight_layout()

        out_png = out_plot_dir / f"smile_curve_segment_{i}.png"
        try:
            fig.savefig(out_png)
            print(f"[INFO] Saved plot: {out_png}")
        finally:
            plt.close(fig)


# ------------------------- Main -------------------------
def main():
    base_out = Path(__file__).resolve().parent / "output"
    base_out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading stitched CSVs from: {INPUT_DIR}")
    all_df = read_all_segments(INPUT_DIR, START_IDX, END_IDX)

    base = interp_fill(all_df["rank_interpolated"].to_numpy())
    methods_done = []

    for method in METHODS_TO_RUN:
        print(f"[INFO] Running method: {method}")
        smoothed = None
        if method == "savgol":
            smoothed = filt_savgol(base, PARAMS["savgol_window"], PARAMS["savgol_poly"])
        elif method == "gaussian":
            smoothed = filt_gaussian(base, PARAMS["gauss_sigma"], PARAMS["gauss_radius"])
        elif method == "moving_average":
            smoothed = filt_moving_average(base, PARAMS["ma_window"])
        elif method == "median":
            smoothed = filt_median(base, PARAMS["med_window"])
        elif method == "butterworth":
            smoothed = filt_butterworth(base, PARAMS["butter_order"], PARAMS["butter_cutoff"])
        elif method == "kalman":
            smoothed = filt_kalman_1d(base, PARAMS["kalman_Q"], PARAMS["kalman_R"])
        elif method == "loess":
            smoothed = filt_loess(base, PARAMS["loess_frac"])
        elif method == "bilateral1d":
            smoothed = filt_bilateral1d(base, PARAMS["bilateral_radius"], PARAMS["bilateral_sigma_space"], PARAMS["bilateral_sigma_value"])
        else:
            print(f"[WARN] Unknown method: {method} (skipped)")

        if smoothed is None:
            print(f"[SKIP] Method {method} not available in this environment.")
            continue

        out_dirs = ensure_dirs(base_out, method)
        tmp = all_df.copy()
        tmp["rank_smoothed"] = smoothed

        # Save combined CSV
        keep_cols = ["frame", "rank_original_0based_or_nan", "rank_interpolated", "rank_smoothed", "is_long_gap", "filename"]
        for c in keep_cols:
            if c not in tmp.columns:
                tmp[c] = np.nan
        out_csv = out_dirs["data"] / "smile_data_smoothed_all.csv"
        tmp[keep_cols].to_csv(out_csv, index=False, float_format="%.6f")
        print(f"[INFO] Saved CSV: {out_csv}")

        # Plot per 900 frames
        plot_segments(tmp, FRAMES_PER_FIG, out_dirs["plots"])
        methods_done.append(method)

    print(f"[INFO] Completed methods: {methods_done}")


if __name__ == "__main__":
    main()
