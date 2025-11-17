#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict smile start/end detector with hard gating to prevent over-detection.

Inputs:
  ./output/data/smile_data_smoothed_all.csv
Outputs:
  ./output/events_strict/smile_segments.dat
  ./output/events_strict/plots/smile_events_segment_{i}.png
"""

import math
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Paths --------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV  = SCRIPT_DIR / "output" / "data" / "smile_data_smoothed_all.csv"
OUT_DIR    = SCRIPT_DIR / "output" / "events_strict"
PLOT_DIR   = OUT_DIR / "plots"
FRAMES_PER_FIG = 900

# -------- Strict params --------
T_CORE = 3.0           # hard smile core threshold (rank <= 3 means smiling)
PROM_MIN = 0.8         # stronger prominence
WIN_PEAK = 5

TIN  = 3.0             # enter when <= 3.0 (tie to your rule)
BASE_LEFT  = (180, 30)
BASE_RIGHT = (30, 180)
BASE_DELTA = 0.2
TOUT_MIN, TOUT_MAX = 4.5, 7.5  # tighter exit band

K_OUT = 10             # consecutive frames to confirm exit
EPS   = 0.02
L_MIN = 18             # min segment length
G_MERGE = 8            # merge close
IEI_MIN = 20           # min inter-event interval after a segment closes

# Hard gates inside candidate segment
AMP_MIN  = 1.0         # baseline - min(rank) >= 1.0
D_CORE   = 4           # at least 4 frames with r <= T_CORE
F_CORE   = 0.10        # at least 10% frames with r <= T_CORE

LONGGAP_RATIO_MAX = 0.4

def _nanmedian(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.nanmedian(x)) if x.size else float("nan")

def _fallback_valley_prominence(r: np.ndarray, idx: int, radius: int = 20) -> float:
    n = len(r)
    L = max(0, idx - radius)
    R = min(n, idx + radius + 1)
    if R - L <= 3:
        return 0.0
    left_min  = np.min(r[L:idx])  if idx > L else r[idx]
    right_min = np.min(r[idx+1:R]) if idx+1 < R else r[idx]
    prom = min(left_min - r[idx], right_min - r[idx])
    return float(max(prom, 0.0))

def _find_valley_peaks(r: np.ndarray, thr: float, prom: float, win: int) -> List[int]:
    try:
        from scipy.signal import find_peaks
        peaks, prop = find_peaks(-r, prominence=prom, width=win)
        return [int(p) for p in peaks if r[p] <= thr]
    except Exception:
        w = max(3, int(win) | 1)
        vals = []
        for i in range(1, len(r)-1):
            if r[i] <= thr and r[i] <= r[i-1] and r[i] <= r[i+1]:
                if _fallback_valley_prominence(r, i, radius=max(10, w//2)) >= prom:
                    vals.append(i)
        return vals

def detect(df: pd.DataFrame) -> List[Tuple[int,int]]:
    r  = df["rank_smoothed"].astype(float).to_numpy()
    n  = len(r)
    dr = np.diff(r, prepend=r[0])
    peaks = _find_valley_peaks(r, T_CORE, PROM_MIN, WIN_PEAK)
    segs: List[Tuple[int,int]] = []
    last_end_for_iei = -10**9

    for p in peaks:
        aL, bL = BASE_LEFT
        aR, bR = BASE_RIGHT
        Lslice = r[max(0, p-aL):max(0, p-bL)]
        Rslice = r[min(n, p+aR):min(n, p+bR)]
        base = _nanmedian(np.concatenate([Lslice, Rslice])) if Lslice.size + Rslice.size > 0 else _nanmedian(r)
        Tout = float(np.clip(base - BASE_DELTA, TOUT_MIN, TOUT_MAX))

        # Left bound
        L0 = p
        for i in range(p, -1, -1):
            if r[i] <= TIN:
                L0 = i
        start = 0
        ok = 0
        i = L0
        while i >= 0:
            cond = (r[i] > Tout) and (dr[i] >= -EPS)
            ok = ok + 1 if cond else 0
            if ok >= K_OUT:
                start = min(i + K_OUT, L0)
                break
            i -= 1
        else:
            start = max(0, i+1)

        # Right bound
        R0 = p
        for i in range(p, n):
            if r[i] <= TIN:
                R0 = i
        end = n - 1
        ok = 0
        i = R0
        while i < n:
            cond = (r[i] > Tout) and (dr[i] <= +EPS)
            ok = ok + 1 if cond else 0
            if ok >= K_OUT:
                end = max(i - K_OUT, R0)
                break
            i += 1

        if end - start + 1 < L_MIN:
            continue
        if start - last_end_for_iei < IEI_MIN:
            # too close to previous; merge later by just appending
            pass

        # Hard gates
        seg = r[start:end+1]
        min_r = float(np.min(seg))
        amp   = base - min_r
        core_cnt = int((seg <= T_CORE).sum())
        core_frac = core_cnt / max(1, len(seg))

        if amp < AMP_MIN:           # not deep enough
            continue
        if core_cnt < D_CORE:       # too few core frames
            continue
        if core_frac < F_CORE:      # not enough duty of core
            continue

        # Long-gap ratio
        if "is_long_gap" in df.columns:
            lg = df["is_long_gap"].to_numpy().astype(bool)
            if float(lg[start:end+1].mean()) > LONGGAP_RATIO_MAX:
                continue

        segs.append((start, end))
        last_end_for_iei = end

    # Merge close
    segs.sort()
    merged: List[Tuple[int,int]] = []
    for s in segs:
        if not merged: merged.append(s); continue
        if s[0] - merged[-1][1] - 1 <= G_MERGE:
            merged[-1] = (merged[-1][0], max(merged[-1][1], s[1]))
        else:
            merged.append(s)
    return merged

def plot(df: pd.DataFrame, segs: List[Tuple[int,int]]):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    total_frames = int(df["frame"].max()) + 1
    num_figs = math.ceil(total_frames / FRAMES_PER_FIG)
    y_min = float(np.nanmin([df["rank_smoothed"].min(), df["rank_interpolated"].min()]))
    y_max = float(np.nanmax([df["rank_smoothed"].max(), df["rank_interpolated"].max()]))
    for i in range(num_figs):
        a = i * FRAMES_PER_FIG
        b = min((i+1) * FRAMES_PER_FIG, total_frames)
        seg_df = df[(df["frame"] >= a) & (df["frame"] < b)].copy()
        if seg_df.empty: continue
        long_mask = seg_df["is_long_gap"].astype(bool) if "is_long_gap" in seg_df.columns else pd.Series(False, index=seg_df.index)
        blue = seg_df["rank_smoothed"].where(~long_mask, np.nan)
        red  = seg_df["rank_smoothed"].where( long_mask, np.nan)
        fig, ax = plt.subplots(figsize=(18,5))
        ax.plot(seg_df["frame"], blue, '-', color='blue', label='Smoothed')
        ax.plot(seg_df["frame"], red,  '--', color='red',  label='Long Gap')
        for (s,e) in segs:
            S, E = max(s, a), min(e, b-1)
            if S <= E:
                ax.axvspan(S, E, facecolor='red', alpha=0.18, zorder=1)
                ax.axvline(S, color='red', linestyle='-', linewidth=1.5, zorder=3)
                ax.axvline(E, color='red', linestyle='-', linewidth=1.5, zorder=3)
        ax.set_title(f"Smile Events (Strict) - Segment {i+1} (Frames {a}-{b-1})")
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Smile Intensity Rank (0=Strongest, higher=Weaker)")
        ax.grid(True, linestyle=':')
        ax.set_xlim(a-10, b+10)
        ax.set_ylim(y_max + 1, y_min - 1)
        ax.legend()
        plt.tight_layout()
        out = PLOT_DIR / f"smile_events_segment_{i}.png"
        fig.savefig(out)
        plt.close(fig)

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    for col in ["frame","rank_interpolated","rank_smoothed"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing column: {col}")
    segs = detect(df)
    # write dat
    dat = OUT_DIR / "smile_segments.dat"
    with open(dat, "w", encoding="utf-8") as f:
        for s,e in segs:
            f.write(f"{s},{e}\n")
    print(f"[INFO] Segments: {len(segs)} -> {dat}")
    plot(df, segs)

if __name__ == "__main__":
    main()
