#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core-based smile detector:
- Uses rank_smoothed <= 3.0 as the only anchor for events.
- Morphological cleanup prevents whole-trace marking.
- Expands boundaries to sustained-above-Tout blocks; longgaps are barriers.

Inputs:
  ./output/data/smile_data_smoothed_all.csv
Outputs:
  ./output/events_core/smile_segments.dat
  ./output/events_core/plots/smile_events_segment_{i}.png
"""

import math
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV  = SCRIPT_DIR / "output" / "data" / "smile_data_smoothed_all.csv"
OUT_DIR    = SCRIPT_DIR / "output" / "events_core"
PLOT_DIR   = OUT_DIR / "plots"
FRAMES_PER_FIG = 900

# --- Parameters ---
R_CORE = 3.0           # core threshold
MIN_CORE_LEN = 4       # remove core runs shorter than this
MAX_CORE_GAP = 6       # fill gaps between core runs if gap <= this and no longgap inside
MERGE_GAP    = 10      # merge neighboring events if gap <= this and no longgap barrier

BASE_LEFT  = (180, 30) # [far, near] for local baseline windows around the core mid
BASE_RIGHT = (30, 180)
BASE_DELTA = 0.3
TOUT_MIN, TOUT_MAX = 4.5, 7.5

K_OUT = 8              # sustained length above Tout to confirm outside
LONGGAP_RATIO_MAX = 0.4

def find_runs(x: np.ndarray) -> List[Tuple[int,int]]:
    """Return list of [start,end] inclusive runs where x is True."""
    n = len(x)
    runs = []
    i = 0
    while i < n:
        if x[i]:
            j = i
            while j+1 < n and x[j+1]: j += 1
            runs.append((i,j))
            i = j+1
        else:
            i += 1
    return runs

def remove_short_true(x: np.ndarray, min_len: int) -> None:
    for s,e in find_runs(x):
        if e - s + 1 < min_len:
            x[s:e+1] = False

def fill_small_gaps(x: np.ndarray, max_gap: int, barrier: np.ndarray) -> None:
    """Fill False gaps between True runs if gap length<=max_gap and no barrier True inside gap."""
    n = len(x)
    i = 0
    while i < n:
        if x[i]:
            j = i
            while j+1 < n and x[j+1]: j += 1
            # j is end of a true run
            k = j + 1
            # gap start at k
            g = 0
            while k < n and not x[k]:
                g += 1
                k += 1
            # now k is next true or end
            if k < n and g > 0 and g <= max_gap:
                if not barrier[j+1:k].any():
                    x[j+1:k] = True
            i = k
        else:
            i += 1

def sustained_blocks(cond: np.ndarray, K: int) -> np.ndarray:
    """Return boolean array where each index marks the end of a length-K all-True block in cond."""
    n = len(cond)
    if K <= 1:
        return cond.copy()
    # sliding window sum
    s = np.convolve(cond.astype(int), np.ones(K, dtype=int), mode='full')
    # ends at idx i where window ends at i -> index i of 's' corresponds to window [i-K+1, i]
    ends = np.zeros(n, dtype=bool)
    for i in range(n):
        w_end = i
        if w_end >= K-1:
            if s[w_end] == K:
                ends[i] = True
    return ends

def compute_local_baseline(r: np.ndarray, center: int) -> float:
    n = len(r)
    aL,bL = BASE_LEFT
    aR,bR = BASE_RIGHT
    L = r[max(0, center-aL):max(0, center-bL)]
    R = r[min(n, center+aR):min(n, center+bR)]
    vec = np.concatenate([L,R]) if L.size+R.size>0 else r
    return float(np.nanmedian(vec))

def detect(df: pd.DataFrame) -> List[Tuple[int,int]]:
    r  = df["rank_smoothed"].astype(float).to_numpy()
    n  = len(r)
    lg = df["is_long_gap"].to_numpy().astype(bool) if "is_long_gap" in df.columns else np.zeros(n, dtype=bool)

    # 1) core mask
    core = r <= R_CORE
    # cleanup
    remove_short_true(core, MIN_CORE_LEN)
    fill_small_gaps(core, MAX_CORE_GAP, barrier=lg)

    # 2) candidate core runs
    runs = find_runs(core)
    events: List[Tuple[int,int]] = []
    if not runs:
        return events

    # precompute cond_exit per dynamic Tout later per run
    for (s0,e0) in runs:
        center = (s0 + e0) // 2
        base = compute_local_baseline(r, center)
        Tout = float(np.clip(base - BASE_DELTA, TOUT_MIN, TOUT_MAX))

        cond_exit = r > Tout
        # longgap are hard barriers: we cannot be outside decision across them; force False to break sustained blocks
        cond_exit_bar = cond_exit.copy()
        cond_exit_bar[lg] = False

        ends = sustained_blocks(cond_exit_bar, K_OUT)
        # left boundary: last index j in [0..s0] where ends[j] True, set start=j+1; else start=0 or stop at longgap
        left_candidates = np.where(ends[:s0+1])[0]
        if left_candidates.size > 0:
            j = int(left_candidates[-1])
            start = j + 1
        else:
            # stop at nearest longgap to the left
            left_bar = np.where(lg[:s0+1])[0]
            start = int(left_bar[-1]+1) if left_bar.size>0 else 0

        # right boundary: first index j in [e0..n-1] where ends[j] True using forward sustained blocks on cond_exit_bar.
        right_candidates = np.where(ends[e0:])[0]
        if right_candidates.size > 0:
            j_rel = int(right_candidates[0])
            j = e0 + j_rel
            end = max(j - K_OUT, e0)  # end before the sustained outside block
        else:
            # stop at nearest longgap to the right
            right_bar = np.where(lg[e0:])[0]
            end = int(e0 + right_bar[0] - 1) if right_bar.size>0 else n-1

        if end >= start:
            events.append((start, end))

    # 3) merge neighbors with small gap and no longgap barrier
    events.sort()
    merged: List[Tuple[int,int]] = []
    for seg in events:
        if not merged:
            merged.append(seg); continue
        prev = merged[-1]
        gap = seg[0] - prev[1] - 1
        if gap <= MERGE_GAP and not lg[prev[1]+1:seg[0]].any():
            merged[-1] = (prev[0], max(prev[1], seg[1]))
        else:
            merged.append(seg)

    # 4) drop segments dominated by longgap
    kept: List[Tuple[int,int]] = []
    for a,b in merged:
        if a <= b:
            if float(lg[a:b+1].mean()) <= LONGGAP_RATIO_MAX:
                kept.append((a,b))
    return kept

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
        ax.set_title(f"Smile Events (Core) - Segment {i+1} (Frames {a}-{b-1})")
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
    dat = OUT_DIR / "smile_segments.dat"
    with open(dat, "w", encoding="utf-8") as f:
        for s,e in segs:
            f.write(f"{s},{e}\n")
    print(f"[INFO] Segments: {len(segs)} -> {dat}")
    plot(df, segs)

if __name__ == "__main__":
    main()
