#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Core-based smile detector:
# - Uses rank_smoothed <= 3.0 as the only anchor for events.
# - Morphological cleanup prevents whole-trace marking.
# - Expands boundaries to sustained-above-Tout blocks; longgaps are barriers.
#
# Inputs:
#   ./output/data/smile_data_smoothed_all.csv
# Outputs:
#   ./output/events_core/smile_segments.dat
#   ./output/events_core/plots/smile_events_segment_{i}.png
#

import math
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 脚本目录设置
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV  = SCRIPT_DIR / "output" / "data" / "smile_data_smoothed_all.csv"
OUT_DIR    = SCRIPT_DIR / "output" / "events_core"
PLOT_DIR   = OUT_DIR / "plots"
FRAMES_PER_FIG = 900 # 每张图表包含的帧数

# --- 参数 ---
R_CORE = 4.0           # 核心阈值
MIN_CORE_LEN = 4       # 移除短于此的核心片段
MAX_CORE_GAP = 6       # 填充核心片段之间的间隙 (如果间隙 <= 此值且内部无longgap)
MERGE_GAP    = 10      # 合并邻近事件 (如果间隙 <= 此值且内部无longgap)

BASE_LEFT  = (180, 30) # [远, 近] 用于核心中点附近的局部基线窗口
BASE_RIGHT = (30, 180)
BASE_DELTA = 0.3
TOUT_MIN, TOUT_MAX = 4.5, 7.5

K_OUT = 8              # 确认在"外部"所需的持续长度 (高于Tout)
LONGGAP_RATIO_MAX = 0.4 # 片段中longgap的最大比例

def find_runs(x: np.ndarray) -> List[Tuple[int,int]]:
    # 返回x为True的 [开始, 结束] (包含) 区间列表
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
    # 移除短于 min_len 的True区间
    for s,e in find_runs(x):
        if e - s + 1 < min_len:
            x[s:e+1] = False

def fill_small_gaps(x: np.ndarray, max_gap: int, barrier: np.ndarray) -> None:
    # 填充True区间之间的False间隙，如果间隙长度<=max_gap且间隙内没有barrier
    n = len(x)
    i = 0
    while i < n:
        if x[i]:
            j = i
            while j+1 < n and x[j+1]: j += 1
            # j 是一个true run的结束点
            k = j + 1
            # 间隙从k开始
            g = 0
            while k < n and not x[k]:
                g += 1
                k += 1
            # k 是下一个true run的开始或数组结尾
            if k < n and g > 0 and g <= max_gap:
                # 检查间隙 [j+1, k-1] 内是否有barrier
                if not barrier[j+1:k].any():
                    x[j+1:k] = True
            i = k
        else:
            i += 1

def sustained_blocks(cond: np.ndarray, K: int) -> np.ndarray:
    # 返回一个布尔数组，其中每个索引标记一个长度为K的全True块的结束
    n = len(cond)
    if K <= 1:
        return cond.copy()
    # 滑动窗口求和
    s = np.convolve(cond.astype(int), np.ones(K, dtype=int), mode='full')
    # 在索引i处结束的窗口对应 's' 的索引i
    ends = np.zeros(n, dtype=bool)
    for i in range(n):
        w_end = i
        if w_end >= K-1:
            if s[w_end] == K:
                ends[i] = True
    return ends

def compute_local_baseline(r: np.ndarray, center: int) -> float:
    # 计算局部基线
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

    # 1) 核心掩码 (core mask)
    core = r <= R_CORE
    # 清理
    remove_short_true(core, MIN_CORE_LEN)
    fill_small_gaps(core, MAX_CORE_GAP, barrier=lg)

    # 2) 候选核心区间
    runs = find_runs(core)
    events: List[Tuple[int,int]] = []
    if not runs:
        return events

    # 对每个run，预计算动态Tout的退出条件
    for (s0,e0) in runs:
        center = (s0 + e0) // 2
        base = compute_local_baseline(r, center)
        Tout = float(np.clip(base - BASE_DELTA, TOUT_MIN, TOUT_MAX))

        cond_exit = r > Tout
        # longgap是硬屏障: 不能跨越它们进行"外部"决策; 强制为False以打破持续块
        cond_exit_bar = cond_exit.copy()
        cond_exit_bar[lg] = False

        ends = sustained_blocks(cond_exit_bar, K_OUT)
        
        # 左边界: 在 [0..s0] 中找到最后一个 ends[j] 为True的索引 j, 设置 start=j+1
        left_candidates = np.where(ends[:s0+1])[0]
        if left_candidates.size > 0:
            j = int(left_candidates[-1])
            start = j + 1
        else:
            # 停止在左侧最近的longgap处
            left_bar = np.where(lg[:s0+1])[0]
            start = int(left_bar[-1]+1) if left_bar.size>0 else 0

        # 右边界: 在 [e0..n-1] 中找到第一个 ends[j] 为True的索引 j
        right_candidates = np.where(ends[e0:])[0]
        if right_candidates.size > 0:
            j_rel = int(right_candidates[0])
            j = e0 + j_rel
            end = max(j - K_OUT, e0)  # 在持续的外部块之前结束
        else:
            # 停止在右侧最近的longgap处
            right_bar = np.where(lg[e0:])[0]
            end = int(e0 + right_bar[0] - 1) if right_bar.size>0 else n-1

        if end >= start:
            events.append((start, end))

    # 3) 合并邻近且间隙小(且无longgap)的事件
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

    # 4) 丢弃主要由longgap构成的片段
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

    # --- 1. 运行原始检测 ---
    segs = detect(df)

    # --- 2. (新) 应用30帧的Padding (填充) ---
    # 定义要填充的帧数
    PADDING_FRAMES = 30
    # 获取最大帧索引 (总行数 - 1, 因为帧索引从0开始)
    max_frame_idx = len(df) - 1
    
    padded_segs = []
    if segs: # 确保segs列表不为空
        # 对每个检测到的区间应用padding
        for s, e in segs:
            # 开始帧 = max(0, s - 30)
            new_s = max(0, s - PADDING_FRAMES)
            # 结束帧 = min(最大帧, e + 30)
            new_e = min(max_frame_idx, e + PADDING_FRAMES)
            padded_segs.append((new_s, new_e))
        
        # (使用这个填充后的列表)
        final_segs = padded_segs
    else:
        # (如果一开始就没有检测到segs, final_segs就是空列表)
        final_segs = []

    # --- 3. 保存和绘制最终结果 ---
    dat = OUT_DIR / "smile_segments.dat"
    with open(dat, "w", encoding="utf-8") as f:
        # (使用 final_segs 替换 segs)
        for s,e in final_segs:
            f.write(f"{s},{e}\n")
    
    # (使用 final_segs 替换 segs)
    print(f"[INFO] Segments (padded): {len(final_segs)} -> {dat}")
    plot(df, final_segs)

if __name__ == "__main__":
    main()