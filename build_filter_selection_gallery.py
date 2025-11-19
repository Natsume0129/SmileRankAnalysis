#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
构建滤波器选择用的对比图集：

- 读取前 10 段原始 CSV: smile_data_segment_0..9.csv
- 对每种滤波器 + 参数组合：
    * 对每段做平滑
    * 用“core logic” 自动选笑容区间
    * 生成对比图：上原始，下平滑+高亮笑段
- 输出目录：./filter-selection/<filter_name>/segment_X_compare.png
"""

import os
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ===================== 基本配置 =====================

SCRIPT_DIR = Path(__file__).resolve().parent

# 1) 原始分段 CSV 所在目录（你需要改成项目内的相对路径或绝对路径）
#    例如：E:\chrome-downloads\20250926_plots\20250926\csv_data
INPUT_CSV_DIR = SCRIPT_DIR / "E:\\chrome-downloads\\20250926_plots\\20250926\\csv_data"   

# 2) 原始 plot PNG 所在目录（你之前的未滤波图）
#    例如：E:\chrome-downloads\20250926_plots\20250926
ORIG_PLOT_DIR = SCRIPT_DIR / "E:\\chrome-downloads\\20250926_plots\\20250926\\plots"  # 自己改

# 3) 输出目录（不在 output 里）
FILTER_SELECTION_DIR = SCRIPT_DIR / "filter-selection"

# 处理的段号（前 10 段）
SEGMENT_IDS = list(range(10))

# ===================== 滤波器配置 =====================

# 你可以在这里添加 / 修改 / 删除滤波配置
# name: 用于文件夹命名
# type: "moving_average" / "gaussian" / "pipeline" / "savgol"(需要scipy, 有降级)
FILTER_CONFIGS: List[Dict[str, Any]] = [
    # ========= 1. Moving Average 系列（8个） =========
    {
        "name": "ma_w3",
        "type": "moving_average",
        "params": {"window": 3},
    },
    {
        "name": "ma_w5",
        "type": "moving_average",
        "params": {"window": 5},
    },
    {
        "name": "ma_w7",
        "type": "moving_average",
        "params": {"window": 7},
    },
    {
        "name": "ma_w9",
        "type": "moving_average",
        "params": {"window": 9},
    },
    {
        "name": "ma_w13",
        "type": "moving_average",
        "params": {"window": 13},
    },
    {
        "name": "ma_w17",
        "type": "moving_average",
        "params": {"window": 17},
    },
    {
        "name": "ma_w25",
        "type": "moving_average",
        "params": {"window": 25},
    },
    {
        "name": "ma_w33",
        "type": "moving_average",
        "params": {"window": 33},
    },

    # ========= 2. Gaussian 平滑系列（8个） =========
    {
        "name": "gauss_r2_s0.8",
        "type": "gaussian",
        "params": {"radius": 2, "sigma": 0.8},
    },
    {
        "name": "gauss_r3_s1.0",
        "type": "gaussian",
        "params": {"radius": 3, "sigma": 1.0},
    },
    {
        "name": "gauss_r4_s1.2",
        "type": "gaussian",
        "params": {"radius": 4, "sigma": 1.2},
    },
    {
        "name": "gauss_r5_s1.5",
        "type": "gaussian",
        "params": {"radius": 5, "sigma": 1.5},
    },
    {
        "name": "gauss_r6_s2.0",
        "type": "gaussian",
        "params": {"radius": 6, "sigma": 2.0},
    },
    {
        "name": "gauss_r8_s2.5",
        "type": "gaussian",
        "params": {"radius": 8, "sigma": 2.5},
    },
    {
        "name": "gauss_r10_s3.0",
        "type": "gaussian",
        "params": {"radius": 10, "sigma": 3.0},
    },
    {
        "name": "gauss_r12_s3.5",
        "type": "gaussian",
        "params": {"radius": 12, "sigma": 3.5},
    },

    # ========= 3. Savitzky–Golay 系列（8个） =========
    # 注意：window 一定是奇数，且 > polyorder
    {
        "name": "savgol_w7_p2",
        "type": "savgol",
        "params": {"window": 7, "polyorder": 2},
    },
    {
        "name": "savgol_w9_p2",
        "type": "savgol",
        "params": {"window": 9, "polyorder": 2},
    },
    {
        "name": "savgol_w11_p2",
        "type": "savgol",
        "params": {"window": 11, "polyorder": 2},
    },
    {
        "name": "savgol_w15_p2",
        "type": "savgol",
        "params": {"window": 15, "polyorder": 2},
    },
    {
        "name": "savgol_w21_p2",
        "type": "savgol",
        "params": {"window": 21, "polyorder": 2},
    },
    {
        "name": "savgol_w9_p3",
        "type": "savgol",
        "params": {"window": 9, "polyorder": 3},
    },
    {
        "name": "savgol_w13_p3",
        "type": "savgol",
        "params": {"window": 13, "polyorder": 3},
    },
    {
        "name": "savgol_w17_p3",
        "type": "savgol",
        "params": {"window": 17, "polyorder": 3},
    },

    # ========= 4. Pipeline 组合滤波（6个） =========
    # 4.1 Gaussian -> Moving Average
    {
        "name": "gauss3_s1_then_ma5",
        "type": "pipeline",
        "stages": [
            {"type": "gaussian", "params": {"radius": 3, "sigma": 1.0}},
            {"type": "moving_average", "params": {"window": 5}},
        ],
    },
    {
        "name": "gauss3_s1_then_ma9",
        "type": "pipeline",
        "stages": [
            {"type": "gaussian", "params": {"radius": 3, "sigma": 1.0}},
            {"type": "moving_average", "params": {"window": 9}},
        ],
    },
    {
        "name": "gauss5_s1_5_then_ma9",
        "type": "pipeline",
        "stages": [
            {"type": "gaussian", "params": {"radius": 5, "sigma": 1.5}},
            {"type": "moving_average", "params": {"window": 9}},
        ],
    },

    # 4.2 Moving Average -> Gaussian
    {
        "name": "ma5_then_gauss3_s1",
        "type": "pipeline",
        "stages": [
            {"type": "moving_average", "params": {"window": 5}},
            {"type": "gaussian", "params": {"radius": 3, "sigma": 1.0}},
        ],
    },
    {
        "name": "ma9_then_gauss3_s1",
        "type": "pipeline",
        "stages": [
            {"type": "moving_average", "params": {"window": 9}},
            {"type": "gaussian", "params": {"radius": 3, "sigma": 1.0}},
        ],
    },

    # 4.3 Moving Average -> Savitzky–Golay
    {
        "name": "ma5_then_savgol9_p2",
        "type": "pipeline",
        "stages": [
            {"type": "moving_average", "params": {"window": 5}},
            {"type": "savgol", "params": {"window": 9, "polyorder": 2}},
        ],
    },
]


# ===================== 滤波函数实现 =====================

def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    kernel = np.ones(w, dtype=float) / w
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    sm = np.convolve(padded, kernel, mode="valid")
    return sm


def gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def gaussian_smooth(values: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return values.copy()
    kernel = gaussian_kernel1d(sigma, radius)
    padded = np.pad(values, (radius, radius), mode="reflect")
    sm = np.convolve(padded, kernel, mode="valid")
    return sm


def savgol_smooth(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """如果没有 scipy，就退化为 gaussian。"""
    try:
        from scipy.signal import savgol_filter
        w = int(window)
        if w % 2 == 0:
            w += 1
        w = min(w, len(values) if len(values) % 2 == 1 else len(values) - 1)
        if w <= polyorder:
            w = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4
        if w < 3:
            return values.copy()
        return savgol_filter(values, window_length=w, polyorder=polyorder, mode="interp")
    except Exception as e:
        print(f"[WARN] Savitzky-Golay unavailable ({e}), fallback to gaussian sigma=1.0 radius=3.")
        return gaussian_smooth(values, radius=3, sigma=1.0)


def apply_filter(values: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """根据 FILTER_CONFIGS 的定义，对 1D 数组做平滑。"""
    v = values.astype(float).copy()
    # 先对 NaN 做插值填充
    s = pd.Series(v)
    v_filled = s.interpolate(method="linear", limit_direction="both").to_numpy()

    t = cfg["type"]
    if t == "moving_average":
        w = cfg["params"]["window"]
        return moving_average(v_filled, w)
    elif t == "gaussian":
        radius = cfg["params"]["radius"]
        sigma = cfg["params"]["sigma"]
        return gaussian_smooth(v_filled, radius, sigma)
    elif t == "savgol":
        w = cfg["params"]["window"]
        p = cfg["params"]["polyorder"]
        return savgol_smooth(v_filled, w, p)
    elif t == "pipeline":
        out = v_filled
        for stage in cfg["stages"]:
            out = apply_filter(out, stage)
        return out
    else:
        raise ValueError(f"Unknown filter type: {t}")


# ===================== 笑段检测（core logic 的简化版） =====================

R_CORE = 4.0
MAX_RANK_INDEX = 10
MIN_CORE_LEN = 4
MAX_CORE_GAP = 6
MERGE_GAP = 10

BASE_LEFT = (180, 30)
BASE_RIGHT = (30, 180)
BASE_DELTA = 0.3
TOUT_MIN, TOUT_MAX = 4.5, 7.5
K_OUT = 8
LONGGAP_RATIO_MAX = 0.4


def find_runs(x: np.ndarray) -> List[Tuple[int, int]]:
    n = len(x)
    runs = []
    i = 0
    while i < n:
        if x[i]:
            j = i
            while j + 1 < n and x[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def remove_short_true(x: np.ndarray, min_len: int) -> None:
    for s, e in find_runs(x):
        if e - s + 1 < min_len:
            x[s:e+1] = False


def fill_small_gaps(x: np.ndarray, max_gap: int, barrier: np.ndarray) -> None:
    n = len(x)
    i = 0
    while i < n:
        if x[i]:
            j = i
            while j + 1 < n and x[j+1]:
                j += 1
            k = j + 1
            gap = 0
            while k < n and not x[k]:
                gap += 1
                k += 1
            if k < n and 0 < gap <= max_gap:
                if not barrier[j+1:k].any():
                    x[j+1:k] = True
            i = k
        else:
            i += 1


def compute_local_baseline(r: np.ndarray, center: int) -> float:
    n = len(r)
    aL, bL = BASE_LEFT
    aR, bR = BASE_RIGHT
    L = r[max(0, center - aL):max(0, center - bL)]
    R = r[min(n, center + aR):min(n, center + bR)]
    vec = np.concatenate([L, R]) if L.size + R.size > 0 else r
    return float(np.nanmedian(vec))


def sustained_blocks(cond: np.ndarray, K: int) -> np.ndarray:
    """返回一个布尔数组：每个 True 表示在该点结束的长度为 K 的连续 True 块。"""
    n = len(cond)
    if K <= 1:
        return cond.copy()
    s = np.convolve(cond.astype(int), np.ones(K, dtype=int), mode="full")
    ends = np.zeros(n, dtype=bool)
    for i in range(n):
        if i >= K - 1 and s[i] == K:
            ends[i] = True
    return ends


def detect_smile_segments_for_segment(seg_df: pd.DataFrame) -> List[Tuple[int, int]]:
    """在单个 segment 内检测笑段，返回的是“全局 frame 号”的区间列表。"""
    frame_vals = seg_df["frame"].to_numpy()
    r = seg_df["rank_smoothed"].astype(float).to_numpy()
    n = len(r)
    lg = seg_df["is_long_gap"].to_numpy().astype(bool) if "is_long_gap" in seg_df.columns else np.zeros(n, dtype=bool)

    core = r <= R_CORE
    remove_short_true(core, MIN_CORE_LEN)
    fill_small_gaps(core, MAX_CORE_GAP, barrier=lg)
    core_runs = find_runs(core)

    if not core_runs:
        return []

    events_idx: List[Tuple[int, int]] = []

    for (s0, e0) in core_runs:
        center = (s0 + e0) // 2
        base = compute_local_baseline(r, center)
        Tout = float(np.clip(base - BASE_DELTA, TOUT_MIN, TOUT_MAX))

        cond_exit = r > Tout
        cond_exit_bar = cond_exit.copy()
        cond_exit_bar[lg] = False
        ends = sustained_blocks(cond_exit_bar, K_OUT)

        # left
        left_idx = np.where(ends[:s0+1])[0]
        if left_idx.size > 0:
            j = int(left_idx[-1])
            start_idx = j + 1
        else:
            left_bar = np.where(lg[:s0+1])[0]
            start_idx = int(left_bar[-1] + 1) if left_bar.size > 0 else 0

        # right
        right_idx = np.where(ends[e0:])[0]
        if right_idx.size > 0:
            j_rel = int(right_idx[0])
            j = e0 + j_rel
            end_idx = max(j - K_OUT, e0)
        else:
            right_bar = np.where(lg[e0:])[0]
            end_idx = int(e0 + right_bar[0] - 1) if right_bar.size > 0 else n - 1

        if end_idx >= start_idx:
            events_idx.append((start_idx, end_idx))

    # merge events_idx
    events_idx.sort()
    merged_idx: List[Tuple[int, int]] = []
    for seg in events_idx:
        if not merged_idx:
            merged_idx.append(seg)
            continue
        prev = merged_idx[-1]
        gap = seg[0] - prev[1] - 1
        if gap <= MERGE_GAP and not lg[prev[1]+1:seg[0]].any():
            merged_idx[-1] = (prev[0], max(prev[1], seg[1]))
        else:
            merged_idx.append(seg)

    # drop segments dominated by longgap
    kept_global: List[Tuple[int, int]] = []
    for (si, ei) in merged_idx:
        if si <= ei:
            if float(lg[si:ei+1].mean()) <= LONGGAP_RATIO_MAX:
                kept_global.append((int(frame_vals[si]), int(frame_vals[ei])))

    return kept_global


# ===================== 绘图 & 图像拼接 =====================

def plot_segment_filtered_with_events(
    seg_df: pd.DataFrame,
    events: List[Tuple[int, int]],
    title: str,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 5))

    long_mask = seg_df["is_long_gap"].astype(bool) if "is_long_gap" in seg_df.columns else pd.Series(False, index=seg_df.index)
    blue = seg_df["rank_smoothed"].where(~long_mask, np.nan)
    red = seg_df["rank_smoothed"].where(long_mask, np.nan)

    ax.plot(seg_df["frame"], blue, '-', color='blue', label='Smoothed')
    ax.plot(seg_df["frame"], red, '--', color='red', label='Long Gap')

    # 不再用局部 min/max，而是固定全局 0~MAX_RANK_INDEX
    for (start_frame, end_frame) in events:
        ax.axvspan(start_frame, end_frame, facecolor='red', alpha=0.18, zorder=1)
        ax.axvline(start_frame, color='red', linestyle='-', linewidth=1.0, zorder=3)
        ax.axvline(end_frame, color='red', linestyle='-', linewidth=1.0, zorder=3)

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Smile Rank (0 strongest)")
    ax.grid(True, linestyle=':')

    # ★ 统一横轴（之前已经说过）
    frame_min = int(seg_df["frame"].min())
    frame_max = int(seg_df["frame"].max())
    ax.set_xlim(frame_min - 10, frame_max + 10)

    # ★ 统一纵轴（和 run_02_interpolate_and_plot 一样）
    ax.set_ylim(MAX_RANK_INDEX + 1, -1)          # 例如 0~10 → (11, -1)
    ax.set_yticks(range(0, MAX_RANK_INDEX + 1))  # 0,1,...,10

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)



def stack_vertical(orig_png: Path, filtered_png: Path, out_png: Path) -> None:
    """把原始图 (上) 和滤波+检测图 (下) 纵向拼接。"""
    if not orig_png.exists():
        print(f"[WARN] original plot missing: {orig_png}")
        return
    if not filtered_png.exists():
        print(f"[WARN] filtered plot missing: {filtered_png}")
        return

    img_top = Image.open(orig_png).convert("RGB")
    img_bottom = Image.open(filtered_png).convert("RGB")

    w = max(img_top.width, img_bottom.width)
    h = img_top.height + img_bottom.height
    
    if img_bottom.width != img_top.width:
        img_bottom = img_bottom.resize((img_top.width, img_bottom.height))

    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(img_top, (0, 0))
    canvas.paste(img_bottom, (0, img_top.height))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    img_top.close()
    img_bottom.close()


# ===================== 主流程 =====================

def main():
    print("[INFO] building filter-selection gallery...")

    for cfg in FILTER_CONFIGS:
        fname = cfg["name"]
        print(f"[INFO] Running filter: {fname}")
        filt_out_dir = FILTER_SELECTION_DIR / fname
        filt_tmp_dir = filt_out_dir / "_tmp_filtered_only"
        filt_tmp_dir.mkdir(parents=True, exist_ok=True)

        for seg_id in SEGMENT_IDS:
            csv_path = INPUT_CSV_DIR / f"smile_data_segment_{seg_id}.csv"
            if not csv_path.exists():
                print(f"[WARN] CSV not found: {csv_path}")
                continue

            seg_df = pd.read_csv(csv_path)
            if "rank_interpolated" not in seg_df.columns:
                print(f"[WARN] rank_interpolated missing in: {csv_path}")
                continue
            if "is_long_gap" not in seg_df.columns:
                seg_df["is_long_gap"] = False

            # 1) 滤波
            base = seg_df["rank_interpolated"].astype(float).to_numpy()
            smoothed = apply_filter(base, cfg)
            seg_df["rank_smoothed"] = smoothed

            # 2) 检测笑段（用 core logic）
            events = detect_smile_segments_for_segment(seg_df)

            # 3) 画“滤波+笑段”的图（下半图用）
            filtered_png = filt_tmp_dir / f"segment_{seg_id}_filtered.png"
            title = f"{fname} - segment {seg_id}"
            plot_segment_filtered_with_events(seg_df, events, title, filtered_png)

            # 4) 读原图 + 拼接
            orig_png = ORIG_PLOT_DIR / f"smile_curve_segment_{seg_id}.png"
            out_png = filt_out_dir / f"segment_{seg_id}_compare.png"
            stack_vertical(orig_png, filtered_png, out_png)

        # 可选择把 _tmp_filtered_only 保留，方便单独查看；如果不想保留，可以在此删除
        # import shutil; shutil.rmtree(filt_tmp_dir, ignore_errors=True)

    print("[INFO] done. All results in:", FILTER_SELECTION_DIR)


if __name__ == "__main__":
    main()
