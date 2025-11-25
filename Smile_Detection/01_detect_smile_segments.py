#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# -------------------- 通用配置解析函数 --------------------

def parse_source_dat(filepath: str) -> Dict[str, Any]:
    """
    解析形如 key = value 的简单配置文件。
    支持:
      - # 注释
      - 字符串两侧引号
      - 将能转成 float/int 的值自动转换数值类型
    """
    conf: Dict[str, Any] = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('=', 1)
                if len(parts) != 2:
                    print(f"警告：跳过格式错误的行: {line}")
                    continue
                key = parts[0].strip()
                value = parts[1].strip()
                # 去掉尾部注释
                if '#' in value:
                    value = value.split('#', 1)[0].strip()
                # 去掉引号
                value = value.strip('\'"')

                # 尝试转为数值
                try:
                    float_val = float(value)
                    if float_val.is_integer():
                        value_cast: Any = int(float_val)
                    else:
                        value_cast = float_val
                    conf[key] = value_cast
                except ValueError:
                    # 非数值，保持为字符串
                    conf[key] = value
    except FileNotFoundError:
        print(f"错误：配置文件未找到: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"读取配置文件 {filepath} 时出错: {e}")
        sys.exit(1)

    return conf


# -------------------- 读取所有 CSV 并合并 --------------------

def load_all_csv(csv_dir: str) -> Tuple[pd.DataFrame, int]:
    """
    从 csv_dir 中读取所有 *.csv，拼成一个完整 DataFrame。
    期望每个 CSV 至少包含这些列：
      - frame
      - rank_original_0based_or_nan
      - rank_interpolated
      - is_long_gap
      - filename
    返回:
      df_concat: 合并后的 DataFrame（按 frame 升序排序）
      num_files: 读取的 CSV 文件数
    """
    if not os.path.isdir(csv_dir):
        print(f"错误：CSV_DATA_DIR 不存在或不是目录: {csv_dir}")
        sys.exit(1)

    all_files = [
        os.path.join(csv_dir, fname)
        for fname in os.listdir(csv_dir)
        if fname.lower().endswith('.csv')
    ]
    if not all_files:
        print(f"错误：目录中未找到任何 CSV 文件: {csv_dir}")
        sys.exit(1)

    all_files.sort()
    dfs: List[pd.DataFrame] = []
    for path in all_files:
        try:
            df = pd.read_csv(path)
            if 'frame' not in df.columns:
                print(f"警告：文件缺少 frame 列，已跳过: {os.path.basename(path)}")
                continue
            dfs.append(df)
        except Exception as e:
            print(f"警告：读取 CSV 失败，已跳过 {os.path.basename(path)}: {e}")

    if not dfs:
        print("错误：没有成功读取任何 CSV 文件。")
        sys.exit(1)

    df_concat = pd.concat(dfs, ignore_index=True)
    df_concat = df_concat.sort_values(by='frame').reset_index(drop=True)

    return df_concat, len(all_files)


# -------------------- Core-based detection 原始算法（加 longgap 放宽开关） --------------------
# 参数签名保持不变，只在内部加 RELAX_LONGGAP 分支

R_CORE = 4.0           # 核心阈值 (作用于 rank_smoothed)
MIN_CORE_LEN = 4       # 移除短于此的 core 段
MAX_CORE_GAP = 6       # 填充 core 段之间的小 gap
MERGE_GAP    = 10      # 合并事件时允许的最大 gap

BASE_LEFT  = (180, 30) # [远, 近] 左局部基线窗口
BASE_RIGHT = (30, 180) # 右局部基线窗口
BASE_DELTA = 0.3
TOUT_MIN, TOUT_MAX = 4.5, 7.5

K_OUT = 8               # 连续高于 Tout 的长度
LONGGAP_RATIO_MAX = 0.4 # 严格模式下的 longgap 最大比例
PADDING_FRAMES = 30     # 最后对事件做 padding 帧数

# 是否放宽 longgap 管理
RELAX_LONGGAP = True


def find_runs(x: np.ndarray) -> List[Tuple[int,int]]:
    # 返回 x 为 True 的 [开始, 结束] (包含) 区间列表
    n = len(x)
    runs: List[Tuple[int,int]] = []
    i = 0
    while i < n:
        if x[i]:
            j = i
            while j+1 < n and x[j+1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def remove_short_true(x: np.ndarray, min_len: int) -> None:
    # 移除短于 min_len 的 True 区间
    for s, e in find_runs(x):
        if e - s + 1 < min_len:
            x[s:e+1] = False


def fill_small_gaps(x: np.ndarray, max_gap: int, barrier: np.ndarray) -> None:
    # 填充 True 区间之间的 False 间隙，如果间隙长度<=max_gap且间隙内没有 barrier
    n = len(x)
    i = 0
    while i < n:
        if x[i]:
            j = i
            while j+1 < n and x[j+1]:
                j += 1
            k = j + 1
            g = 0
            while k < n and not x[k]:
                g += 1
                k += 1
            if k < n and g > 0 and g <= max_gap:
                if not barrier[j+1:k].any():
                    x[j+1:k] = True
            i = k
        else:
            i += 1


def sustained_blocks(cond: np.ndarray, K: int) -> np.ndarray:
    # 返回一个布尔数组，其中每个索引标记一个长度为 K 的全 True 块的结束
    n = len(cond)
    if K <= 1:
        return cond.copy()
    s = np.convolve(cond.astype(int), np.ones(K, dtype=int), mode='full')
    ends = np.zeros(n, dtype=bool)
    for i in range(n):
        w_end = i
        if w_end >= K-1 and s[w_end] == K:
            ends[i] = True
    return ends


def compute_local_baseline(r: np.ndarray, center: int) -> float:
    # 计算局部基线
    n = len(r)
    aL, bL = BASE_LEFT
    aR, bR = BASE_RIGHT
    L = r[max(0, center-aL):max(0, center-bL)]
    R = r[min(n, center+aR):min(n, center+bR)]
    vec = np.concatenate([L, R]) if L.size + R.size > 0 else r
    return float(np.nanmedian(vec))


def detect(df: pd.DataFrame) -> List[Tuple[int,int]]:
    """
    原始 core-based 检测函数（参数签名保持不变）。
    这里假定 df 至少包含列：
      - frame
      - rank_smoothed
      - rank_interpolated
      - is_long_gap
    """
    r  = df["rank_smoothed"].astype(float).to_numpy()
    n  = len(r)
    if "is_long_gap" in df.columns:
        lg = df["is_long_gap"].to_numpy().astype(bool)
    else:
        lg = np.zeros(n, dtype=bool)

    # 1) 核心掩码 core = (r <= R_CORE)
    core = r <= R_CORE
    remove_short_true(core, MIN_CORE_LEN)
    fill_small_gaps(core, MAX_CORE_GAP, barrier=lg)

    # 2) 候选核心区间
    runs = find_runs(core)
    events: List[Tuple[int,int]] = []
    if not runs:
        return events

    for (s0, e0) in runs:
        center = (s0 + e0) // 2
        base = compute_local_baseline(r, center)
        Tout = float(np.clip(base - BASE_DELTA, TOUT_MIN, TOUT_MAX))

        cond_exit = r > Tout

        # 放宽 longgap：不再强行打断“外部”持续块
        if RELAX_LONGGAP:
            cond_exit_bar = cond_exit
        else:
            cond_exit_bar = cond_exit.copy()
            cond_exit_bar[lg] = False

        ends = sustained_blocks(cond_exit_bar, K_OUT)

        # 左边界
        left_candidates = np.where(ends[:s0+1])[0]
        if left_candidates.size > 0:
            j = int(left_candidates[-1])
            start = j + 1
        else:
            if RELAX_LONGGAP:
                # 放宽：不在 longgap 处截断，直接从 0 开始
                start = 0
            else:
                left_bar = np.where(lg[:s0+1])[0]
                start = int(left_bar[-1]+1) if left_bar.size>0 else 0

        # 右边界
        right_candidates = np.where(ends[e0:])[0]
        if right_candidates.size > 0:
            j_rel = int(right_candidates[0])
            j = e0 + j_rel
            end = max(j - K_OUT, e0)
        else:
            if RELAX_LONGGAP:
                # 放宽：不在 longgap 处截断，直接延到末尾
                end = n - 1
            else:
                right_bar = np.where(lg[e0:])[0]
                end = int(e0 + right_bar[0] - 1) if right_bar.size>0 else n-1

        if end >= start:
            events.append((start, end))

    # 3) 合并邻近且 gap 小的事件
    events.sort()
    merged: List[Tuple[int,int]] = []
    for seg in events:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        gap = seg[0] - prev[1] - 1
        if RELAX_LONGGAP:
            # 放宽：只要 gap 不大就合并，不再检查 longgap
            if gap <= MERGE_GAP:
                merged[-1] = (prev[0], max(prev[1], seg[1]))
            else:
                merged.append(seg)
        else:
            # 严格：gap 内不能有 longgap
            if gap <= MERGE_GAP and not lg[prev[1]+1:seg[0]].any():
                merged[-1] = (prev[0], max(prev[1], seg[1]))
            else:
                merged.append(seg)

    # 4) 丢弃 longgap 比例太高的片段（放宽模式直接保留）
    kept: List[Tuple[int,int]] = []
    for a, b in merged:
        if a <= b:
            if RELAX_LONGGAP:
                kept.append((a, b))
            else:
                if float(lg[a:b+1].mean()) <= LONGGAP_RATIO_MAX:
                    kept.append((a, b))

    return kept


# -------------------- 写 .dat --------------------

def write_segments_dat(
    segments: List[Tuple[int, int]],
    output_dir: str,
    dest_dat_name: str,
    meta: Dict[str, Any],
    num_csv_files: int
) -> str:
    """
    将片段写入 .dat 文件。
    每行: start_frame end_frame
    文件开头写若干 # 注释记录元信息。
    返回 .dat 文件完整路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    dest_path = os.path.join(output_dir, dest_dat_name)

    person = meta.get('PERSON', '')
    date_tag = meta.get('DATE_TAG', '')
    csv_dir = meta.get('CSV_DATA_DIR', '')
    rank_column = meta.get('RANK_COLUMN', '')
    rank_thresh = meta.get('RANK_THRESH', '')

    with open(dest_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f"# PERSON = {person}\n")
        f.write(f"# DATE_TAG = {date_tag}\n")
        f.write(f"# CSV_DATA_DIR = {csv_dir}\n")
        f.write(f"# NUM_CSV_FILES = {num_csv_files}\n")
        f.write(f"# RANK_COLUMN = {rank_column}\n")
        f.write(f"# RANK_THRESH = {rank_thresh}\n")
        f.write("# FORMAT: start_frame end_frame\n")
        f.write("#\n")

        for start_frame, end_frame in segments:
            f.write(f"{start_frame} {end_frame}\n")

    return dest_path


# -------------------- 主流程 --------------------

def main():
    # 1) 读取配置文件路径
    if len(sys.argv) >= 2:
        conf_path = sys.argv[1]
    else:
        conf_path = "01_detect_smile_events.dat"

    print(f"使用配置文件: {conf_path}")
    conf = parse_source_dat(conf_path)

    # 2) 提取必要配置
    person = str(conf.get('PERSON', 'unknown_person'))
    date_tag = str(conf.get('DATE_TAG', 'unknown_date'))
    csv_data_dir = str(conf.get('CSV_DATA_DIR', '')).strip()
    output_dir_conf = str(conf.get('OUTPUT_DIR', '')).strip()
    rank_column = str(conf.get('RANK_COLUMN', 'rank_interpolated')).strip()
    rank_thresh = float(conf.get('RANK_THRESH', 3.0))  # 仅用于记录到 meta
    dest_dat_name = f"smile_segments_rank_{person}_{date_tag}.dat"

    if not csv_data_dir:
        print("错误：CSV_DATA_DIR 未在配置文件中指定。")
        sys.exit(1)

    # 决定输出目录：如果配置为空，则在 csv_data_dir 的上一级创建 "Smile_Segments" 目录
    if output_dir_conf:
        output_dir = output_dir_conf
    else:
        parent_dir = os.path.dirname(csv_data_dir.rstrip(r"\\/"))
        output_dir = os.path.join(parent_dir, "Smile_Segments")

    segment_folder_name = f"{person}_{date_tag}_SEGMENTS"
    output_dir = os.path.join(output_dir, segment_folder_name)

    print(f"PERSON = {person}")
    print(f"DATE_TAG = {date_tag}")
    print(f"CSV_DATA_DIR = {csv_data_dir}")
    print(f"OUTPUT_DIR = {output_dir}")
    print(f"RANK_COLUMN = {rank_column}")
    print(f"RANK_THRESH = {rank_thresh}")
    print(f"DEST_DAT_NAME = {dest_dat_name}")
    print(f"RELAX_LONGGAP = {RELAX_LONGGAP}")

    # 3) 读取并拼接所有 CSV
    df, num_csv_files = load_all_csv(csv_data_dir)
    print(f"已读取并合并 {num_csv_files} 个 CSV 文件, 总行数 = {len(df)}")
    print(f"frame 范围: {int(df['frame'].min())} ~ {int(df['frame'].max())}")

    # 补一列 rank_smoothed（旧算法用这个）——先用 rank_column 来代替
    if rank_column not in df.columns:
        print(f"错误：DataFrame 中不存在列: {rank_column}")
        sys.exit(1)
    df['rank_smoothed'] = df[rank_column].astype(float)

    # 4) 使用 core-based 算法检测笑容片段（返回的是“索引区间”）
    segs_index = detect(df)
    print(f"核心算法检测到片段数（未 padding）: {len(segs_index)}")

    # 5) 对区间做 padding（转换为“帧号区间”）
    if segs_index:
        frame_series = df['frame'].astype(int)
        frame_min = int(frame_series.min())
        frame_max = int(frame_series.max())
        segments: List[Tuple[int,int]] = []
        for s_idx, e_idx in segs_index:
            s_frame = int(frame_series.iloc[s_idx])
            e_frame = int(frame_series.iloc[e_idx])
            new_s = max(frame_min, s_frame - PADDING_FRAMES)
            new_e = min(frame_max, e_frame + PADDING_FRAMES)
            segments.append((new_s, new_e))
        print(f"padding 后片段数: {len(segments)}")
    else:
        segments = []
        print("未检测到任何片段。")

    # 6) 写入 .dat
    meta = {
        'PERSON': person,
        'DATE_TAG': date_tag,
        'CSV_DATA_DIR': csv_data_dir,
        'RANK_COLUMN': rank_column,
        'RANK_THRESH': rank_thresh,
    }
    dest_path = write_segments_dat(
        segments=segments,
        output_dir=output_dir,
        dest_dat_name=dest_dat_name,
        meta=meta,
        num_csv_files=num_csv_files
    )

    print(f"片段已写入: {dest_path}")
    print("完成。")


if __name__ == "__main__":
    main()
