#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import math
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np


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


# -------------------- 核心功能 --------------------

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

    # 收集所有 .csv 文件
    all_files = [
        os.path.join(csv_dir, fname)
        for fname in os.listdir(csv_dir)
        if fname.lower().endswith('.csv')
    ]
    if not all_files:
        print(f"错误：目录中未找到任何 CSV 文件: {csv_dir}")
        sys.exit(1)

    # 按文件名排序，保证顺序可控
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


def detect_smile_segments(
    df: pd.DataFrame,
    rank_column: str,
    rank_thresh: float
) -> List[Tuple[int, int]]:
    """
    在 df 中，根据 rank_column 和 rank_thresh 检测所有连续的 rank < rank_thresh 的片段。
    条件：
      - rank 非 NaN
      - rank < rank_thresh
    连续性定义：frame 逐帧连续（当前 frame == 上一个 frame + 1）。
    返回：
      segments: [(start_frame, end_frame), ...]
    """
    if rank_column not in df.columns:
        print(f"错误：DataFrame 中不存在列: {rank_column}")
        sys.exit(1)

    # 确保 frame 为 int，rank 为 float
    df = df.copy()
    df['frame'] = df['frame'].astype(int)
    df[rank_column] = df[rank_column].astype(float)

    df = df.sort_values(by='frame').reset_index(drop=True)

    segments: List[Tuple[int, int]] = []

    in_segment = False
    start_frame = None
    prev_frame = None

    for _, row in df.iterrows():
        frame = int(row['frame'])
        rank = row[rank_column]

        # 判定是否“在笑”
        is_smile = (not math.isnan(rank)) and (rank < rank_thresh)

        if is_smile:
            if not in_segment:
                # 开始新片段
                in_segment = True
                start_frame = frame
            else:
                # 已在片段中：如果 frame 不连续，先结束上一个片段再重新开始
                if prev_frame is not None and frame != prev_frame + 1:
                    # 结束旧片段
                    segments.append((start_frame, prev_frame))
                    # 开启新片段
                    start_frame = frame
        else:
            if in_segment:
                # 刚刚结束一个片段
                segments.append((start_frame, prev_frame))
                in_segment = False

        prev_frame = frame

    # 收尾：如果最后还在片段中
    if in_segment and start_frame is not None and prev_frame is not None:
        segments.append((start_frame, prev_frame))

    return segments


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
        # 默认配置文件名
        conf_path = "01_detect_smile_events.dat"

    print(f"使用配置文件: {conf_path}")
    conf = parse_source_dat(conf_path)

    # 2) 提取必要配置
    person = str(conf.get('PERSON', 'unknown_person'))
    date_tag = str(conf.get('DATE_TAG', 'unknown_date'))
    csv_data_dir = str(conf.get('CSV_DATA_DIR', '')).strip()
    output_dir_conf = str(conf.get('OUTPUT_DIR', '')).strip()
    rank_column = str(conf.get('RANK_COLUMN', 'rank_interpolated')).strip()
    rank_thresh = float(conf.get('RANK_THRESH', 3.0))
    dest_dat_name = f"smile_segments_rank_{person}_{date_tag}.dat"

    if not csv_data_dir:
        print("错误：CSV_DATA_DIR 未在配置文件中指定。")
        sys.exit(1)

    # 决定输出目录：如果配置为空，则在 csv_data_dir 的上一级创建 "Smile_Segments" 目录
    if output_dir_conf:
        output_dir = output_dir_conf
    else:
        parent_dir = os.path.dirname(csv_data_dir.rstrip(r"\/"))
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

    # 3) 读取并拼接所有 CSV
    df, num_csv_files = load_all_csv(csv_data_dir)
    print(f"已读取并合并 {num_csv_files} 个 CSV 文件, 总行数 = {len(df)}")
    print(f"frame 范围: {int(df['frame'].min())} ~ {int(df['frame'].max())}")

    # 4) 检测笑容片段
    segments = detect_smile_segments(df, rank_column=rank_column, rank_thresh=rank_thresh)
    print(f"检测到连续的 rank<{rank_thresh} 片段数: {len(segments)}")

    # 5) 写入 .dat 文件
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
