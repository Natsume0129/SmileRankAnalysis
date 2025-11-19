#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 02 的绘图方法，从 csv_data 段文件重绘笑容曲线，
并叠加 smile_segments_rank_{PERSON}_{DATE_TAG}.dat 中的笑容区间，使用红色高亮。

依赖配置文件：01_detect_smile_events.dat
  PERSON
  DATE_TAG
  CSV_DATA_DIR
  OUTPUT_DIR      (可空)
  RANK_COLUMN
  RANK_THRESH
"""

import os
import sys
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------------------- 通用配置解析 --------------------

def parse_source_dat(filepath: str) -> Dict[str, Any]:
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

                # 尝试转数值
                try:
                    v_float = float(value)
                    if v_float.is_integer():
                        conf[key] = int(v_float)
                    else:
                        conf[key] = v_float
                except ValueError:
                    conf[key] = value
    except FileNotFoundError:
        print(f"错误：配置文件未找到: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"读取配置文件 {filepath} 时出错: {e}")
        sys.exit(1)

    return conf


# -------------------- 工具函数 --------------------

def load_segments_dat(dat_path: str) -> List[Tuple[int, int]]:
    """
    从 dat 文件读取笑容区间。
    dat 格式：
      - 以 # 开头的行为注释
      - 其他行：start_frame end_frame
    """
    if not os.path.isfile(dat_path):
        print(f"错误：笑容区间 dat 文件不存在: {dat_path}")
        sys.exit(1)

    segments: List[Tuple[int, int]] = []
    with open(dat_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"警告：跳过格式异常的行: {line}")
                continue
            try:
                start = int(parts[0])
                end = int(parts[1])
                if end < start:
                    start, end = end, start
                segments.append((start, end))
            except ValueError:
                print(f"警告：无法解析区间行: {line}")
                continue

    segments.sort(key=lambda x: x[0])
    print(f"从 {os.path.basename(dat_path)} 读取到 {len(segments)} 个笑容区间。")
    return segments


def frames_in_any_segment(frames: np.ndarray,
                          segments: List[Tuple[int, int]]) -> np.ndarray:
    """
    给定帧号数组 frames (1D)，以及若干 [start, end] 区间，
    返回一个布尔数组 mask，标记每一帧是否落在任一区间之内。
    """
    mask = np.zeros_like(frames, dtype=bool)
    if not segments:
        return mask

    for start, end in segments:
        mask |= ((frames >= start) & (frames <= end))
    return mask


# -------------------- 主绘图逻辑 --------------------

def plot_with_segments_for_all_csv(
    csv_data_dir: str,
    out_plot_dir: str,
    segments: List[Tuple[int, int]],
    person: str,
    date_tag: str,
    num_dp_ranks: int = 11
) -> None:
    """
    遍历 csv_data_dir 下的每个 CSV 段文件，按“02”的方式绘图，
    并叠加 segments 中的笑容区间（用红色高亮 rank_interpolated 曲线）。
    """
    if not os.path.isdir(csv_data_dir):
        print(f"错误：CSV_DATA_DIR 不是合法目录: {csv_data_dir}")
        sys.exit(1)

    os.makedirs(out_plot_dir, exist_ok=True)

    all_csv_files = [
        os.path.join(csv_data_dir, fname)
        for fname in os.listdir(csv_data_dir)
        if fname.lower().endswith('.csv')
    ]
    if not all_csv_files:
        print(f"错误：目录中未找到任何 CSV 文件: {csv_data_dir}")
        sys.exit(1)

    all_csv_files.sort()
    print(f"将在 {len(all_csv_files)} 个 CSV 段文件上绘制笑容曲线并叠加区间。")

    max_dp_rank_index = int(num_dp_ranks) - 1

    for idx, csv_path in enumerate(all_csv_files, start=1):
        try:
            seg_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"警告：读取 CSV 失败，跳过 {os.path.basename(csv_path)}: {e}")
            continue

        required_cols = [
            'frame',
            'rank_original_0based_or_nan',
            'rank_interpolated',
            'is_long_gap'
        ]
        if not all(c in seg_df.columns for c in required_cols):
            print(f"警告：文件缺少必要列，已跳过: {os.path.basename(csv_path)}")
            continue

        # 确保类型正确
        seg_df = seg_df.copy()
        seg_df['frame'] = seg_df['frame'].astype(int)
        seg_df['rank'] = seg_df['rank_original_0based_or_nan'].astype(float)
        seg_df['rank_interpolated'] = seg_df['rank_interpolated'].astype(float)
        # is_long_gap 如果不是 bool，转 bool
        if seg_df['is_long_gap'].dtype != bool:
            seg_df['is_long_gap'] = seg_df['is_long_gap'].astype(bool)

        seg_df = seg_df.sort_values(by='frame').reset_index(drop=True)

        if seg_df.empty:
            print(f"  段 {idx}: {os.path.basename(csv_path)} 为空，跳过。")
            continue

        frames = seg_df['frame'].values
        frame_min = int(frames.min())
        frame_max = int(frames.max())

        print(f"  段 {idx}: {os.path.basename(csv_path)} (帧 {frame_min} ~ {frame_max})")

        # ---- 原始“02风格”绘图 ----
        fig, ax = plt.subplots(figsize=(18, 5))

        is_original = seg_df['rank'].notna()
        is_long_gap = seg_df['is_long_gap']
        is_short_gap = (~is_original) & (~is_long_gap)

        # 1) 原始测量点（蓝色点）
        ax.plot(
            seg_df.loc[is_original, 'frame'],
            seg_df.loc[is_original, 'rank'],
            linestyle='',
            marker='.',
            markersize=4,
            color='blue',
            label='Measured',
            zorder=3
        )

        # 2) 蓝色实线：仅连接“连续原始”区间
        measured_only = seg_df['rank_interpolated'].copy()
        measured_only[~is_original] = np.nan
        ax.plot(
            seg_df['frame'],
            measured_only,
            linestyle='-',
            marker='',
            color='blue',
            alpha=0.8,
            label='Continuous Measured',
            zorder=2
        )

        # 3) 绿色实线：短缺口插值区间
        short_only = seg_df['rank_interpolated'].copy()
        short_only[~is_short_gap] = np.nan
        ax.plot(
            seg_df['frame'],
            short_only,
            linestyle='-',
            marker='',
            color='green',
            alpha=0.8,
            label='Short Gap Interp.',
            zorder=1
        )

        # 4) 红色虚线：长缺口插值区间（保持与 02 一致）
        long_only = seg_df['rank_interpolated'].copy()
        long_only[~is_long_gap] = np.nan
        ax.plot(
            seg_df['frame'],
            long_only,
            linestyle='--',
            marker='',
            color='red',
            alpha=0.7,
            label='Long Gap Interp.',
            zorder=0
        )

        # ---- 叠加笑容区间（使用 dat：红色竖线 + 区间底色） ----
        for (s, e) in segments:
            # 和当前图的帧范围 [frame_min, frame_max] 做交集
            S = max(s, frame_min)
            E = min(e, frame_max)
            if S > E:
                continue

            # 区间底色
            ax.axvspan(S, E, facecolor='red', alpha=0.18, zorder=1)
            # 左右两条红色竖线
            ax.axvline(S, color='red', linestyle='-', linewidth=1.5, zorder=3)
            ax.axvline(E, color='red', linestyle='-', linewidth=1.5, zorder=3)


        # 轴与样式
        ax.set_xlabel('Frame Number')
        ax.set_ylabel(f'Smile Intensity Rank (0=Strongest, {max_dp_rank_index}=Weakest)')
        ax.set_title(
            f'{person.capitalize()} Smile Intensity ({date_tag}) - '
            f'Segment {idx} (Frames {frame_min}-{frame_max})'
        )
        ax.grid(True, linestyle=':')

        ax.invert_yaxis()
        ax.set_ylim(max_dp_rank_index + 1, -1)
        ax.set_xlim(frame_min - 10, frame_max + 10)
        ax.set_yticks(range(0, max_dp_rank_index + 1))

        ax.legend()
        plt.tight_layout()

        # 输出文件名：沿用原 segment 编号 + 后缀
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        out_png = os.path.join(out_plot_dir, f'{base_name}_with_segments.png')
        try:
            plt.savefig(out_png)
            print(f"    图已保存: {out_png}")
        except Exception as e:
            print(f"    错误：保存图失败: {e}")
        plt.close(fig)


# -------------------- 主流程 --------------------

def main():
    # 配置文件路径
    if len(sys.argv) >= 2:
        conf_path = sys.argv[1]
    else:
        conf_path = "01_detect_smile_events.dat"

    print(f"使用配置文件: {conf_path}")
    conf = parse_source_dat(conf_path)

    person = str(conf.get('PERSON', 'unknown_person'))
    date_tag = str(conf.get('DATE_TAG', 'unknown_date'))
    csv_data_dir = str(conf.get('CSV_DATA_DIR', '')).strip()
    output_dir_conf = str(conf.get('OUTPUT_DIR', '')).strip()
    rank_column = str(conf.get('RANK_COLUMN', 'rank_interpolated')).strip()
    rank_thresh = float(conf.get('RANK_THRESH', 3.0))
    num_dp_ranks = int(conf.get('NUM_DP_RANKS', 11))

    if not csv_data_dir:
        print("错误：CSV_DATA_DIR 未在配置文件中指定。")
        sys.exit(1)

    print(f"PERSON = {person}")
    print(f"DATE_TAG = {date_tag}")
    print(f"CSV_DATA_DIR = {csv_data_dir}")
    print(f"RANK_COLUMN = {rank_column}")
    print(f"RANK_THRESH = {rank_thresh}")
    print(f"NUM_DP_RANKS = {num_dp_ranks}")

    # 计算 BASE_OUT_DIR：与检测脚本逻辑保持一致，只是基名用 Smile_Segments
    if output_dir_conf:
        base_out_dir = output_dir_conf
    else:
        parent_dir = os.path.dirname(csv_data_dir.rstrip(r"\/"))
        base_out_dir = os.path.join(parent_dir, "Smile_Segments")

    # 区间 dat 文件所在目录：<BASE_OUT_DIR>/<PERSON_DATE_TAG_SEGMENTS>/
    segment_folder_name = f"{person}_{date_tag}_SEGMENTS"
    segment_dir = os.path.join(base_out_dir, segment_folder_name)

    # dat 文件名：与前一个脚本保持一致
    auto_dat_name = f"smile_segments_rank_{person}_{date_tag}.dat"
    conf_dat_name = str(conf.get('DEST_DAT_NAME', 'smile_segments_rank_lt3.dat')).strip()

    # 依次尝试自动名与配置名，哪个存在用哪个
    candidate_paths = [
        os.path.join(segment_dir, auto_dat_name),
        os.path.join(segment_dir, conf_dat_name)
    ]
    dat_path = None
    for p in candidate_paths:
        if p and os.path.isfile(p):
            dat_path = p
            break

    if dat_path is None:
        print("错误：在以下位置均未找到笑容区间 dat 文件：")
        for p in candidate_paths:
            print(f"  - {p}")
        sys.exit(1)

    print(f"区间 dat 文件: {dat_path}")

    # 输出图目录：<BASE_OUT_DIR>/<PERSON_DATE_TAG_SEGMENTS>/plots_with_segments/
    out_plot_dir = os.path.join(segment_dir, "plots_with_segments")
    print(f"绘图输出目录: {out_plot_dir}")

    # 读取区间
    segments = load_segments_dat(dat_path)

    # 绘图
    plot_with_segments_for_all_csv(
        csv_data_dir=csv_data_dir,
        out_plot_dir=out_plot_dir,
        segments=segments,
        person=person,
        date_tag=date_tag,
        num_dp_ranks=num_dp_ranks
    )

    print("全部绘图完成。")


if __name__ == "__main__":
    main()
