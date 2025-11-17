#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import csv

# (parse_source_dat 函数保持不变)
def parse_source_dat(filepath):
    conf = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().strip("'\"")
                        value = value.split('#', 1)[0].strip()
                        try:
                            float_val = float(value)
                            if float_val.is_integer():
                                value = int(float_val)
                            else:
                                value = float_val
                        except ValueError:
                            pass
                        conf[key] = value
                    else:
                        print(f"警告：跳过格式错误的行: {line}")
    except FileNotFoundError:
        print(f"错误：源文件 {filepath} 未找到。")
        exit(1)
    except Exception as e:
        print(f"读取源文件 {filepath} 时出错: {e}")
        exit(1)
    return conf

# --- 读取配置与默认值 ---
DEFAULT_FRAMES_PER_CHUNK = 1800
DEFAULT_DASHED_GAP = 12

try:
    conf = parse_source_dat('00_source_plot.dat')
    DATE_TAG = str(conf.get('DATE_TAG', 'unknown_date'))
    PLOT_OUTPUT_DIR_BASE = os.path.join('/workspace', conf.get('PLOT_OUTPUT_DIR_BASE', ''))
    PLOT_OUTPUT_DIR = os.path.join(PLOT_OUTPUT_DIR_BASE, DATE_TAG)
    AREA_CSV_FILE = os.path.join('/workspace', conf.get('AREA_CSV_FILE', ''))
    FRAMES_PER_CHUNK = int(conf.get('FRAMES_PER_CHUNK', DEFAULT_FRAMES_PER_CHUNK))
    DASHED_GAP_THRESHOLD = int(conf.get('DASHED_GAP_THRESHOLD', DEFAULT_DASHED_GAP))
    PERSON = conf.get('PERSON', 'matsuda')
    NUM_DP_RANKS = conf.get('NUM_DP_RANKS', 11)
    MAX_DP_RANK_INDEX = int(NUM_DP_RANKS) - 1

    raw_rank_input_file = os.path.join(PLOT_OUTPUT_DIR, 'intermediate', f'frontal_ranks_raw_{PERSON}.dat')
    plot_output_subdir = os.path.join(PLOT_OUTPUT_DIR, 'plots')
    csv_output_subdir  = os.path.join(PLOT_OUTPUT_DIR, 'csv_data')

    os.makedirs(plot_output_subdir, exist_ok=True)
    os.makedirs(csv_output_subdir,  exist_ok=True)

    if not AREA_CSV_FILE or not PLOT_OUTPUT_DIR_BASE or not PERSON or not DATE_TAG:
        raise ValueError("AREA_CSV_FILE, PLOT_OUTPUT_DIR_BASE, PERSON, 或 DATE_TAG 缺失。")

except Exception as e:
    print(f"处理配置时出错: {e}")
    print(f"--- 步骤 2: 插值并分段绘图 (按 {DEFAULT_FRAMES_PER_CHUNK} 帧分块) ---")
    exit(1)

print(f"--- 步骤 2: 插值并分段绘图 (按 {FRAMES_PER_CHUNK} 帧分块) ---")

# --- 1. 读取总帧数 ---
try:
    with open(AREA_CSV_FILE, 'r') as f:
        line = f.readline()
        total_frames = int(line.strip().split(',')[-1])
    print(f"从 {os.path.basename(AREA_CSV_FILE)} 读取到总帧数: {total_frames}")
except FileNotFoundError:
    print(f"错误：总帧数文件未找到: {AREA_CSV_FILE}"); exit(1)
except (IndexError, ValueError, Exception) as e:
    print(f"错误：无法从 {os.path.basename(AREA_CSV_FILE)} 解析总帧数: {e}"); exit(1)
if total_frames <= 0:
    print("错误：总帧数必须为正数。"); exit(1)

# --- 2. 读取正面帧排名 ---
ranks_data = {}
try:
    with open(raw_rank_input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                filename = row[0].strip()
                try:
                    frame_num = int(filename.split('_')[-1].split('.')[0])
                    original_rank = float(row[1].strip())
                    if 1.0 <= original_rank <= float(NUM_DP_RANKS):
                        desired_rank = (original_rank - 1.0)
                        ranks_data[frame_num] = {'rank': desired_rank, 'filename': filename}
                    elif original_rank == -1.0:
                        ranks_data[frame_num] = {'rank': np.nan, 'filename': filename}
                    else:
                        print(f"警告: 帧 {frame_num} 原始排名 {original_rank:.1f} 超出范围 [1,{int(NUM_DP_RANKS)}]，已忽略。")
                        ranks_data[frame_num] = {'rank': np.nan, 'filename': filename}
                except (IndexError, ValueError):
                    print(f"警告：无法解析行: {row}")
            else:
                print(f"警告：跳过格式错误的行: {row}")
except FileNotFoundError:
    print(f"错误：原始排名文件未找到: {raw_rank_input_file}"); exit(1)
except Exception as e:
    print(f"读取原始排名文件时出错: {e}"); exit(1)
print(f"读取了 {len(ranks_data)} 个正面帧的排名信息。")

# --- 3. 构造完整时间序列 + 插值 + 长缺口判定 ---
all_frames = pd.DataFrame({'frame': range(total_frames)})
frontal_ranks_list = [{'frame': k, **v} for k, v in ranks_data.items()]
frontal_ranks_df = pd.DataFrame(frontal_ranks_list)
full_data = pd.merge(all_frames, frontal_ranks_df, on='frame', how='left')
full_data['rank'] = full_data['rank'].astype(float)
full_data['rank_interpolated'] = full_data['rank'].interpolate(method='linear', limit_direction='both', limit_area=None)
full_data['is_long_gap'] = False

nan_mask = full_data['rank'].isnull()
group_ids = (nan_mask != nan_mask.shift()).cumsum()
gap_lengths = nan_mask.groupby(group_ids).transform('sum')
full_data.loc[nan_mask & (gap_lengths > DASHED_GAP_THRESHOLD), 'is_long_gap'] = True
full_data = full_data[['frame', 'rank', 'rank_interpolated', 'is_long_gap', 'filename']].copy()
print(f"已完成插值与长缺口标记 (阈值={DASHED_GAP_THRESHOLD} 帧)。")

# --- 4. 分块绘图和保存 CSV ---
num_chunks = math.ceil(total_frames / FRAMES_PER_CHUNK)
print(f"将数据分为 {num_chunks} 个片段 (每段 {FRAMES_PER_CHUNK} 帧) 进行处理...")

for i in range(num_chunks):
    start_frame = i * FRAMES_PER_CHUNK
    end_frame = min((i + 1) * FRAMES_PER_CHUNK, total_frames)
    segment_df = full_data.loc[start_frame:end_frame-1].copy()

    if segment_df.empty:
        print(f"  处理片段 {i+1}/{num_chunks} (帧 {start_frame} 到 {end_frame-1})... 无数据，跳过。")
        continue

    print(f"  处理片段 {i+1}/{num_chunks} (帧 {start_frame} 到 {end_frame-1})...")

    # --- 绘图：分离连续原始段与短缺口段 ---
    fig, ax = plt.subplots(figsize=(18, 5))
    is_original = segment_df['rank'].notna()
    is_long_gap = segment_df['is_long_gap']
    is_short_gap = (~is_original) & (~is_long_gap)

    # 1) 原始测量点（蓝色点）
    ax.plot(segment_df.loc[is_original, 'frame'],
            segment_df.loc[is_original, 'rank'],
            linestyle='', marker='.', markersize=4, color='blue',
            label='Measured', zorder=3)

    # 2) 蓝色实线：仅连接“连续原始”区间
    #   将非原始帧置为 NaN，避免跨缺口连接
    measured_only = segment_df['rank_interpolated'].copy()
    measured_only[~is_original] = np.nan
    ax.plot(segment_df['frame'], measured_only,
            linestyle='-', marker='', color='blue', alpha=0.8,
            label='Continuous Measured', zorder=2)

    # 3) 绿色实线：短缺口插值区间
    #   仅在短缺口帧上绘制，避免与原始段混线
    short_only = segment_df['rank_interpolated'].copy()
    short_only[~is_short_gap] = np.nan
    ax.plot(segment_df['frame'], short_only,
            linestyle='-', marker='', color='green', alpha=0.8,
            label=f'Short Gap (≤ {DASHED_GAP_THRESHOLD} frames) Interp.', zorder=1)

    # 4) 红色虚线：长缺口插值区间
    long_only = segment_df['rank_interpolated'].copy()
    long_only[~is_long_gap] = np.nan
    ax.plot(segment_df['frame'], long_only,
            linestyle='--', marker='', color='red',
            label=f'Long Gap (> {DASHED_GAP_THRESHOLD} frames) Interp.', zorder=0)

    # 轴与样式
    ax.set_xlabel('Frame Number')
    ax.set_ylabel(f'Smile Intensity Rank (0=Strongest, {MAX_DP_RANK_INDEX}=Weakest)')
    ax.set_title(f'{PERSON.capitalize()} Smile Intensity ({DATE_TAG}) - Segment {i+1} (Frames {start_frame}-{end_frame-1})')
    ax.grid(True, linestyle=':')
    ax.invert_yaxis()
    ax.set_ylim(MAX_DP_RANK_INDEX + 1, -1)
    ax.set_xlim(start_frame - 10, end_frame + 10)
    ax.set_yticks(range(0, MAX_DP_RANK_INDEX + 1))
    ax.legend()
    plt.tight_layout()

    # 保存图
    plot_filename = os.path.join(plot_output_subdir, f'smile_curve_segment_{i}.png')
    try:
        plt.savefig(plot_filename)
        print(f"    图表已保存: {os.path.basename(plot_filename)}")
    except Exception as e:
        print(f"    错误：保存图表失败: {e}")
    plt.close(fig)

    # 保存 CSV
    csv_filename = os.path.join(csv_output_subdir, f'smile_data_segment_{i}.csv')
    try:
        output_df = segment_df[['frame', 'rank', 'rank_interpolated', 'is_long_gap', 'filename']].rename(
            columns={'rank': 'rank_original_0based_or_nan'}
        )
        output_df.to_csv(csv_filename, index=False, float_format='%.3f')
        print(f"    CSV 数据已保存: {os.path.basename(csv_filename)}")
    except Exception as e:
        print(f"    错误：保存 CSV 失败: {e}")

print("--- 步骤 2 完成 ---")
