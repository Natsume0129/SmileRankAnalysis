#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze.py (pooled, multi-day)

- 输入：segments_dir（.dat 标注文件目录），smilerank_dir（smile rank CSV 目录）
- 自动根据文件名中的 8 位日期进行一一匹配，例如：
    segments:      output20250926.dat
    smilerank:     smile_data_merged_20250926.csv
- 输出：跨所有日期 pooled 的统计，只按 label 区分，不看日期。

每个区间计算：
- duration_frames
- mean_rank
- peak_rank (强度最强处)
- peak_frame
- peak_pos_frac (峰值在区间内的位置 0~1)
- std_rank
- dynamic_range
- peak_position_category (early/middle/late)
- variation_category (flat/dynamic)
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_date_from_name(name: str) -> Optional[str]:
    """从文件名中提取 8 位日期（例如 20250926）。"""
    m = re.search(r"(\d{8})", name)
    return m.group(1) if m else None


def parse_segments_file(path: Path) -> pd.DataFrame:
    """解析 .dat 片段文件：
    # start_frame end_frame label remark
    155 299 false_smile ...
    """
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=3)
            if len(parts) < 3:
                continue
            start = int(parts[0])
            end = int(parts[1])
            label = parts[2]
            remark = parts[3] if len(parts) == 4 else ""
            rows.append((start, end, label, remark))

    return pd.DataFrame(rows, columns=["start_frame", "end_frame", "label", "remark"])


def load_smilerank_file(path: Path, frame_col: str, rank_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if frame_col not in df.columns:
        raise ValueError(f"{path} missing frame column '{frame_col}'")
    if rank_col not in df.columns:
        raise ValueError(f"{path} missing rank column '{rank_col}'")

    df = df[[frame_col, rank_col]].copy()
    df = df.sort_values(frame_col).reset_index(drop=True)
    return df


def analyze_segment(
    seg_row: pd.Series,
    sr_df: pd.DataFrame,
    frame_col: str,
    rank_col: str,
    lower_is_stronger: bool,
) -> dict:
    start = int(seg_row["start_frame"])
    end = int(seg_row["end_frame"])
    label = seg_row["label"]
    remark = seg_row["remark"]

    mask = (sr_df[frame_col] >= start) & (sr_df[frame_col] <= end)
    seg = sr_df.loc[mask]

    duration = end - start

    if seg.empty:
        return {
            "start_frame": start,
            "end_frame": end,
            "label": label,
            "remark": remark,
            "duration_frames": duration,
            "mean_rank": np.nan,
            "peak_rank": np.nan,
            "peak_frame": np.nan,
            "peak_pos_frac": np.nan,
            "std_rank": np.nan,
            "dynamic_range": np.nan,
            "peak_position_category": "missing",
            "variation_category": "missing",
        }

    ranks = seg[rank_col].to_numpy()
    frames = seg[frame_col].to_numpy()

    mean_rank = float(np.mean(ranks))
    std_rank = float(np.std(ranks))
    dynamic_range = float(np.max(ranks) - np.min(ranks))

    # 峰值 = 笑容最强的位置
    if lower_is_stronger:
        peak_idx = int(np.argmin(ranks))
    else:
        peak_idx = int(np.argmax(ranks))

    peak_rank = float(ranks[peak_idx])
    peak_frame = int(frames[peak_idx])

    if duration > 0:
        frac = (peak_frame - start) / duration
        peak_pos_frac = float(np.clip(frac, 0.0, 1.0))
    else:
        peak_pos_frac = np.nan

    # 峰值位置类别
    if np.isnan(peak_pos_frac):
        peak_pos_category = "missing"
    elif peak_pos_frac < 1.0 / 3:
        peak_pos_category = "early"
    elif peak_pos_frac > 2.0 / 3:
        peak_pos_category = "late"
    else:
        peak_pos_category = "middle"

    # 变化程度
    variation_category = "flat" if dynamic_range < 0.5 else "dynamic"

    return {
        "start_frame": start,
        "end_frame": end,
        "label": label,
        "remark": remark,
        "duration_frames": duration,
        "mean_rank": mean_rank,
        "peak_rank": peak_rank,
        "peak_frame": peak_frame,
        "peak_pos_frac": peak_pos_frac,
        "std_rank": std_rank,
        "dynamic_range": dynamic_range,
        "peak_position_category": peak_pos_category,
        "variation_category": variation_category,
    }


def make_plots(df: pd.DataFrame, outdir: Path, invert_rank_axis: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    def boxplot(metric: str, filename: str, ylabel: str, invert: bool = False):
        data = []
        labels = []
        for lbl, g in df.groupby("label"):
            vals = g[metric].dropna().to_numpy()
            if len(vals) == 0:
                continue
            labels.append(lbl)
            data.append(vals)

        if not data:
            return

        plt.figure()
        # 注意：Matplotlib 3.9 起推荐 tick_labels
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by label")
        if invert:
            plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(outdir / filename)
        plt.close()

    # 时长：不反转
    boxplot(
        "duration_frames",
        "duration_by_label.png",
        "Duration (frames)",
        invert=False,
    )
    # 平均强度：越小越强 → 如果 lower_is_stronger，则反转纵轴
    boxplot(
        "mean_rank",
        "mean_rank_by_label.png",
        "Mean smile rank (lower = stronger)",
        invert=invert_rank_axis,
    )
    # 峰值位置 0~1：不反转
    boxplot(
        "peak_pos_frac",
        "peak_pos_by_label.png",
        "Peak position inside segment (0=start, 1=end)",
        invert=False,
    )


def summarize_by_label(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "duration_frames",
        "mean_rank",
        "peak_rank",
        "peak_pos_frac",
        "std_rank",
        "dynamic_range",
    ]
    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
        agg[f"{m}_count"] = (m, "count")

    summary = df.groupby("label").agg(**agg).reset_index()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Pooled smile segment analysis.")
    parser.add_argument(
        "--segments-dir",
        type=str,
        required=True,
        help="Directory containing segment .dat files (e.g., output20250926.dat).",
    )
    parser.add_argument(
        "--smilerank-dir",
        type=str,
        required=True,
        help="Directory containing smile-rank CSV files (e.g., smile_data_merged_20250926.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write pooled analysis outputs.",
    )
    parser.add_argument(
        "--frame-col",
        type=str,
        default="frame",
        help="Frame index column name in smilerank CSV.",
    )
    parser.add_argument(
        "--rank-col",
        type=str,
        default="rank_interpolated",
        help="Smile rank column name in smilerank CSV.",
    )
    parser.add_argument(
        "--lower-is-stronger",
        action="store_true",
        help="If set, smaller rank value means stronger smile.",
    )

    args = parser.parse_args()

    segments_dir = Path(args.segments_dir)
    smilerank_dir = Path(args.smilerank_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 构建 smilerank 映射：date -> path
    smilerank_map = {}
    for p in smilerank_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".csv":
            continue
        date_str = extract_date_from_name(p.name)
        if date_str:
            smilerank_map[date_str] = p

    all_results = []

    # 遍历 segments 目录，按日期匹配
    for seg_path in sorted(segments_dir.iterdir()):
        if not seg_path.is_file():
            continue
        if seg_path.suffix.lower() not in (".dat", ".txt"):
            continue

        date_str = extract_date_from_name(seg_path.name)
        if not date_str:
            print(f"[WARN] Cannot extract date from segments file: {seg_path.name}")
            continue

        if date_str not in smilerank_map:
            print(
                f"[WARN] No smilerank CSV found for date {date_str} (segments: {seg_path.name})"
            )
            continue

        sr_path = smilerank_map[date_str]
        print(f"[INFO] Processing date {date_str}:")
        print(f"       segments  = {seg_path}")
        print(f"       smilerank = {sr_path}")

        seg_df = parse_segments_file(seg_path)
        sr_df = load_smilerank_file(sr_path, args.frame_col, args.rank_col)

        for idx, row in seg_df.iterrows():
            res = analyze_segment(
                row,
                sr_df,
                args.frame_col,
                args.rank_col,
                args.lower_is_stronger,
            )
            res["session_date"] = date_str
            res["segment_file"] = seg_path.name
            res["smilerank_file"] = sr_path.name
            res["segment_index"] = idx
            all_results.append(res)

    if not all_results:
        print("[ERROR] No segments analyzed. Check your directories and naming.")
        return

    segments_df = pd.DataFrame(all_results)
    segments_csv = outdir / "segments_analysis.csv"
    segments_df.to_csv(segments_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved pooled per-segment analysis to: {segments_csv}")

    summary_df = summarize_by_label(segments_df)
    summary_csv = outdir / "label_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved pooled label summary to: {summary_csv}")

    make_plots(segments_df, outdir, invert_rank_axis=args.lower_is_stronger)
    print(f"[INFO] Plots saved in: {outdir}")


if __name__ == "__main__":
    main()
