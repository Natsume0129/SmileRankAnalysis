#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze.py  (version: no-name-matching)

Analyze smile segments using explicitly provided segment files and smile-rank files.

=== INPUT ===
You provide:
    --segments    seg_file1 seg_file2 seg_file3 ...
    --smileranks  rank_file1 rank_file2 rank_file3 ...
These lists are positionally matched:
    segments[i]  <->  smileranks[i]

=== OUTPUT ===
    - segments_analysis.csv
    - label_summary.csv
    - duration_by_label.png
    - mean_rank_by_label.png
    - peak_pos_by_label.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_segments_file(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
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


def load_smilerank_file(path: Path, frame_col: str, rank_col: str):
    df = pd.read_csv(path)
    if frame_col not in df.columns:
        raise ValueError(f"{path} missing frame column '{frame_col}'")
    if rank_col not in df.columns:
        raise ValueError(f"{path} missing rank column '{rank_col}'")

    df = df[[frame_col, rank_col]].copy()
    df = df.sort_values(frame_col).reset_index(drop=True)
    return df


def analyze_segment(
    seg_row,
    sr_df,
    frame_col: str,
    rank_col: str,
    lower_is_stronger: bool,
):
    start = seg_row["start_frame"]
    end = seg_row["end_frame"]
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

    # Peak = strongest
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

    # Basic curve categories
    if np.isnan(peak_pos_frac):
        peak_pos_category = "missing"
    elif peak_pos_frac < 1 / 3:
        peak_pos_category = "early"
    elif peak_pos_frac > 2 / 3:
        peak_pos_category = "late"
    else:
        peak_pos_category = "middle"

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


def make_plots(df: pd.DataFrame, outdir: Path):

    def boxplot(metric, filename, ylabel):
        d = []
        labels = []
        for lbl, g in df.groupby("label"):
            val = g[metric].dropna().to_numpy()
            if len(val) == 0:
                continue
            labels.append(lbl)
            d.append(val)

        if not d:
            return

        plt.figure()
        plt.boxplot(d, labels=labels, showmeans=True)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by label")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(outdir / filename)
        plt.close()

    boxplot("duration_frames", "duration_by_label.png", "Duration (frames)")
    boxplot("mean_rank", "mean_rank_by_label.png", "Mean Smile Rank")
    boxplot("peak_pos_frac", "peak_pos_by_label.png", "Peak Position Fraction")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", nargs="+", required=True,
                        help="List of segment .dat files (in order).")
    parser.add_argument("--smileranks", nargs="+", required=True,
                        help="List of smile-rank CSV files (in order).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-col", dest="frame_col", default="frame")
    parser.add_argument("--rank-col", dest="rank_col", default="rank_smoothed")
    parser.add_argument("--lower-is-stronger", action="store_true")

    args = parser.parse_args()

    if len(args.segments) != len(args.smileranks):
        raise ValueError("segments and smileranks must have same length.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seg_path_str, sr_path_str in zip(args.segments, args.smileranks):
        seg_path = Path(seg_path_str)
        sr_path = Path(sr_path_str)

        print(f"[INFO] Processing pair:")
        print(f"       segments = {seg_path}")
        print(f"       smilerank = {sr_path}")

        seg_df = parse_segments_file(seg_path)
        sr_df = load_smilerank_file(sr_path, args.frame_col, args.rank_col)

        for idx, row in seg_df.iterrows():
            res = analyze_segment(
                row,
                sr_df,
                args.frame_col,
                args.rank_col,
                args.lower_is_stronger
            )
            res["segment_file"] = seg_path.name
            res["smilerank_file"] = sr_path.name
            res["segment_index"] = idx

            all_results.append(res)

    df = pd.DataFrame(all_results)
    df.to_csv(outdir / "segments_analysis.csv", index=False, encoding="utf-8-sig")

    # Summary by label
    summary = df.groupby("label").agg({
        "duration_frames": "mean",
        "mean_rank": "mean",
        "peak_rank": "mean",
        "peak_pos_frac": "mean",
    }).reset_index()
    summary.to_csv(outdir / "label_summary.csv", index=False, encoding="utf-8-sig")

    make_plots(df, outdir)

    print(f"[INFO] All outputs saved under {outdir}")


if __name__ == "__main__":
    main()
