from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moviepy import VideoFileClip, VideoClip
from moviepy.video.compositing.CompositeVideoClip import clips_array


FPS = 30  # 固定帧率 30fps


def _build_segments_for_plot(frames, ranks, is_long_gap):
    """
    根据 is_long_gap 把曲线拆成若干连续区间：
    返回 [(start_idx, end_idx, is_long_gap_flag), ...]
    """
    segments = []
    if len(frames) == 0:
        return segments

    current_start = 0
    current_flag = bool(is_long_gap[0])

    for i in range(1, len(frames)):
        flag = bool(is_long_gap[i])
        if flag != current_flag:
            segments.append((current_start, i - 1, current_flag))
            current_start = i
            current_flag = flag

    segments.append((current_start, len(frames) - 1, current_flag))
    return segments


def make_smile_demo(
    video_path: str,
    csv_path: str,
    start_frame: int,
    end_frame: int,
    output_path: str,
    playback_speed: float = 1.0,
    margin_frames: int = 30,
):
    if playback_speed <= 0:
        raise ValueError("playback_speed must be > 0")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    clip = VideoFileClip(video_path)
    fps = FPS
    total_frames = int(round(clip.duration * fps))

    # ========= 帧范围 =========
    global_start = max(0, start_frame - margin_frames)
    global_end = min(total_frames - 1, end_frame + margin_frames)
    if global_end < global_start:
        raise ValueError(f"Invalid frame range: {global_start} ~ {global_end}")

    n_frames_raw = global_end - global_start + 1
    duration_raw = n_frames_raw / fps
    duration_out = duration_raw / playback_speed

    # ========= 上半部分视频 =========
    t_start = global_start / fps
    t_end = (global_end + 1) / fps  # +1 保证最后一帧包含

    subclip = clip.subclipped(t_start, t_end)
    if playback_speed != 1.0:
        subclip = subclip.with_speed_scaled(playback_speed)

    video_w, video_h = subclip.size

    # ========= 读取 SmileRanking CSV =========
    df = pd.read_csv(csv_path)

    required = {"frame", "rank_interpolated", "is_long_gap"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required}")

    df_seg = df[(df["frame"] >= global_start) & (df["frame"] <= global_end)].copy()
    if df_seg.empty:
        raise ValueError(
            f"No ranking data in specified range [{global_start}, {global_end}]."
        )

    df_seg.sort_values("frame", inplace=True)

    frames_arr = df_seg["frame"].to_numpy(int)
    ranks_arr = df_seg["rank_interpolated"].to_numpy(float)
    is_long_gap_arr = (
        df_seg["is_long_gap"].astype(str).str.lower().eq("true").to_numpy(bool)
    )

    segments = _build_segments_for_plot(frames_arr, ranks_arr, is_long_gap_arr)

    # ========= 下半部分图像（动画） =========
    lower_h = video_h // 2
    dpi = 100
    fig_w = video_w / dpi
    fig_h = lower_h / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.18)
    x_min, x_max = global_start, global_end
    y_min, y_max = 0, 10

    def make_plot_frame(t):
        # 当前帧位置（全局帧号）
        current_frame = global_start + int(round(t * fps * playback_speed))
        current_frame = int(np.clip(current_frame, global_start, global_end))

        ax.clear()
        ax.tick_params(labelsize=18)   # 刻度数字变大
        ax.set_xlabel("Frame", fontsize=20)
        ax.set_ylabel("SmileRanking (0-10)", fontsize=20)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.invert_yaxis()

        # 蓝实线 / 红虚线
        for s_idx, e_idx, long_gap in segments:
            x_seg = frames_arr[s_idx : e_idx + 1]
            y_seg = ranks_arr[s_idx : e_idx + 1]
            if len(x_seg) < 2:
                continue

            if long_gap:
                ax.plot(x_seg, y_seg, linestyle="--", color="red", linewidth=1)
            else:
                ax.plot(x_seg, y_seg, linestyle="-", color="blue", linewidth=1)

        # 当前帧黑色虚线
        ax.axvline(x=current_frame, linestyle="--", color="black", linewidth=1)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()

        # matplotlib 新版用 buffer_rgba，得到 RGBA，再丢掉 alpha
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))[..., :3]

        return buf

    lower_clip = VideoClip(make_plot_frame, duration=duration_out)
    lower_clip = lower_clip.resized(width=video_w)

    # ========= 拼接 =========
    final_clip = clips_array([[subclip], [lower_clip]])
    final_clip = final_clip.with_audio(subclip.audio)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    final_clip.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
    )

    clip.close()
    final_clip.close()
    plt.close(fig)
