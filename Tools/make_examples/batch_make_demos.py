"""
batch_make_demos.py

批量处理 dat 文件中带有 "tag" 标记的片段，
对每个片段调用 smile_demo_core.make_smile_demo 生成一个演示视频。

输出文件命名规则："{start_frame}-{end_frame}.mp4"
"""

from __future__ import annotations

import os

from smile_demo_core import make_smile_demo


# ====================== 可配置参数 ======================

VIDEO_PATH = r"E:\Matsuda_data\20251008\20251008.mp4"
CSV_PATH = r"E:\Matsuda_data\20251008\smile_data_merged_20251008.csv"
DAT_PATH = r"E:\Matsuda_data\examples\20251008.dat"
OUTPUT_DIR = r"E:\Matsuda_data\example_outputs\20251008"

PLAYBACK_SPEED = 0.5  # 播放速度
MARGIN_FRAMES = 30    # 前后 margin 帧数（默认 30 = 1 秒）

# 如需要，只生成前 N 个 tag 片段用于测试；设为 None 则处理全部
MAX_SAMPLES = None

# ========================================================


def parse_dat_file(dat_path: str):
    """
    解析 dat 文件，返回一个列表：
    [
        {
            "start_frame": int,
            "end_frame": int,
            "label": str,
            "remark": str,
        },
        ...
    ]
    仅返回 remark 中包含 "tag"（大小写不敏感）的条目。
    """
    samples = []
    with open(dat_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 4:
                # 格式不符合约定，跳过
                continue

            try:
                start_frame = int(parts[0])
                end_frame = int(parts[1])
            except ValueError:
                # 非整数，跳过
                continue

            label = parts[2]
            remark = " ".join(parts[3:])

            if "tag" not in remark.lower():
                continue

            samples.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "label": label,
                    "remark": remark,
                }
            )
    return samples


def main() -> None:
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"VIDEO_PATH not found: {VIDEO_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV_PATH not found: {CSV_PATH}")
    if not os.path.exists(DAT_PATH):
        raise FileNotFoundError(f"DAT_PATH not found: {DAT_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    samples = parse_dat_file(DAT_PATH)
    if not samples:
        print("No tagged samples found in dat file.")
        return

    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]

    print(f"Found {len(samples)} tagged segments. Start processing...")

    for idx, item in enumerate(samples, start=1):
        start_frame = item["start_frame"]
        end_frame = item["end_frame"]
        label = item["label"]
        remark = item["remark"]

        # 输出文件名：start_frame-end_frame.mp4
        filename = f"{start_frame}-{end_frame}.mp4"
        output_path = os.path.join(OUTPUT_DIR, filename)

        print(
            f"[{idx}/{len(samples)}] "
            f"Processing {start_frame}-{end_frame} "
            f"label={label} remark={remark!r}"
        )

        make_smile_demo(
            video_path=VIDEO_PATH,
            csv_path=CSV_PATH,
            start_frame=start_frame,
            end_frame=end_frame,
            output_path=output_path,
            playback_speed=PLAYBACK_SPEED,
            margin_frames=MARGIN_FRAMES,
        )

    print("All tagged segments processed.")


if __name__ == "__main__":
    main()
