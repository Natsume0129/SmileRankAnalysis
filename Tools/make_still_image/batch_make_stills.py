"""
batch_make_stills.py

读取一个仅包含 start_frame / end_frame 的 dat 文件，
对每一行片段生成一张静止画。

依赖：
- make_still_image_core.py (同目录)
- Pillow
"""

from __future__ import annotations

import os

from make_still_image_core import make_still_image


# ================== 可配置参数 ==================

# 存放单帧图片的目录
FRAMES_DIR = r"E:\Matsuda_data\single_frame\20251029\DetectedFaces\20251029\0\0"

# 文件名中的 DATE 前缀，例如 20251029
DATE_PREFIX = "20251029"

# 仅包含起止帧的 dat 文件
# 格式示例：
#   # start_frame end_frame
#   29023 29297
#   75329 75506
DAT_PATH = r"E:\Matsuda_data\classification\20251029.dat"

# 输出静止画目录
OUTPUT_DIR = r"E:\Matsuda_data\classification\still_examples\20251029"

# 帧率（目前接口保留，实际用不到复杂逻辑，你现在统一 30 即可）
FPS = 30

# 帧间隔（gap=2 → 每隔一帧取一张）
GAP = 2

# 每一行拼接多少张图片
PICS_PER_LINE = 15

# 若只想先测试前 N 个片段，设为正整数；设为 None 则处理全部
MAX_SAMPLES = None

# 单帧图片尺寸（你现在是 224）
TILE_SIZE = 224

# ===============================================


def parse_dat_file(dat_path: str):
    """
    解析只包含 start_frame / end_frame 的 dat 文件。
    返回列表：[ (start_frame, end_frame), ... ]。
    """
    segments = []
    with open(dat_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                start_frame = int(parts[0])
                end_frame = int(parts[1])
            except ValueError:
                continue

            segments.append((start_frame, end_frame))

    return segments


def main() -> None:
    if not os.path.exists(FRAMES_DIR):
        raise FileNotFoundError(f"FRAMES_DIR not found: {FRAMES_DIR}")
    if not os.path.exists(DAT_PATH):
        raise FileNotFoundError(f"DAT_PATH not found: {DAT_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    segments = parse_dat_file(DAT_PATH)
    if not segments:
        print("No segments found in dat file.")
        return

    if MAX_SAMPLES is not None:
        segments = segments[:MAX_SAMPLES]

    print(f"Found {len(segments)} segments. Start making still images...")

    for idx, (start_frame, end_frame) in enumerate(segments, start=1):
        filename = f"{start_frame}-{end_frame}_gap{GAP}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)

        print(
            f"[{idx}/{len(segments)}] "
            f"segment {start_frame}-{end_frame} -> {output_path}"
        )

        make_still_image(
            start_frame=start_frame,
            end_frame=end_frame,
            fps=FPS,
            gap=GAP,
            pics_per_line=PICS_PER_LINE,
            frames_dir=FRAMES_DIR,
            date_prefix=DATE_PREFIX,
            output_path=output_path,
            tile_size=TILE_SIZE,
        )

    print("All still images generated.")


if __name__ == "__main__":
    main()
