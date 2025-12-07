"""
make_still_image_core.py

从一段帧区间中，按 gap 采样若干帧，
将对应的 224x224 单帧图片按网格拼成一张静止画。

如果某帧对应的图片不存在，则用 224x224 的白色占位图代替。
"""

from __future__ import annotations

import os
import math
from typing import List

from PIL import Image


def _ensure_dir(path: str) -> None:
    """确保输出目录存在。"""
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _load_frame_image_or_white(
    frame: int,
    frames_dir: str,
    date_prefix: str,
    size: int = 224,
) -> Image.Image:
    """
    尝试读取某一帧对应的单帧图片；
    若不存在，则返回一张白色占位图。
    """
    filename = f"{date_prefix}_0_0_{frame}.png"
    full_path = os.path.join(frames_dir, filename)

    if os.path.exists(full_path):
        try:
            img = Image.open(full_path).convert("RGB")
            if img.size != (size, size):
                img = img.resize((size, size))
            return img
        except Exception:
            # 读图失败时也使用白色占位图
            pass

    # 白色占位图
    return Image.new("RGB", (size, size), color=(255, 255, 255))


def make_still_image(
    start_frame: int,
    end_frame: int,
    fps: int,
    gap: int,
    pics_per_line: int,
    frames_dir: str,
    date_prefix: str,
    output_path: str,
    tile_size: int = 224,
) -> None:
    """
    生成一张静止画，由若干 224x224 的单帧图片拼接而成。

    参数
    ----
    start_frame : 片段起始帧（包含）
    end_frame   : 片段结束帧（包含）
    fps         : 帧率（目前仅保留接口，你现在可固定传 30）
    gap         : 帧间隔（1=每帧，2=隔一帧，等）
    pics_per_line : 每一行拼接多少张图片
    frames_dir  : 存放单帧图片的目录
    date_prefix : 文件名前缀，即 DATE 部分
    output_path : 输出静止画路径（.png）
    tile_size   : 单帧图片尺寸（默认 224）
    """
    if gap <= 0:
        raise ValueError("gap must be a positive integer")

    if pics_per_line <= 0:
        raise ValueError("pics_per_line must be a positive integer")

    # 选取帧列表
    frames: List[int] = list(range(start_frame, end_frame + 1, gap))
    if not frames:
        raise ValueError(f"No frames selected in range [{start_frame}, {end_frame}]")

    # 加载所有单帧图片（或白色占位图）
    tiles: List[Image.Image] = [
        _load_frame_image_or_white(frame, frames_dir, date_prefix, size=tile_size)
        for frame in frames
    ]

    num_tiles = len(tiles)
    rows = math.ceil(num_tiles / pics_per_line)

    # 目标大图尺寸
    total_width = pics_per_line * tile_size
    total_height = rows * tile_size

    # 背景默认使用白色（即使最后一行不足，也是白底）
    canvas = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))

    for idx, tile in enumerate(tiles):
        row = idx // pics_per_line
        col = idx % pics_per_line
        x = col * tile_size
        y = row * tile_size
        canvas.paste(tile, (x, y))

    _ensure_dir(output_path)
    canvas.save(output_path)
