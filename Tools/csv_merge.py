#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
csv_merge.py

把同一个文件夹里的多个 CSV（如 smile_data_segment_0.csv, smile_data_segment_1.csv, ...）
按编号顺序拼接成一个大的 CSV。
"""

import os
import csv
import re
from typing import List, Tuple

# ================== 可修改区域 ==================
# 输入 CSV 所在目录
INPUT_DIR = r"E:\Matsuda_data\20251029\20251029_plots\20251029\csv_data"

# 输出 CSV 目录
OUTPUT_DIR = r"E:\Matsuda_data\20251029"

# 输入文件的前缀（例如：smile_data_segment_0.csv）
FILENAME_PREFIX = "smile_data_segment_"

# 输出文件名
OUTPUT_FILENAME = "smile_data_merged_20251029.csv"

# 是否认为每个小 CSV 都带有相同的表头
HAS_HEADER = True
# ================== 可修改区域 ==================


def find_segment_files(
    input_dir: str,
    prefix: str,
) -> List[Tuple[int, str]]:
    """
    在 input_dir 中查找形如 prefix + <数字>.csv 的文件，
    返回 [(编号, 完整路径), ...] 列表。
    """
    pattern = re.compile(r"^" + re.escape(prefix) + r"(\d+)\.csv$", re.IGNORECASE)
    files: List[Tuple[int, str]] = []

    for name in os.listdir(input_dir):
        match = pattern.match(name)
        if match:
            idx = int(match.group(1))
            full_path = os.path.join(input_dir, name)
            files.append((idx, full_path))

    return sorted(files, key=lambda x: x[0])


def merge_csv_files(
    files_with_index: List[Tuple[int, str]],
    output_path: str,
    has_header: bool = True,
) -> None:
    """
    按顺序合并 CSV。
    has_header=True 时：只保留第一个文件的表头，后面的文件跳过第一行。
    """
    if not files_with_index:
        print("没有找到匹配的 CSV 文件。")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = None

    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        for i, (idx, csv_path) in enumerate(files_with_index):
            with open(csv_path, "r", newline="", encoding="utf-8") as fin:
                reader = csv.reader(fin)

                # 处理表头
                if has_header:
                    header = next(reader, None)
                    if writer is None:
                        writer = csv.writer(fout)
                        if header is not None:
                            writer.writerow(header)
                else:
                    if writer is None:
                        writer = csv.writer(fout)

                # 写入数据行
                for row in reader:
                    # 如果有空行，可以根据需要选择是否过滤
                    # 这里简单跳过完全空的行
                    if not row:
                        continue
                    writer.writerow(row)

    print(f"合并完成，共合并 {len(files_with_index)} 个文件。")
    print(f"输出文件：{output_path}")


def main():
    input_dir = os.path.abspath(INPUT_DIR)
    output_dir = os.path.abspath(OUTPUT_DIR)
    output_path = os.path.join(output_dir, OUTPUT_FILENAME)

    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_path}")

    files_with_index = find_segment_files(input_dir, FILENAME_PREFIX)

    if not files_with_index:
        print("未在输入目录中找到符合命名规则的文件："
              f"{FILENAME_PREFIX}<数字>.csv")
        return

    print("将按以下顺序合并：")
    for idx, path in files_with_index:
        print(f"  index={idx}: {os.path.basename(path)}")

    merge_csv_files(files_with_index, output_path, has_header=HAS_HEADER)


if __name__ == "__main__":
    main()
