# Smile Rank Analysis Toolkit  
### 微笑强度分析工具集｜笑顔ランク解析ツールセット

## 🇨🇳 中文说明
本项目用于对 **SmileRank（微笑强度曲线）** 数据进行插值、滤波、检测与可视化分析。
适用于基于帧序列的笑容动态分析研究（例如面部表情实验、情感识别等）。

### 文件说明
| 文件名 | 功能描述 |
|--------|-----------|
| run_02_interpolate_and_plot.py | 读取原始 `.dat` 文件，对笑容排名进行插值与分段绘图（每900帧一张图）。 |
| smile_rank_filter_and_plot.py | 对所有分段CSV进行拼接、滤波与重新绘图，输出平滑后的曲线图与合并CSV。 |
| smile_rank_filter_compare.py | 对比多种滤波算法（Savitzky-Golay、Gaussian、Moving Average等）的效果。 |
| detect_and_plot_smile_events.py | 早期笑容检测版本：基于突出度（prominence）与阈值检测笑容段。 |
| detect_and_plot_smile_events_corelogic.py | 最新稳定版笑容检测逻辑，基于核心阈值（≤3）与局部基线扩展确定笑容起止。输出 `.dat` 文件与绘图。 |
| pairwise_stitch_plots.py | 将滤波前后对应图像上下拼接对比。 |
| build_labeled_compare_stacks_0_9.py | 生成0–9段的多算法对比图表，每个算法对应一张标注有算法名称的子图。 |
| index.html | 可选的可视化入口，用于网页展示结果图。 |

## 🇯🇵 日本語説明
このプロジェクトは **SmileRank（微笑強度曲線）** データを用いた補間・平滑化・笑顔区間検出・可視化のためのツールセットです。

### スクリプト説明
| ファイル名 | 説明 |
|-------------|------|
| run_02_interpolate_and_plot.py | 元データからランクを補間し、900フレームごとに描画。 |
| smile_rank_filter_and_plot.py | 全CSVを結合し、フィルタリング＋再描画。平滑化後のデータを出力。 |
| smile_rank_filter_compare.py | 複数の平滑化手法（Savitzky–Golay・Gaussian・移動平均など）の比較。 |
| detect_and_plot_smile_events.py | 初期の笑顔検出ロジック。ピークとしきい値を用いて笑顔区間を抽出。 |
| detect_and_plot_smile_events_corelogic.py | 改良版：rank≤3 を核とし、局所ベースラインで笑顔の開始・終了を決定。 |
| pairwise_stitch_plots.py | 元画像とフィルタ後画像を上下に連結して比較。 |
| build_labeled_compare_stacks_0_9.py | 0〜9セグメントの各フィルタ結果を縦に並べた比較画像を生成。 |
| index.html | 結果可視化用の簡易HTML。 |

## 🇬🇧 English Description
A toolkit for analyzing **SmileRank (smile intensity curve)** data — including interpolation, filtering, smile-event detection, and visualization.

### Script Overview
| File | Description |
|------|-------------|
| run_02_interpolate_and_plot.py | Interpolates raw rank data and plots per 900 frames. |
| smile_rank_filter_and_plot.py | Concatenates all CSV segments, applies filtering, and redraws smooth curves. |
| smile_rank_filter_compare.py | Compares multiple smoothing filters. |
| detect_and_plot_smile_events.py | Early smile detection logic. |
| detect_and_plot_smile_events_corelogic.py | Final version using rank ≤ 3 + local baseline expansion. |
| pairwise_stitch_plots.py | Stitches original vs filtered plots vertically. |
| build_labeled_compare_stacks_0_9.py | Generates multi-algorithm comparison charts for segments 0–9. |
| index.html | Optional viewer for displaying generated plots. |
