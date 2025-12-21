# SmileRankAnalysis

SmileRank（微笑强度/笑顔ランク曲线）数据的分析工具集：**插值 → 平滑滤波 → 笑容区间检测 → 可视化对比 → 结果汇总/展示**。适用于基于帧序列的表情动态分析研究（例如 Zoom/实验录像的面部表情时间序列分析）。 :contentReference[oaicite:0]{index=0}

---

## What this repo does

Given SmileRank time-series (typically per-frame rank values), this toolkit helps you:

- **Interpolate** raw rank signals and split long sequences into manageable segments (e.g., 900 frames per plot). :contentReference[oaicite:1]{index=1}
- **Filter / smooth** the curve with multiple methods and export merged CSV + plots. :contentReference[oaicite:2]{index=2}
- **Compare filters** (Savitzky–Golay / Gaussian / Moving Average, etc.) side-by-side. :contentReference[oaicite:3]{index=3}
- **Detect smile events** using a stable “core-threshold + local baseline expansion” logic (rank ≤ 3 as core). :contentReference[oaicite:4]{index=4}
- **Generate visual reports** (stitched before/after plots, stacked comparisons, simple HTML viewer). :contentReference[oaicite:5]{index=5}

---

## Repository structure (high level)

Top-level scripts (main workflow utilities):

- `run_02_interpolate_and_plot.py`  
  Interpolate raw `.dat` rank data; plot per segment (e.g., 900 frames). :contentReference[oaicite:6]{index=6}
- `smile_rank_filter_and_plot.py`  
  Concatenate all segment CSVs → filter/smooth → redraw → export merged CSV + plots. :contentReference[oaicite:7]{index=7}
- `smile_rank_filter_compare.py`  
  Compare multiple smoothing filters. :contentReference[oaicite:8]{index=8}
- `detect_and_plot_smile_events.py`  
  Early smile-event detection (peak prominence + thresholds). :contentReference[oaicite:9]{index=9}
- `detect_and_plot_smile_events_corelogic.py`  
  Stable event detection: **rank ≤ 3** as core; expand with local baseline to determine start/end; outputs `.dat` + plots. :contentReference[oaicite:10]{index=10}
- `pairwise_stitch_plots.py`  
  Stitch “before vs after” plots vertically for visual comparison. :contentReference[oaicite:11]{index=11}
- `build_labeled_compare_stacks_0_9.py`  
  Build stacked comparison images for segments 0–9 (each filter labeled). :contentReference[oaicite:12]{index=12}
- `build_filter_selection_gallery.py`  
  Build a gallery to support filter selection/inspection. :contentReference[oaicite:13]{index=13}
- `index.html`  
  Optional lightweight viewer for generated images. :contentReference[oaicite:14]{index=14}

Subfolders (as shown in repo root):

- `AnnotationTools/` :contentReference[oaicite:15]{index=15}
- `Smile_Detection/` :contentReference[oaicite:16]{index=16}
- `Tools/` :contentReference[oaicite:17]{index=17}
- `filter-selection/` :contentReference[oaicite:18]{index=18}
- `output/` (generated artifacts) :contentReference[oaicite:19]{index=19}

---

## Typical workflow

1. **Prepare input SmileRank data**
   - Start from your raw SmileRank `.dat` (or exported rank series).
   - Keep frame index aligned with your video/frame pipeline.

2. **Interpolate + segment plots**
   - Run `run_02_interpolate_and_plot.py` to interpolate the raw curve and produce per-segment plots. :contentReference[oaicite:20]{index=20}

3. **Filter + export**
   - Run `smile_rank_filter_and_plot.py` to merge CSV segments, smooth the curve, and output plots + merged CSV. :contentReference[oaicite:21]{index=21}

4. **Compare filters (optional but recommended)**
   - Run `smile_rank_filter_compare.py` and/or `build_labeled_compare_stacks_0_9.py` to visually decide the best filter. :contentReference[oaicite:22]{index=22}

5. **Detect smile events**
   - Use `detect_and_plot_smile_events_corelogic.py` as the stable detection pipeline (rank ≤ 3 core + baseline expansion). :contentReference[oaicite:23]{index=23}

6. **Reporting / presentation**
   - Use `pairwise_stitch_plots.py` for before/after comparisons and `index.html` for quick browsing. :contentReference[oaicite:24]{index=24}

---

## Outputs

Depending on the scripts you run, you will typically get:

- Interpolated and/or filtered **CSV** (merged across segments). :contentReference[oaicite:25]{index=25}
- Segment-level and summary **plots** (PNG/JPG).
- Smile-event detection results as **`.dat`** + plots (corelogic script). :contentReference[oaicite:26]{index=26}
- Stitched/stacked comparison images for quick qualitative evaluation. :contentReference[oaicite:27]{index=27}

---

## Requirements

This repo is primarily Python + a small HTML viewer. The exact dependencies are not listed in the repo root view I can access right now, but based on the functionality, it typically requires:

- Python 3.x
- Common scientific stack: `numpy`, `pandas`, `matplotlib`
- Filtering utilities likely using: `scipy` (Savitzky–Golay / Gaussian) :contentReference[oaicite:28]{index=28}

Suggested setup (create your own `requirements.txt` once you confirm imports):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install numpy pandas matplotlib scipy
