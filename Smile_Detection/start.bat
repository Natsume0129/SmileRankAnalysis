@echo off
echo ===== Step 1: Detect Smile Segments =====
python 01_detect_smile_segments.py 01_detect_smile_events.dat
echo.

echo ===== Step 2: Plot Smile Curves with Marked Segments =====
python 02_plot_smile_segments_with_events.py 01_detect_smile_events.dat
echo.

echo ===== All Steps Finished =====
pause
