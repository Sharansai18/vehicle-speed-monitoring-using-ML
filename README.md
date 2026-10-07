# Vehicle Speed Monitoring System using Machine Learning

YOLOv8 detects vehicles, a centroid tracker follows them, and speed is computed
from the time a vehicle takes to travel between the red and blue lines.

## Setup
1. Install Python 3.9 – 3.12 (tick "Add Python to PATH").
2. In this folder run:
       pip install -r requirements.txt
3. Put a traffic video in this folder named `highway.mp4`
   (any road video filmed from above/behind works).

## Run
    python speed_monitor.py
    python speed_monitor.py --video highway.mp4 --speed-limit 80

It asks for the speed limit if you don't pass `--speed-limit`.
Press `q` or `Esc` to stop.

## Outputs
- `output.avi`        – processed video with boxes and speeds
- `speed_log.csv`     – every measured vehicle (time, id, type, direction, speed, overspeed yes/no)
- `overspeed_images/` – cropped photo of each vehicle over the limit
- `detected_frames/`  – sample frames

## Calibration (important for accurate speed)
`--distance` is the REAL distance in metres between the red and blue lines on the
road (default 10). `--red-y` / `--blue-y` move the lines (frame is 1020x500).
Measure the real road distance for your camera for correct results.

## Troubleshooting
- "Video file not found": place `highway.mp4` here or use `--video path`.
- Slow? Use `--model yolov8n.pt` (smaller/faster) or `--no-display`.
- First run needs internet once to download the YOLO weights.
