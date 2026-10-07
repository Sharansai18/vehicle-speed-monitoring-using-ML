"""
Vehicle Speed Monitoring System using Machine Learning (YOLOv8 + OpenCV)

Usage examples:
    python speed_monitor.py                              # asks for speed limit, uses highway.mp4
    python speed_monitor.py --video highway.mp4 --speed-limit 80
    python speed_monitor.py --video 0 --speed-limit 60   # webcam
    python speed_monitor.py --no-display                 # headless (no window)
Press 'q' or ESC in the video window to quit.
"""
import argparse
import csv
import os
import sys
from datetime import datetime

import cv2
from ultralytics import YOLO

from tracker import Tracker

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FRAME_W, FRAME_H = 1020, 500


def parse_args():
    p = argparse.ArgumentParser(description="Vehicle speed monitoring")
    p.add_argument("--video", default="highway.mp4", help="video path, or 0 for webcam")
    p.add_argument("--speed-limit", type=float, default=None, help="speed limit in km/h")
    p.add_argument("--distance", type=float, default=10.0,
                   help="real-world distance in metres between red and blue line")
    p.add_argument("--red-y", type=int, default=198, help="y position of red line")
    p.add_argument("--blue-y", type=int, default=268, help="y position of blue line")
    p.add_argument("--model", default="yolov8s.pt", help="YOLO weights (auto-downloaded)")
    p.add_argument("--conf", type=float, default=0.4, help="detection confidence")
    p.add_argument("--no-display", action="store_true", help="do not open a window")
    return p.parse_args()


def get_speed_limit(value):
    if value is not None:
        if value <= 0:
            sys.exit("Speed limit must be greater than 0.")
        return value
    while True:
        try:
            v = float(input("Enter speed limit for this street (km/h): "))
            if v > 0:
                return v
        except ValueError:
            pass
        print("Please enter a valid positive number.")


def crossed(prev_y, cur_y, line_y):
    """True if the vehicle centre moved across the horizontal line."""
    return prev_y != cur_y and (prev_y - line_y) * (cur_y - line_y) <= 0


def main():
    args = parse_args()
    speed_limit = get_speed_limit(args.speed_limit)

    source = int(args.video) if args.video.isdigit() else args.video
    if isinstance(source, str) and not os.path.exists(source):
        sys.exit(f"Video file not found: {source}\n"
                 f"Put your video in this folder or use --video <path>.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit("Could not open the video source.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0

    print("Loading YOLO model (first run downloads weights)...")
    model = YOLO(args.model)
    tracker = Tracker()

    os.makedirs("overspeed_images", exist_ok=True)
    os.makedirs("detected_frames", exist_ok=True)
    log_exists = os.path.exists("speed_log.csv")
    log_file = open("speed_log.csv", "a", newline="")
    log = csv.writer(log_file)
    if not log_exists:
        log.writerow(["timestamp", "vehicle_id", "type", "direction",
                      "speed_kmh", "speed_limit", "overspeeding", "image"])

    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"),
                          fps, (FRAME_W, FRAME_H))

    red_y, blue_y = args.red_y, args.blue_y
    prev_cy = {}          # id -> previous centre y
    t_red = {}            # id -> frame index when red line crossed
    t_blue = {}           # id -> frame index when blue line crossed
    done = set()          # ids whose speed has been measured
    speeds = {}           # id -> (speed, direction)
    counter_down, counter_up = set(), set()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        # ---- detect vehicles ----
        result = model.predict(frame, conf=args.conf, verbose=False)[0]
        boxes, types = [], {}
        for x1, y1, x2, y2, conf, cls in result.boxes.data.cpu().numpy():
            cls = int(cls)
            if cls in VEHICLE_CLASSES:
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                types[(int(x1), int(y1))] = VEHICLE_CLASSES[cls]

        # ---- track + speed ----
        for x1, y1, x2, y2, vid in tracker.update(boxes):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            vtype = types.get((x1, y1), "vehicle")
            py = prev_cy.get(vid, cy)

            if crossed(py, cy, red_y):
                t_red.setdefault(vid, frame_idx)
            if crossed(py, cy, blue_y):
                t_blue.setdefault(vid, frame_idx)

            if vid in t_red and vid in t_blue and vid not in done:
                done.add(vid)
                seconds = abs(t_blue[vid] - t_red[vid]) / fps
                if seconds > 0:
                    kmh = args.distance / seconds * 3.6
                    going_down = t_red[vid] < t_blue[vid]
                    direction = "down" if going_down else "up"
                    (counter_down if going_down else counter_up).add(vid)
                    over = kmh > speed_limit
                    speeds[vid] = (kmh, direction, over)

                    img_path = ""
                    if over:
                        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                        img_path = os.path.join(
                            "overspeed_images",
                            f"id{vid}_{int(kmh)}kmh_{datetime.now():%Y%m%d_%H%M%S}.jpg")
                        if crop.size:
                            cv2.imwrite(img_path, crop)
                    log.writerow([datetime.now().isoformat(timespec="seconds"), vid,
                                  vtype, direction, round(kmh, 1), speed_limit,
                                  over, img_path])
                    log_file.flush()
            prev_cy[vid] = cy

            # ---- draw ----
            info = speeds.get(vid)
            color = (0, 0, 255) if (info and info[2]) else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(vid), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
            if info:
                label = f"{int(info[0])} Km/h" + (" OVERSPEED" if info[2] else "")
                cv2.putText(frame, label, (x2, y2),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

        # ---- overlay ----
        cv2.rectangle(frame, (0, 0), (260, 90), (0, 255, 255), -1)
        cv2.putText(frame, f"Going Down - {len(counter_down)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Going Up - {len(counter_up)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Limit - {speed_limit:g} km/h", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (172, red_y), (774, red_y), (0, 0, 255), 2)
        cv2.putText(frame, "Red Line", (172, red_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (8, blue_y), (927, blue_y), (255, 0, 0), 2)
        cv2.putText(frame, "Blue Line", (8, blue_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        out.write(frame)
        if frame_idx % 30 == 0:
            cv2.imwrite(os.path.join("detected_frames", f"frame_{frame_idx}.jpg"), frame)

        if not args.no_display:
            cv2.imshow("Vehicle Speed Monitoring", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    cap.release()
    out.release()
    log_file.close()
    cv2.destroyAllWindows()
    over_n = sum(1 for s in speeds.values() if s[2])
    print(f"\nDone. Vehicles measured: {len(speeds)}, overspeeding: {over_n}")
    print("Saved: output.avi, speed_log.csv, overspeed_images/, detected_frames/")


if __name__ == "__main__":
    main()
