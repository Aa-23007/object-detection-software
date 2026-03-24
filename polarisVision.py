from ultralytics import YOLO
import cv2
import time
import math
import torch

# -------------------- DEVICE CONFIG --------------------

if torch.cuda.is_available():
    DEVICE = "0"
    USE_HALF = True
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    USE_HALF = False
else:
    DEVICE = "cpu"
    USE_HALF = False

IMG_SIZE = 704
VERBOSE_ULTRA = False

# -------------------- VIDEO CONFIG --------------------

video_path = "Carsdriving-night.mp4"

cap = cv2.VideoCapture(video_path)
VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
cap.release()

# -------------------- SPEED CONFIG --------------------

METERS_PER_PIXEL = 0.05
STATIONARY_PIXEL_THRESHOLD = 2.0
MIN_SPEED_KMH = 1.0

SPEED_SMOOTHING_ALPHA = 0.4
BOX_SMOOTHING_ALPHA = 0.4

prev_centers = {}
prev_speeds = {}
prev_boxes = {}

# -------------------- MODEL --------------------

model = YOLO("best_448px.pt")
model.to(DEVICE)

if USE_HALF:
    model.model.half()

# -------------------- COLORS --------------------

colors = {
    "person": (0, 0, 255),
    "pedestrian": (0, 0, 255),
    "car": (255, 0, 0),
    "truck": (0, 255, 255),
    "bus": (0, 165, 255),
    "motorcycle": (255, 0, 255),
    "bicycle": (128, 0, 128),
    "building": (128, 128, 128)
}

# -------------------- TIMING --------------------

prev_time = time.time()
start_time = time.time()
paused_total = 0.0

# -------------------- MAIN LOOP --------------------

for result in model.track(
    source=video_path,
    stream=True,
    conf=0.25,
    iou=0.45,
    imgsz=IMG_SIZE,
    tracker="bytetrack.yaml",
    device=DEVICE,
    half=USE_HALF,
    verbose=VERBOSE_ULTRA
):

    frame = result.orig_img
    names = result.names

    # -------- TRACK IDS (SAFE) --------
    if result.boxes.id is not None:
        ids = result.boxes.id.cpu().numpy().astype(int)
    else:
        ids = []

    # -------- OBJECT COUNTER --------
    object_count = len(ids)

    for i, box in enumerate(result.boxes):

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        cls_id = int(box.cls)
        label = names[cls_id]

        track_id = ids[i] if i < len(ids) else -1

        # -------- BOX SMOOTHING --------
        if track_id != -1 and track_id in prev_boxes:
            px1, py1, px2, py2 = prev_boxes[track_id]

            sx1 = BOX_SMOOTHING_ALPHA * x1 + (1 - BOX_SMOOTHING_ALPHA) * px1
            sy1 = BOX_SMOOTHING_ALPHA * y1 + (1 - BOX_SMOOTHING_ALPHA) * py1
            sx2 = BOX_SMOOTHING_ALPHA * x2 + (1 - BOX_SMOOTHING_ALPHA) * px2
            sy2 = BOX_SMOOTHING_ALPHA * y2 + (1 - BOX_SMOOTHING_ALPHA) * py2
        else:
            sx1, sy1, sx2, sy2 = x1, y1, x2, y2

        if track_id != -1:
            prev_boxes[track_id] = (sx1, sy1, sx2, sy2)

        sx1_i, sy1_i, sx2_i, sy2_i = int(sx1), int(sy1), int(sx2), int(sy2)

        # -------- SPEED --------
        speed_kmh = 0.0

        if track_id != -1:
            cx = (sx1 + sx2) / 2
            cy = (sy1 + sy2) / 2

            raw_speed_kmh = 0.0

            if track_id in prev_centers:
                px, py = prev_centers[track_id]
                dx, dy = cx - px, cy - py

                pixel_distance = math.hypot(dx, dy)

                if pixel_distance >= STATIONARY_PIXEL_THRESHOLD:
                    pixels_per_second = pixel_distance * VIDEO_FPS
                    meters_per_second = pixels_per_second * METERS_PER_PIXEL
                    raw_speed_kmh = meters_per_second * 3.6

            prev_centers[track_id] = (cx, cy)

            if track_id in prev_speeds:
                speed_kmh = (
                    SPEED_SMOOTHING_ALPHA * raw_speed_kmh
                    + (1 - SPEED_SMOOTHING_ALPHA) * prev_speeds[track_id]
                )
            else:
                speed_kmh = raw_speed_kmh

            if speed_kmh < MIN_SPEED_KMH:
                speed_kmh = 0.0

            prev_speeds[track_id] = speed_kmh

        # -------- LABEL --------
        label_text = (
            f"{label} #{track_id} {int(round(speed_kmh))} km/h"
            if track_id != -1 else label
        )

        color = colors.get(label, (0, 255, 0))

        cv2.rectangle(frame, (sx1_i, sy1_i), (sx2_i, sy2_i), color, 2)
        cv2.putText(frame, label_text, (sx1_i, sy1_i - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------- FPS + TIMER --------

    current_time = time.time()
    dt = current_time - prev_time
    fps = 1.0 / dt if dt > 0 else 0.0
    prev_time = current_time

    elapsed_time = current_time - start_time - paused_total
    mins = int(elapsed_time // 60)
    secs = int(elapsed_time % 60)

    # -------- HUD --------

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.putText(frame, f"Time: {mins:02}:{secs:02}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.putText(frame, f"Objects: {object_count}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    cv2.imshow("UniDrone Optimized", frame)

    # -------- PAUSE --------

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        pause_start = time.time()

        while True:
            key2 = cv2.waitKey(0) & 0xFF
            if key2 == ord(' '):
                paused_total += time.time() - pause_start
                break
            if key2 == ord('q'):
                cv2.destroyAllWindows()
                exit()

cv2.destroyAllWindows()
