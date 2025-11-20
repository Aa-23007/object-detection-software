from ultralytics import YOLO
import cv2
import time
import math
import torch

# -------------------- DEVICE & MODEL SPEED CONFIG --------------------

if torch.cuda.is_available():
    DEVICE = "0"          # CUDA GPU
    USE_HALF = True
elif torch.backends.mps.is_available():
    DEVICE = "mps"        # Apple Silicon GPU
    USE_HALF = False      # MPS doesn't support half nicely
else:
    DEVICE = "cpu"
    USE_HALF = False

IMG_SIZE = 704            # smaller than 704 => faster
VERBOSE_ULTRA = False     # turn off Ultralytics logging for speed

# -------------------- CONFIG --------------------

video_path = "Carsdriving-night.mp4"    #Change to any video you want to upload

# Get the FPS of the input video (for speed calculation)
cap = cv2.VideoCapture(video_path)
VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0  # fallback if 0
cap.release()

# Approximate real-world scale (meters per pixel) – TUNE THIS
METERS_PER_PIXEL = 0.05   # try 0.02–0.08 depending on camera

# Thresholds to avoid fake speed from jitter
STATIONARY_PIXEL_THRESHOLD = 2.0   # pixels per frame; below this = not moving
MIN_SPEED_KMH = 1.0                # below this speed = treated as 0 km/h

# Smoothing factor for speeds (0–1). Lower = smoother, slower to react.
SPEED_SMOOTHING_ALPHA = 0.4        # try 0.3–0.6

# Smoothing factor for bounding boxes (0–1).
BOX_SMOOTHING_ALPHA = 0.4          # try 0.3–0.6

# track_id -> (cx, cy) for speed estimation
prev_centers = {}

# track_id -> smoothed speed (km/h)
prev_speeds = {}

# track_id -> smoothed box (x1, y1, x2, y2)
prev_boxes = {}

# -------------------- MODEL --------------------

model = YOLO("best_448px.pt")

# Move model to the right device
model.to(DEVICE)

# Optionally set half precision on CUDA for extra speed
if USE_HALF:
    model.model.half()

# 2. Colors
colors = {
    "person": (0, 0, 255),
    "pedestrian": (0, 0, 255),
    "light_vehicles": (255, 0, 0),
    "car": (255, 0, 0),
    "heavy_vehicles": (0, 255, 255),
    "truck": (0, 255, 255),
    "bus": (0, 165, 255),
    "motorcycle": (255, 0, 255),
    "bicycle": (128, 0, 128),
    "boat": (255, 255, 0),
    "building": (128, 128, 128)
}

prev_time = time.time()

# -------------------- MAIN LOOP --------------------

for result in model.track(
        source=video_path,
        stream=True,
        conf=0.25,
        iou=0.45,
        imgsz=IMG_SIZE,          # reduced for speed
        tracker="bytetrack.yaml",
        device=DEVICE,
        half=USE_HALF,
        verbose=VERBOSE_ULTRA
    ):

    # DO NOT COPY: draw directly on original image
    frame = result.orig_img
    names = result.names

    # tracking IDs
    if result.boxes.id is not None:
        ids = result.boxes.id.cpu().numpy().astype(int)
    else:
        ids = []

    for i, box in enumerate(result.boxes):
        # Raw box from YOLO
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        cls_id = int(box.cls)
        label = names[cls_id]

        track_id = ids[i] if i < len(ids) else -1

        # ------------ BOX SMOOTHING PER OBJECT ------------

        if track_id != -1 and track_id in prev_boxes:
            px1, py1, px2, py2 = prev_boxes[track_id]

            # EMA on box coords
            sx1 = BOX_SMOOTHING_ALPHA * x1 + (1.0 - BOX_SMOOTHING_ALPHA) * px1
            sy1 = BOX_SMOOTHING_ALPHA * y1 + (1.0 - BOX_SMOOTHING_ALPHA) * py1
            sx2 = BOX_SMOOTHING_ALPHA * x2 + (1.0 - BOX_SMOOTHING_ALPHA) * px2
            sy2 = BOX_SMOOTHING_ALPHA * y2 + (1.0 - BOX_SMOOTHING_ALPHA) * py2
        else:
            # No previous box: use raw
            sx1, sy1, sx2, sy2 = x1, y1, x2, y2

        # Save smoothed box for next frame
        if track_id != -1:
            prev_boxes[track_id] = (sx1, sy1, sx2, sy2)

        # Convert smoothed box to int for drawing
        sx1_i, sy1_i, sx2_i, sy2_i = int(sx1), int(sy1), int(sx2), int(sy2)

        # ------------ SPEED ESTIMATION PER OBJECT ------------

        speed_kmh = 0.0

        if track_id != -1:
            # Use smoothed box center for speed calc
            cx = (sx1 + sx2) / 2.0
            cy = (sy1 + sy2) / 2.0

            raw_speed_kmh = 0.0  # unsmoothed speed

            if track_id in prev_centers:
                px, py = prev_centers[track_id]
                dx = cx - px
                dy = cy - py

                # distance in pixels between this frame and previous frame
                pixel_distance = math.hypot(dx, dy)

                # if movement is tiny, treat as stationary (noise/jitter)
                if pixel_distance < STATIONARY_PIXEL_THRESHOLD:
                    raw_speed_kmh = 0.0
                else:
                    # pixels per second based on VIDEO_FPS (not processing FPS)
                    pixels_per_second = pixel_distance * VIDEO_FPS

                    # convert to m/s then km/h
                    meters_per_second = pixels_per_second * METERS_PER_PIXEL
                    raw_speed_kmh = meters_per_second * 3.6

            # update stored center for this track ID (using smoothed center)
            prev_centers[track_id] = (cx, cy)

            # ---- EXPONENTIAL MOVING AVERAGE SMOOTHING FOR SPEED ----
            if track_id in prev_speeds:
                # s_t = α * new + (1-α) * old
                speed_kmh = (
                    SPEED_SMOOTHING_ALPHA * raw_speed_kmh
                    + (1.0 - SPEED_SMOOTHING_ALPHA) * prev_speeds[track_id]
                )
            else:
                speed_kmh = raw_speed_kmh

            # clamp tiny speeds to 0 after smoothing
            if speed_kmh < MIN_SPEED_KMH:
                speed_kmh = 0.0

            # store smoothed speed
            prev_speeds[track_id] = speed_kmh

        # If you only want speeds for vehicles, you can do:
        # if label not in ["car", "truck", "bus", "motorcycle"]:
        #     speed_kmh = 0.0

        # build label text (rounded to integer km/h)
        if track_id != -1:
            label_text = f"{label} #{track_id} {int(round(speed_kmh))} km/h"
        else:
            label_text = f"{label}"

        color = colors.get(label, (0, 255, 0))

        cv2.rectangle(frame, (sx1_i, sy1_i), (sx2_i, sy2_i), color, 2)
        cv2.putText(
            frame,
            label_text,
            (sx1_i, sy1_i - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,  #size
            color, 
            2     #thickness
        )

    # ------------ FPS CALC (PROCESSING FPS) ------------

    current_time = time.time()
    dt = current_time - prev_time
    fps = 1.0 / dt if dt > 0 else 0.0
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # comment this out if you want maximum speed
    # print(f"FPS: {fps:.2f}")

    cv2.imshow("UniDrone FAST + Tracking IDs + Smoothed Speed", frame)

    # ------------ PAUSE / RESUME LOGIC ------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        print("PAUSED")

        # Freeze frame and draw "PAUSED" in bottom-right corner
        pause_frame = frame.copy()

        text = "PAUSED"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thickness = 3

        # Get text size
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)

        # Bottom right placement
        x = pause_frame.shape[1] - text_w - 20   # 20px from right edge
        y = pause_frame.shape[0] - 20            # 20px from bottom edge

        cv2.putText(
            pause_frame,
            text,
            (x, y),
            font,
            scale,
            (0, 0, 255),
            thickness
        )

        cv2.imshow("UniDrone FAST + Tracking IDs + Smoothed Speed", pause_frame)

        # Wait for unpause / quit
        while True:
            key2 = cv2.waitKey(0) & 0xFF
            if key2 == ord(' '):
                print("RESUMED")
                break
            if key2 == ord('q'):
                cv2.destroyAllWindows()
                exit()

cv2.destroyAllWindows()
