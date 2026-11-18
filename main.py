import os
import sys
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")

FILTERS = {
    "1": "Invert",
    "2": "Pixelate",
    "3": "Thermal",
    "4": "Neon Edges",
    "5": "Noir Blur",
    "6": "RGB Glitch",
}


def ensure_model() -> str:
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading hand landmark model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def apply_filter(frame: np.ndarray, mode: str) -> np.ndarray:
    h, w = frame.shape[:2]

    if mode == "1":  # Invert
        return cv2.bitwise_not(frame)

    if mode == "2":  # Pixelate
        block = max(12, min(w, h) // 36)
        small = cv2.resize(frame, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    if mode == "3":  # Thermal
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.applyColorMap(blur, cv2.COLORMAP_INFERNO)

    if mode == "4":  # Neon Edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 140)
        glow = cv2.GaussianBlur(edges, (9, 9), 0)
        out = (frame * 0.18).astype(np.uint8)
        out[:, :, 0] = np.clip(out[:, :, 0].astype(np.int16) + glow * 1.4, 0, 255).astype(np.uint8)
        out[:, :, 1] = np.clip(out[:, :, 1].astype(np.int16) + edges * 1.0, 0, 255).astype(np.uint8)
        out[:, :, 2] = np.clip(out[:, :, 2].astype(np.int16) + glow * 1.1, 0, 255).astype(np.uint8)
        return out

    if mode == "5":  # Noir Blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (35, 35), 0)
        contrast = cv2.convertScaleAbs(blurred, alpha=1.25, beta=-15)
        return cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)

    if mode == "6":  # RGB Glitch
        shift = max(8, w // 70)
        b, g, r = cv2.split(frame)
        b_shifted = np.roll(b, -shift, axis=1)
        r_shifted = np.roll(r, shift, axis=1)
        merged = cv2.merge([b_shifted, g, r_shifted])
        merged[::4, :] = (merged[::4, :] * 0.72).astype(np.uint8)
        return merged

    return frame


def segments_intersect(p1, p2, p3, p4) -> bool:
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def order_quad(thumb_a, index_a, thumb_b, index_b) -> np.ndarray:
    """Connect thumb_a-index_a, index_a-index_b, index_b-thumb_b, thumb_b-thumb_a without self-intersection."""
    if segments_intersect(thumb_a, index_a, index_b, thumb_b) or segments_intersect(index_a, index_b, thumb_b, thumb_a):
        return np.array([thumb_a, index_a, thumb_b, index_b], dtype=np.int32)
    return np.array([thumb_a, index_a, index_b, thumb_b], dtype=np.int32)


def draw_hud(frame: np.ndarray, active_mode: str, hands_found: int) -> None:
    h, w = frame.shape[:2]
    bar_h = 44
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    x = 16
    y = h - 16
    for key, label in FILTERS.items():
        text = f"[{key}] {label}"
        is_active = key == active_mode
        color = (80, 240, 180) if is_active else (180, 185, 195)
        thickness = 2 if is_active else 1
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, thickness, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, thickness)
        x += tw + 18

    status = "Active" if hands_found >= 2 else f"Show both hands ({hands_found}/2)"
    status_color = (80, 240, 180) if hands_found >= 2 else (140, 180, 255)
    (sw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, status, (w - sw - 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)


def main() -> int:
    model_path = ensure_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera (/dev/video0).", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.55,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    active_mode = "1"
    smoothed_quad = None
    missing_frames = 0
    alpha = 0.55
    start_ts = time.monotonic()
    last_ts_ms = -1

    win_name = "Hand Lens (1-6 filters | Q quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts_ms = int((time.monotonic() - start_ts) * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            res = landmarker.detect_for_video(mp_image, ts_ms)

            hands_found = 0
            quad = None

            if res.hand_landmarks:
                hands_found = len(res.hand_landmarks)
                if hands_found >= 2:
                    pts = []
                    for hand_lm in res.hand_landmarks[:2]:
                        thumb = hand_lm[4]
                        index = hand_lm[8]
                        pts.append((
                            (int(thumb.x * w), int(thumb.y * h)),
                            (int(index.x * w), int(index.y * h)),
                        ))
                    pts.sort(key=lambda pair: (pair[0][0] + pair[1][0]) * 0.5)
                    raw_quad = order_quad(pts[0][0], pts[0][1], pts[1][0], pts[1][1]).astype(np.float32)

                    if smoothed_quad is None:
                        smoothed_quad = raw_quad
                    else:
                        smoothed_quad = alpha * raw_quad + (1.0 - alpha) * smoothed_quad
                    quad = smoothed_quad.astype(np.int32)
                    missing_frames = 0

            if quad is None and smoothed_quad is not None and missing_frames < 5:
                missing_frames += 1
                quad = smoothed_quad.astype(np.int32)
            elif quad is None:
                smoothed_quad = None

            if quad is not None:
                filtered = apply_filter(frame, active_mode)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [quad], 255)
                mask_3ch = mask[:, :, None] > 0
                frame = np.where(mask_3ch, filtered, frame)

                # Clean quad border + 4 fingertip anchor dots (no hand skeleton)
                cv2.polylines(frame, [quad], isClosed=True, color=(80, 240, 180), thickness=2, lineType=cv2.LINE_AA)
                for pt in quad:
                    cv2.circle(frame, tuple(pt), 6, (80, 240, 180), -1, cv2.LINE_AA)
                    cv2.circle(frame, tuple(pt), 8, (255, 255, 255), 1, cv2.LINE_AA)

            draw_hud(frame, active_mode, hands_found)
            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if chr(key) in FILTERS:
                active_mode = chr(key)
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
