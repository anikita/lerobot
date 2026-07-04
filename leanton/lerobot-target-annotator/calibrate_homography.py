"""
Camera-to-Plate Homography Calibration  [v1]
=============================================
Computes a 3x3 planar homography mapping overview-camera pixels to
base-plate coordinates (millimeters). Uses ArUco markers for automatic
pixel-coordinate detection with manual-click fallback.

Output: homography_calibration.json

Usage:
    python calibrate_homography.py                    # calibration run
    python calibrate_homography.py --generate-markers  # print markers to PDF
    python calibrate_homography.py --camera 1          # use specific camera index

Setup:
    conda activate lerobot
    pip install opencv-contrib-python pyzmq matplotlib
"""

import cv2
import sys
import os
import glob
import platform
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# ═════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════

ARUCO_DICT           = cv2.aruco.DICT_4X4_50
MARKER_IDS           = [0, 1, 2, 3]      # calibration markers
VALIDATION_MARKER_ID = 4                  # held-out validation marker
ALL_MARKER_IDS       = [0, 1, 2, 3, 4]   # all markers to detect
MARKER_SIZE_MM       = 25        # printed marker side length (mm)
NUM_REF_POINTS       = 4
DEFAULT_CAMERA       = 0
CALIBRATION_FILE     = "homography_calibration.json"
CALIBRATION_VERSION  = 1
_actual_resolution   = (0, 0)   # set by _open_camera_raw at camera open time

# ── Manual-fallback colours (BGR) ───────────────────────────
CLICK_COLORS = [
    (0, 0, 255),     # red
    (0, 165, 255),   # orange
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
]

# ═════════════════════════════════════════════════════════════


# ── Camera ────────────────────────────────────────────────────────────────────

def _open_camera_raw(idx):
    """Open camera at its native (maximum) resolution. Requests 1920x1080 with
    MJPG; V4L2 clamps to the camera's maximum. Stores the delivered resolution
    in _actual_resolution. Returns cap or None.

    Identical behaviour to annotate_stream_multi.py's open_camera + _open_camera_raw.
    """
    global _actual_resolution
    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # First frame after mode switch is often junk — discard it (matches annotator)
    cap.read()
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        # 1920x1080 not supported — fall back to 640x480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.read()  # discard stale
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            cap.release()
            return None
    actual_h, actual_w = frame.shape[:2]
    _actual_resolution = (actual_w, actual_h)
    return cap


def probe_cameras():
    """Return list of available camera indices (same convention as annotate_stream.py)."""
    available = []
    for idx in range(8):
        cap = _open_camera_raw(idx)
        if cap is not None:
            available.append(idx)
            cap.release()
    return available


def interactive_camera_select(available, start_idx=None):
    """Cycle through cameras with 'c' key, confirm with Enter/Space.

    Shows a live feed from the current camera with ArUco detection overlaid.
    Returns the selected camera index, or None if cancelled (ESC/q).
    """
    if start_idx is None or start_idx not in available:
        cam_idx = available[0]
    else:
        cam_idx = start_idx

    cap = _open_camera_raw(cam_idx)
    if cap is None:
        print(f"[ERROR] Cannot open camera {cam_idx}")
        return None

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(dictionary, parameters)

    cv2.namedWindow("Select Camera — press C to cycle, ENTER to confirm", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Camera — press C to cycle, ENTER to confirm", 960, 720)

    print("Camera selection:")
    print("  C / Right Arrow  — next camera")
    print("  Left Arrow       — previous camera")
    print("  ENTER            — confirm selection")
    print("  ESC / Q          — quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Camera {cam_idx}: NO SIGNAL", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # run ArUco detection on live feed
        corners, ids, _ = detector.detectMarkers(frame)
        found = {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                pts = corners[i][0]
                cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
                found[int(mid)] = (cx, cy)

        # draw
        display = frame.copy()

        # camera label
        cv2.putText(display, f"Camera {cam_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.88, (255, 255, 255), 2)

        # draw detected markers
        for mid, (cx, cy) in found.items():
            cv2.circle(display, (int(cx), int(cy)), 8, (0, 255, 0), -1)
            cv2.putText(display, f"ID {mid}", (int(cx) + 12, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # status bar
        found_ids = sorted(found.keys())
        missing = [m for m in ALL_MARKER_IDS if m not in found]
        nc = len([m for m in found_ids if m in MARKER_IDS])
        nv = 1 if VALIDATION_MARKER_ID in found else 0
        status = f"Calib: {nc}/4  |  Valid: {nv}/1  (IDs: {found_ids})"
        if missing:
            status += f"  missing: {missing}"
        cv2.putText(display, status, (10, display.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 255, 0) if nc == 4 and nv == 1 else (0, 165, 255), 1)

        # hints
        cv2.putText(display, "c=cycle  ENTER=confirm  ESC=quit",
                    (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # available cameras bar
        cam_bar = "CAM [" + "  ".join(
            f">{c}<" if c == cam_idx else str(c) for c in available
        ) + "]"
        cv2.putText(display, cam_bar, (display.shape[1] - 300, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)

        cv2.imshow("Select Camera — press C to cycle, ENTER to confirm", display)

        key = cv2.waitKey(100) & 0xFF

        if key in (ord('c'), 83):   # c or right arrow → next
            pos = available.index(cam_idx)
            next_idx = available[(pos + 1) % len(available)]
            cap.release()
            cap = _open_camera_raw(next_idx)
            if cap is not None:
                cam_idx = next_idx
            else:
                print(f"  Cannot open camera {next_idx}")

        elif key == 81:   # left arrow → previous
            pos = available.index(cam_idx)
            prev_idx = available[(pos - 1) % len(available)]
            cap.release()
            cap = _open_camera_raw(prev_idx)
            if cap is not None:
                cam_idx = prev_idx
            else:
                print(f"  Cannot open camera {prev_idx}")

        elif key in (13, 32):   # ENTER or SPACE → confirm
            break

        elif key in (ord('q'), 27):   # q or ESC → quit
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSelected camera {cam_idx}")
    return cam_idx


def capture_frame(camera_idx=DEFAULT_CAMERA):
    """Open camera, capture one frame, return (BGR image or None, error string)."""
    cap = _open_camera_raw(camera_idx)
    if cap is None:
        return None, f"Camera index {camera_idx} not found. Check connection."
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None, f"Camera {camera_idx} opened but frame capture failed."
    return frame, None


# ── ArUco ─────────────────────────────────────────────────────────────────────

def detect_markers(frame):
    """Detect ArUco markers in frame.

    Returns:
        markers: dict {marker_id: (cx, cy)}  — centroid of marker corners
        corners: list of detected corner arrays (for drawing)
        ids: list of detected marker IDs
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, _ = detector.detectMarkers(frame)

    markers = {}
    corner_list = []
    id_list = []

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i][0]  # (4, 2) array of corner pixel coordinates
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            markers[int(marker_id)] = (cx, cy)
            corner_list.append(pts)
            id_list.append(int(marker_id))

    return markers, corner_list, id_list


def draw_detected_markers(frame, markers, detected_ids):
    """Annotate frame with marker centroids and IDs. Returns annotated copy."""
    out = frame.copy()
    for mid, (cx, cy) in markers.items():
        cv2.circle(out, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        cv2.putText(out, f"ID {mid}", (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # label missing IDs
    for mid in ALL_MARKER_IDS:
        if mid not in detected_ids:
            role = "VALIDATION" if mid == VALIDATION_MARKER_ID else "calib"
            cv2.putText(out, f"ID {mid} ({role}): NOT FOUND",
                        (10, 60 + mid * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 255), 1)
    return out


# ── Multi-frame marker averaging ──────────────────────────────────────────────

AVERAGING_FRAMES = 60


def capture_averaged_markers(camera_idx, expected_ids=ALL_MARKER_IDS, num_frames=AVERAGING_FRAMES):
    """Collect marker centroids over N frames and return median per marker.

    Opens the camera, shows a live view with a progress bar, and accumulates
    centroid positions frame by frame. The median across frames suppresses
    pixel jitter from sensor noise and lighting flicker.

    Returns:
        markers: dict {marker_id: (cx_median, cy_median)}, or None if cancelled.
    """
    cap = _open_camera_raw(camera_idx)
    if cap is None:
        print(f"[ERROR] Cannot open camera {camera_idx}")
        return None

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(dictionary, parameters)

    # accumulator: {marker_id: [[cx1, cx2, ...], [cy1, cy2, ...]]}
    accum = {mid: [[], []] for mid in expected_ids}

    cv2.namedWindow("Averaging markers — 60 frames", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Averaging markers — 60 frames", 960, 720)

    print(f"\nCollecting {num_frames} frames for marker position averaging...")
    print("  Hold still. Press ESC to cancel.\n")

    for frame_i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        corners, ids, _ = detector.detectMarkers(frame)
        found_this_frame = {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                pts = corners[i][0]
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                found_this_frame[int(mid)] = (cx, cy)

        # accumulate
        for mid in expected_ids:
            if mid in found_this_frame:
                cx, cy = found_this_frame[mid]
                accum[mid][0].append(cx)
                accum[mid][1].append(cy)

        # draw
        display = frame.copy()
        for mid, (cx, cy) in found_this_frame.items():
            cv2.circle(display, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            cv2.putText(display, f"ID {mid}", (int(cx) + 10, int(cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1)

        # progress bar
        h, w = display.shape[:2]
        bar_w = w - 20
        bar_h = 12
        bx, by = 10, h - 50
        progress = (frame_i + 1) / num_frames
        cv2.rectangle(display, (bx, by), (bx + bar_w, by + bar_h), (60, 60, 60), -1)
        cv2.rectangle(display, (bx, by),
                      (bx + int(bar_w * progress), by + bar_h), (0, 220, 120), -1)
        cv2.rectangle(display, (bx, by), (bx + bar_w, by + bar_h), (140, 140, 140), 1)
        cv2.putText(display, f"Averaging {frame_i+1}/{num_frames}  (ESC to cancel)",
                    (bx + 4, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (200, 200, 200), 1)

        # show which markers have been consistently detected
        y_off = h - 70
        for mid in expected_ids:
            n = len(accum[mid][0])
            pct = n / (frame_i + 1) * 100 if frame_i > 0 else 0
            color = (0, 200, 0) if pct > 80 else (0, 140, 240) if pct > 40 else (0, 80, 200)
            cv2.putText(display, f"ID {mid}: {n}/{frame_i+1} ({pct:.0f}%)",
                        (bx, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1)
            y_off -= 16

        cv2.imshow("Averaging markers — 60 frames", display)

        key = cv2.waitKey(10) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            cap.release()
            cv2.destroyWindow("Averaging markers — 60 frames")
            print("Averaging cancelled.")
            return None

    cap.release()
    cv2.destroyWindow("Averaging markers — 60 frames")

    # compute medians
    result = {}
    for mid in expected_ids:
        if len(accum[mid][0]) > 0:
            cx_med = float(np.median(accum[mid][0]))
            cy_med = float(np.median(accum[mid][1]))
            result[mid] = (cx_med, cy_med)
            print(f"  Marker {mid}: median pixel ({cx_med:.1f}, {cy_med:.1f})  "
                  f"({len(accum[mid][0])}/{num_frames} frames)")
        else:
            print(f"  Marker {mid}: NOT DETECTED in any frame!")

    return result


# ── Manual click fallback ─────────────────────────────────────────────────────

def manual_click_calibration(frame):
    """Fallback: operator clicks 4 reference points in an OpenCV window.

    Returns:
        image_points: list of (u, v) pixel coordinates, or None if cancelled.
    """
    image_points = []

    cv2.namedWindow("Calibration: Click 4 reference points", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration: Click 4 reference points", 960, 720)

    click_pos = [None]  # mutable container for callback

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)

    cv2.setMouseCallback("Calibration: Click 4 reference points", on_mouse)

    for i in range(NUM_REF_POINTS):
        color = CLICK_COLORS[i]
        display = frame.copy()
        for j, (px, py) in enumerate(image_points):
            cv2.circle(display, (int(px), int(py)), 6, CLICK_COLORS[j], -1)
            cv2.putText(display, str(j + 1), (int(px) + 10, int(py) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.88, CLICK_COLORS[j], 2)
        cv2.putText(display, f"Click point {i+1}/{NUM_REF_POINTS}  (Q=quit, R=redo last)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.69, color, 2)
        cv2.imshow("Calibration: Click 4 reference points", display)

        click_pos[0] = None
        while click_pos[0] is None:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                return None
            if key == ord('r') and image_points:
                # redo last point
                removed = image_points.pop()
                print(f"  Redoing point {len(image_points)+1} (was pixel ({removed[0]:.0f}, {removed[1]:.0f}))")
                click_pos[0] = (-1, -1)  # sentinel to break inner loop
                break

        if click_pos[0] == (-1, -1):
            # redo this iteration
            i -= 1
            continue

        px, py = click_pos[0]
        image_points.append((px, py))
        print(f"  Point {i+1} clicked at pixel ({px}, {py})")

    cv2.destroyAllWindows()
    return image_points


POINTS_CONFIG_FILE = "calibration_points.json"


def is_config_ready(filename=POINTS_CONFIG_FILE):
    """Check if config exists and has all plate coordinates filled in."""
    if not os.path.exists(filename):
        return False
    try:
        data = json.loads(open(filename).read())
    except (json.JSONDecodeError, Exception):
        return False
    for pt in data.get("points", []):
        if pt.get("plate_x_mm") is None or pt.get("plate_y_mm") is None:
            return False
        if pt.get("pixel_u") is None or pt.get("pixel_v") is None:
            return False
    return True


def live_validation_view(camera_idx, available, config_file=POINTS_CONFIG_FILE):
    """Interactive view showing homography predictions vs ground truth.

    Detects markers live, projects through H, and displays predicted plate
    coordinates alongside ground truth for all markers — especially the
    held-out validation marker (ID 4).

    Keys:
        c/arrows — cycle cameras
        r        — reload config JSON, recompute H (after editing measurements)
        SPACE    — re-average marker pixel positions (60 frames), update JSON
        s        — save homography_calibration.json and exit
        q/ESC    — quit without saving
    """
    cap = _open_camera_raw(camera_idx)
    if cap is None:
        print(f"[ERROR] Cannot open camera {camera_idx}")
        return

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Load config and compute H
    def load_and_fit():
        calib_img, calib_plt, val_img, val_plt, _ = read_points_config(config_file)
        if calib_img is None:
            return None, None, None, None, None
        H, errors, rms, mask = compute_homography(calib_img, calib_plt)
        if H is None:
            print("[ERROR] Homography computation failed.")
            return None, None, None, None, None
        return H, calib_img, calib_plt, val_img, val_plt

    H, calib_img, calib_plt, val_img, val_plt = load_and_fit()
    if H is None:
        cap.release()
        return

    print("\nLive validation view:")
    print("  r     = reload JSON, recompute H")
    print("  SPACE = re-average pixel positions (60 frames)")
    print("  s     = save homography_calibration.json")
    print("  q/ESC = quit\n")

    cv2.namedWindow("Calibration — Camera Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration — Camera Feed", 960, 720)
    cv2.namedWindow("Calibration — Readout", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration — Readout", 340, 330)

    status_msg = ""  # feedback line shown on readout panel

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        corners, ids, _ = detector.detectMarkers(frame)
        found = {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                pts = corners[i][0]
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                found[int(mid)] = (cx, cy)

        display = frame.copy()
        h, w = display.shape[:2]

        # Draw detected markers and their predicted positions
        for mid, (cx, cy) in found.items():
            vpt = np.array([[[cx, cy]]], dtype=np.float32)
            pred = cv2.perspectiveTransform(vpt, H)
            px_mm, py_mm = float(pred[0][0][0]), float(pred[0][0][1])

            is_val = (mid == VALIDATION_MARKER_ID)
            color = (0, 255, 255) if is_val else (0, 255, 0)

            cv2.circle(display, (int(cx), int(cy)), 6, color, -1)
            cv2.putText(display, f"ID {mid}", (int(cx) + 10, int(cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)
            cv2.putText(display, f"({px_mm:.1f}, {py_mm:.1f}) mm",
                        (int(cx) + 10, int(cy) + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # hints bar on feed
        cv2.putText(display, "r=reload  SPACE=re-average  s=save  q=quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

        cv2.imshow("Calibration — Camera Feed", display)

        # ── Readout panel (separate window) ─────────────────────
        panel_w, panel_h = 340, 280
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        y = 20

        # title
        cv2.putText(panel, "Homography Validation", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)
        y += 22

        # calibration markers summary
        for i in range(NUM_REF_POINTS):
            mid = MARKER_IDS[i]
            if mid in found:
                cx, cy = found[mid]
                vpt = np.array([[[cx, cy]]], dtype=np.float32)
                pred = cv2.perspectiveTransform(vpt, H)
                px_mm, py_mm = float(pred[0][0][0]), float(pred[0][0][1])
                tx, ty = calib_plt[i]
                err = np.sqrt((px_mm - tx)**2 + (py_mm - ty)**2)
                cv2.putText(panel, f"ID{mid}: pred({px_mm:.0f},{py_mm:.0f}) true({tx:.0f},{ty:.0f}) e={err:.1f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                            (0, 220, 0) if err < 5 else (0, 140, 255), 1)
            else:
                cv2.putText(panel, f"ID{mid}: not visible",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (80, 80, 80), 1)
            y += 15

        y += 6
        cv2.line(panel, (10, y), (panel_w - 10, y), (80, 80, 80), 1)
        y += 12

        # validation marker
        cv2.putText(panel, "Validation (held out)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
        y += 20
        if VALIDATION_MARKER_ID in found and val_img is not None and val_plt is not None:
            cx, cy = found[VALIDATION_MARKER_ID]
            vpt = np.array([[[cx, cy]]], dtype=np.float32)
            pred = cv2.perspectiveTransform(vpt, H)
            px_mm, py_mm = float(pred[0][0][0]), float(pred[0][0][1])
            tx, ty = val_plt
            val_err = np.sqrt((px_mm - tx)**2 + (py_mm - ty)**2)
            cv2.putText(panel, f"Pixel:  ({cx:.0f}, {cy:.0f})", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 16
            cv2.putText(panel, f"Pred:   ({px_mm:.1f}, {py_mm:.1f}) mm", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 16
            cv2.putText(panel, f"Truth:  ({tx:.1f}, {ty:.1f}) mm", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 22
            cv2.putText(panel, f"ERROR: {val_err:.1f} mm  {'PASS' if val_err <= 5 else 'FAIL'}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.69,
                        (0, 255, 0) if val_err <= 5 else (0, 0, 255), 1)
        elif val_img is None or val_plt is None:
            cv2.putText(panel, "Not configured — add ID 4 to JSON", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (80, 80, 80), 1)
        else:
            cv2.putText(panel, "ID 4: not visible", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (80, 80, 200), 1)

        # status message + key controls
        panel_y = panel_h - 42
        if status_msg:
            cv2.putText(panel, status_msg, (10, panel_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 200), 1)
            panel_y += 14
        cv2.putText(panel, "c=cycle cam  r=reload  SPACE=re-avg  s=save  q=quit",
                    (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        cv2.imshow("Calibration — Readout", panel)

        key = cv2.waitKey(100) & 0xFF
        status_msg = ""  # clear after one frame

        if key in (ord('c'), 83, 81):  # c, right arrow, left arrow
            pos = available.index(camera_idx)
            if key == 81:  # left arrow
                next_idx = available[(pos - 1) % len(available)]
            else:
                next_idx = available[(pos + 1) % len(available)]
            new_cap = _open_camera_raw(next_idx)
            if new_cap is not None:
                cap.release()
                cap = new_cap
                camera_idx = next_idx
                status_msg = f"Switched to camera {camera_idx}"
                print(f"  Camera -> {camera_idx}")
            else:
                status_msg = f"Cannot open camera {next_idx}"

        elif key == ord('r'):
            print("Reloading config...")
            H_new, calib_img, calib_plt, val_img, val_plt = load_and_fit()
            if H_new is not None:
                H = H_new
                status_msg = "Config reloaded — H recomputed"
                print("  H recomputed.")
            else:
                status_msg = "Reload FAILED — check JSON"
                print("  [ERROR] Reload failed — check JSON.")

        elif key == 32:  # SPACE — re-average
            print("Re-averaging marker positions (60 frames)...")
            cap.release()  # free device so capture_averaged_markers can open it
            avg = capture_averaged_markers(camera_idx, ALL_MARKER_IDS)
            # re-acquire for live view
            cap = _open_camera_raw(camera_idx)
            if cap is None:
                print(f"[ERROR] Cannot re-open camera {camera_idx} after averaging.")
                break
            if avg is not None:
                # update pixel positions in the JSON
                data = json.loads(open(config_file).read())
                for pt in data["points"]:
                    mid = pt["marker_id"]
                    if mid in avg:
                        pt["pixel_u"] = round(avg[mid][0], 1)
                        pt["pixel_v"] = round(avg[mid][1], 1)
                with open(config_file, "w") as f:
                    json.dump(data, f, indent=2)
                status_msg = "Pixel positions updated"
                print("  Pixel positions updated in JSON.")
                # recompute H with new pixel positions + existing plate coords
                calib_img, calib_plt, val_img, val_plt, _ = read_points_config(config_file)
                if calib_img is not None:
                    H_new, _, _, _ = compute_homography(calib_img, calib_plt)
                    if H_new is not None:
                        H = H_new
                        status_msg = "Re-averaged + H recomputed"
                        print("  H recomputed.")

        elif key == ord('s'):
            calib_img, calib_plt, val_img, val_plt, _ = read_points_config(config_file)
            if calib_img is None:
                status_msg = "Save FAILED — config invalid"
                print("[ERROR] Cannot save — config not valid.")
                continue
            H_final, errors, rms, _ = compute_homography(calib_img, calib_plt)
            if H_final is None:
                status_msg = "Save FAILED — homography error"
                print("[ERROR] Cannot save — homography failed.")
                continue
            save_calibration(H_final, calib_img, calib_plt, rms,
                             camera_idx=camera_idx)
            status_msg = f"Saved! RMS={rms:.1f}mm"
            print("Saved. Press q to exit.")

        elif key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


def write_points_config(markers, filename=POINTS_CONFIG_FILE):
    """Write detected marker positions with empty plate-coordinate placeholders.

    Never overwrites an existing config file — the user's measurements are
    preserved. If the file already exists, informs the user and exits.
    """
    out_path = Path(filename)

    if out_path.exists():
        print(f"\n{filename} already exists — preserving your measurements.")
        print(f"To regenerate pixel positions, delete or rename the file and re-run.")
        return

    points = []
    for mid in ALL_MARKER_IDS:
        role = "calibration" if mid in MARKER_IDS else "validation"
        if mid in markers:
            cx, cy = markers[mid]
            points.append({
                "marker_id": mid,
                "role": role,
                "pixel_u": round(cx, 1),
                "pixel_v": round(cy, 1),
                "plate_x_mm": None,
                "plate_y_mm": None,
            })
        else:
            points.append({
                "marker_id": mid,
                "role": role,
                "pixel_u": None,
                "pixel_v": None,
                "plate_x_mm": None,
                "plate_y_mm": None,
            })

    data = {
        "description": (
            "Fill in plate_x_mm and plate_y_mm for each marker with the "
            "physical (x, y) coordinates measured from the base-plate origin "
            "(middle of top edge of base plate). Replace null with the measured value. "
            "Markers 0-3 are used for homography fit; marker 4 is a held-out "
            "validation point. Then run: python calibrate_homography.py --from-config"
        ),
        "camera": markers.get("_camera", 0),
        "points": points,
    }

    out_path.write_text(json.dumps(data, indent=2))
    print(f"\nWrote {out_path.resolve()}")
    print("\nNext steps:")
    print(f"  1. Edit {filename} — fill in plate_x_mm and plate_y_mm for ALL 5 markers")
    print(f"     (measure from base-plate origin with framing square + ruler)")
    print(f"  2. Run: python calibrate_homography.py --from-config {filename}")


def read_points_config(filename=POINTS_CONFIG_FILE):
    """Read calibration points config.

    Returns:
        (calib_image, calib_plate, val_image, val_plate, camera_idx)
        calib_image/plate: lists for markers 0-3 (homography fit)
        val_image/plate:   single (px,py) and (x,y) for validation marker 4, or (None, None)
        camera_idx: int
        All None on error.
    """
    if not os.path.exists(filename):
        print(f"[ERROR] {filename} not found.")
        print("  Run without --from-config first to detect markers and generate the file.")
        return None, None, None, None, None

    try:
        with open(filename) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse {filename}: {e}")
        print("  Check the file for syntax errors (missing comma, trailing comma, etc).")
        return None, None, None, None, None
    except Exception as e:
        print(f"[ERROR] Failed to read {filename}: {e}")
        return None, None, None, None, None

    calib_image = []
    calib_plate = []
    val_image   = None
    val_plate   = None

    for pt in data.get("points", []):
        mid  = pt.get("marker_id", "?")
        role = pt.get("role", "calibration")
        px, py = pt.get("pixel_u"), pt.get("pixel_v")
        x, y   = pt.get("plate_x_mm"), pt.get("plate_y_mm")

        if px is None or py is None:
            print(f"[ERROR] Marker {mid}: pixel position missing. Re-run detection to regenerate.")
            return None, None, None, None, None

        if x is None or y is None:
            print(f"[ERROR] Marker {mid} ({role}): plate coordinates not filled in.")
            print(f"  Edit {filename} and replace null with measured (x, y) for ALL markers.")
            return None, None, None, None, None

        if role == "validation":
            val_image = (float(px), float(py))
            val_plate = (float(x), float(y))
        else:
            calib_image.append((float(px), float(py)))
            calib_plate.append((float(x), float(y)))

    if len(calib_image) != NUM_REF_POINTS:
        print(f"[ERROR] Expected {NUM_REF_POINTS} calibration markers, found {len(calib_image)}.")
        return None, None, None, None, None

    camera_idx = data.get("camera", DEFAULT_CAMERA)
    return calib_image, calib_plate, val_image, val_plate, camera_idx

def compute_homography(image_points, plate_points):
    """Fit homography with RANSAC and compute per-point reprojection errors.

    Returns:
        H: (3, 3) homography matrix
        errors: list of (point_index, error_mm)
        rms: root-mean-square reprojection error (mm)
        mask: RANSAC inlier mask
    """
    img_pts = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    plt_pts = np.array(plate_points, dtype=np.float32).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(img_pts, plt_pts, method=cv2.RANSAC,
                                  ransacReprojThreshold=3.0)

    if H is None:
        return None, [], float('inf'), None

    # per-point reprojection error
    projected  = cv2.perspectiveTransform(img_pts, H)  # (N, 1, 2)
    errors_mm  = np.linalg.norm(projected - plt_pts, axis=2).flatten()
    rms        = float(np.sqrt(np.mean(errors_mm ** 2)))

    errors = [(i, float(e)) for i, e in enumerate(errors_mm)]

    return H, errors, rms, mask


# ── Calibration file ──────────────────────────────────────────────────────────

def save_calibration(H, image_points, plate_points, rms, camera_idx):
    """Write homography_calibration.json."""
    data = {
        "version": CALIBRATION_VERSION,
        "date": datetime.now().isoformat(),
        "camera": camera_idx,
        "camera_resolution": list(_actual_resolution),
        "origin_description": (
            "Middle of top edge of robot base plate "
            "(camera-near edge, gripper home position)"
        ),
        "reference_points_plate_mm": [[x, y] for x, y in plate_points],
        "reference_points_pixel":   [[u, v] for u, v in image_points],
        "homography_matrix": H.tolist(),
        "reprojection_error_rms_mm": round(rms, 2),
        "table_height_mm": 0,
    }
    out_path = Path(CALIBRATION_FILE)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"\nCalibration saved: {out_path.resolve()}")


# ── Marker PDF generator ──────────────────────────────────────────────────────

def generate_markers_pdf():
    """Generate a printable A4 PDF with 4 ArUco markers (IDs 0-3)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("[ERROR] matplotlib is required for PDF generation.")
        print("  pip install matplotlib")
        sys.exit(1)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    # A4: 210 x 297 mm
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=72)  # 72 dpi -> figsize in inches
    ax.set_xlim(0, 210)
    ax.set_ylim(0, 297)
    ax.set_aspect("equal")
    ax.axis("off")

    # layout: 2x2 grid for calibration markers + 1 validation marker centered below
    positions_mm = [
        (35, 200),   # top-left     — ID 0
        (120, 200),  # top-right    — ID 1
        (35, 95),    # bottom-left  — ID 2
        (120, 95),   # bottom-right — ID 3
        (77, 15),    # centered below — ID 4 (validation)
    ]

    for mid, (pos_x, pos_y) in zip(ALL_MARKER_IDS, positions_mm):
        marker_img = cv2.aruco.generateImageMarker(
            dictionary, mid, MARKER_SIZE_MM * 10
        )  # generateImageMarker uses pixels — scale up for clean rendering
        marker_img = marker_img[1:-1, 1:-1]  # strip the 1px white border (OpenCV adds it)

        # matplotlib imshow with y-axis flipped (image origin is top-left)
        ax.imshow(marker_img, cmap="gray", origin="upper",
                  extent=(pos_x, pos_x + MARKER_SIZE_MM, pos_y, pos_y + MARKER_SIZE_MM))

        # ID label below marker
        role = "VALIDATION" if mid == VALIDATION_MARKER_ID else "calibration"
        ax.text(pos_x + MARKER_SIZE_MM / 2, pos_y - 6,
                f"ID {mid} ({role})", ha="center", va="top",
                fontsize=8, fontweight="bold", family="monospace")

    # scale-check square (10mm)
    scale_x, scale_y = 30, 275
    ax.add_patch(Rectangle((scale_x, scale_y), 10, 10,
                            edgecolor="black", facecolor="none", lw=1))
    ax.text(scale_x + 5, scale_y - 4, "10mm",
            ha="center", va="top", fontsize=8, family="monospace")

    # instructions
    ax.text(105, 285, "Print at 100% scale (no scaling / no fit-to-page).\n"
                      "Verify the 10mm square with a ruler.\n"
                      "Place each marker centered on its measurement dot.",
            ha="center", va="bottom", fontsize=7, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("ArUco Calibration Markers  —  DICT_4X4_50, IDs 0-3 (calib) + ID 4 (validation), 25mm",
                 fontsize=10, family="monospace", y=0.98)

    out_pdf = Path("calibration_markers.pdf")
    fig.savefig(str(out_pdf), dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Wrote {out_pdf.resolve()}")
    print("Print at 100% scale (no scaling / no fit-to-page).")
    print("Verify the 10mm square with a ruler before placing markers.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Camera-to-Plate Homography Calibration v1"
    )
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA,
                        help=f"Camera index (default: {DEFAULT_CAMERA})")
    parser.add_argument("--list", action="store_true",
                        help="Print available cameras and exit")
    parser.add_argument("--generate-markers", action="store_true",
                        help="Generate a printable PDF with 4 ArUco markers and exit")
    parser.add_argument("--from-config", default=None, const=POINTS_CONFIG_FILE, nargs="?",
                        help="Launch live validation from a specific config file")
    args = parser.parse_args()

    # ── Marker generation mode ──────────────────────────────────
    if args.generate_markers:
        generate_markers_pdf()
        sys.exit(0)

    # ── Camera probing ──────────────────────────────────────────
    print("Scanning cameras...")
    available = probe_cameras()
    if not available:
        print("[ERROR] No cameras found.")
        sys.exit(1)
    print(f"Available cameras: {available}")
    if args.list:
        sys.exit(0)

    print("""
╔══════════════════════════════════════════════════════════════╗
║          Camera-to-Plate Homography Calibration              ║
╠══════════════════════════════════════════════════════════════╣
║  Keys (same throughout):                                    ║
║    C / →       — next camera                                ║
║    ←           — previous camera                            ║
║    ENTER       — confirm selection / proceed                ║
║    SPACE       — re-average pixel positions (60 frames)     ║
║    S           — save calibration                           ║
║    R           — reload config + recompute homography       ║
║    Q / ESC     — quit / cancel                              ║
╠══════════════════════════════════════════════════════════════╣
║  Workflow:                                                  ║
║    1. Select camera                                         ║
║    2. Detect markers → SPACE to average (suppresses jitter) ║
║    3. Fill plate_x_mm / plate_y_mm in calibration_points.json║
║    4. Re-run → enters live validation view                  ║
║    5. Press S to save homography_calibration.json           ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Determine config file
    config_file = args.from_config if args.from_config else POINTS_CONFIG_FILE

    # ═════════════════════════════════════════════════════════════
    # CASE 1: Config is fully filled → live validation view
    # ═════════════════════════════════════════════════════════════
    if is_config_ready(config_file):
        print(f"\n{config_file} is fully filled — launching live validation.")
        cam_idx = interactive_camera_select(available, start_idx=args.camera)
        if cam_idx is None:
            print("Cancelled.")
            sys.exit(0)
        live_validation_view(cam_idx, available, config_file)
        sys.exit(0)

    # ═════════════════════════════════════════════════════════════
    # CASE 2: Config exists but not ready → inform user
    # ═════════════════════════════════════════════════════════════
    if os.path.exists(config_file):
        print(f"\n{config_file} exists but has unfilled plate coordinates.")
        print("Edit the file and replace null with measured (x, y) mm values for all markers,")
        print(f"then re-run. Or delete {config_file} to re-detect markers from scratch.\n")

        # Check if this is --from-config (user explicitly asked for it)
        if args.from_config:
            print("Cannot proceed — config is incomplete. Fill in all plate_x_mm / plate_y_mm fields.")
            sys.exit(1)

        # Offer to re-detect anyway
        resp = input("Delete config and re-detect markers? [y/N] ").strip().lower()
        if resp == 'y':
            os.remove(config_file)
            print(f"Deleted {config_file}.")
        else:
            print("Aborted. Edit the file or delete it and re-run.")
            sys.exit(0)

    # ═════════════════════════════════════════════════════════════
    # CASE 3: No config → detect markers, write config for user to edit
    # ═════════════════════════════════════════════════════════════
    cam_idx = interactive_camera_select(available, start_idx=args.camera)
    if cam_idx is None:
        print("Cancelled.")
        sys.exit(0)

    frame, err = capture_frame(cam_idx)
    if frame is None:
        print(f"[ERROR] {err}")
        sys.exit(1)
    print(f"Captured frame: {frame.shape[1]}x{frame.shape[0]}")

    print("Detecting ArUco markers (DICT_4X4_50)...")
    markers, corners, detected_ids = detect_markers(frame)

    found_ids = sorted(markers.keys())
    missing_ids = [mid for mid in ALL_MARKER_IDS if mid not in markers]
    nc = len([m for m in found_ids if m in MARKER_IDS])
    nv = 1 if VALIDATION_MARKER_ID in found_ids else 0
    print(f"  Found markers: IDs {found_ids if found_ids else 'none'}  "
          f"(calib: {nc}/{NUM_REF_POINTS}, valid: {nv}/1)")
    if missing_ids:
        print(f"  Missing markers: IDs {missing_ids}")

    # Visual feedback + averaging option
    display = draw_detected_markers(frame, markers, found_ids)
    cv2.namedWindow("Calibration: Detected Markers", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration: Detected Markers", 960, 720)
    if nc == NUM_REF_POINTS:
        vmsg = " + validation" if nv == 1 else " (validation marker missing)"
        cv2.putText(display, f"All 4 calib markers found{vmsg}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
        cv2.putText(display, "SPACE = re-average over 60 frames (recommended)  |  any other key = use single frame",
                    (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    else:
        cv2.putText(display, f"Calib: {nc}/4, Valid: {nv}/1 — press any key to continue",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.69, (0, 165, 255), 2)
    cv2.imshow("Calibration: Detected Markers", display)
    print("\nReview the detected markers in the camera view.")
    if nc == NUM_REF_POINTS:
        print("  SPACE = re-average pixel positions over 60 frames (suppresses jitter)")
        print("  any other key = use single-frame positions as-is")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("Calibration: Detected Markers")

    # If all 4 calib found and user pressed SPACE → multi-frame averaging
    if nc == NUM_REF_POINTS and key == 32:  # SPACE
        avg_markers = capture_averaged_markers(cam_idx, ALL_MARKER_IDS)
        if avg_markers is None:
            print("Averaging cancelled — using single-frame positions.")
        else:
            markers = avg_markers
            still_missing = [mid for mid in MARKER_IDS if mid not in markers]
            if still_missing:
                print(f"\nAfter averaging, calibration markers {still_missing} were never detected.")
                print("Check lighting/placement and re-run.")
                sys.exit(1)

    # Write config
    if nc == NUM_REF_POINTS:
        markers["_camera"] = cam_idx
        write_points_config(markers)
        if nv == 0:
            print("Note: validation marker (ID 4) not detected — independent accuracy check unavailable.")
    else:
        print(f"\nNeed {NUM_REF_POINTS} calibration markers (IDs {MARKER_IDS}). "
              f"Found: {nc}/{NUM_REF_POINTS}.")
        extra = [mid for mid in found_ids if mid not in ALL_MARKER_IDS]
        if extra:
            print(f"Unknown markers detected: IDs {extra}. Remove them and re-run.")
        resp = input("Fall back to manual click? [Y/n] ").strip().lower()
        if resp == 'n':
            print("Aborted. Adjust lighting or marker placement and re-run.")
            sys.exit(0)
        image_points = manual_click_calibration(frame)
        if image_points is None:
            print("Cancelled.")
            sys.exit(0)
        markers = {i: (u, v) for i, (u, v) in enumerate(image_points)}
        markers["_camera"] = cam_idx
        write_points_config(markers)
        print("Note: manual click does not include a validation marker.")


if __name__ == "__main__":
    main()
