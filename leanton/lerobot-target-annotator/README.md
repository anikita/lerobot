# lerobot-target-annotator

**Real-time visual goal conditioning for [LeRobot](https://github.com/huggingface/lerobot) data collection — multi-stream ZMQ annotation + calibrated coordinate output.**

Publishes three simultaneous ZMQ streams (overview + wrist + target patch) directly into `lerobot-record` as native camera channels. Adds a calibrated coordinate overlay — target object position in millimeters relative to the robot base plate — via planar homography. No LeRobot modifications required.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  annotate_stream_multi.py                       │
│                                                 │
│  Stream 0 [detect]       camera 4 → ZMQ :5555  │
│    YOLO-World + buffer → annotated overview     │
│    + H-transform (mm coords on frozen bbox)     │
│                                                 │
│  Stream 1 [passthrough]  camera 0 → ZMQ :5556  │
│    Raw wrist/gripper feed                       │
│                                                 │
│  Stream 2 [patch]         derived → ZMQ :5557   │
│    Static target template (256×256)             │
│    Snapshot at freeze, replay until unfreeze    │
└─────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
   lerobot-record       SmolVLA training
   camera1/2/3          camera1/2/3 slots
```

- **Stream 0 (detect):** YOLO-World open-vocabulary detection on the overview camera. Stabilization buffer (IoU-gated rolling median + hysteresis) locks onto the largest object. Bounding box drawn on the frame, published as `camera1`.
- **Stream 1 (passthrough):** Raw wrist/gripper camera feed — no detection, no overlay. Published as `camera2`.
- **Stream 2 (patch):** Static 256×256 crop of the frozen target, replayed every frame until unfreeze. Published as `camera3`. SmolVLA's vision encoder already has a `camera3` slot — this populates it with a visual template.

All cameras open at native resolution (1920×1080 requested, V4L2 clamps to max). Frames are center-cropped to the ZMQ target aspect ratio (4:3), then uniformly resized — no stretching, distortion-free regardless of camera native aspect ratio.

---

## Calibrated Coordinates (H-Transform)

The annotator can compute the target object's (x, y) position in millimeters — robot base-plate frame, table plane — via a pre-calibrated planar homography.

| Feature | Detail |
|:---|:---|
| **Calibration** | `calibrate_homography.py` — 5 ArUco markers, planar homography, one-time per camera/table setup |
| **Activation** | Press `H` to toggle ON/OFF. Default OFF. |
| **Coordinate output** | Only when FROZEN — coordinate is the bbox center at freeze moment, held constant until unfreeze |
| **On-screen** | `(143, 267) mm` displayed next to frozen bbox. HUD shows `H: ON` / `H: OFF` / `H: N/A` |
| **ZMQ** | `target_coord_mm` field embedded in the annotated ZMQ message (FROZEN only) |
| **Accuracy** | ±5mm target (RSS budget ~±4mm with manual measurement) |
| **Persistence** | H-toggle state saved across sessions. Forced OFF if calibration file is missing. |

When H is OFF or the detection is unfrozen, the ZMQ message is identical to standard v6 format — no extra fields, fully backward-compatible.

---

## Quick Start

```bash
# Pre-flight — identify your cameras
python probe_cameras.py

# Terminal 1 — multi-stream annotator (default: detect + passthrough + patch)
conda activate lerobot
cd lerobot-target-annotator
python annotate_stream_multi.py

# Terminal 2 — record with 3 ZMQ cameras
conda activate lerobot
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem<FOLLOWER_ID> \
  --robot.cameras='{
    "annotated":    {"type": "zmq", "server_address": "localhost", "port": 5555, "camera_name": "annotated",    "width": 640, "height": 480, "fps": 30},
    "wrist":        {"type": "zmq", "server_address": "localhost", "port": 5556, "camera_name": "wrist",        "width": 640, "height": 480, "fps": 30},
    "target_patch": {"type": "zmq", "server_address": "localhost", "port": 5557, "camera_name": "target_patch", "width": 256, "height": 256, "fps": 30}
  }' \
  --dataset.repo_id=<HF_USERNAME>/<DATASET_NAME> \
  --dataset.single_task="pick the target object and place it in the basket" \
  --dataset.num_episodes=50
```

**Start Terminal 1 before Terminal 2.** LeRobot's `ZMQCamera` blocks on warmup until the publisher is available.

On subsequent runs state is auto-restored. Pass `--fresh` to start from defaults.

---

## Calibration Workflow

```bash
# One-time setup
python calibrate_homography.py --generate-markers   # print ArUco markers
python calibrate_homography.py                       # detect markers → write calibration_points.json

# Edit calibration_points.json — fill in plate_x_mm / plate_y_mm for all 5 markers
# Measure from the base-plate origin (middle of top edge) with a framing square and ruler

python calibrate_homography.py                       # re-run → live validation view
#   SPACE = re-average pixel positions (60 frames)
#   S     = save homography_calibration.json
#   R     = reload JSON + recompute H after editing measurements
#   Q     = quit
```

After calibration, launch the annotator — the HUD shows `H: OFF`. Press `H` to activate calibrated coordinates.

---

## Controls

### Annotator (`annotate_stream_multi.py`)

| Key | Action |
|:---|:---|
| `TAB` | Cycle active stream (which stream receives keystrokes) |
| `F` | Freeze / unfreeze target on active detect stream |
| `H` | Toggle calibrated coordinate overlay (detect streams only) |
| `P` | Toggle passthrough / YOLO annotation (detect streams only) |
| `[` / `]` / PgUp / PgDn | Cycle through detected objects |
| `A` | Toggle show-all secondary detections in HUD |
| `R` | Toggle raw / annotated view |
| `S` | Save snapshot of active stream pane to `snapshots/` |
| `Z` | Draw pick zone (detect streams only) / clear |
| `X` | Draw exclusion zone (detect streams only) / clear |
| `C` / ← → | Cycle camera on active camera-backed stream |
| `Q` | Quit (auto-saves state) |

Modifier keys target specific streams: no modifier → active stream, Shift → stream 1, Ctrl+Shift → stream 2.

### Calibrator (`calibrate_homography.py`)

| Key | Action |
|:---|:---|
| `C` / → | Next camera |
| ← | Previous camera |
| `ENTER` | Confirm camera selection / proceed |
| `SPACE` | Re-average pixel positions over 60 frames (suppresses jitter) |
| `S` | Save `homography_calibration.json` (live validation view) |
| `R` | Reload JSON + recompute homography (live validation view) |
| `Q` / `ESC` | Quit / cancel |

---

## CLI Options

### Multi-stream annotator

```
--modes MODES           Comma-separated stream modes: detect,passthrough,patch (default)
--cameras IDX,IDX,      Camera indices per stream (empty for patch)
--zmq-ports P,P,P       ZMQ PUB ports (default: 5555,5556,5557)
--zmq-names N,N,N       ZMQ camera_names (default: annotated,wrist,target_patch)
--zmq-res WxH,WxH,WxH   ZMQ output resolution per stream (default: 640x480,640x480,640x480)
--capture-res WxH,WxH,  Capture resolution per stream (default: empty = native)
--patch-sources ,,0     Source stream index for patch streams
--classes "a,b,c"       YOLO-World class list (applied to all detect streams)
--device auto|cuda|mps  Torch device (default: auto)
--no-zmq                Disable all ZMQ publishers, HUD only
--fresh                 Skip loading saved state
--show-all              Show all detections in HUD at startup
--list                  Print available cameras and exit
--legacy                Run in v5 single-stream compat mode
```

### Calibrator

```
--camera N              Camera index (default: auto-detect)
--list                  Print available cameras and exit
--generate-markers      Print ArUco marker PDF and exit
--from-config FILE      Launch live validation from a specific config file
```

---

## ZMQ Wire Format

Standard message (H OFF or unfrozen):

```json
{
  "timestamps": {"annotated": 1712345678.901},
  "images":     {"annotated": "<base64 JPEG, RGB>"}
}
```

Enriched message (H ON + FROZEN):

```json
{
  "timestamps": {"annotated": 1712345678.901},
  "images":     {"annotated": "<base64 JPEG, RGB>"},
  "target_coord_mm": [143, 267],
  "status": "FROZEN"
}
```

`target_coord_mm` is `[x_mm, y_mm]` rounded to nearest integer millimeter. Stock LeRobot `ZMQCamera` ignores unknown keys — backward-compatible. A custom camera class extracts the coordinate into `observation.target_x_mm` / `observation.target_y_mm` columns (see `4_annotator_user_requirements.md`).

---

## Data-Collection Protocol

1. Launch annotator → wait for green stability bar (STABLE)
2. Verify target — cycle objects with `[`/`]` or draw a pick zone if needed
3. Press `H` to activate coordinate output (if calibrated)
4. **Press `F` to freeze** — bbox locks, patch stream snapshots, coordinate computed (if H is ON)
5. Launch `lerobot-record` → begin teleoperation
6. **Unfreeze immediately after grasp** — the object has moved, pre-grasp coordinate is stale
7. Press `F` again for next object

The coordinate covers the approach phase only: freeze before grasp, unfreeze right after grasp. Keeping it frozen through transport and placement mixes pre-grasp and post-grasp modalities.

---

## State Persistence

Auto-saved on quit and on every meaningful state change (freeze, zone draw, H-toggle, camera cycle). Restored on next launch.

- **State file:** `annotate_stream_multi_state.json`
- **What persists:** per-stream camera index, classes, pick/exclusion zones, frozen bbox + label, H-toggle state
- **Skip restore:** `--fresh`
- **Calibration file:** `homography_calibration.json` (separate, loaded at startup)

---

## Stabilization Design

| Mechanism | What it does |
|:---|:---|
| **IoU-gated rolling median** | 15-frame buffer. Only accumulates frames where new detection overlaps the current median. Median is spatially stable. |
| **Hysteresis** | Once stable, target held for 8 seconds after last valid detection. Arm occlusion tolerated. |
| **Loss cooldown** | After target lost, 8-second cooldown suppresses false re-locks on empty table. |

States: FILLING → STABLE → HOLD (8s) → COOLDOWN (8s) → FILLING.

---

## Files

| File | Role |
|:---|:---|
| `annotate_stream_multi.py` | Multi-stream annotator (v6, default) |
| `annotate_stream.py` | Legacy single-stream annotator (v5 compat) |
| `calibrate_homography.py` | Camera-to-plate homography calibration |
| `probe_cameras.py` | Camera detection utility |
| `coord_subscriber.py` | Reference subscriber — logs coordinates to CSV |
| `homography_calibration.json` | Calibration output (auto-generated) |
| `calibration_points.json` | Intermediate config — marker positions + plate measurements |
| `annotate_stream_multi_state.json` | Runtime state (auto-saved, auto-restored) |

---

## Tested On

- Ubuntu Linux, `device=cuda`
- macOS (Apple Silicon M3), `device=mps`
- LeRobot main branch (post v0.5)
- SO-101 follower/leader arms
- Logitech C920 (overview) + generic USB camera (wrist)
- `yolov8s-worldv2.pt` (auto-downloaded by ultralytics on first run)

---

## Related Work

- [`lerobot-annotate`](https://github.com/huggingface/lerobot-annotate) — official HuggingFace tool for post-hoc language annotation
- [`any4lerobot`](https://github.com/Tavish9/any4lerobot) — dataset format conversion utilities
- [CLIPort](https://cliport.github.io/) — language-conditioned spatial goal conditioning
- [RoboPoint](https://robo-point.github.io/) — VLM-generated point annotations as manipulation goals

---

## License

MIT
