# Depth Context: LeRobot Target Annotator

A real-time visual goal conditioning tool for the LeRobot project — multi-stream ZMQ annotation + calibrated coordinate output. Enables open-vocabulary object targeting via YOLO-World with stabilized bounding box overlays, wrist passthrough, and a static target patch for SmolVLA's camera3 slot.

## Document Summaries (Local Files)
| File | Role & Significance (Summary) | Freshness | Status |
| :--- | :--- | :--- | :--- |
| `README.md` | **[Primary SSoT]** Full guide — architecture, quick start, calibration workflow, controls, CLI, ZMQ format, data-collection protocol. | 2026-07-04 | ACTIVE |
| `annotate_stream_multi.py` | **[Core Script — v6]** Multi-stream annotator (default). Three streams: detect (YOLO + H-transform), passthrough (raw wrist), patch (static template). Supports calibrated coordinate output via planar homography. | 2026-07-04 | ACTIVE |
| `annotate_stream.py` | **[Legacy — v5]** Single-stream annotator. Preserved for backward compatibility via `--legacy` flag. | 2026-06-20 | STABLE |
| `calibrate_homography.py` | **[Calibration]** Camera-to-plate homography calibration. 5 ArUco markers, native resolution detection, live validation view. Outputs `homography_calibration.json`. | 2026-07-04 | ACTIVE |
| `probe_cameras.py` | **[Utility]** Pre-flight camera probe — tiles all connected cameras into a single window with index labels. | 2026-05-08 | ACTIVE |
| `coord_subscriber.py` | **[Utility]** Reference subscriber — logs coordinate ZMQ stream to CSV alongside recording sessions. | 2026-07-02 | ACTIVE |
| `annotate_stream_multi_state.json` | **[Config]** Auto-generated runtime state (camera index, classes, zones, frozen bbox, H-toggle). Restored on launch unless `--fresh`. | 2026-07-04 | ACTIVE |
| `homography_calibration.json` | **[Config]** Auto-generated calibration file (homography matrix, camera resolution, reference points). Loaded at annotator startup. | 2026-07-04 | ACTIVE |
| `calibration_points.json` | **[Config]** Intermediate calibration config — marker pixel positions + user-filled plate coordinates. | 2026-07-02 | ACTIVE |
| `requirements.txt` | **[Config]** Project dependencies (ultralytics, pyzmq, opencv-contrib-python). | 2026-04-28 | ACTIVE |
| `LICENSE` | **[Legal]** MIT License. | 2026-04-28 | ACTIVE |

## Sub-Folder Index
| Folder | Purpose & Strategy | Key SSoT | Depth |
| :--- | :--- | :--- | :--- |
| `archive/` | Superseded script versions (v2, v4) and timestamped backups. Not used at runtime. | - | 0f |
| `examples/` | Usage examples and shell scripts for integrating with `lerobot-record`. | `record_command.sh` | 1f |
| `snapshots/` | HUD snapshots saved with `S` key. Git-ignored. | - | 0f |

## Camera Convention
Camera indices are dynamic — run `python probe_cameras.py` to identify available cameras. Default setup:
- Stream 0 (detect): overview camera → ZMQ `annotated` → SmolVLA `camera1`
- Stream 1 (passthrough): wrist/gripper camera → ZMQ `wrist` → SmolVLA `camera2`
- Stream 2 (patch): derived from stream 0 → ZMQ `target_patch` → SmolVLA `camera3`

All cameras open at native resolution (1920×1080 requested, V4L2 clamps). Frames are center-cropped to target aspect ratio, uniformly resized — no stretching.

## Vault Design Docs
Detailed requirements and technical specs live in the vault:
`Knowledge_Drafts/LeRobot/Environment_and_SW/lerobot-fork/calibrated-coord-annotator/`
- `0_annotator_architecture_overview.md` — architecture overview
- `1_requirements_and_options.md` — coordinate frame conventions, calibration options
- `2_calibration_technical_specs.md` — ArUco workflow, UI flow
- `3_build_plan.md` — implementation steps
- `4_annotator_user_requirements.md` — user requirements
- `5_technical_requirements.md` — technical spec
