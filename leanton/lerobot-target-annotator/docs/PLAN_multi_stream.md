# Plan: Multi-Stream Annotator (Overview + Wrist + Target Patch)

> **Status:** READY — all design questions resolved, reviewer feedback incorporated
> **Target file:** `annotate_stream.py` (refactor from single-stream to multi-stream, v5 → v6)
> **SmolVLA compatibility:** Confirmed — `camera3` slot already exists in trained model config (`[3, 256, 256]`), currently receiving dummy/black frames. SmolVLA's vision encoder processes all camera slots through the same frozen trunk; cross-attention downstream decides what's important. Populating `camera3` with real patch data requires no architecture changes, only a rename_map entry.

---

## 1. Objective

Scale the annotator from 1 camera → N simultaneous ZMQ streams (default 3), each configured by the user from a single unified stream abstraction. Every stream is an equal citizen running the same code, differentiated only by its **mode**:

| Mode | Source | Behavior |
|:---|:---|:---|
| `detect` | USB camera | YOLO detection → DetectionBuffer → annotated ZMQ output. Full operator controls (freeze, zones, manual selection, raw, show-all). |
| `passthrough` | USB camera | Raw camera feed → ZMQ output. No YOLO, no buffer. Minimal HUD. |
| `patch` | Derived from another stream's freeze event | No camera. **At freeze moment:** snapshots a crop of the source stream's hi-res frame at the frozen bbox. **While frozen:** replays the cached snapshot every frame. **When unfrozen:** publishes black. No operator controls. |

### Mode × passthrough_active matrix

A `detect`-mode stream has a *runtime* `passthrough_active` flag (toggled via `P`). This is distinct from `passthrough` *mode*. The three states and what ZMQ receives:

| Mode | `passthrough_active` | ZMQ output |
|:---|:---|:---|
| `detect` | `False` | Annotated frame (YOLO bbox + overlays) |
| `detect` | `True` | Raw camera frame (YOLO bypassed, HUD minimal) |
| `passthrough` | N/A (ignored) | Raw camera frame (always) |
| `patch` | N/A (ignored) | Cached snapshot or black frame |

**Default 3-stream setup serving SmolVLA's camera1/camera2/camera3:**

| Stream | Mode | Camera | ZMQ port | ZMQ name | SmolVLA slot | Purpose |
|:---|:---|:---|:---|:---|:---|:---|
| 0 | `detect` | /dev/video0 (1024×768→640×480) | 5555 | `annotated` | `camera1` | Coarse approach — YOLO bbox guides arm |
| 1 | `passthrough` | /dev/video2 (640×480) | 5556 | `wrist` | `camera2` | Fine positioning — gripper-eye view |
| 2 | `patch` | derived from stream 0 | 5557 | `target_patch` | `camera3` | Visual template — what to grasp |

**The user configures the allocation** — which camera goes to which stream, which mode each runs. All combinations are valid. Adding a 4th stream of any mode requires zero new code, just another config entry.

**The key insight:** Instead of running YOLO on the wrist or implementing explicit template matching, we give SmolVLA's vision head the raw materials — wrist view + frozen target patch — and let the model learn cross-view correspondence implicitly through DAGGER fine-tuning. `camera3` was always receiving dummy black frames; now it gets the target's appearance exactly when it matters (grasp phase). The patch is a true static template: snapshot at freeze moment, cached, replayed every frame until unfreeze. No flicker, no live-pixel-at-fixed-bbox drift.

---

## 2. Modifier Key Dispatch

Three-tier targeting via modifier keys, detected from `cv2.waitKeyEx()`:

| Modifier bits | Modifier | Target stream | Default role |
|:---|:---|:---|:---|
| `0x000000` | *(none)* | stream 0 | Overview (detect) |
| `0x100000` | Shift | stream 1 | Wrist (passthrough) |
| `0x300000` | Ctrl+Shift | stream 2 | Target Patch (patch) |

All shortcut keys dispatch to the targeted stream. What the key *does* depends on that stream's mode — `F` on a `passthrough` stream is a no-op, `F` on a `patch` stream is a no-op. The dispatch is uniform; the stream decides whether to handle it.

| Key | No modifier (stream 0) | Shift (stream 1) | Ctrl+Shift (stream 2) |
|:---|:---|:---|:---|
| `F` | Freeze/unfreeze stream 0 | Freeze/unfreeze stream 1 | *(no-op — patch mode)* |
| `P` | Passthrough toggle stream 0 | Passthrough toggle stream 1 | *(no-op)* |
| `[` `]` / PgUp PgDn | Cycle objects stream 0 | Cycle objects stream 1 | *(no-op)* |
| `A` | Show-all toggle stream 0 | Show-all toggle stream 1 | *(no-op)* |
| `R` | Raw toggle stream 0 | Raw toggle stream 1 | *(no-op)* |
| `Z` | Pick zone stream 0 | Pick zone stream 1 | *(no-op)* |
| `X` | Exclusion zone stream 0 | Exclusion zone stream 1 | *(no-op)* |
| `S` | Snapshot stream 0 | Snapshot stream 1 | Snapshot stream 2 |
| `C` / ← → | Cycle camera stream 0 | Cycle camera stream 1 | *(no-op — patch has no camera)* |

### Modifier detection + key normalization (fixes R2-A2, R2-A4)

```python
key = cv2.waitKeyEx(1)
modifiers = key & 0xFF0000

# ── Determine target stream ──
if modifiers == 0x300000:         # Ctrl+Shift → stream 2
    target_idx = 2
elif modifiers == 0x100000:       # Shift → stream 1
    target_idx = 1
else:                              # no modifier → stream 0
    target_idx = 0
target = streams[target_idx]

# ── Normalize shifted letter keys ──
# Shift+F arrives as 'F' (0x46), but the handler compares against 'f' (0x66).
# Normalize: uppercase ASCII → lowercase, but ONLY when Ctrl is NOT held
# (Ctrl+letter is a control code, not a shifted letter).
if modifiers == 0x100000:                     # Shift only, no Ctrl
    if ord('A') <= (key & 0xFF) <= ord('Z'):
        key_ascii = (key & 0xFF) | 0x20       # 'F' → 'f'
    else:
        key_ascii = key & 0xFF
else:
    key_ascii = key & 0xFF

# ── Extended-key value (modifier bits stripped, NOT masked to 0xFFFF) ──
# PgUp=0x210000, PgDn=0x220000 have no bits in 0-15; masking to 0xFFFF
# would destroy them.  Strip only the two modifier bits, preserving the
# full extended keycode for dispatch.
key_ext = key & ~0x300000   # clear Shift + Ctrl bits only
```

**Key points:**
- Extended keys are dispatched on `key_ext` = the full key with only modifier bits cleared. **Never `& 0xFFFF`** — that destroys PgUp (`0x210000`) and PgDn (`0x220000`).
- ASCII keys are dispatched on `key_ascii` (already shift-normalized to lowercase).
- The handler checks extended keys first (against `key_ext`), then ASCII keys (against `key_ascii`).
- The literal `0x210000` / `0x220000` constants were inherited from v5 where they were already dead code (masked to `& 0xFF`). **Empirically verify** the actual `cv2.waitKeyEx` return values for PgUp/PgDn on the target Linux backend (and macOS) in Phase 1c before relying on these literals. Update the constants to match reality.
- On macOS, modifier bits may differ; add an OS-detect branch if testing reveals remapping.

---

## 3. Composite Window Layout

Single OpenCV window `"Annotation Stream v6"` containing all views stacked vertically. The patch is shown as a 256×256 PiP inset on the wrist pane.

```
┌──────────────────────────────────────────┐
│ YOLO-World (cuda)  24.3 fps  r:3 i:12   │  ← shared top bar
│──────────────────────────────────────────│
│  ┌────────────────────────────────┐      │
│  │  OVERVIEW — ZMQ :5555         │      │  ← stream label + ZMQ address
│  │  "annotated" → camera1        │      │     so user can match lerobot config
│  │                                │      │
│  │     (640×480 frame)            │      │  ← captured at 1024×768,
│  │                                │      │     resized to 640×480 for stream
│  │  ┌─ stability bar ──────────┐  │      │
│  │  │████████████████████████  │  │      │
│  │  └──────────────────────────┘  │      │
│  │  info: 3 objects [STABLE]      │      │
│  │  bbox [234,156,389,412]        │      │
│  │  CAM [>0<  2  3]               │      │
│  └────────────────────────────────┘      │
│  ── separator ───────────────────────    │
│  ┌────────────────────────────────┐      │
│  │  WRIST — ZMQ :5556            │      │
│  │  "wrist" → camera2            │ ┌────┼──┐
│  │                                │ │TARGET│  ← 256×256 PiP inset
│  │     (640×480 raw frame)        │ │ PATCH │     appears when frozen
│  │                                │ │:5557  │     black when not frozen
│  │  proc: 2.1ms  CAM [0  >2<  3] │ └──────┘│
│  └────────────────────────────────┘      │
│──────────────────────────────────────────│
│ [no mod]:stream0  [SHIFT]:stream1  [CTRL+SHIFT]:stream2  │
└──────────────────────────────────────────┘
```

**Why composite:** OpenCV `waitKey` only reads from the focused window. A single composited frame guarantees all keystrokes reach the annotator.

**Per-stream timing display:** Since all streams tick in one sequential loop, there is one wall-clock FPS. Per-stream panes show **processing latency** (e.g. `proc: 2.1ms`) instead of independent FPS, which is honest about single-threaded execution.

**Canvas construction:** Panes are letterboxed/padded to a common width (max pane width) before vertical stacking. `build_composite` pre-allocates the canvas once and slice-assigns into it each frame (avoids ~1.8 MB allocation per iteration).

---

## 4. Architecture

### 4.1 Unified stream — one class, three modes

Every stream is the same `StreamState` object. The `mode` field determines which code path runs. No inheritance, no polymorphism. Just an `if/elif` dispatch on `cfg.mode`.

| Mode | Camera? | YOLO? | Buffer? | HUD level | Key handling |
|:---|:---|:---|:---|:---|:---|
| `detect` | ✅ | ✅ | ✅ | Full | Full |
| `passthrough` | ✅ | ❌ | ❌ | Minimal | Passthrough toggle, snapshot, camera cycle |
| `patch` | ❌ (derived) | ❌ | ❌ | Bare (label only) | Snapshot only |

### 4.2 StreamConfig

```python
@dataclass
class StreamConfig:
    id: int                          # 0, 1, 2, ... (position in streams list, modifier-targetable)
    name: str                        # "overview", "wrist", "target_patch"
    mode: str                        # "detect" | "passthrough" | "patch"
    zmq_port: int                    # e.g. 5555, 5556, 5557
    zmq_camera_name: str             # "annotated", "wrist", "target_patch"
    zmq_width: int                   # output width for ZMQ (e.g. 640, 256)
    zmq_height: int                  # output height for ZMQ (e.g. 480, 256)
                                     # ^ explicit H/W avoids tuple-order ambiguity (fixes R2-A5)

    # ── Camera-backed modes (detect, passthrough) ──
    camera_idx: int | None           # /dev/video index
    capture_width: int | None        # acquisition width before downscale (e.g. 1024)
    capture_height: int | None       # acquisition height (e.g. 768)
    # NOTE: actual capture dims are read from cap.get(CAP_PROP_FRAME_WIDTH/HEIGHT) at open
    # time and stored in stream.actual_capture_w / stream.actual_capture_h.
    # Patch bbox scaling uses ACTUAL, not requested, dimensions. (fixes R1-#7)

    # ── detect mode only ──
    classes: list[str] | None        # YOLO classes (None → DEFAULT_CLASSES)
    passthrough_start: bool          # start in passthrough? (runtime toggle via P)

    # ── patch mode only ──
    source_stream: int | None        # stream id whose freeze event drives this patch
```

**Per-stream classes constraint (fixes R2-A3):** `classes` is a per-stream field for future flexibility (equal citizens). However, with one shared YOLO-World instance, all detect streams MUST use identical class lists. At startup: collect unique class lists from all detect streams. If they differ → raise `ValueError("per-stream class lists require per-stream YOLO instances (not yet implemented). Use --classes to set a common list.")` . The `--classes` CLI flag sets classes for ALL detect streams uniformly.

**Resolution convention:** All resolution fields are stored as explicit `width`/`height` integers, never as tuples. `frame.shape` returns `(height, width, channels)` — always decompose explicitly: `h, w = frame.shape[:2]`. Black frames are constructed as `np.zeros((height, width, 3), dtype=np.uint8)`.

### 4.3 StreamState — unified

```python
@dataclass
class StreamState:
    cfg: StreamConfig

    # ── Camera (None for patch mode) ──
    cap: cv2.VideoCapture | None
    capture_frame_hi: np.ndarray | None   # latest hi-res frame (for patch cropping)
    actual_capture_w: int | None          # from cap.get(CAP_PROP_FRAME_WIDTH)
    actual_capture_h: int | None          # from cap.get(CAP_PROP_FRAME_HEIGHT)

    # ── ZMQ ──
    zmq_ctx: zmq.Context
    zmq_sock: zmq.Socket
    last_publish_monotonic: float         # timestamp of last published frame (for patch inheritance)

    # ── patch mode: cached snapshot ──
    patch_cache: np.ndarray | None        # snapshot at freeze moment, replayed until unfreeze

    # ── detect-mode runtime state ──
    buf: DetectionBuffer | None
    frozen: bool
    frozen_bbox: tuple | None
    frozen_label: str
    frozen_conf: float
    passthrough_active: bool              # runtime toggle (P key) on detect streams
    show_raw: bool
    show_all: bool
    selected_det_idx: int | None
    prev_sel_idx: int | None
    pick_zone: tuple | None
    excl_zone: tuple | None

    # ── Per-stream timing (processing latency, not independent FPS) ──
    frame_count: int
    proc_ms: float                        # last frame processing time in ms
    buf_read: deque
    buf_infer: deque
    buf_total: deque
    avg_read: float
    avg_infer: float
    avg_total: float
    info_str: str
    display: np.ndarray | None            # last rendered frame for compositing
```

### 4.4 Mode dispatch

```python
def process_stream(stream, streams, shared_model):
    """Process one stream for one iteration. Mode dispatch is explicit."""

    if stream.cfg.mode == "detect":
        # ── Acquire ──
        frame_hi = stream.cap.read()
        stream.capture_frame_hi = frame_hi
        h, w = frame_hi.shape[:2]
        if (w, h) != (stream.cfg.zmq_width, stream.cfg.zmq_height):
            frame = cv2.resize(frame_hi, (stream.cfg.zmq_width, stream.cfg.zmq_height))
        else:
            frame = frame_hi

        # ── Detect ──
        if stream.passthrough_active:
            output = frame
        elif stream.frozen:
            output = draw_annotated_stream(frame, stream.frozen_bbox, stream.buf, frozen=True)
        else:
            detections = raw_detect(frame, shared_model, stream.pick_zone, stream.excl_zone)
            best_match = resolve_selection(detections, stream)
            stream.buf.update(best_match)
            output = draw_annotated_stream(frame, stream.buf.median(), stream.buf, frozen=False)

        ts = time.monotonic()
        stream.zmq_publish(output, timestamp=ts)
        stream.last_publish_monotonic = ts
        stream.display = draw_detect_hud(frame, stream, detections)

    elif stream.cfg.mode == "passthrough":
        frame = stream.cap.read()
        ts = time.monotonic()
        stream.zmq_publish(frame, timestamp=ts)
        stream.last_publish_monotonic = ts
        stream.display = draw_passthrough_hud(frame, stream)

    elif stream.cfg.mode == "patch":
        source = streams[stream.cfg.source_stream]
        if source.frozen and source.frozen_bbox is not None:
            # ── On freeze: snapshot once into cache ──
            if source.capture_frame_hi is not None and stream.patch_cache is None:
                # scale bbox from stream coords → actual capture coords
                sx = source.actual_capture_w / source.cfg.zmq_width
                sy = source.actual_capture_h / source.cfg.zmq_height
                bx1, by1, bx2, by2 = source.frozen_bbox
                cx1 = int(bx1 * sx); cy1 = int(by1 * sy)
                cx2 = int(bx2 * sx); cy2 = int(by2 * sy)
                crop = source.capture_frame_hi[cy1:cy2, cx1:cx2]
                stream.patch_cache = cv2.resize(crop, (stream.cfg.zmq_width, stream.cfg.zmq_height))
            patch = stream.patch_cache if stream.patch_cache is not None else \
                    np.zeros((stream.cfg.zmq_height, stream.cfg.zmq_width, 3), dtype=np.uint8)
        else:
            # ── Unfrozen: clear cache, publish black ──
            stream.patch_cache = None
            patch = np.zeros((stream.cfg.zmq_height, stream.cfg.zmq_width, 3), dtype=np.uint8)

        # Inherit source's timestamp so camera3 aligns with camera2/camera1
        ts = source.last_publish_monotonic if source.last_publish_monotonic else time.monotonic()
        stream.zmq_publish(patch, timestamp=ts)
        stream.last_publish_monotonic = ts
        stream.display = patch
```

Key fixes from review:
- **Patch is true static template (fixes R2-A1, R1-#1):** snapshot cached at freeze moment via `patch_cache`. Replayed every frame until unfreeze, then cleared to None and black published. No live-pixel-at-fixed-bbox, no flicker.
- **H/W explicit (fixes R2-A5):** all comparisons use explicit width/height, not tuple ordering.
- **Timestamp inheritance (fixes R1-#5, R2-C6):** `zmq_publish` accepts optional `timestamp`; patch passes source's `last_publish_monotonic`. Black frames also inherit source timestamp (no jitter between "inherited" and "now"). `zmq_publish(sock, name, frame, quality, timestamp=None)` — if timestamp is None, uses `time.monotonic()`.
- **Actual capture resolution (fixes R1-#7):** bbox scaling uses `actual_capture_w/h` from `cap.get()`, not requested config values.

### 4.5 Main loop

```python
streams = build_streams(args)        # list of StreamState, any length
shared_model = load_yolo(...)         # used by all detect-mode streams

# Validate: all detect streams must share the same classes (single YOLO instance)
detect_classes = [tuple(s.cfg.classes) for s in streams if s.cfg.mode == "detect" and s.cfg.classes]
if len(set(detect_classes)) > 1:
    raise ValueError("Per-stream class lists require per-stream YOLO instances (not yet implemented). "
                     "Use --classes to set a common class list for all detect streams.")

# ── Stream ordering: patch streams must appear after their source (fixes R2-B2) ──
for s in streams:
    if s.cfg.mode == "patch" and s.cfg.source_stream is not None:
        if streams.index(s) <= s.cfg.source_stream:
            raise ValueError(f"Patch stream '{s.cfg.name}' (idx {s.cfg.id}) must appear after "
                             f"its source stream (idx {s.cfg.source_stream}).")

while True:
    for stream in streams:
        t0 = time.perf_counter()
        process_stream(stream, streams, shared_model)
        stream.proc_ms = (time.perf_counter() - t0) * 1000

    composite = build_composite(streams)
    cv2.imshow("Annotation Stream v6", composite)

    key = cv2.waitKeyEx(1)
    modifiers = key & 0xFF0000

    # ── Target selection ──
    if modifiers == 0x300000:         # Ctrl+Shift → stream 2
        target = streams[2]
    elif modifiers == 0x100000:       # Shift → stream 1
        target = streams[1]
    else:
        target = streams[0]

    # ── Normalize shifted letters ──
    if modifiers == 0x100000:
        if ord('A') <= (key & 0xFF) <= ord('Z'):
            key_ascii = (key & 0xFF) | 0x20
        else:
            key_ascii = key & 0xFF
    else:
        key_ascii = key & 0xFF

    # Extended key with only modifier bits stripped (NOT & 0xFFFF)
    key_ext = key & ~0x300000

    dispatch_key(key_ext, key_ascii, target, streams)
    # ^ extended keys (PgUp, arrows) checked via key_ext first,
    #   ASCII keys checked via key_ascii (already normalized for Shift)
```

### 4.6 Threading model

Single-threaded sequential execution. All streams tick at the slowest stream's rate. In the default config (1 detect + 1 passthrough + 1 patch):

| Stream | Work per iteration | Approx. time |
|:---|:---|:---|
| detect (overview) | cap.read + YOLO inference + draw + ZMQ publish | ~35 ms |
| passthrough (wrist) | cap.read + ZMQ publish | ~2 ms |
| patch | crop/resize + ZMQ publish | ~1 ms |

Effective FPS ≈ 1000 / (35 + 2 + 1) ≈ 26 fps. The passthrough stream's effective rate is bottlenecked by the detect stream — it publishes at ~26 fps, not camera-native 30 fps. This is acceptable.

**Constraint:** 1 or fewer detect streams is recommended. Adding a 2nd detect stream doubles YOLO time → ~13 fps.

### 4.7 Target patch — design rationale

The `patch` mode implements a **true static template** (fixes R2-A1, confirmed by user):

1. At freeze moment (`F` on the source detect stream): crop source's hi-res frame at the frozen bbox → resize → store in `stream.patch_cache`. Publish cached frame.
2. Every subsequent frame while frozen: replay `patch_cache`. No live cropping, no fixed-window-onto-changing-scene.
3. On unfreeze: clear `patch_cache` to None, publish black frame.

**What SmolVLA sees:**
- Before freeze: `camera3` = black frame
- After freeze: `camera3` = static high-res crop of the locked target (same image every frame, stable template)
- After unfreeze: `camera3` = black frame (back to baseline)
- During a single episode, camera3 has a bimodal distribution (black → crop → black). The model learns "crop present = attend to this object in camera2." Operator should unfreeze promptly after grasp so the slot doesn't carry a stale template through placement.

**Data-collection protocol note:** To maximize signal quality, the operator should freeze just before the grasp approach, and unfreeze after the object is released. The patch being a stable (non-flickering) template is critical — if it changed every frame, the model would learn to ignore it as noise.

### 4.8 Zone ROI in composite window (fixes R2-B1)

`cv2.selectROI` is modal and opens its own window. In the composite world, pressing `Z` or `X`:

1. Pops a temporary window `"Draw Zone — <stream.name>"` showing the targeted stream's current display frame
2. Operator drags the ROI as before (SPACE/ENTER to confirm, ESC to cancel)
3. The ROI is stored in `stream.pick_zone` / `stream.excl_zone`
4. The temporary window is destroyed, compositing resumes

This preserves the existing interaction model — the operator draws on the stream they targeted, not on the composite.

### 4.9 Shared model + per-stream classes

Single YOLO-World instance shared across all `detect`-mode streams. Per-stream `classes` is a config field (equal citizens, future-proof for per-stream YOLO instances), but at startup we validate all detect streams have identical class lists. Mismatch → clear error. Default: all detect streams use `DEFAULT_CLASSES` or the value of `--classes`.

### 4.10 LeRobot / SmolVLA integration

SmolVLA's trained config already defines `camera3` with shape `[3, 256, 256]`. The vision encoder (SigLIP trunk) processes all camera slots through the same frozen weights; cross-attention downstream learns what to attend to. No encoder unfreezing is needed — the model already has the capacity to use camera3, it just never received meaningful data there.

**New rename map:**
```json
"rename_map": {
    "observation.images.annotated":    "observation.images.camera1",
    "observation.images.wrist":        "observation.images.camera2",
    "observation.images.target_patch": "observation.images.camera3"
}
```
`empty_cameras` = 0. All 3 slots populated.

**LeRobot robot config:**
```python
cameras={
    "annotated":    ZMQCameraConfig(type="zmq", server_address="localhost", port=5555, camera_name="annotated", width=640, height=480, color_mode="rgb"),
    "wrist":        ZMQCameraConfig(type="zmq", server_address="localhost", port=5556, camera_name="wrist", width=640, height=480, color_mode="rgb"),
    "target_patch": ZMQCameraConfig(type="zmq", server_address="localhost", port=5557, camera_name="target_patch", width=256, height=256, color_mode="rgb"),
}
```

---

## 5. Per-Stream Data Flow

### detect mode
```
cap.read() → hi-res frame stored in capture_frame_hi
    │
    ▼
resize → zmq_width×zmq_height  [if capture dims ≠ stream dims; uses actual cap dims]
    │
    ▼
[passthrough_active?] ──yes──▶ publish raw → minimal HUD
    │ no
    ▼
[frozen?] ──yes──▶ use frozen_bbox → draw bbox → publish → HUD (frozen overlay)
    │ no
    ▼
raw_detect(frame, shared_model, pick_zone, excl_zone)
    │
    ▼
Manual selection override? → buf.hard_reset() if selection changed
    │
    ▼
DetectionBuffer.update(best_match)
    │
    ▼
draw_annotated_stream() → ZMQ publish (with timestamp)
draw_target_hud() + draw_stability_bar() + draw_zone_overlays() + info
    │
    ▼
→ stream.display
```

### passthrough mode
```
cap.read() → frame
    │
    ▼
ZMQ publish raw frame (no overlay, with timestamp)
    │
    ▼
draw minimal HUD: proc time, camera indicator, ZMQ address
    │
    ▼
→ stream.display
```

### patch mode
```
source = streams[cfg.source_stream]
    │
    ▼
source.frozen?
    │
    ├── YES ──▶ patch_cache set? ──▶ replay patch_cache
    │           patch_cache None? ──▶ crop source.capture_frame_hi at source.frozen_bbox
    │                                 (using actual capture dims for scaling)
    │                                 → resize → store in patch_cache → publish
    │
    └── NO ───▶ patch_cache = None → publish black frame
    │
    ▼
ZMQ publish (timestamp = source.last_publish_monotonic)
    │
    ▼
→ stream.display (bare frame, no HUD)
```

---

## 6. Camera Assignment

Camera cycling only applies to camera-backed streams. A camera can only be assigned to one stream at a time. Cycling re-opens the target stream's `cap` (releases old, opens new) — does not affect other streams' caps (fixes R1-#14).

| Key | Action |
|:---|:---|
| `C` / `←` / `→` | Cycle camera on stream 0 |
| `Shift+C` / `Shift+←` / `Shift+→` | Cycle camera on stream 1 |

Cycling skips indices already assigned to another stream.

---

## 7. State Persistence

Single file `annotate_stream_state.json`. On first load of an old flat-format state, a backup `annotate_stream_state.v5.bak` is written before migration (fixes R1-#8).

```json
{
  "version": 6,
  "streams": {
    "overview": {
      "camera": 0,
      "capture_width": 1024,
      "capture_height": 768,
      "classes": ["small object", "cloth toy"],
      "pick_zone": null,
      "excl_zone": null,
      "frozen": false,
      "frozen_bbox": null,
      "frozen_label": ""
    },
    "wrist": {
      "camera": 2,
      "capture_width": 640,
      "capture_height": 480
    }
  }
}
```

**Patch stream on restore:** If the source stream is restored with `frozen=True` and a saved `frozen_bbox`, the patch stream will snapshot the crop on the first iteration (since `patch_cache` is None and source is frozen) and begin publishing immediately. If the source is restored unfrozen, the patch starts black (fixes R1-#9).

**`selected_det_idx` on restore:** Always reset to None (fixes R1-#12).

---

## 8. CLI Changes

### Multi-stream arguments

| Argument | Default | Description |
|:---|:---|:---|
| `--modes` | `None` (v5 single-stream compat) | `"detect,passthrough,patch"` — explicitly activates multi-stream |
| `--cameras` | `None` | `"0,2,"` — comma-separated indices (empty for patch) |
| `--zmq-ports` | `"5555,5556,5557"` | ZMQ PUB ports |
| `--zmq-names` | `"annotated,wrist,target_patch"` | ZMQ camera_names |
| `--zmq-res` | `"640x480,640x480,256x256"` | ZMQ output WxH per stream |
| `--capture-res` | `"1024x768,640x480,"` | Capture WxH per stream (empty for patch) |
| `--patch-sources` | `",,0"` | Source stream index per stream |

### Shared arguments (apply across streams)

| Argument | Default | Description |
|:---|:---|:---|
| `--classes` | `DEFAULT_CLASSES` | YOLO classes (applied to ALL detect-mode streams) |
| `--device` | `"auto"` | Torch device |
| `--no-zmq` | `False` | Disable ALL ZMQ publishers |
| `--fresh` | `False` | Skip loading saved state |

### v5 backward-compat flags (preserved exactly)

`--camera`, `--zmq-port`, `--zmq-name`, `--passthrough`, `--show-all`, `--list`, `--device` — all work unchanged when `--modes` is not provided. `--no-zmq` and `--fresh` are promoted to shared arguments (apply to all streams in multi-stream mode).

### Backward compatibility

**`--modes` is the gate.** If `--modes` is NOT provided, the script runs in v5 single-stream mode regardless of any other argument. `python annotate_stream.py` = v5 behavior, auto-probe one camera, single detect stream. `python annotate_stream.py --camera 2` = v5, camera 2.

To activate multi-stream mode, `--modes` MUST be passed explicitly: `python annotate_stream.py --modes detect,passthrough,patch`. This is a deliberate non-breaking change (fixes R2-B5).

### Examples

```bash
# v5 backward compat (identical to current behavior)
python annotate_stream.py
python annotate_stream.py --camera 0 --passthrough

# Default 3-stream setup
python annotate_stream.py --modes detect,passthrough,patch

# Custom 3-stream: overview passthrough (no YOLO), wrist detect (YOLO on wrist)
python annotate_stream.py --modes passthrough,detect --cameras 0,2

# Two detect streams + one patch
python annotate_stream.py --modes detect,detect,patch --cameras 0,2, --patch-sources ,,0 --classes "cup,bottle"
```

---

## 9. Implementation Phases

### Phase 1a: Extract StreamState (structural, no behavior change)
- Create `StreamConfig` and `StreamState` dataclasses with `mode` field
- Move all per-stream variables from `main()` into a single `StreamState` instance
- Extract `process_stream(stream, streams, shared_model)` with mode dispatch
- Wrap in `streams = [stream_state]` list, loop over `streams`
- Fix H/W: replace tuple resolutions with explicit width/height; audit all `.shape` comparisons
- Delete dead `_ChallengerBuffer` class (fixes R1-#15)
- Bump version: docstring, imshow window → "Annotation Stream v6" (fixes R1-#20)
- Single stream, identical behavior — **smoke-test before proceeding**

### Phase 1b: Composite frame assembly
- Build composite from N stream displays stacked vertically
- Pre-allocate canvas, slice-assign each frame (fixes R2-C3)
- Letterbox panes to common width (fixes R2-B4)
- Shared top bar (device, FPS, aggregate timing)
- Stream label showing mode + ZMQ address + SmolVLA slot per pane
- Per-stream pane shows `proc: Xms` instead of independent FPS (fixes R2-B3)
- Shared hint bar with 3-tier modifier indicator

### Phase 1c: Modifier key dispatch (3-tier)
- Switch `cv2.waitKey(1)` → `cv2.waitKeyEx(1)`
- 3-tier routing: none → stream 0, Shift → stream 1, Ctrl+Shift → stream 2
- Normalize Shift+letter to lowercase (fixes R2-A2)
- Extended keys dispatched as `key & ~0x300000`, never `& 0xFFFF` (fixes R2-A4)
- **Empirically verify:** print `cv2.waitKeyEx()` return values for PgUp, PgDn, arrows on the target Linux backend (and macOS). The `0x210000`/`0x220000` constants were dead code in v5 (masked to `& 0xFF`); they may be incorrect. Update dispatch constants to match reality.
- Per-mode key handling: `detect` full set, `passthrough` subset, `patch` snapshot only
- Test on macOS; add OS-detect branch if modifier bits differ from Linux

### Phase 1d: Multi-stream CLI + passthrough mode
- `--modes` gate: absent → v5 single-stream compat; present → multi-stream
- Parse comma-separated config arrays
- `passthrough` mode: camera capture → raw ZMQ → minimal HUD
- Per-stream camera cycling (cap reopen isolated per stream)
- Multi-stream state persistence with v5 backup migration
- Audit all v5 argparse flags against new CLI (fixes R2-C1)
- Passthrough: if `zmq_resolution` is None, publish at native capture resolution (fixes R1-#18)

### Phase 1e: Patch mode (static template)
- `patch` mode: references `source_stream`, crops from source's `capture_frame_hi`
- **Static template:** snapshot at freeze moment → `patch_cache`; replay until unfreeze; clear + black on unfreeze (fixes R2-A1)
- Bbox scaling uses `actual_capture_w/h` from `cap.get()`, not requested config (fixes R1-#7)
- Timestamp inheritance from source stream (fixes R1-#5, R2-C6)
- Patch goes black on source camera failure → `capture_frame_hi` None guard (fixes R2-C2)
- PiP compositing on configured pane
- Stream ordering validation: patch must appear after its source (fixes R2-B2)
- Zone ROI: `cv2.selectROI` on popped-out temp window showing targeted stream's frame (fixes R2-B1)
- Preserve `MAX_BBOX_AREA_RATIO` guard + comments in refactored detect path (fixes R1-#19)

### Phase 1f: Documentation + testing
- Update `README.md`: Quick Start, ZMQ Wire Format, Controls table, v6 changes (fixes R1-#16)
- Update `AGENTS.md`: bump freshness dates (fixes R1-#16)
- Add `examples/record_command_3cam.sh` with 3-camera ZMQ variant (fixes R1-#16)
- Update `examples/AGENTS.md` (fixes R1-#16)
- Add `smoke_test.py`: open annotator with a video file or static image, confirm ZMQ subscribers on each port, snapshot composite for visual inspection (fixes R1-#17)

### Phase 2: DAGGER data collection with 3 streams
- Record episodes with all 3 ZMQ cameras active
- Train with `rename_map` mapping `target_patch→camera3`, `empty_cameras=0`
- Document bimodal camera3 distribution in data-collection runbook

---

## 10. Files Modified

| File | Change |
|:---|:---|
| `annotate_stream.py` | Full refactor — single-stream v5 → multi-stream v6 |
| `annotate_stream_state.json` | Format change — flat dict → `{"version": 6, "streams": {...}}` |
| `annotate_stream_state.v5.bak` | Auto-created on first migration |
| `README.md` | Rewrite Quick Start, Controls table, ZMQ Wire Format for v6 |
| `AGENTS.md` | Bump freshness dates |
| `examples/record_command_3cam.sh` | New — 3-camera ZMQ recording variant |
| `examples/AGENTS.md` | Refresh |
| `smoke_test.py` | New — basic smoke test |

---

## 11. Resolved Design Questions & Reviewer Findings

| ID | Source | Issue | Resolution |
|:---|:---|:---|:---|
| Q1 | Plan | Modifier detection | `cv2.waitKeyEx()` — none→S0, Shift→S1, Ctrl+Shift→S2. Shift+letter normalized to lowercase. Extended keys preserved as `key_raw`. |
| Q2 | Plan | Window focus | Composite single window |
| Q3 | Plan | Camera assignment | Arrow keys + `C` cycle per-stream camera |
| Q4 | Plan | LeRobot multi-ZMQ | Confirmed — N ZMQ cameras natively supported |
| Q5 | Plan | Resolution | Overview: 1024×768→640×480. Wrist: 640×480. Patch: 256×256. Explicit W/H fields. |
| Q6 | Plan | Model threading | Single-threaded sequential. 1 detect stream recommended. |
| Q7 | Plan | Stream architecture | Unified — one class, mode-dispatch, equal citizens |
| R1-#1 | reviewer1 | Patch semantics | Resolved by R2-A1 static template fix |
| R1-#2 | reviewer1 | macOS modifier bits | Addressed by key normalization; OS-detect branch if needed |
| R1-#3 | reviewer1 | SmolVLA encoder frozen | Confirmed: frozen SigLIP trunk processes all slots identically; cross-attention learns what to use. No encoder unfreezing needed. |
| R1-#4 | reviewer1 | Frame rate coupling | Documented in §4.6 threading model |
| R1-#5 | reviewer1 | ZMQ timestamp drift | Patch inherits source `last_publish_monotonic` |
| R1-#6 | reviewer1 | Mode vs passthrough_active naming | Matrix added in §1 |
| R1-#7 | reviewer1 | Capture resolution mismatch | Use `cap.get(CAP_PROP_FRAME_WIDTH/HEIGHT)` for actual dims |
| R1-#8 | reviewer1 | State migration rollback | Backup `.v5.bak` written before migration |
| R1-#9 | reviewer1 | Patch restore behavior | Explicit: snapshots on first iteration if source restored frozen |
| R1-#10 | reviewer1 | Backward compat ambiguity | `--modes` gate: absent = v5, present = multi-stream |
| R1-#11 | reviewer1 | Snapshot semantics | `S` saves `stream.display` (per-stream pane), not composite |
| R1-#12 | reviewer1 | selected_det_idx restore | Reset to None on load |
| R1-#13 | reviewer1 | Zones in passthrough | Zones are detect-mode only; HUD overlays not enforced in passthrough |
| R1-#14 | reviewer1 | Camera cycling isolation | Per-stream cap reopen, doesn't affect other streams |
| R1-#15 | reviewer1 | Dead ChallengerBuffer | Deleted in Phase 1a |
| R1-#16 | reviewer1 | Docs update | README, AGENTS.md, examples in Phase 1f |
| R1-#17 | reviewer1 | No test plan | smoke_test.py in Phase 1f |
| R1-#18 | reviewer1 | Passthrough zmq_resolution | zmq_resolution=None → publish at capture res |
| R1-#19 | reviewer1 | MAX_BBOX_AREA_RATIO | Preserved in refactored detect path |
| R1-#20 | reviewer1 | Version bump | v5 → v6 in docstring, imshow window, README |
| R2-A1 | reviewer2 | Patch = live pixels at fixed bbox | **Fixed.** True static template: snapshot→cache→replay→clear. |
| R2-A2 | reviewer2 | Shift+letter silently dropped | **Fixed.** Lowercase normalization for Shift-modified letters. |
| R2-A3 | reviewer2 | Per-stream classes + shared model | **Fixed.** Per-stream config kept (equal citizens); startup validation enforces identical classes; clear error if violated. |
| R2-A4 | reviewer2 | PgUp/PgDn break under masking | **Fixed.** Extended keys dispatched as `key_ext = key & ~0x300000` (modifier bits only stripped, NOT `& 0xFFFF` which destroys `0x21xxxx` codes). Empirically verify real PgUp/PgDn values from `waitKeyEx` on Linux/macOS in Phase 1c. |
| R2-A5 | reviewer2 | H/W tuple order bugs | **Fixed.** Explicit width/height fields; all frame ops decompose `.shape` explicitly. |
| R2-B1 | reviewer2 | Zone ROI in composite | **Fixed.** Popped temp window for `selectROI` on targeted stream. |
| R2-B2 | reviewer2 | Patch ordering dependency | **Fixed.** Startup validation: patch streams must appear after source. |
| R2-B3 | reviewer2 | Per-stream FPS dishonest | **Fixed.** Per-stream panes show `proc_ms` latency, not independent FPS. |
| R2-B4 | reviewer2 | Non-uniform pane widths | **Fixed.** Letterbox to common width in `build_composite`. |
| R2-B5 | reviewer2 | Default modes vs compat | **Fixed.** `--modes` default=None → v5 compat. Multi-stream requires explicit `--modes`. |
| R2-C1 | reviewer2 | Unported v5 flags | Audited. `--no-zmq` and `--fresh` promoted to shared; all v5 flags preserved in compat mode. |
| R2-C2 | reviewer2 | Camera failure + patch | Patch goes black if source `capture_frame_hi` is None. |
| R2-C3 | reviewer2 | Canvas allocation | Pre-allocate once, slice-assign each frame. |
| R2-C4 | reviewer2 | Bimodal camera3 distribution | Noted in §4.7 and data-collection protocol. |
| R2-C5 | reviewer2 | LeRobot config shape | Config includes `type`, `server_address`, `width`, `height`, `color_mode` fields. |
| R2-C6 | reviewer2 | Black-frame timestamp | Patch black frames inherit source timestamp (not `time.monotonic()`). |
