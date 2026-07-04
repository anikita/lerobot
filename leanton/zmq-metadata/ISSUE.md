# Feature Request: ZMQCamera metadata passthrough

**Title:** ZMQCamera: expose non-image JSON fields as `latest_metadata`

## Motivation

`ZMQCamera._read_from_hardware()` parses the full ZMQ JSON message but only extracts `images` — all other keys (`target_coord_mm`, `status`, sensor readings, etc.) are silently discarded. This makes it impossible for ZMQ publishers to enrich frames with arbitrary sensor data that downstream consumers (robot, dataset writer) can access.

Use case: a visual target annotator publishes calibrated object coordinates alongside annotated frames. The robot should log these coordinates into the dataset as scalar observation features.

## Proposed solution

1. **ZMQCamera** stores extra JSON keys (everything except `timestamps` and `images`) in a new `latest_metadata` dict, updated under the existing `frame_lock`, with a `read_metadata()` public accessor. Zero breaking change — cameras without metadata return `{}`.

2. **Robot subclasses** (e.g. `SOFollower`) declare sensor features in `observation_features` and populate them from `camera.read_metadata()` in `get_observation()`.

3. **Rollout context** accepts all `float`-typed observation features in the dataset schema (currently restricted to `.pos` motor positions). This is backward-compatible — the `.pos` filter was overly restrictive and already had a TODO for broader sensor support.

## Implementation

- `src/lerobot/cameras/zmq/camera_zmq.py` — 18 lines: `latest_metadata` init, store in `_read_from_hardware()`, copy under lock in `_read_loop()`, reset in thread lifecycle, `read_metadata()` accessor
- `src/lerobot/robots/so_follower/so_follower.py` — 18 lines: `_sensor_ft` property, populate `target_x_mm`/`target_y_mm` in `get_observation()`
- `src/lerobot/rollout/context.py` — 6 lines: expand filter from `v is float and k.endswith(".pos")` to `v is float`

Diff: `leanton/zmq-metadata/zmq-metadata.diff` (156 lines, applies cleanly to `origin/main`)

## Scope / limitations

- Camera layer only exposes the metadata dict — how robots interpret specific keys is robot-specific.
- `target_x_mm`/`target_y_mm` defaults are hardcoded to `0.0` when no ZMQ camera carries coordinates. A generic robot base class would need a more configurable approach.
- Only `SOFollower` is updated. Other robot classes (Koch, OpenArm, etc.) would need similar changes to support ZMQ metadata.

## Testing

- Wire-level: `zmq_inspect.py` confirms ZMQ messages carry `target_coord_mm` and `status`.
- Integration: recording session (TBD) to verify coordinates survive through `lerobot-record` → dataset parquet.
