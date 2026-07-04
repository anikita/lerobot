# ZMQ metadata passthrough

**Target:** `src/lerobot/cameras/zmq/camera_zmq.py`, `src/lerobot/robots/so_follower/so_follower.py`, `src/lerobot/rollout/context.py`, `src/lerobot/utils/feature_utils.py`
**Status:** `active`
**GitHub:** Not filed (validating on hardware first)
**Diff basis:** `origin/main` @ `6f0ba4be` (clean upstream, 2026-07)

## What

Four-layer change so ZMQ camera metadata (e.g. calibrated target coordinates) flows from the annotator through the robot observation to the dataset as a separate column:

1. **ZMQCamera** exposes non-image JSON fields as `latest_metadata` (dict), with a thread-safe `read_metadata()` accessor.
2. **SOFollower** declares `target_coord_mm: (2,)` as a sensor vector feature (non-camera tuple) in `observation_features`, populates it from the first ZMQ camera that carries `target_coord_mm` in its metadata.
3. **Rollout context** splits tuple features by arity — 3-element tuples are cameras, others are sensor vectors. Sensor vectors bypass `hw_to_dataset_features` and are injected directly as individual dataset columns (`observation.target_coord_mm`, dtype int64, shape (2,)).
4. **build_dataset_frame** handles individual features (`names: None`) — previously only bundled state vectors and images were handled.

Key design decision: `target_coord_mm` is a **separate dataset column**, NOT bundled into `observation.state`. This keeps the state vector at 6D (motors only), so the policy's normalizer stats remain valid.

## Why

ZMQCamera's `_read_from_hardware()` parses the full JSON message but only extracts `images` — all other keys (`target_coord_mm`, `status`, etc.) are silently discarded. Downstream callers (robot, dataset writer) have no way to access sensor metadata published alongside frames.

Without this patch, calibrated target coordinates from the annotator cannot reach the dataset without fragile post-hoc timestamp alignment.

## Validate

**User:**
1. Launch annotator with H ON, freeze a target object
2. Run `lerobot-record` with ZMQ cameras (`--robot.type=so101_follower`)
3. Record 1 episode, then inspect the dataset:
   ```python
   from lerobot.datasets import LeRobotDataset
   ds = LeRobotDataset("anikitakis/<name>")
   obs = ds[0]
   print(obs["observation.state"])            # 6D: motors only
   print(obs["observation.target_coord_mm"])  # [x, y] int64
   ```

**Agent:**
```bash
grep -q "latest_metadata" ~/lerobot/src/lerobot/cameras/zmq/camera_zmq.py && echo "✅ camera_zmq" || echo "MISSING"
grep -q "target_coord_mm" ~/lerobot/src/lerobot/robots/so_follower/so_follower.py && echo "✅ so_follower" || echo "MISSING"
grep -q "sensor_features_hw" ~/lerobot/src/lerobot/rollout/context.py && echo "✅ context" || echo "MISSING"
grep -q "names.*is None" ~/lerobot/src/lerobot/utils/feature_utils.py && echo "✅ feature_utils" || echo "MISSING"
```
