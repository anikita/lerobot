# Feature Request: Render all scalar dataset columns generically in lerobot-dataset-viz

## Motivation

`lerobot-dataset-viz` is the standard tool for inspecting LeRobot datasets in Rerun. Currently it only renders hardcoded scalar features: `action` dimensions, `observation.state` dimensions, `next.done`, `next.reward`, `next.success`. Any additional scalar columns in the dataset — whether computed (Q-values, metrics, custom annotations) or user-defined — are silently invisible in the viewer, even though they are present in the dataset's parquet data and `meta/info.json` feature schema.

This makes it impossible to inspect enriched datasets without modifying the viz script. Users who embed custom columns (e.g., Q-value targets for RL training, intervention flags for DAGGER data, task-progress scores) have no way to visually verify them alongside camera feeds and actions.

## Proposed Solution

After rendering the known hardcoded features (cameras, action, state, done, reward, next.success), add a generic pass that iterates over all remaining keys in the batch and logs any scalar-valued feature to Rerun as a time-series. Skip metadata columns (`index`, `timestamp`, `episode_index`, `frame_index`, `task_index`, `task`) and already-rendered keys.

**Files changed:** `src/lerobot/scripts/lerobot_dataset_viz.py` (+13 lines)

**Implementation:** See `leanton/dataset-viz-generic-scalars/dataset-viz-generic-scalars.diff`
(commit `55421a11` on `anikita/lerobot:tmp-dataset-viz-generic-scalars`)

```python
# display any additional scalar features (e.g. q_target, intervention)
known_keys = {ACTION, OBS_STATE, DONE, REWARD, "next.success"}
known_keys.update(dataset.meta.camera_keys)
for key in batch:
    if key in known_keys or key in ("index", "timestamp", "episode_index", "frame_index", "task_index", "task"):
        continue
    val = batch[key][i]
    try:
        scalar = float(val.item() if hasattr(val, 'item') else val)
        rr.log(key, rr.Scalars(scalar))
    except (TypeError, ValueError, AttributeError):
        pass  # skip non-scalar features (vectors, images)
```

## Scope / Limitations

- Only scalar features are rendered. Vector/tensor features (e.g., action chunks, embeddings) are silently skipped via the try/except.
- The patch does not change any existing behavior — known keys render exactly as before.
- Performance impact is negligible: a single pass over batch keys with float conversion, guarded by a try/except.

## Testing

Tested with `anikitakis/rollout_pick_n_place_dagger_r1_with_q` which contains two custom scalar columns (`q_target`: float32, `intervention`: bool). Both columns appeared as time-series plots in Rerun alongside camera feeds and action/state signals.
