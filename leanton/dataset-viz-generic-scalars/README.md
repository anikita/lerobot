# Render all scalar columns generically in dataset viz

**Target:** `src/lerobot/scripts/lerobot_dataset_viz.py`
**Status:** `active`
**GitHub:** [#3880](https://github.com/huggingface/lerobot/issues/3880)
**Diff basis:** `origin/main` @ `6f0ba4be` (clean upstream, 2026-06-26)

## What

`lerobot-dataset-viz` currently only renders hardcoded scalar features (`action`, `observation.state`, `next.done`, `next.reward`, `next.success`). Any additional scalar columns in the dataset (e.g. `q_target`, `intervention`) are invisible in the Rerun viewer despite being present in the parquet data.

This patch adds a generic pass that iterates over all remaining batch keys and logs any scalar-valued feature to Rerun. Known keys (action, state, cameras, metadata) are skipped to avoid duplicate logging.

## Why

When embedding computed columns (Q-values, intervention flags, custom metrics) into a LeRobot dataset for inspection, `lerobot-dataset-viz` silently drops them. There is no way to visualize custom scalar features without modifying the viz script. The hardcoded feature list makes the tool unnecessarily restrictive — it should render whatever columns the dataset provides.

## Validate

**User:** Run `lerobot-dataset-viz` on any dataset with custom scalar columns (e.g. `anikitakis/rollout_pick_n_place_dagger_r1_with_q`). Verify that `q_target` and `intervention` appear as additional time-series plots in the Rerun viewer alongside the action/state signals.

**Agent:**
```bash
grep -q "display any additional scalar features" ~/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py && echo "✅" || echo "MISSING"
```
