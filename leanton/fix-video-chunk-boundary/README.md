# fix-video-chunk-boundary

**Target:** `src/lerobot/datasets/` (merge tool + dataset reader)
**Status:** `not-implemented` (issue filed, fix pending)
**GitHub:** [#3883](https://github.com/huggingface/lerobot/issues/3883)
**Diff basis:** N/A (no code fix yet — issue stage)

## What

`lerobot-edit-dataset merge` copies video files from source datasets but doesn't reconcile `chunks_size` with actual file frame counts, causing `IndexError` at read time near chunk boundaries.

## Why

The merge tool inherits `chunks_size` from the first source dataset or defaults to 1000, but the copied video files retain their original frame counts (which may not divide evenly by the merge's `chunks_size`). The dataset reader computes file boundaries using `chunks_size` and overshoots into frames that don't exist.

## Validate

Reproduced on a merge of 21 datasets with mixed 2-camera and 3-camera footage — `chunks_size: 1000` but video files with 11,724 frames. Frame 11,724 crashes with `IndexError: Invalid frame index=11724 for streamIndex=0; must be less than 11724`.
