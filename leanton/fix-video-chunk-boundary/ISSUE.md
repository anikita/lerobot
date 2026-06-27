# `lerobot-edit-dataset merge` produces video chunk boundaries that don't match actual video file lengths

## Motivation

When merging multiple LeRobot datasets with `lerobot-edit-dataset merge`, the resulting dataset has a `chunks_size` in `meta/info.json` that may not reflect the actual frame counts of the copied video files. This causes `IndexError` at read time when the dataset reader attempts to access a frame near a chunk boundary — the reader computes the target video file using `chunks_size`, but the file has fewer frames than the math expects.

## Steps to reproduce

1. Create two datasets with different video file lengths (e.g., one with 2 cameras, another with 3 cameras, different frame counts per round).
2. Merge them: `lerobot-edit-dataset --operation.type merge --operation.repo_ids [dataset_a, dataset_b]`
3. Load the merged dataset and iterate through frames sequentially. At some point, the reader throws:

```
IndexError: Invalid frame index=<N> for streamIndex=0; must be less than <N>
```

Where `<N>` is the actual frame count of one of the copied video files — one less than the boundary the reader computed from `chunks_size`.

## Root cause

The merge operation copies video files from source datasets into the merged dataset's `videos/` directory, but:

1. `chunks_size` in the merged `meta/info.json` is inherited from the first source dataset (or defaults to 1000) and doesn't reflect the actual frame distribution across copied video files.
2. The video files are concatenated across chunks (e.g., `chunk-000/file-000.mp4`, `chunk-000/file-001.mp4`, ...) but each file's frame count comes from the source dataset's chunking, which may differ from the merge's chosen `chunks_size`.
3. The dataset reader (`dataset_reader.py`) uses `chunks_size` to map a frame index to a video file and local frame index. When the computed local index exceeds the file's actual frame count, the read fails.

## Proposed solution

The merge tool should reconcile video chunk boundaries with the declared `chunks_size`. Options:

**A) Re-chunk on merge (most robust).** Re-encode video files so each chunk has exactly `chunks_size` frames (except the last). This guarantees consistency but is expensive for large datasets.

**B) Compute `chunks_size` from actual file lengths (least invasive).** After copying all video files, scan their frame counts and set `chunks_size` to the least common multiple or the per-chunk maximum. This avoids re-encoding but requires the reader to handle variable chunk sizes.

**C) Validate and warn on mismatch (minimum viable).** At merge time, check that each copied video file's frame count divides evenly by `chunks_size` (or is the last chunk). Emit a warning or error if not. This catches the problem early without fixing it automatically.

## Scope / limitations

- This affects any merge where source datasets have different video chunking from each other or from the merge's `chunks_size`.
- Read-only operations (dataset inspection, training) are affected — the dataset appears valid until frame access crosses the mismatched boundary.
- The `data/` parquet files are not affected (they are re-chunked correctly by the merge tool). Only `videos/` are affected.

## Testing

Reproduced on a merge of 21 datasets with mixed 2-camera and 3-camera footage. The merged dataset had `chunks_size: 1000` but video files with 11,724 frames. Reading past frame 11,723 from the affected file raised the `IndexError` above.
