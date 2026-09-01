# analysis/

One-off scripts behind [`TRAINING_PERFORMANCE.md`](../TRAINING_PERFORMANCE.md) and the
`dice_white_pnp_nodip` dataset rebuild. They are working scripts, not a library: paths to datasets,
runs and checkpoints are constants at the top of each file — edit them for your environment.

## Benchmarks

| script | answers |
| --- | --- |
| `bench_precision.py` | fp32 / fp32+TF32 / bf16 / bf16+compile, samples/s and VRAM at a fixed batch |
| `bench_imle_gpu.py` | throughput vs batch size (shows where the GPU saturates) |
| `bench_split.py` | how compute divides between the ResNet encoders and the generator |
| `bench_loader.py` | dataloader ceiling at N workers, and frame-cache sizing |

## Frame cache

| script | purpose |
| --- | --- |
| `build_frame_cache.py` | pre-decodes a dataset's videos into a uint8 memmap for `--dataset.video_backend=memmap`, then verifies a sample byte-for-byte against the video decoder |

## Dataset surgery (`dice_white_pnp_100` → `dice_white_pnp_nodip`)

Run in this order:

| script | purpose |
| --- | --- |
| `dip_analysis.py` | locates the pre-grasp dive-and-return in every episode and writes `dip_rows.json` |
| `splice_feasibility.py` | checks whether the excision splices cleanly, and rules out the alternative cut |
| `rotation_phase.py` | shows the wrist-yaw alignment happens during the hover hold — the reason that cut is the wrong one |
| `cut_points.py` | picks the per-episode frame range to remove, minimising the splice discontinuity |
| `build_nodip.py` | rebuilds data, videos and meta (`data` / `video` / `meta` stages) |
| `validate_nodip.py` | proves new frame *i* is source frame `keep[i]`, pixel-level, including across the seam |
| `make_j7_2cam.py` | derives the 7-dim / 2-camera training variant (hard-links videos) |
| `publish_nodip.py` | uploads the result to the Hub |

## Figures

`aggregate_plot.py` (all 99 episodes' z-profiles aligned on the first touch-down),
`render_dip_clip.py` and `render_new_grasp.py` (annotated before/after clips of the grasp).
