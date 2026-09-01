# Training Performance Guide

Everything we measured while making IMLE and ACT train faster on an 8×A100 box, written down so the
next run starts from the answers instead of rediscovering them. Every number here came from this
hardware and these policies — treat the *method* as portable and the *numbers* as ours.

Headline: the same IMLE run went from **11 h 20 m to ~3.8 h**, and ACT from 16.4 h to ~9.6 h, with no
change to what the models learn. Two of the four wins were configuration bugs, not optimizations.

---

## 1. Measure before you tune

Every LeRobot training step logs its own breakdown. Read it before touching anything:

```
step:100K ... data_s:0.051 prep_s:0.009 updt_s:0.292 step_s:0.366 smp/s:194 mem_gb:13.73
```

| field | meaning | what it tells you |
| --- | --- | --- |
| `updt_s` | forward + backward + optimizer | compute cost; compare against a synthetic benchmark to find loop overhead |
| `data_s` | time the step *waited* on the dataloader | already post-prefetch, so anything above ~0.02 s is real starvation |
| `step_s` | wall clock per step | `updt_s + data_s + prep_s` plus logging |
| `mem_gb` | torch-allocated VRAM | ours was 13.7 of 80 GB — memory headroom is not a speed signal |

Then compare against the compute ceiling: run forward+backward on synthetic tensors with no dataloader.
Ours said 332 samples/s against 175 realized — i.e. **half the GPU was going to waiting and loop
overhead**, which is where the entire speedup came from.

`nvidia-smi` utilization is a poor guide on a shared box: sample it 6–8 times over ~20 s. A card at
"0 %" may be a neighbour's data-starved job that wakes up in bursts.

## 2. The four levers, ranked by measured effect

### 2.1 A second GPU — 1.9×

LeRobot v0.6.2 has accelerate-based DDP. All GPUs on this box are NV12-NVLinked, so the allreduce for
an 87 M-param model (~350 MB) is a few milliseconds — effectively free.

```
accelerate launch --num_processes 2 --gpu_ids 4,1 lerobot-train ...
```

**`--batch_size` is per rank.** Two ranks at 64 is an effective batch of 128, which halves the optimizer
steps for a given number of epochs and changes optimization, not just speed. Either scale the LR
(√2 is the usual heuristic) or halve the per-rank batch to reproduce the single-GPU recipe exactly.
Measured cost of the safe path: per-rank 32 runs at 260 samples/s vs 338 at 64, so exactness costs ~23 %.

Two DDP flags worth setting — LeRobot defaults `find_unused_parameters=True`, which pays for an extra
autograd-graph traversal every iteration that our model never needs:

```
--accelerator.ddp.find_unused_parameters=false
--accelerator.ddp.gradient_as_bucket_view=true
```

BatchNorm note: with `use_group_norm=false` the ResNet encoders carry BN. Under DDP, buffers diverge per
rank and rank 0's are saved. That is fine — rank 0 still sees millions of samples — but be aware the
saved running stats come from half the data stream.

### 2.2 The data path — up to 1.9×, and it's the variance killer

This is the difference between a predictable run and one whose speed depends on what your colleagues
are doing. Our three observations of the *same* pipeline:

| context | `data_s` | share of step |
| --- | --- | --- |
| quiet box | 0.051 s | 14 % |
| busy box (6 neighbouring jobs) | 0.23–0.27 s | ~45 % |
| with a raw frame cache | **0.022 s** | 7 % |

Per-frame costs we measured (480×640, AV1 CRF 30, GOP 2):

| access method | cost |
| --- | --- |
| AV1 seek + decode, container reopened | 7.1 ms |
| AV1 seek + decode, container kept open | **4.0 ms** |
| PNG decode | **4.6 ms** — no better than the video |
| raw uint8 memmap | **0.1 ms** |

Conclusions that cost us time to learn:

- **Converting a video dataset to PNG buys nothing.** PNG decode is as expensive as AV1 random access at
  short GOP. Only a raw uncompressed cache is worth building.
- **GPU decode is not an option on A100.** GA100's NVDEC has no AV1 decoder (AV1 decode on Ampere landed
  only on GA10x/RTX-30), so torchcodec-CUDA cannot help. On this box torchcodec fails to load at all and
  silently falls back to pyav — check your logs for `Falling back to 'pyav'`.
- **Multi-timestep image history is the expensive case.** IMLE requests `n_obs_steps=2` × 2 cameras =
  4 random-access frames per sample; ACT with 1 obs step requests 2. This is the documented split in
  [huggingface/lerobot#1488](https://github.com/huggingface/lerobot/issues/1488), where ACT loads fine
  and Diffusion-Policy-shaped models stall.

The cache we built (`analysis/build_frame_cache.py` + `video_backend="memmap"` in `dataset_reader.py`):
a `(total_frames, H, W, 3)` uint8 memmap per camera, indexed by dataset-global frame index, 117 GB for
2 cameras × 68 k frames. It lives in page cache after the first epoch. **Verify it byte-for-byte against
the video decoder before training on it** — a silent frame-mapping bug would poison the run invisibly.

Worker counts follow from this: ~24 per rank without a cache, ~8 with one. Fewer workers also means you
stop competing with everyone else on the box for CPU.

### 2.3 Mixed precision — 1.12×, and one config trap

**`--policy.use_amp=true` does nothing during training.** It is only read at eval/inference
(`lerobot_eval.py`, `rollout/inference/sync.py`). The training loop takes precision from accelerate:

```
--accelerator.mixed_precision=bf16     # default is "no"
```

Two of our completed runs logged `'mixed_precision': 'no'` while their policy configs said
`use_amp: True`. Check the logged value, not the flag you passed.

Measured at batch 64 (IMLE, A100):

| config | samples/s | VRAM |
| --- | --- | --- |
| fp32, TF32 forced off | 112.0 | 20.8 GB |
| fp32 + TF32 (**the real default** — cuDNN TF32 is on unless disabled) | 298.7 | 20.8 GB |
| bf16 | 335.0 | 12.8 GB |
| bf16 + `torch.compile` on the U-Net | 375.6 | 8.4 GB |

Note the second row: PyTorch already enables `cudnn.allow_tf32`, so a conv-dominated model is *not*
sitting at the 112 row. The honest bf16 gain is ~12 %, not 3×. (`torch.backends.cuda.matmul.allow_tf32`
does default to off and is worth setting, but it only touches linear layers.)

### 2.4 `torch.compile` — 1.12× on top of bf16

Our IMLE port compiles the U-Net when `--policy.compile_model=true` (`modeling_imle.py`); it defaults
off. Use `compile_mode="default"` for training — `max-autotune` is an inference recommendation.

Costs to plan for: **15–20 minutes of CPU-bound Inductor codegen before the first step** on a loaded
box, and one extra recompile per distinct batch shape. LeRobot's train dataloader uses
`drop_last=False`, so the ragged last batch of each epoch triggers exactly one of those.

## 3. Precision is policy-specific — check your loss before enabling bf16

Generic advice ("bf16 is numerically safe") is not enough when the loss has a threshold in it.

RS-IMLE rejects candidate trajectories within ε = 0.03 of the target and trains on the nearest survivor.
bf16 rounding of the generator's output perturbs each candidate chunk by **L2 0.017 mean / 0.022 p99** in
normalized action space — 57 % of ε, and ~3× the smallest distances the converged model produces
(`distance_min` 0.006). Simulated consequences: 2.7 % of candidates flip their rejection decision (6.2 %
near the boundary), 45 % of rows select a different surviving candidate, and the loss shifts ~7 %.
ε-rejection is the mechanism that keeps the generator from collapsing, so we don't fuzz it.

The fix costs nothing, because of where the compute actually is:

| component | fp32 | bf16 |
| --- | --- | --- |
| 2 × ResNet18 encoders (4 passes) | 139.3 ms | **120.2 ms** |
| U-Net × 20 candidates | 63.2 ms | 65.2 ms (no gain) |

The encoders are 69 % of compute and the only part bf16 accelerates. So: **bf16 everywhere, fp32 for the
generator and the loss** (`--policy.fp32_generator=true`, on by default in this fork). The conditioning
vector still carries bf16 rounding, which is harmless — it's a network input, not part of the metric.

`torch.cdist` is already on autocast's fp32 promotion list, so the distance computation was never the
risk; the candidates entering it were.

Transferable rule: if your loss compares distances against a fixed threshold, compute the quantization
noise of your dtype in the same units as that threshold before enabling it.

## 4. What is *not* a lever

- **Bigger batches.** Measured flat from 64 to 512: 332 / 367 / 333 / 387 / 362 samples/s at
  9.3 / 17.4 / 33.5 / 49.6 / 65.7 GB. The GPU saturates at 64 because each step already pushes B × 20
  trajectories through the U-Net. Spare VRAM is not free speed; it just costs you an LR retune.
- **FSDP / sharding** at 87 M parameters.
- **Reducing `n_samples_per_condition`** (20). It would nearly halve compute, but it *is* the RS-IMLE
  algorithm, not a knob.
- **Fused AdamW** — real but tiny; a published UNet study measured ~0.6 s per epoch.

## 5. Operational gotchas

- **torchcodec** is broken in both our envs and falls back to pyav with a multi-screen traceback at
  startup. Harmless, but it means log-watching monitors that grep for `Traceback` fire on every run —
  baseline the count on first poll and alert only on increases.
- **tmux launches don't inherit your exports.** `tmux new-session` attaches to the *existing* server, so
  `export FOO` in the launching script does not reach the new session. Expand variables in the outer
  shell instead. And quote carefully: `\\"` inside a double-quoted tmux command closes the outer string
  and silently empties the argument — we lost two launches to `--dataset.episodes=""`.
- **Check which filesystem you're filling.** MLflow's `system/disk_usage_percentage` reports the root
  volume, which on our box sits at 94 % while `/mnt/data` has 1.3 TB free. Point
  `TORCHINDUCTOR_CACHE_DIR` and `TMPDIR` at the big volume.
- **Shared boxes:** sample GPU utilization repeatedly, check *who* owns the resident memory
  (`nvidia-smi --query-compute-apps` + `ps -o user=`), and prefer GPUs on one NUMA node for a DDP pair
  (`nvidia-smi topo -m`).

## 6. Reference configuration

What we run now, on the cleaned dataset:

```bash
accelerate launch --num_processes 2 --gpu_ids 4,1 lerobot-train \
  --policy.type=imle \
  --policy.compile_model=true --policy.compile_mode=default \
  --policy.fp32_generator=true \
  --accelerator.mixed_precision=bf16 \
  --accelerator.ddp.find_unused_parameters=false \
  --accelerator.ddp.gradient_as_bucket_view=true \
  --dataset.repo_id=dice_nodip_j7 --dataset.root=/path/to/dice_nodip_j7 \
  --dataset.video_backend=memmap \
  --batch_size=64 --num_workers=8 \
  --steps=45000 --save_freq=2500 --ema.enable=false \
  --mlflow.enable=true --mlflow.experiment=imle_nodip
```

## 7. Results

| | before | after |
| --- | --- | --- |
| IMLE throughput | 175 samples/s | 545 solo / ~416 sharing the box |
| IMLE `step_s` | 0.366 | 0.23–0.31 |
| IMLE `data_s` | 0.051 (0.25 busy) | 0.022 |
| IMLE VRAM | 13.7 GB | 10.0 GB |
| IMLE wall clock (85 epochs) | 11 h 20 m | **~3.8 h** |
| ACT rate | 2.2–2.5 step/s | 2.9 step/s |
| ACT wall clock (100 k steps) | 16 h 24 m | **~9.6 h** |

Reproduce any of this with the benchmark scripts in `analysis/`: `bench_precision.py` (dtype sweep),
`bench_split.py` (encoder vs generator), `bench_loader.py` (dataloader ceiling),
`build_frame_cache.py` (cache + byte-level verification).
