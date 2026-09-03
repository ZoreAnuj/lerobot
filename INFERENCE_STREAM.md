# Inference Guide — Stream-Dataset Policies (Fanuc CRX-5iA)

How to deploy the policies trained on
[`azorematter/dice_white_pnp_stream_100`](https://huggingface.co/datasets/azorematter/dice_white_pnp_stream_100):
the exact observation/action contract, which checkpoints to load, working code, and the per-policy
failure modes we measured. **Training and inference must agree on every property in §2** — most
real-robot failures we've debugged traced to a mismatch here, not to the weights.

This supersedes [INFERENCE.md](./INFERENCE.md) for these policies. That document still applies to the
older `dice_white_pnp_100` checkpoints, which were trained on data with a timing defect (see §7).

---

## 1. What was trained, and how it scores

All policies: 100-episode `stream_100` dataset, 94 episodes for training, **6 held out**
(7, 23, 41, 58, 76, 91), reduced to two cameras and joint-space 7-dim state/action.

**All MAE numbers below are measured on the held-out episodes** — data no policy ever saw. This is the
first honest generalisation measurement in this project; earlier reported figures were scored on
training data and are not comparable.

| policy | Hub repo | best held-out MAE | bit-flip | dwell escape |
| --- | --- | --- | --- | --- |
| **RS-IMLE original** | [`imle_stream_orig`](https://huggingface.co/azorematter/imle_stream_orig) | **0.269°** | 0.29 | 0.25 |
| **ACT** | [`act_stream`](https://huggingface.co/azorematter/act_stream) | 0.516° | **0.04–0.19** | **0.68–0.75** |
| RS-IMLE + gripper (fine-tune) | [`imle_ft_stream`](https://huggingface.co/azorematter/imle_ft_stream) | see §5 | — | — |
| RS-IMLE + gripper (from scratch) | [`imle_stream_grip`](https://huggingface.co/azorematter/imle_stream_grip) | 0.738° | 0.31 | 0.27 |

Metric definitions:

- **held-out MAE** — mean absolute joint error (degrees, J1–J6) between the predicted action chunk and
  the demonstrated one, on unseen episodes. Precision.
- **bit-flip** — flip the observed gripper bit in the input and measure how much the *commanded*
  gripper channel changes. **Lower is better**: high values mean the policy is copying the input bit
  rather than deciding from vision, which deadlocks if your executor feeds its own latch back in.
- **dwell escape** — from a stationary hover, does the chunk command motion? **Higher is better**, but
  see the caveat in §6: it is meaningless for undertrained checkpoints.

## 2. Inputs and resolution — the full specification

Get this section exactly right and the policies perform as measured; deviate and they degrade in ways
that look like bad weights. Everything here was read out of the trained checkpoints and the model code,
not from memory.

### 2.1 The two required keys

| key | shape | dtype | range | meaning |
| --- | --- | --- | --- | --- |
| `observation.images.gripper` | `(B, 3, 480, 640)` | `float32` | `[0.0, 1.0]` | wrist camera, **RGB**, channel-first |
| `observation.images.cam0` | `(B, 3, 480, 640)` | `float32` | `[0.0, 1.0]` | elevated board view, **RGB**, channel-first |
| `observation.state` | `(B, 7)` | `float32` | raw units | `[J1, J2, J3, J4, J5, J6, gripper]` |

For IMLE with explicit history the image keys are `(B, 2, 3, 480, 640)` and state is `(B, 2, 7)` (§4.1).
No other keys are needed — these policies are **not** language-conditioned, so unlike SmolVLA there is
no `task` string to pass.

**State units, exactly:**

- `J1…J6` — **absolute joint angles in degrees**, as the controller reports them. Not radians, not
  deltas, not Cartesian. Approximate ranges seen in training: J1 43…93, J2 −28…21, J3 −52…19,
  J4 −19…37, J5 −86…−21, J6 −122…−16.
- `gripper` — **1.0 = OPEN, 0.0 = CLOSED**. This polarity is easy to invert by accident and the policy
  gives no obvious sign of it; a flipped bit is one of the failure modes in §6.

**Image format, exactly:**

- **RGB, not BGR.** OpenCV gives you BGR — convert it. A swapped channel order silently costs accuracy
  because the encoders were trained on ImageNet-initialised weights and dataset RGB statistics.
- **Channel-first** `(3, H, W)`, not `(H, W, 3)`.
- **`float32` scaled to `[0, 1]`** — divide `uint8` by 255. Do not pass `uint8`, and do not
  pre-apply ImageNet mean/std: normalization happens inside the pipeline (§2.4).

### 2.2 Resolution — feed native 480×640, and why it differs per policy

**Feed full 480×640 to both policies. Never resize or crop outside the model.** What happens inside
differs, which matters if you are tempted to "help" by downscaling:

| | RS-IMLE | ACT |
| --- | --- | --- |
| `resize_shape` | `[240, 320]` | **none — consumes native resolution** |
| `crop_shape` | `None` (no crop) | none |
| Internal resize | `torchvision.transforms.Resize([240,320])`, **bilinear, antialias=True** | — |
| Resolution the ResNet18 sees | **240×320** | **480×640** |
| `layer4` feature map | `512 × 8 × 10` | `512 × 15 × 20` |
| Encoder count per sample | 4 (2 cameras × 2 obs steps, separate encoder per camera) | 2 (2 cameras × 1 step) |

Consequences to respect:

- **ACT has no resize step at all.** Hand it anything other than 480×640 and its feature map changes
  shape (a 240×320 input yields 8×10 instead of 15×20), which silently changes the spatial resolution
  its transformer attends over. It may not even error — it will just be wrong.
- **IMLE will accept other sizes** because it resizes internally, but the *field of view and aspect
  ratio* must still match collection. Downscaling before you pass it means resizing twice (yours, then
  the model's bilinear pass), which throws away detail the wrist view needs for fine alignment.
- **Do not letterbox or pad.** Neither policy expects it (that is a SmolVLA behaviour).
- **Aspect ratio is fixed at 4:3.** Cropping to a different aspect changes the geometry the policy
  learned to map onto joint angles.

### 2.3 Timing and history

| | RS-IMLE | ACT |
| --- | --- | --- |
| `n_obs_steps` | **2** | **1** |
| Spacing between the two steps | **33.3 ms** (one frame at 30 Hz) | — |
| `horizon` / `chunk_size` | 16 | 30 |
| `n_action_steps` executed per replan | 8 (0.27 s) | 30 (1.0 s) |

The 30 Hz observation rate is not incidental: it is the spacing the policies saw, and IMLE's two-frame
pair encodes **velocity**. A pair of frames 100 ms apart reads as faster motion than anything in
training; a duplicated frame reads as stationary (see §4.1 and §6.2).

### 2.4 Normalization — do not do it yourself

The preprocessor pipeline is **rename → add batch dim → to device → normalize**, built from the
checkpoint's own statistics. The output pipeline is **unnormalize → to CPU**, so the actions you get
back are already in real units (degrees).

The two policies use **different normalization modes**, which is precisely why you must load the
processors from the checkpoint rather than reconstruct them:

| feature | RS-IMLE | ACT |
| --- | --- | --- |
| `VISUAL` | MEAN_STD | MEAN_STD |
| `STATE` | **MIN_MAX** → scaled to `[-1, 1]` | **MEAN_STD** → zero mean, unit variance |
| `ACTION` | **MIN_MAX** | **MEAN_STD** |

Feed raw degrees and raw `[0,1]` images; the pipeline handles the rest. Feeding pre-normalized values
double-normalizes them and the policy will behave as if the arm is somewhere it is not.

### 2.5 Image chain and camera identity

- **Match the capture pipeline.** Training frames passed through **AV1, CRF 30, yuv420p** video
  encoding, so they carry that codec's texture-level artifacts. Frames straight from the sensor are
  *cleaner* than training data, which is still a domain shift. If behaviour looks off with raw frames,
  A/B an AV1 round-trip offline before suspecting the weights.
- **Camera assignment is not interchangeable.** `gripper` must be the wrist-mounted view and `cam0` the
  elevated board view. Swapping them is catastrophic and produces confident nonsense. Verify by serial
  number, not by enumeration order, which can change across reboots.
- **Mounting and intrinsics must match collection.** These are 2D-image policies with no calibration
  input; a moved or re-lensed camera is out-of-distribution.
- **Gate on frame freshness.** A wedged feed yields a confident policy driving on a still image. Assert
  both streams advanced since the last tick, and assert the `(3, 480, 640)` shape on every frame rather
  than trusting the driver.

### 2.6 Minimum viable checklist

Before an armed run, assert all of these — each one has bitten us:

- [ ] images are **RGB**, channel-first, `float32` in `[0,1]`, shape exactly `(3, 480, 640)`
- [ ] `gripper` camera is the **wrist**, `cam0` is the **overhead**, verified by serial
- [ ] state is `[J1..J6 degrees, gripper]` with **1.0 = OPEN**
- [ ] no resizing, cropping, letterboxing or normalization applied outside the model
- [ ] observations arrive at **30 Hz**; for IMLE the two history frames are **real** and 33 ms apart
- [ ] processors loaded **from the checkpoint folder** (§3), not rebuilt
- [ ] both camera streams verified fresh this tick
- [ ] gripper command thresholded with hysteresis (close < 0.4, open > 0.6), not a single 0.5 cut

## 3. Loading a checkpoint — read this first

Download the **whole** `pretrained_model` folder. The normalization statistics live in
`policy_pre/postprocessor_step_*.safetensors`, **not** in the JSON files; loading without them
silently feeds the network unnormalized degrees and the policy will look broken for no visible reason.

```bash
hf download azorematter/imle_stream_orig \
  --include "checkpoints/092500/pretrained_model/*" --local-dir ./ckpt
```

Note the two policies use **different normalization modes** — IMLE uses MIN_MAX for state/action,
ACT uses MEAN_STD. This is handled automatically by loading the processors from the checkpoint, and is
another reason never to rebuild them by hand.

**ACT additionally requires this fork.** It was trained with `use_separate_backbone_per_camera=true`,
which upstream `lerobot` does not have — stock LeRobot cannot load the checkpoint:

```bash
pip install "lerobot @ git+https://github.com/ZoreAnuj/lerobot@main"
```

## 4. Per-policy usage

### 4.1 RS-IMLE (original, and both gripper variants)

`n_obs_steps=2`, `horizon=16`, `n_action_steps=8`, `n_samples_per_condition=20`, `rs_epsilon=0.03`.
Generates a 16-step chunk in one forward pass; 8 steps are executed per replan (0.27 s at 30 Hz).

```python
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

CKPT = "./ckpt/checkpoints/092500/pretrained_model"
cfg = PreTrainedConfig.from_pretrained(CKPT)
cfg.pretrained_path = CKPT
cfg.device = "cuda"
policy = get_policy_class(cfg.type).from_pretrained(CKPT, config=cfg)
policy.eval()
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": cfg.device}},
)

policy.reset()                                   # once per episode
while not done:
    obs = pre({
        "observation.state": state_1x7,          # (1, 7) [J1..J6 deg, gripper 1=OPEN]
        "observation.images.gripper": wrist_rgb,  # (1, 3, 480, 640) float32 in [0,1]
        "observation.images.cam0": board_rgb,
    })
    action = post(policy.select_action(obs))     # (1, 7), one row per 30 Hz tick
    send_joint_targets(action[0, :6])            # absolute degrees
    set_gripper(action[0, 6])                    # hysteresis: close < 0.4, open > 0.6
```

`select_action` maintains the 2-step observation history and the action queue internally: it re-runs
the network every 8 ticks and pops one row per call.

**Never duplicate a frame to fill the 2-step history.** The pair encodes velocity; a duplicated frame
reads as "stationary" and measurably weakens descents. Keep a rolling buffer of real frames 33 ms
apart. Caveat: `select_action` itself fills its history by copying the first observation after
`reset()`, so the *first* chunk of each episode is conditioned on a stationary pair regardless — either
accept it, discard the first replan, or build the first chunk explicitly (below).

For chunk-at-once execution (e.g. interpolating to 125 Hz), call `predict_action_chunk` with an
explicitly stacked history — state `(1, 2, 7)`, images `(1, 2, 3, 480, 640)`, the two steps 33 ms apart.
It returns the executed window already sliced to `n_action_steps`. Two traps: run inputs through
`pre(...)` and outputs through `post(...)` exactly as above, and **never mix `select_action` with
explicit-history `predict_action_chunk` in one episode** — populated queues silently override the batch
you pass. Call `policy.reset()` first.

Optional: `cfg.use_traj_consistency = True` samples `n_consistency_candidates=32` chunks per replan and
picks the one whose start best matches the previous chunk's tail, which reduces mode switching between
replans. It is **off** in these checkpoints; enable it at load time if you see chunk-to-chunk jumps.

### 4.2 ACT

`n_obs_steps=1`, `chunk_size=30`, `n_action_steps=30`, `temporal_ensemble_coeff=None`.
Generates 1.0 s of motion and executes the whole chunk before replanning.

```python
CKPT = "./ckpt/checkpoints/040000/pretrained_model"
# ... identical load block to §4.1 ...

policy.reset()
while not done:
    obs = pre({
        "observation.state": state_1x7,           # (1, 7) — single step, no history stacking
        "observation.images.gripper": wrist_rgb,
        "observation.images.cam0": board_rgb,
    })
    action = post(policy.select_action(obs))      # (1, 7) per tick; re-plans every 30 ticks
    send_joint_targets(action[0, :6]); set_gripper(action[0, 6])
```

Temporal ensembling stays **off** — that is deliberate per the tuning guide (query frequency equals the
chunk length). Do not enable it without re-evaluating.

## 5. Which checkpoints to test

Three per policy, chosen to span the axes that actually differ rather than three near-identical models.

**RS-IMLE original** — `azorematter/imle_stream_orig`

| pick | checkpoint | MAE° | bit-flip | escape |
| --- | --- | --- | --- | --- |
| best precision | `092500` | **0.269** | 0.292 | 0.250 |
| best balance | `065000` | 0.280 | **0.224** | 0.233 |
| runner-up | `090000` | 0.279 | 0.261 | 0.233 |

**ACT** — `azorematter/act_stream`

| pick | checkpoint | MAE° | bit-flip | escape |
| --- | --- | --- | --- | --- |
| best precision | `040000` | **0.516** | 0.187 | 0.750 |
| best balance | `085000` | 0.632 | 0.068 | 0.717 |
| most gripper-independent | `075000` | 0.826 | **0.039** | 0.683 |

ACT's held-out MAE degrades monotonically after ~40k steps (0.516 → 0.632 → 0.826): **it overfits, and
the final checkpoint is not the best.** Pick from this table, not from `last`.

**RS-IMLE + gripper (from scratch)** — `azorematter/imle_stream_grip`: best is `072500` (MAE 0.738,
bit-flip 0.314). **Not recommended for robot time** — 2.7× less precise than plain IMLE with no
bit-flip improvement. Those four interventions were validated as a *fine-tune on a converged base*;
applied from cold start they cost precision and buy nothing.

**RS-IMLE + gripper (fine-tune)** — `azorematter/imle_ft_stream`: 13k steps at lr 1e-5 from
`imle_stream_orig/100000` with `rs_gripper_weight=5`, `transition_oversample=10`,
`motion_onset_min_dwell=5`, `gripper_obs_dropout=0.2`. Training-side `gripper_err_selected` reached
**0.010** (best of any run; base 0.012). Checkpoint ranking to be appended here.

## 6. Deployment notes, in order of how much they have cost us

1. **The gripper observation is a feedback loop.** If the observed gripper bit is rebuilt from your
   executor's latch, a policy that merely copies that bit can never close. RS-IMLE's bit-flip is
   **0.22–0.29** versus ACT's **0.04–0.19** — meaning IMLE leans considerably more on that input.
   Before an armed run with an IMLE checkpoint, run a bit-flip probe: invert the input bit and confirm
   the commanded gripper still tracks the scene. If your executor closes the loop on its own latch,
   prefer ACT or the fine-tuned IMLE.
2. **Avoid zero-motion observation pairs at replan time.** Demonstrations contain stationary dwells
   (17.4 % of frames repeat the previous action), so "currently stationary" biases every policy toward
   "stay". Replan *while the arm is still moving*, or build the pair from the last two in-motion frames.
   RS-IMLE's dwell escape (0.25) is materially worse than ACT's (0.72) here.
3. **Camera health is asymmetric — and ACT is the fragile one.** Measured by blanking one camera to
   mid-grey and re-scoring: losing the **wrist** view multiplies ACT's error ~4.5× but IMLE's only
   ~1.1×; losing the **overhead** view costs both far less (~0.6–1.3×). So ACT carries most of its
   alignment information in the wrist view. Practical upshot: if the wrist camera can be occluded by
   the workpiece or the gripper itself in your cell, prefer IMLE, and gate armed runs on that feed's
   freshness and exposure either way (§2.5).
4. **Match the image chain.** Training frames passed through AV1 (CRF 30) video encoding. Prefer the
   same camera pipeline used for collection; if feeding raw frames and behaviour looks off, A/B an AV1
   round-trip offline — texture-level domain shift is real.
5. **Pin geometry and identity.** Assert 480×640, verify camera serials map to the right views, and
   gate on frame freshness — a wedged feed yields a confident policy driving on a still image.
6. **Loss cannot rank these policies.** A bit-copying model still scores ~99 % gripper step-accuracy.
   Use the probes in `analysis/rank_imle_ckpts.py`, and ignore high `dwell escape` on early checkpoints
   (escape ≈1.0 at step 2,500 with MAE ~1.2 just means an undertrained model emits motion from any
   input).

## 7. Dataset provenance — why these supersede the older checkpoints

`stream_100` fixed a timing defect present in the original `dice_white_pnp_100`. In the old data the
pose stream was **waypoint-interpolated**: 98.9 % of frames sat inside perfectly constant-velocity
segments with only ~19 direction changes per episode, so image↔state pairs were misaligned by up to
several hundred milliseconds during fast motion. `stream_100` shows 22.7 % and 471 kinks per episode —
genuine per-frame motion. Anything trained on the old data inherits that misalignment.

Still present in `stream_100`, and worth knowing: a **scripted pre-grasp dive** in 97/100 episodes —
the arm descends to the taught grasp depth (−126.5 mm), retracts 35 mm, holds 1.6–3.3 s while the wrist
rotates to the dice yaw, then closes and drops. We measured that excising it does **not** improve
policy behaviour (dwell escape 0.188 vs 0.150 for the uncut baseline, within probe noise), so these
policies are trained on the data as recorded.

## 8. Latency (A100-class GPU, batch 1)

| policy | network passes per replan | ballpark | replan budget |
| --- | --- | --- | --- |
| RS-IMLE | 1 (one-step generator) | ~10 ms | 267 ms (8 steps) |
| ACT | 1 | ~15 ms | 1000 ms (30 steps) |

Both comfortably replan within their executed window at 30 Hz.
