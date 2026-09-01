# Inference Guide — Dice Pick-and-Place Policies (Fanuc CRX-5iA)

How to run the trained policies from this repo at inference time: the exact observation contract
they were trained on, which checkpoint to load, working code, and the deployment pitfalls found
during real-robot evaluation. **Training and inference must agree on every property below** — most
real-world failures traced back to a mismatch here, not to the weights.

All checkpoints were trained on
[`azorematter/dice_white_pnp_100`](https://huggingface.co/datasets/azorematter/dice_white_pnp_100)
(99 episodes, 30 fps, task: *"Pick up the dice and place it on the empty white block."*), reduced to
two cameras and joint-space state/action.

---

## 1. Checkpoints

All repos are private — authenticate first with `hf auth login`.

| Policy | Hub repo | Recommended checkpoint | Notes |
| --- | --- | --- | --- |
| **IMLE (fine-tuned)** — recommended | [`azorematter/imle_ft_gripperz_dice`](https://huggingface.co/azorematter/imle_ft_gripperz_dice) | `checkpoints/015000/pretrained_model` (alt: `017500`) | Gripper-robust + dwell-exit fine-tune of the base IMLE. Best precision/robustness balance: joint MAE 0.14°, dwell-escape 0.84, gripper transition commit 0.97. `checkpoints/005000` is a higher-escape (0.95) fallback at lower precision. |
| IMLE (base) | [`azorematter/imle_dice_white_pnp`](https://huggingface.co/azorematter/imle_dice_white_pnp) | `checkpoints/100000/pretrained_model_ema` (or `pretrained_model`) | Best open-loop precision (0.051°) but **known hover-stall**: from a stationary observation pair only ~35% of samples command motion. Use only with replan-while-moving (§6). |
| SmolVLA (fine-tuned from `lerobot/smolvla_base`) | [`azorematter/smolvla_dice_white_pnp`](https://huggingface.co/azorematter/smolvla_dice_white_pnp) | `checkpoints/030000/pretrained_model` | Language-conditioned; needs the task string and camera renaming (§3). Top-3 by open-loop MAE: 30k (0.153°), 25k, 20k. |
| ACT | [`azorematter/act_dice_white_pnp`](https://huggingface.co/azorematter/act_dice_white_pnp) | latest synced (run in progress; final `checkpoints/100000`) | Tuned per the ACT guide: chunk 30 (=1 s), separate backbone per camera, kl 10, L1. Checkpoint ranking to be published after training completes. |

Download one checkpoint folder:

```bash
hf download azorematter/imle_ft_gripperz_dice \
  --include "checkpoints/015000/pretrained_model/*" --local-dir ./ckpt
```

Every checkpoint folder is self-contained: `config.json`, `model.safetensors`,
`train_config.json`, `policy_preprocessor.json` / `policy_postprocessor.json` **plus their
`policy_pre/postprocessor_step_*.safetensors` companions — the step safetensors, not the JSONs,
hold the normalization statistics**, so always download the whole folder (the `--include` pattern
above does). Load the processors from the checkpoint; rebuilding them from a dataset (or skipping
them) silently feeds the network unnormalized degrees/millimetres.

## 2. Observation contract

| Property | Value |
| --- | --- |
| Cameras | `observation.images.gripper` (wrist), `observation.images.cam0` (elevated board view) |
| Frame format | 480×640, **RGB** (convert from BGR!), channel-first `(3, 480, 640)`, `float32` in `[0, 1]` |
| Resolution handling | Feed **full 480×640**. IMLE resizes to 240×320 *inside* the model; SmolVLA letterboxes to 512×512 inside the model (`prepare_images`/`resize_with_pad`). Do not crop or resize outside. |
| State `observation.state` | 7-dim `float32`: `[J1..J6, gripper]` — joints in **absolute degrees**, gripper **1.0 = OPEN, 0.0 = CLOSED** |
| Rate | 30 Hz (observation spacing of 33 ms is what the policies saw) |
| History | IMLE: **2 observation steps, 33 ms apart** — see §6, this must be two *real* frames. ACT / SmolVLA: 1 step |
| Task string (SmolVLA only) | exactly `Pick up the dice and place it on the empty white block.` |

## 3. Action contract

All policies emit 7-dim rows: `[J1..J6, gripper]` — **absolute joint targets in degrees** (not
deltas, not radians, not Cartesian) plus the gripper channel (1 = open, 0 = close). Execute with
joint position control; threshold the gripper with hysteresis (close below 0.4, open above 0.6)
rather than a single 0.5 cut.

Chunking per policy (at 30 Hz):

| Policy | Chunk generated | Execute per replan |
| --- | --- | --- |
| IMLE | 16 steps (0.53 s) | 8 |
| ACT | 30 steps (1.0 s) | 30 (full chunk, no temporal ensembling) |
| SmolVLA | 50 steps (1.67 s) | 25–50 |

## 4. Minimal working example (IMLE fine-tuned)

```python
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

CKPT = "./ckpt/checkpoints/015000/pretrained_model"

cfg = PreTrainedConfig.from_pretrained(CKPT)
cfg.pretrained_path = CKPT
cfg.device = "cuda"                      # or "cpu"
policy = get_policy_class(cfg.type).from_pretrained(CKPT, config=cfg)
policy.eval()

# Processors come from the checkpoint; only the device is overridden.
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": cfg.device}},
)

def observe():
    """Return one observation in the training contract (see §2)."""
    return {
        "observation.state": torch.tensor([[j1, j2, j3, j4, j5, j6, grip_open]]),  # (1, 7) degrees
        "observation.images.gripper": wrist_rgb_chw.unsqueeze(0),   # (1, 3, 480, 640) in [0,1]
        "observation.images.cam0": board_rgb_chw.unsqueeze(0),
    }

policy.reset()                            # once per episode
while not done:
    obs = pre(observe())
    action = post(policy.select_action(obs))   # (1, 7): one row per 30 Hz tick
    send_joint_targets(action[0, :6])          # absolute degrees
    set_gripper(action[0, 6])                  # 1=open, 0=close (hysteresis 0.4/0.6)
```

`select_action` manages the observation history and the action queue internally: it re-runs the
network every `n_action_steps` ticks and pops one row per call. For chunk-at-once streaming (e.g.
125 Hz interpolated execution), call `policy.predict_action_chunk(batch)` with an explicitly
stacked history — state `(1, 2, 7)`, images `(1, 2, 3, 480, 640)` per camera, the two steps 33 ms
apart. It returns the executed window **already sliced and aligned to the current step**:
`(1, n_action_steps=8, 7)` for IMLE (want a longer window per replan? set `cfg.n_action_steps` up
to `horizon - n_obs_steps + 1 = 15` before loading). Two caveats: run inputs through `pre(...)`
and outputs through `post(...)` exactly as with `select_action`; and this offline path is only
taken while the policy's internal queues are empty — call `policy.reset()` first and never mix
`select_action` with explicit-history `predict_action_chunk` in one episode, or the populated
queues silently override the batch you pass.

**SmolVLA differences:** batch must also contain `"task": ["Pick up the dice and place it on the
empty white block."]`, and camera keys must be renamed to the base model's slots
(`gripper → observation.images.camera1`, `cam0 → observation.images.camera2`). The rename map is
serialized in the checkpoint's `policy_preprocessor.json` — read it from there rather than
hardcoding.

## 5. Latency expectations (A100-class GPU, batch 1)

| Policy | Network passes per replan | Ballpark |
| --- | --- | --- |
| IMLE | 1 (one-step generator) | ~10 ms |
| ACT | 1 | ~15 ms |
| SmolVLA | 10 flow steps | ~100–150 ms |

All comfortably replan within their executed-window budget at 30 Hz.

## 6. Deployment pitfalls (each one cost us a real debugging session)

1. **Never duplicate a frame to fill IMLE's 2-step history.** The pair encodes velocity; a
   duplicated frame reads as "stationary" and measurably weakens descents. Keep a rolling buffer of
   real frames 33 ms apart. Caveat: `select_action` itself fills its internal history by copying
   the first observation after `reset()`, so the *first* chunk of every episode is conditioned on a
   stationary pair no matter what you buffer — either accept it (fine-tuned checkpoint handles it),
   discard the first replan, or generate the first chunk via `reset()` +
   `predict_action_chunk` with two real stacked frames.
2. **Avoid zero-motion observation pairs at replan time.** Demonstrations contain stationary dwells,
   so "currently stationary" biases every policy toward "stay" — the base IMLE checkpoint hover-stalls
   because of exactly this. Replan *while the arm is still executing*, or build the pair from the
   last two in-motion frames. The fine-tuned checkpoint is trained to escape dwells (escape rate
   0.8–0.98 vs 0.35 base) but the inference-side fix is still recommended.
3. **The gripper observation is a feedback loop.** If the observed gripper bit is rebuilt from your
   executor's latch, a policy that merely copies the bit can never close. The fine-tuned IMLE was
   explicitly trained (gripper-weighted loss, bit dropout) to make the close decision from vision;
   verify any *other* checkpoint with a bit-flip probe before an armed run: flip the input bit and
   check the commanded gripper still tracks the scene, not the bit.
4. **Match the image chain.** Training frames passed through AV1 (CRF 30, yuv420) video encoding.
   Prefer the same camera pipeline used for data collection; if feeding raw frames, A/B an
   AV1 round-trip offline if behavior looks off — texture-level domain shift is real.
5. **Pin geometry and identity.** Assert 480×640 (don't let a backbone silently absorb another
   resolution), verify camera serials map to the right views, and gate on frame freshness — a wedged
   camera feed yields a confident policy driving on a still image.
6. **Evaluate several checkpoints.** Loss cannot rank these policies (a gripper-bit-copying model
   scores 99% step-accuracy). Use behavior probes: transition recall (does it close near
   demonstrated closes?), dwell escape (does it move from stationary hovers?), and open-loop joint
   MAE for precision.

## 7. Recommended defaults per policy

- **IMLE ft-15k**: `use_traj_consistency=true` for replans (reduces mode switching between chunks),
  execute 8 of 16 steps, replan in motion.
- **ACT**: execute the full 30-step chunk, then replan (temporal ensembling stays off, per config).
- **SmolVLA**: execute 25 of 50 steps; expect ~10× IMLE's latency per replan.
