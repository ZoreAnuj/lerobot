"""Rank IMLE checkpoints on precision AND the two behaviours that decide real-robot success.

Loss cannot rank these policies (a gripper-bit-copying model scores ~99% step accuracy), so we run
behaviour probes instead:

  open-loop MAE   precision, measured on the data the policy was trained to reproduce
  transition      near a demonstrated close, does the chunk actually command the close?
  commit          having closed, does it stay closed for the rest of the chunk?
  bit-flip        flip the observed gripper bit: does the command follow the scene or just copy the bit?
  dwell escape    from a stationary hover, does it command motion or sit there? (the hover-stall)

The dwell-escape probe deliberately samples its stationary hovers from the ORIGINAL (uncut) dataset,
because that is what the published baseline numbers were measured on — changing the probe set and the
checkpoint at the same time would make the comparison meaningless.

Usage: rank_imle_ckpts.py <run_dir_or_ckpt_glob> [--probe-root ORIGINAL_DS] [--mae-root TRAIN_DS]
"""
import argparse
import glob
import inspect
import json
import os

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

ap = argparse.ArgumentParser()
ap.add_argument("run", help="run dir containing checkpoints/, or a glob of pretrained_model dirs")
ap.add_argument("--mae-root", default="/mnt/data/zero/datasets/dice_nodip_j7")
ap.add_argument("--probe-root", default="/mnt/data/zero/datasets/dice_j7_2cam")
ap.add_argument("--mae-episodes", default=None,
                help="comma-separated episode indices for the MAE split; use the HELD-OUT ones")
ap.add_argument("--n-mae", type=int, default=120)
ap.add_argument("--n-probe", type=int, default=80)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default=None)
args = ap.parse_args()

CLOSE_T, OPEN_T = 0.4, 0.6          # deployment hysteresis
MOVE_DEG = 0.5                       # a chunk "commands motion" if any joint moves more than this


def load(ckpt):
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = args.device
    # widen the returned window to the full usable horizon so the probes see the whole plan.
    # IMLE calls it `horizon`, ACT calls it `chunk_size`.
    horizon = getattr(cfg, "horizon", None) or cfg.chunk_size
    cfg.n_action_steps = horizon - getattr(cfg, "n_obs_steps", 1) + 1
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )
    return cfg, policy, pre, post


def make_ds(root, cfg, episodes=None):
    meta = LeRobotDatasetMetadata(repo_id=os.path.basename(root), root=root)
    dt = resolve_delta_timestamps(cfg, meta)
    return LeRobotDataset(os.path.basename(root), root=root, delta_timestamps=dt, episodes=episodes)


@torch.no_grad()
def chunk_for(policy, pre, post, item, state_override=None, noise=None):
    """Predict one action chunk from a dataset item's stacked history.

    `noise` pins the generator's latent so two calls on the same frame (baseline vs. a blanked
    camera) differ only by the input, not by the RS-IMLE sampling spread."""
    batch = {}
    for k, v in item.items():
        if k.startswith("observation."):
            batch[k] = v.unsqueeze(0).to(args.device)
    if state_override is not None:
        batch["observation.state"] = state_override.unsqueeze(0).to(args.device)
    policy.reset()
    # IMLE's generator takes an explicit latent; ACT's predict_action_chunk has no `noise` arg.
    kw = {"noise": noise} if (noise is not None and "noise" in
                              inspect.signature(policy.predict_action_chunk).parameters) else {}
    return post(policy.predict_action_chunk(pre(batch), **kw))[0].float().cpu().numpy()


def evaluate(ckpt):
    cfg, policy, pre, post = load(ckpt)
    eps = [int(x) for x in args.mae_episodes.split(",")] if args.mae_episodes else None
    mae_ds = make_ds(args.mae_root, cfg, episodes=eps)
    probe_ds = make_ds(args.probe_root, cfg)
    rng = np.random.default_rng(7)
    row = {"ckpt": os.path.basename(os.path.dirname(ckpt))}
    # predict_action_chunk returns rows aligned to the CURRENT step; the dataset's action window
    # starts at action_delta_indices[0] = t-(n_obs_steps-1) for IMLE, so the target is offset.
    off = int(getattr(cfg, "n_obs_steps", 1)) - 1
    act_dim = cfg.action_feature.shape[0]

    # ---- open-loop precision on the training distribution
    idxs = rng.integers(0, len(mae_ds), args.n_mae)
    j_err, g_ok = [], []
    noises = {}
    for i in idxs:
        it = mae_ds[int(i)]
        _h = getattr(cfg, "horizon", None) or cfg.chunk_size
        noises[int(i)] = torch.randn(1, _h, act_dim, device=args.device)
        pred = chunk_for(policy, pre, post, it, noise=noises[int(i)])
        tgt = it["action"].numpy()[off : off + len(pred)]
        j_err.append(np.abs(pred[:, :6] - tgt[:, :6]).mean())
        g_ok.append(((pred[:, 6] > 0.5) == (tgt[:, 6] > 0.5)).mean())
    row["mae_deg"] = float(np.mean(j_err))
    row["grip_acc"] = float(np.mean(g_ok))

    # ---- camera reliance: the same frames with ONE camera blanked to mid-grey. A policy that
    # carries alignment in the wrist view loses far more precision without the wrist than
    # without the global view; a policy that ignores the wrist barely notices it going.
    img_keys = list(cfg.image_features)
    blank_err = {k: [] for k in img_keys}
    for i in idxs:
        it = mae_ds[int(i)]
        tgt = it["action"].numpy()
        for k in img_keys:
            it_b = dict(it)
            it_b[k] = torch.full_like(it[k], 0.5)
            pred = chunk_for(policy, pre, post, it_b, noise=noises[int(i)])
            blank_err[k].append(np.abs(pred[:, :6] - tgt[off : off + len(pred), :6]).mean())
    for k in img_keys:
        row[f"mae_no_{k.split('.')[-1]}"] = float(np.mean(blank_err[k]))

    # ---- gripper transitions + bit-flip dependence, on the probe set
    st = np.stack(probe_ds.hf_dataset["observation.state"]).astype(float) \
        if hasattr(probe_ds, "hf_dataset") and probe_ds.hf_dataset is not None else None
    if st is None:
        probe_ds[0]
        st = np.stack(probe_ds.hf_dataset["observation.state"]).astype(float)
    g = st[:, -1]
    closes = np.where((g[:-1] > 0.5) & (g[1:] <= 0.5))[0]
    closes = closes[(closes > 4) & (closes < len(g) - 20)]
    if len(closes) > args.n_probe:
        closes = rng.choice(closes, args.n_probe, replace=False)
    recall, commit, flip = [], [], []
    for t in closes:
        it = probe_ds[int(t) - 2]
        pred = chunk_for(policy, pre, post, it)
        closed = pred[:, 6] < CLOSE_T
        recall.append(float(closed.any()))
        if closed.any():
            k = int(np.argmax(closed))
            commit.append(float(closed[k:].mean()))
        flipped_state = it["observation.state"].clone()
        # IMLE stacks history -> (n_obs, dim); ACT uses n_obs_steps=1 -> (dim,)
        if flipped_state.ndim == 1:
            flipped_state[-1] = 1.0 - flipped_state[-1]
        else:
            flipped_state[:, -1] = 1.0 - flipped_state[:, -1]
        pred_f = chunk_for(policy, pre, post, it, state_override=flipped_state)
        flip.append(float(np.abs(pred_f[:, 6] - pred[:, 6]).mean()))
    row["transition_recall"] = float(np.mean(recall)) if recall else float("nan")
    row["commit"] = float(np.mean(commit)) if commit else float("nan")
    row["bitflip_delta"] = float(np.mean(flip)) if flip else float("nan")

    # ---- dwell escape: stationary hovers with the gripper still open
    dv = np.abs(np.diff(st[:, :6], axis=0)).max(axis=1)
    stat = np.where(dv < 1e-3)[0]
    stat = stat[(g[stat] > 0.5) & (stat > 4) & (stat < len(g) - 20)]
    if len(stat) > args.n_probe:
        stat = rng.choice(stat, args.n_probe, replace=False)
    moved = []
    for t in stat:
        it = probe_ds[int(t)]
        pred = chunk_for(policy, pre, post, it)
        _st = it["observation.state"].numpy()
        base = _st[:6] if _st.ndim == 1 else _st[-1, :6]
        moved.append(float(np.abs(pred[:, :6] - base).max() > MOVE_DEG))
    row["dwell_escape"] = float(np.mean(moved)) if len(moved) else float("nan")
    row["n_probe"] = int(len(closes))
    del policy
    torch.cuda.empty_cache()
    return row


ckpts = sorted(glob.glob(os.path.join(args.run, "checkpoints", "*", "pretrained_model"))) \
    if os.path.isdir(os.path.join(args.run, "checkpoints")) else sorted(glob.glob(args.run))
print(f"{len(ckpts)} checkpoints | MAE on {os.path.basename(args.mae_root)} | "
      f"probes on {os.path.basename(args.probe_root)}\n")
hdr = (f"{'ckpt':>8} {'MAE deg':>8} {'grip_acc':>9} {'recall':>7} {'commit':>7} {'bitflip':>8} {'escape':>7}"
       f" {'noCam0':>7} {'noWrist':>8}")
print(hdr)
print("-" * len(hdr))
rows = []
for c in ckpts:
    try:
        r = evaluate(c)
    except Exception as e:
        print(f"{os.path.basename(os.path.dirname(c)):>8}  FAILED {type(e).__name__}: {e}")
        continue
    rows.append(r)
    print(f"{r['ckpt']:>8} {r['mae_deg']:8.3f} {r['grip_acc']:9.3f} {r['transition_recall']:7.3f} "
          f"{r['commit']:7.3f} {r['bitflip_delta']:8.3f} {r['dwell_escape']:7.3f}"
          f" {r.get('mae_no_cam0', float('nan')):7.3f} {r.get('mae_no_gripper', float('nan')):8.3f}", flush=True)

if args.out and rows:
    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
