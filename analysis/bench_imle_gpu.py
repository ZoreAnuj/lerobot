"""Compute-only scaling curve for IMLE: forward+backward samples/s vs batch size on one A100."""
import argparse
import time

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/mnt/data/zero/runs/imle_dice_j7_2cam/checkpoints/100000/pretrained_model")
ap.add_argument("--batches", type=int, nargs="+", default=[64, 128, 256, 384])
ap.add_argument("--iters", type=int, default=12)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

cfg = PreTrainedConfig.from_pretrained(args.ckpt)
cfg.device = args.device
policy = get_policy_class(cfg.type)(cfg).to(args.device)
policy.train()
opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"IMLE: {n_params/1e6:.1f}M params | resize {cfg.resize_shape} crop {cfg.crop_shape} "
      f"| n_obs {cfg.n_obs_steps} horizon {cfg.horizon} | candidates m={cfg.n_samples_per_condition}")
print(f"{'batch':>6} {'samples/s':>10} {'ms/step':>9} {'peak GB':>8}  (fwd+bwd only, bf16 autocast)")

for B in args.batches:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    batch = {
        "observation.state": torch.randn(B, cfg.n_obs_steps, 7, device=args.device),
        "action": torch.randn(B, cfg.horizon, 7, device=args.device),
        "action_is_pad": torch.zeros(B, cfg.horizon, dtype=torch.bool, device=args.device),
    }
    for k in cfg.image_features:
        batch[k] = torch.rand(B, cfg.n_obs_steps, 3, 480, 640, device=args.device)
    try:
        for i in range(args.iters):
            if i == 2:
                torch.cuda.synchronize()
                t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(batch)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / (args.iters - 2)
        print(f"{B:6d} {B/dt:10.1f} {dt*1000:9.1f} {torch.cuda.max_memory_allocated()/1e9:8.1f}")
    except torch.cuda.OutOfMemoryError:
        print(f"{B:6d}  OOM")
        break
    del batch
