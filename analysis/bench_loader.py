"""Measure the data pipeline ceiling for IMLE-style sampling: video decode vs a raw frame cache."""
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/mnt/data/zero/datasets/dice_j7_2cam")
ap.add_argument("--repo", default="azorematter/dice_j7_2cam")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--workers", type=int, nargs="+", default=[8, 24, 48])
ap.add_argument("--steps", type=int, default=40)
args = ap.parse_args()

dt = {
    "observation.state": [-1 / 30, 0.0],
    "observation.images.gripper": [-1 / 30, 0.0],
    "observation.images.cam0": [-1 / 30, 0.0],
    "action": [i / 30 for i in range(16)],
}
ds = LeRobotDataset(args.repo, root=args.root, delta_timestamps=dt)
print(f"dataset: {ds.num_frames} frames, {ds.num_episodes} episodes, cams {ds.meta.camera_keys}")

# ---- single-sample decode cost (no workers)
t0 = time.time()
idxs = np.random.default_rng(0).integers(0, len(ds) - 20, 60)
for i in idxs:
    _ = ds[int(i)]
per = (time.time() - t0) / len(idxs)
print(f"single-process: {per*1000:.1f} ms/sample  "
      f"({4/per:.0f} frames/s, 4 frames per sample: 2 cams x 2 obs steps)")

# ---- dataloader throughput at various worker counts
for nw in args.workers:
    dl = DataLoader(ds, batch_size=args.batch, num_workers=nw, shuffle=True,
                    pin_memory=True, drop_last=True, persistent_workers=True,
                    prefetch_factor=2)
    it = iter(dl)
    next(it)                                    # warm up workers
    t0 = time.time()
    n = 0
    for _ in range(args.steps):
        b = next(it)
        n += b["observation.state"].shape[0]
    dtsec = time.time() - t0
    print(f"workers={nw:3d}: {n/dtsec:7.1f} samples/s  "
          f"({n/dtsec*4:.0f} frames/s)  -> {args.batch/(n/dtsec)*1000:.0f} ms per batch of {args.batch}")
    del it, dl

# ---- what a raw uint8 frame cache would cost/deliver
f = ds.num_frames
bytes_full = f * 2 * 480 * 640 * 3
print(f"\nframe cache sizing: {f} frames x 2 cams")
print(f"  full res 480x640 uint8 : {bytes_full/1e9:.1f} GB")
print(f"  model res 240x320 uint8: {bytes_full/4/1e9:.1f} GB")
a = np.zeros((4096, 480, 640, 3), dtype=np.uint8)
idx = np.random.default_rng(0).integers(0, 4096, 4 * args.batch)
t0 = time.time()
for _ in range(20):
    x = torch.from_numpy(a[idx])
print(f"  memmap-style gather of {4*args.batch} frames: "
      f"{(time.time()-t0)/20*1000:.1f} ms per batch (RAM-resident)")
