"""Pre-decode a LeRobotDataset's videos into a uint8 memmap keyed by global frame index.

Serves the `video_backend="memmap"` path in dataset_reader.py: ~0.1 ms/frame instead of ~4 ms
for a seek-and-decode, so the dataloader stops competing for CPU with everything else on the box.

Usage: build_frame_cache.py <dataset_root> [--jobs N] [--verify N]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import av
import numpy as np
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--jobs", type=int, default=12)
ap.add_argument("--verify", type=int, default=200)
args = ap.parse_args()

ROOT = args.root
CACHE = f"{ROOT}/frame_cache"
info = json.load(open(f"{ROOT}/meta/info.json"))
FPS = info["fps"]
KEYS = [k for k, v in info["features"].items() if v["dtype"] == "video"]
H, W = info["features"][KEYS[0]]["shape"][:2]
if H == 3:                                    # channel-first declaration
    H, W = info["features"][KEYS[0]]["shape"][1:3]
N = info["total_frames"]
import glob as _glob
_ep_files = sorted(_glob.glob(f"{ROOT}/meta/episodes/*/*.parquet"))
assert _ep_files, f"no episodes parquet under {ROOT}/meta/episodes"
import pyarrow as _pa
ep_meta = _pa.concat_tables([pq.read_table(f) for f in _ep_files]).to_pandas()
assert len(ep_meta) == info["total_episodes"], (
    f"episode metadata is short: {len(ep_meta)} rows vs info.json {info['total_episodes']}")

os.makedirs(CACHE, exist_ok=True)
print(f"cache: {len(KEYS)} cameras x {N} frames x {H}x{W}x3 = "
      f"{len(KEYS)*N*H*W*3/1e9:.1f} GB -> {CACHE}")


def fill(task):
    key, ep_idx = task
    row = ep_meta[ep_meta.episode_index == ep_idx].iloc[0]
    lo, hi = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    fidx = int(row[f"videos/{key}/file_index"])
    cidx = int(row[f"videos/{key}/chunk_index"])
    t0 = float(row[f"videos/{key}/from_timestamp"])
    mm = np.memmap(f"{CACHE}/{key}.uint8", dtype=np.uint8, mode="r+", shape=(N, H, W, 3))
    c = av.open(f"{ROOT}/videos/{key}/chunk-{cidx:03d}/file-{fidx:03d}.mp4")
    st = c.streams.video[0]
    st.thread_type = "AUTO"
    if t0 > 0:
        c.seek(int(max(0, t0 - 1.0) / st.time_base), stream=st)
    written = 0
    for fr in c.decode(st):
        k = int(round((float(fr.pts * st.time_base) - t0) * FPS))
        if k < 0:
            continue
        if k >= hi - lo:
            break
        mm[lo + k] = fr.to_ndarray(format="rgb24")
        written += 1
    c.close()
    mm.flush()
    del mm
    if written != hi - lo:
        raise RuntimeError(f"{key} ep{ep_idx}: wrote {written} frames, expected {hi - lo}")
    return written


for key in KEYS:
    path = f"{CACHE}/{key}.uint8"
    if not os.path.exists(path) or os.path.getsize(path) != N * H * W * 3:
        np.memmap(path, dtype=np.uint8, mode="w+", shape=(N, H, W, 3)).flush()

tasks = [(k, int(e)) for k in KEYS for e in ep_meta.episode_index]
done = 0
with ProcessPoolExecutor(max_workers=args.jobs) as ex:
    for n in ex.map(fill, tasks):
        done += 1
        if done % 40 == 0:
            print(f"  {done}/{len(tasks)} episode-camera pairs", flush=True)

json.dump({"fps": FPS, "total_frames": N,
           "keys": {k: [N, H, W, 3] for k in KEYS}},
          open(f"{CACHE}/index.json", "w"))
print(f"wrote {CACHE}/index.json")

# ---- verify against the video decoder, byte for byte
if args.verify:
    from lerobot.datasets.video_utils import decode_video_frames
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(args.verify):
        key = KEYS[rng.integers(len(KEYS))]
        ep_idx = int(ep_meta.episode_index.iloc[rng.integers(len(ep_meta))])
        row = ep_meta[ep_meta.episode_index == ep_idx].iloc[0]
        lo, hi = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        f = int(rng.integers(0, hi - lo))
        t0 = float(row[f"videos/{key}/from_timestamp"])
        path = (f"{ROOT}/videos/{key}/chunk-{int(row[f'videos/{key}/chunk_index']):03d}/"
                f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4")
        dec = decode_video_frames(path, [t0 + f / FPS], 1e-4, "pyav", return_uint8=True)
        dec = dec.squeeze(0).permute(1, 2, 0).numpy()
        mm = np.memmap(f"{CACHE}/{key}.uint8", dtype=np.uint8, mode="r", shape=(N, H, W, 3))
        if not np.array_equal(dec, np.asarray(mm[lo + f])):
            bad += 1
            print(f"  MISMATCH {key} ep{ep_idx} frame {f}")
        del mm
    print(f"verify: {args.verify - bad}/{args.verify} frames byte-identical to the video decoder")
    if bad:
        raise SystemExit(1)
