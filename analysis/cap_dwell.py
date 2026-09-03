"""Cap stationary dwells in a LeRobotDataset variant, memmap-cache first.

Why: in the dice corpus the gripper closes after a median 43 fully stationary frames at the hover
height, with nothing in the observation marking the moment. From any single hover frame only ~8 % of
8-step windows contain the close, so a policy learns "stationary at hover -> keep waiting" - the
hover-stall seen on the robot. Capping each stationary run turns the close into an event the policy
can see coming.

What it does, per episode:
  * a frame is STATIONARY when max|dJ1..J6| < --thresh vs the previous frame AND the gripper is
    unchanged (a gripper flip is an event, never dwell);
  * every run of consecutive stationary frames longer than --cap keeps only its LAST --cap frames
    (the ones adjacent to whatever ended the dwell), the rest are dropped;
  * action[t] = state[t+1] over the KEPT frames (last action = last state) - the removed frames are
    stationary, so the spliced sequence is continuous;
  * timestamps are regenerated uniform (frame_index / fps); frame_index and the global index re-flow;
  * meta/episodes and meta/stats.json are recomputed for state, action, timestamp, frame_index, index;
    image stats are copied (pixels are a subset of the source's).

VIDEOS ARE HARD-LINKED UNCUT and therefore do NOT match the regenerated timestamps. The variant is
only valid with `--dataset.video_backend=memmap`: the frame cache is derived from the source's
verified cache by row selection (an exact copy, re-verified here). A README in the output says so.

Usage: cap_dwell.py <src_root> <dst_root> [--cap 5] [--thresh 0.05] [--verify 200]
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--cap", type=int, default=5)
ap.add_argument("--thresh", type=float, default=0.05, help="deg/frame below which a joint is still")
ap.add_argument("--verify", type=int, default=200)
args = ap.parse_args()
SRC, DST, CAP = args.src, args.dst, args.cap
RECOMPUTE = ("observation.state", "action", "timestamp", "frame_index", "index")

info = json.load(open(f"{SRC}/meta/info.json"))
FPS = float(info["fps"])
os.makedirs(f"{DST}/data/chunk-000", exist_ok=True)
os.makedirs(f"{DST}/meta/episodes/chunk-000", exist_ok=True)

# ---- data: choose kept frames per episode
t = pq.read_table(f"{SRC}/data/chunk-000/file-000.parquet").to_pandas().sort_values("index")
S = np.stack(t["observation.state"].values).astype(np.float32)
ep_ids = t["episode_index"].values
keep_global = []             # source global indices kept, in order
per_ep = {}
for e in np.unique(ep_ids):
    rows = np.where(ep_ids == e)[0]
    s = S[rows]
    dj = np.abs(np.diff(s[:, :6], axis=0)).max(axis=1)
    dg = s[1:, 6] != s[:-1, 6]
    stationary = np.concatenate([[False], (dj < args.thresh) & ~dg])
    keep = np.ones(len(rows), bool)
    i = 0
    while i < len(rows):
        if stationary[i]:
            j = i
            while j + 1 < len(rows) and stationary[j + 1]:
                j += 1
            run_len = j - i + 1
            if run_len > CAP:
                keep[i : j + 1 - CAP] = False      # keep the LAST `CAP` frames of the run
            i = j + 1
        else:
            i += 1
    per_ep[int(e)] = (rows, keep)
    keep_global.extend(rows[keep].tolist())
keep_global = np.asarray(keep_global)
n_new = len(keep_global)
print(f"kept {n_new} / {len(t)} frames ({100 * (1 - n_new / len(t)):.1f} % removed), "
      f"{len(per_ep)} episodes, cap={CAP} thresh={args.thresh}")

new_state, new_action, new_ts, new_fi, new_ep = [], [], [], [], []
ep_meta = {}
cursor = 0
for e, (rows, keep) in per_ep.items():
    ks = S[rows[keep]]
    n = len(ks)
    act = np.vstack([ks[1:], ks[-1:]])              # action[t] = state[t+1], last = last state
    new_state.append(ks)
    new_action.append(act)
    new_ts.append(np.arange(n, dtype=np.float64) / FPS)
    new_fi.append(np.arange(n, dtype=np.int64))
    new_ep.append(np.full(n, e, dtype=np.int64))
    ep_meta[e] = (cursor, cursor + n, n)
    cursor += n
new_state = np.vstack(new_state)
new_action = np.vstack(new_action)
new_ts = np.concatenate(new_ts)
new_fi = np.concatenate(new_fi)
new_ep = np.concatenate(new_ep)
new_index = np.arange(n_new, dtype=np.int64)
D = new_state.shape[1]
tab = pa.table({
    "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(new_state.ravel(), pa.float32()), D),
    "action": pa.FixedSizeListArray.from_arrays(pa.array(new_action.ravel(), pa.float32()), D),
    "timestamp": pa.array(new_ts.astype(np.float32)),
    "frame_index": pa.array(new_fi),
    "episode_index": pa.array(new_ep),
    "index": pa.array(new_index),
    "task_index": pa.array(t["task_index"].values[keep_global].astype(np.int64)),
})
pq.write_table(tab, f"{DST}/data/chunk-000/file-000.parquet")

# ---- stats helpers (mirror the source's shapes: vector features -> arrays, scalars -> scalars)
def stats_for(x: np.ndarray) -> dict:
    x = x.astype(np.float64)
    q = lambda p: np.quantile(x, p, axis=0)  # noqa: E731
    return {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0),
            "count": np.array([len(x)]), "q01": q(0.01), "q10": q(0.10), "q50": q(0.50),
            "q90": q(0.90), "q99": q(0.99)}

def shaped_like(new_val, old_val):
    old = np.asarray(old_val)
    new = np.asarray(new_val)
    if old.ndim == 0:
        return float(new.ravel()[0]) if new.dtype.kind == "f" else int(new.ravel()[0])
    return new.reshape(old.shape).astype(old.dtype if old.dtype.kind in "fi" else np.float64).tolist()

cols = {"observation.state": new_state, "action": new_action, "timestamp": new_ts[:, None],
        "frame_index": new_fi[:, None], "index": new_index[:, None]}

# ---- meta/episodes
ep = pq.read_table(f"{SRC}/meta/episodes/chunk-000/file-000.parquet").to_pandas().set_index("episode_index")
video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
for e, (a, b, n) in ep_meta.items():
    ep.at[e, "length"] = n
    ep.at[e, "dataset_from_index"] = a
    ep.at[e, "dataset_to_index"] = b
    for vk in video_keys:
        ep.at[e, f"videos/{vk}/from_timestamp"] = 0.0
        ep.at[e, f"videos/{vk}/to_timestamp"] = n / FPS
    sl = slice(a, b)
    for feat, arr in cols.items():
        st = stats_for(arr[sl])
        for sname, sval in st.items():
            col = f"stats/{feat}/{sname}"
            if col in ep.columns:
                ep.at[e, col] = shaped_like(sval, ep.at[e, col])
ep = ep.reset_index()
pq.write_table(pa.Table.from_pandas(ep, preserve_index=False), f"{DST}/meta/episodes/chunk-000/file-000.parquet")

# ---- meta/stats.json, info.json, tasks
gstats = json.load(open(f"{SRC}/meta/stats.json"))
for feat, arr in cols.items():
    if feat in gstats:
        st = stats_for(arr)
        gstats[feat] = {k: shaped_like(st[k], v) if k in st else v for k, v in gstats[feat].items()}
json.dump(gstats, open(f"{DST}/meta/stats.json", "w"))
info["total_frames"] = int(n_new)
info["total_episodes"] = int(len(per_ep))
info["splits"] = {"train": f"0:{len(per_ep)}"}
json.dump(info, open(f"{DST}/meta/info.json", "w"), indent=2)
shutil.copy(f"{SRC}/meta/tasks.parquet", f"{DST}/meta/tasks.parquet")

# ---- videos: hard-link UNCUT (see module docstring)
for vk in video_keys:
    sdir, ddir = f"{SRC}/videos/{vk}/chunk-000", f"{DST}/videos/{vk}/chunk-000"
    os.makedirs(ddir, exist_ok=True)
    for f in sorted(os.listdir(sdir)):
        d = f"{ddir}/{f}"
        if not os.path.exists(d):
            try:
                os.link(os.path.realpath(f"{sdir}/{f}"), d)
            except OSError:  # cross-device (e.g. a scratch dst): a symlink keeps the same semantics
                os.symlink(os.path.realpath(f"{sdir}/{f}"), d)

# ---- frame cache: exact row selection from the source cache
src_cache = f"{SRC}/frame_cache"
if os.path.exists(f"{src_cache}/index.json"):
    spec = json.load(open(f"{src_cache}/index.json"))
    dst_cache = f"{DST}/frame_cache"
    os.makedirs(dst_cache, exist_ok=True)
    new_spec = {"fps": spec["fps"], "total_frames": int(n_new), "keys": {}}
    for key, shape in spec["keys"].items():
        N, H, W, C = shape
        src_mm = np.memmap(f"{src_cache}/{key}.uint8", dtype=np.uint8, mode="r", shape=(N, H, W, C))
        dst_mm = np.memmap(f"{dst_cache}/{key}.uint8", dtype=np.uint8, mode="w+", shape=(n_new, H, W, C))
        for a in range(0, n_new, 1024):
            idx = keep_global[a : a + 1024]
            dst_mm[a : a + len(idx)] = src_mm[idx]
        dst_mm.flush()
        new_spec["keys"][key] = [int(n_new), H, W, C]
        rng = np.random.default_rng(0)
        probe = rng.integers(0, n_new, args.verify)
        bad = sum(int(not np.array_equal(dst_mm[i], src_mm[keep_global[i]])) for i in probe)
        print(f"cache {key}: {n_new} rows, verify {args.verify} probes -> {bad} mismatches")
        assert bad == 0, "cache row selection mismatch"
    json.dump(new_spec, open(f"{dst_cache}/index.json", "w"))
else:
    print("no source frame_cache - videos are UNCUT, do not train on this variant without a cache")

open(f"{DST}/README.md", "w").write(
    f"# {os.path.basename(DST)}\n\nDerived from `{os.path.basename(SRC)}` by `analysis/cap_dwell.py` "
    f"(cap {CAP}, thresh {args.thresh} deg/frame): {len(t) - n_new} stationary frames removed, "
    f"{n_new} kept.\n\n**Videos are hard-linked and UNCUT; timestamps were regenerated uniform. "
    f"Only load this variant with `video_backend=memmap`** (frame_cache/ is exact and verified).\n")

# ---- self-check: parquet vs meta vs cache
chk = pq.read_table(f"{DST}/data/chunk-000/file-000.parquet").to_pandas()
epm = pq.read_table(f"{DST}/meta/episodes/chunk-000/file-000.parquet").to_pandas()
assert len(chk) == n_new == int(epm["length"].sum()) == info["total_frames"]
for _, r in epm.iterrows():
    seg = chk.iloc[int(r.dataset_from_index) : int(r.dataset_to_index)]
    assert (seg.episode_index.values == r.episode_index).all() and (seg.frame_index.values == np.arange(len(seg))).all()
    assert np.allclose(seg.timestamp.values, np.arange(len(seg)) / FPS, atol=1e-4)
a_chk = np.stack(chk["action"].values); s_chk = np.stack(chk["observation.state"].values)
inner = chk.groupby("episode_index").cumcount(ascending=False).values > 0
assert np.array_equal(a_chk[inner], s_chk[np.where(inner)[0] + 1]), "action[t] != state[t+1]"
print(f"OK -> {DST}")
