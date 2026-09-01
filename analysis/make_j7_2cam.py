"""Derive the training variant from a 13-dim / 4-camera LeRobotDataset.

state & action -> [J1..J6, gripper]; cameras -> gripper + cam0 only. Videos are hard-linked
(no re-encode), so the derived set stays bit-identical to its parent.

Usage: make_j7_2cam.py <src_root> <dst_root>
"""
import json
import os
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC, DST = sys.argv[1], sys.argv[2]
DIMS = [0, 1, 2, 3, 4, 5, 12]                 # J1..J6 + gripper
NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "gripper"]
KEEP_CAMS = ["observation.images.gripper", "observation.images.cam0"]
DROP_CAMS = ["observation.images.cam1", "observation.images.cam2"]

os.makedirs(f"{DST}/data/chunk-000", exist_ok=True)
os.makedirs(f"{DST}/meta/episodes/chunk-000", exist_ok=True)

# ---- data
t = pq.read_table(f"{SRC}/data/chunk-000/file-000.parquet")
s = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)[:, DIMS]
a = np.stack(t["action"].to_numpy(zero_copy_only=False)).astype(np.float32)[:, DIMS]
tab = pa.table({
    "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(s.ravel(), pa.float32()), len(DIMS)),
    "action": pa.FixedSizeListArray.from_arrays(pa.array(a.ravel(), pa.float32()), len(DIMS)),
    **{c: t[c] for c in ("timestamp", "frame_index", "episode_index", "index", "task_index")},
})
pq.write_table(tab, f"{DST}/data/chunk-000/file-000.parquet")

# ---- videos: hard-link the kept cameras
for cam in KEEP_CAMS:
    src_dir = f"{SRC}/videos/{cam}/chunk-000"
    dst_dir = f"{DST}/videos/{cam}/chunk-000"
    os.makedirs(dst_dir, exist_ok=True)
    for f in sorted(os.listdir(src_dir)):
        d = f"{dst_dir}/{f}"
        if not os.path.exists(d):
            os.link(os.path.realpath(f"{src_dir}/{f}"), d)

# ---- meta
shutil.copy(f"{SRC}/meta/tasks.parquet", f"{DST}/meta/tasks.parquet")

info = json.load(open(f"{SRC}/meta/info.json"))
for cam in DROP_CAMS:
    info["features"].pop(cam, None)
for k in ("observation.state", "action"):
    info["features"][k]["shape"] = [len(DIMS)]
    if "names" in info["features"][k]:
        info["features"][k]["names"] = list(NAMES)
json.dump(info, open(f"{DST}/meta/info.json", "w"), indent=2)

st = json.load(open(f"{SRC}/meta/stats.json"))
for cam in DROP_CAMS:
    st.pop(cam, None)
for k in ("observation.state", "action"):
    st[k] = {sk: (np.asarray(v)[DIMS].tolist() if np.asarray(v).size >= 13 else v)
             for sk, v in st[k].items()}
json.dump(st, open(f"{DST}/meta/stats.json", "w"))

ep = pq.read_table(f"{SRC}/meta/episodes/chunk-000/file-000.parquet").to_pandas()
drop = [c for c in ep.columns if any(c.startswith(f"videos/{d}/") or c.startswith(f"stats/{d}/")
                                     for d in DROP_CAMS)]
ep = ep.drop(columns=drop)
for c in ep.columns:
    if c.startswith("stats/observation.state/") or c.startswith("stats/action/"):
        ep[c] = ep[c].map(lambda v: np.asarray(v)[DIMS] if np.asarray(v).size >= 13 else v)
pq.write_table(pa.Table.from_pandas(ep, preserve_index=False),
               f"{DST}/meta/episodes/chunk-000/file-000.parquet")

print(f"{DST}: {len(tab)} frames, {len(ep)} episodes, state/action dim {len(DIMS)}, "
      f"cameras {[c.split('.')[-1] for c in KEEP_CAMS]}")
