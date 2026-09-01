"""Choose, per episode, the frame range to excise: the pointless down-and-back-up before the grasp.

Keep everything up to the frame where the arm is still parked at the hover height (before it
dives at the dice), then jump straight to the frame where it is back at that same hover pose.
The wrist yaw alignment happens AFTER the return, so it is preserved.

Writes cut_points.json: {ep: {"cut_from": a+1, "cut_to": b-1, "keep": [[0,a],[b,n-1]], ...}}
"""
import glob
import json

import numpy as np
import pyarrow.parquet as pq

SCRATCH = "/tmp/claude-1000/-home-zero-matter-imle/1ee9319f-ee95-4f6f-8a2d-25bcbccae2d3/scratchpad"
ROOT = glob.glob("/home/zero/.cache/huggingface/hub/datasets--azorematter--dice_white_pnp_100/snapshots/*")[0]
FPS = 30
NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "x", "y", "z", "W", "P", "R", "grip"]
TOL = 0.30          # max allowed splice discontinuity, in deg (joints/WPR) or mm (xyz)

tab = pq.read_table(f"{ROOT}/data/chunk-000/file-000.parquet")
sa = np.stack(tab["observation.state"].to_numpy(zero_copy_only=False)).astype(float)
epi, fi = tab["episode_index"].to_numpy(), tab["frame_index"].to_numpy()
rows = {r["ep"]: r for r in json.load(open(f"{SCRATCH}/dip_rows.json"))}

out, skipped, worst_dims = {}, [], []
for e in sorted(rows):
    m = epi == e
    s = sa[m][np.argsort(fi[m])]
    z = s[:, 8]
    r = rows[e]
    t_dip, t_close = r["t_dip"], r["t_close"]

    # descent start: last frame before the dip at which z is still flat
    k = t_dip
    while k > 0 and z[k - 1] > z[k] + 1e-3:
        k -= 1
    a0 = k - 1                                  # last parked frame
    # return: first frame after the dip back within 0.1 mm of the parked height
    b0 = t_dip
    while b0 < t_close and z[b0] < z[a0] - 0.1:
        b0 += 1

    cands = []
    for a in range(max(0, a0 - 20), a0 + 1):
        for b in range(b0, min(t_close, b0 + 20) + 1):
            d = np.abs(s[b, :12] - s[a, :12])
            if s[b, 12] != s[a, 12]:            # gripper bit must not change across the splice
                continue
            if d.max() > TOL:
                continue
            cands.append((b - a, float(d.max()), a, b, d))
    if not cands:
        skipped.append(e)
        continue
    longest = max(c[0] for c in cands)
    # among near-maximal cuts, take the smoothest splice
    _, resid, a, b, dvec = min((c for c in cands if c[0] >= longest - 8), key=lambda c: c[1])
    worst_dims.append(dvec)
    out[int(e)] = dict(a=int(a), b=int(b), removed=int(b - a - 1), resid=float(resid),
                       n=int(len(s)), t_dip=int(t_dip), t_close=int(t_close))

rem = np.array([v["removed"] for v in out.values()])
res = np.array([v["resid"] for v in out.values()])
print(f"episodes cut: {len(out)}/99   skipped (no clean splice): {skipped}")
print(f"frames removed: total {rem.sum()} of {len(sa)} ({100*rem.sum()/len(sa):.1f}%)  "
      f"per episode mean {rem.mean():.0f} ({rem.mean()/FPS:.2f}s) min {rem.min()} max {rem.max()}")
print(f"splice residual (worst dim): mean {res.mean():.4f}  max {res.max():.4f}  (tolerance {TOL})")
D = np.array(worst_dims)
print("per-dim splice discontinuity:  " + "  ".join(f"{n} {D[:,i].max():.3f}" for i, n in enumerate(NAMES[:12])))
print("\nep    n   keep[0..a]  resume@b  removed  resid")
for e in sorted(out)[:10]:
    v = out[e]
    print(f"{e:3d} {v['n']:5d} {v['a']:10d} {v['b']:9d} {v['removed']:8d} {v['resid']:7.3f}")

json.dump(out, open(f"{SCRATCH}/cut_points.json", "w"))
print(f"\nwrote {SCRATCH}/cut_points.json")
