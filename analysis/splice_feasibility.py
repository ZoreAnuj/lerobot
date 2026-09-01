"""Check whether the pre-grasp excursion can be excised with a continuous splice.

Proposed cut: keep ... t_dip (arm at grasp depth, gripper OPEN), then jump straight to
t_bottom2 (arm at grasp depth, gripper CLOSED), dropping everything between.
Also checks the place side for the same artefact.
"""
import glob
import json

import numpy as np
import pyarrow.parquet as pq

SCRATCH = "/tmp/claude-1000/-home-zero-matter-imle/1ee9319f-ee95-4f6f-8a2d-25bcbccae2d3/scratchpad"
ROOT = glob.glob("/home/zero/.cache/huggingface/hub/datasets--azorematter--dice_white_pnp_100/snapshots/*")[0]
FPS = 30
NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "x", "y", "z", "W", "P", "R", "grip"]

tab = pq.read_table(f"{ROOT}/data/chunk-000/file-000.parquet")
sa = np.stack(tab["observation.state"].to_numpy(zero_copy_only=False)).astype(float)
epi, fi = tab["episode_index"].to_numpy(), tab["frame_index"].to_numpy()
rows = {r["ep"]: r for r in json.load(open(f"{SCRATCH}/dip_rows.json"))}

jump, cut_len, place = [], [], []
for e in sorted(rows):
    m = epi == e
    s = sa[m][np.argsort(fi[m])]
    z, g = s[:, 8], s[:, 12]
    r = rows[e]
    t_dip = r["t_dip"]
    t_bottom2 = r["t_close"] + int(np.argmin(z[r["t_close"]:r["t_close"] + 60]))
    jump.append(np.abs(s[t_bottom2] - s[t_dip]))
    cut_len.append(t_bottom2 - t_dip - 1)

    # ---- place side: gripper OPEN transition (release)
    op = np.where((g[:-1] <= .5) & (g[1:] > .5))[0]
    if len(op):
        tr = int(op[0]) + 1
        lo = max(0, tr - 6 * FPS)
        zz = z[lo:tr + 1]
        # a retract-and-return before the release would show up as a local max between
        # the first arrival at the release depth and the release itself
        z_rel = z[tr]
        at = np.where(zz <= z_rel + 2)[0]
        t_first = lo + int(at[0]) if len(at) else tr
        place.append((tr - t_first) / FPS, )
        if e < 6:
            seg = z[t_first:tr + 1]
            print(f"  ep {e}: release at f{tr} (z {z_rel:.1f}); first at that depth f{t_first} "
                  f"({(tr-t_first)/FPS:.2f}s earlier); max in between {seg.max():.1f} "
                  f"(excursion {seg.max()-z_rel:.1f} mm)")

J = np.array(jump)
print("\nsplice discontinuity (|state[t_bottom2] - state[t_dip]|), over 99 episodes:")
print("  dim      max      mean")
for i, n in enumerate(NAMES):
    print(f"  {n:4s} {J[:,i].max():8.3f} {J[:,i].mean():9.3f}")
cl = np.array(cut_len)
print(f"\nframes removed per episode: mean {cl.mean():.0f} median {np.median(cl):.0f} "
      f"min {cl.min()} max {cl.max()}  -> total {cl.sum()} of {len(sa)} ({100*cl.sum()/len(sa):.1f}%)")
pl = np.array([p[0] for p in place])
print(f"\nplace side: time from first reaching release depth to the release itself: "
      f"mean {pl.mean():.2f}s median {np.median(pl):.2f}s max {pl.max():.2f}s (n={len(pl)})")
