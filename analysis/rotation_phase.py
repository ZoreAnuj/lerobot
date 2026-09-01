"""Where does the wrist rotation happen? Phase-by-phase deltas around the pre-grasp excursion."""
import glob
import json

import numpy as np
import pyarrow.parquet as pq

SCRATCH = "/tmp/claude-1000/-home-zero-matter-imle/1ee9319f-ee95-4f6f-8a2d-25bcbccae2d3/scratchpad"
ROOT = glob.glob("/home/zero/.cache/huggingface/hub/datasets--azorematter--dice_white_pnp_100/snapshots/*")[0]
FPS = 30
NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "x", "y", "z", "W", "P", "R", "grip"]
WATCH = [3, 5, 11]        # J4, J6, R

tab = pq.read_table(f"{ROOT}/data/chunk-000/file-000.parquet")
sa = np.stack(tab["observation.state"].to_numpy(zero_copy_only=False)).astype(float)
epi, fi = tab["episode_index"].to_numpy(), tab["frame_index"].to_numpy()
rows = {r["ep"]: r for r in json.load(open(f"{SCRATCH}/dip_rows.json"))}

phases = {"approach->dip (descent 1)": [], "dip->rec (retract)": [],
          "rec->close (hold at hover)": [], "close->bottom2 (drop)": []}
splice_align_rec, splice_dip_bot = [], []
for e in sorted(rows):
    m = epi == e
    s = sa[m][np.argsort(fi[m])]
    r = rows[e]
    ta, td, tr, tc = r["t_align"], r["t_dip"], r["t_rec"], r["t_close"]
    tb = tc + int(np.argmin(s[tc:tc + 60, 8]))
    for (k, (a, b)) in zip(phases, ((ta, td), (td, tr), (tr, tc), (tc, tb))):
        phases[k].append(np.abs(s[b] - s[a]))
    splice_align_rec.append(np.abs(s[tr] - s[ta]))
    splice_dip_bot.append(np.abs(s[tb] - s[td]))

print("rotation magnitude per phase (deg), over 99 episodes")
print(f"{'phase':30s} " + " ".join(f"{NAMES[i]:>7s}mean {NAMES[i]:>5s}max" for i in WATCH))
for k, v in phases.items():
    V = np.array(v)
    print(f"{k:30s} " + " ".join(f"{V[:,i].mean():11.2f} {V[:,i].max():8.2f}" for i in WATCH))

for label, arr in (("cut A: splice t_align -> t_rec (drop the down-up, keep the hold)", splice_align_rec),
                   ("cut B: splice t_dip -> t_bottom2 (drop retract+hold)", splice_dip_bot)):
    A = np.array(arr)
    print(f"\n{label}")
    print("  dim     mean      max     #eps>1deg/mm")
    for i, n in enumerate(NAMES[:12]):
        print(f"  {n:4s} {A[:,i].mean():8.3f} {A[:,i].max():8.3f}   {int((A[:,i]>1).sum()):3d}")

# how much total rotation is there at all, per episode?
tot = []
for e in sorted(rows):
    m = epi == e
    s = sa[m][np.argsort(fi[m])]
    r = rows[e]
    tot.append(abs(s[r["t_close"], 11] - s[r["t_align"], 11]))
tot = np.array(tot)
print(f"\ntool roll R change between alignment and the close: mean {tot.mean():.1f} deg, "
      f"max {tot.max():.1f}, >5 deg in {int((tot>5).sum())}/99 episodes, >1 deg in {int((tot>1).sum())}/99")
