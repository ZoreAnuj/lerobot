"""All 99 episodes' tool-z profiles, aligned on the first (no-grasp) touch-down."""
import glob
import json

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = "/tmp/claude-1000/-home-zero-matter-imle/1ee9319f-ee95-4f6f-8a2d-25bcbccae2d3/scratchpad"
ROOT = glob.glob("/home/zero/.cache/huggingface/hub/datasets--azorematter--dice_white_pnp_100/snapshots/*")[0]
FPS = 30

tab = pq.read_table(f"{ROOT}/data/chunk-000/file-000.parquet")
sa = np.stack(tab["observation.state"].to_numpy(zero_copy_only=False)).astype(float)
epi, fi = tab["episode_index"].to_numpy(), tab["frame_index"].to_numpy()
rows = json.load(open(f"{SCRATCH}/dip_rows.json"))

PRE, POST = 3 * FPS, 6 * FPS
curves, close_pts = [], []
for r in rows:
    m = epi == r["ep"]
    z = sa[m][np.argsort(fi[m])][:, 8]
    t0 = r["t_dip"]
    seg = np.full(PRE + POST + 1, np.nan)
    lo, hi = max(0, t0 - PRE), min(len(z) - 1, t0 + POST)
    seg[PRE - (t0 - lo): PRE + (hi - t0) + 1] = z[lo:hi + 1]
    curves.append(seg)
    close_pts.append(((r["t_close"] - t0) / FPS, z[r["t_close"]]))
C = np.array(curves)
t = np.arange(-PRE, POST + 1) / FPS

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=110,
                              gridspec_kw={"width_ratios": [2.0, 1]})
for c in C:
    ax.plot(t, c, color="#2d7ff9", alpha=.12, lw=1.1)
nmed = int(2.4 * FPS) + PRE
ax.plot(t[:nmed], np.nanmedian(C[:, :nmed], 0), color="#0b57d0", lw=2.8,
        label="median (up to the hold, where episodes diverge)")
cx, cy = zip(*close_pts)
ax.scatter(cx, cy, s=26, color="#d93025", zorder=5, label="gripper CLOSE (one per episode)")
ax.axhline(-91.48, color="#9aa0a6", lw=1, ls=":")
ax.axhline(-126.58, color="#9aa0a6", lw=1, ls=":")
ax.text(5.9, -88.5, "hover / retract height  −91.5 mm", fontsize=8.5, color="#5f6368", ha="right")
ax.text(5.9, -124.0, "taught grasp depth  −126.6 mm", fontsize=8.5, color="#5f6368", ha="right")
ax.annotate("touch-down #1\ngripper still OPEN", xy=(0, -126.6), xytext=(-2.9, -112),
            fontsize=9, color="#e8710a", arrowprops=dict(arrowstyle="->", color="#e8710a", lw=1.3))
ax.annotate("back up 35 mm,\nthen wait 0.7–4.1 s", xy=(1.15, -91.5), xytext=(1.5, -70),
            fontsize=9, color="#e8710a", arrowprops=dict(arrowstyle="->", color="#e8710a", lw=1.3))
ax.annotate("close, then drop\nback onto the dice", xy=(3.2, -126.0), xytext=(3.5, -108),
            fontsize=9, color="#d93025", arrowprops=dict(arrowstyle="->", color="#d93025", lw=1.3))
ax.set_xlabel("time relative to the first touch-down (s)")
ax.set_ylabel("tool z (mm)")
ax.set_ylim(-140, -55)
ax.set_xlim(-3, 6)
ax.set_title("99/99 demos: down to the dice, back up 35 mm, wait, then close and drop again", fontsize=11)
ax.grid(alpha=.25, lw=.6)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

dip = np.array([r["dip_depth"] for r in rows])
hold = np.array([(r["t_close"] - r["t_rec"]) / FPS for r in rows])
ax2.scatter(hold, dip, s=24, color="#0b57d0", alpha=.75, edgecolor="white", linewidth=.6)
ax2.set_xlabel("hold at hover before the close (s)")
ax2.set_ylabel("first touch-down, mm below hover")
ax2.set_title("wasted motion per episode: 2.7 s on average", fontsize=11)
ax2.grid(alpha=.25, lw=.6)
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)

fig.tight_layout()
fig.savefig(f"{SCRATCH}/pregrasp_dip_all_episodes.png")
print("wrote", f"{SCRATCH}/pregrasp_dip_all_episodes.png")
