"""Characterise the pre-grasp vertical excursion (down-up-hold-down) in dice_white_pnp_100."""
import json
import numpy as np
import pyarrow.parquet as pq

P = ("/home/zero/.cache/huggingface/hub/datasets--azorematter--dice_white_pnp_100/"
     "snapshots/dcb48688dc988d2731b7051defb3f3c809c3d09f/data/chunk-000/file-000.parquet")
FPS = 30
X, Y, Z, GRIP = 6, 7, 8, 12
SCRATCH = "/tmp/claude-1000/-home-zero-matter-imle/1ee9319f-ee95-4f6f-8a2d-25bcbccae2d3/scratchpad"

tab = pq.read_table(P)
state = np.stack(tab["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
epi = tab["episode_index"].to_numpy()
fi = tab["frame_index"].to_numpy()


def episode(e):
    m = epi == e
    return state[m][np.argsort(fi[m])]


rows = []
for e in np.unique(epi):
    s = episode(e)
    g, x, y, z = s[:, GRIP], s[:, X], s[:, Y], s[:, Z]
    close = np.where((g[:-1] > 0.5) & (g[1:] <= 0.5))[0]
    if not len(close):
        continue
    tc = int(close[0]) + 1                       # first frame with gripper CLOSED

    # alignment = last frame before the close where XY still moves appreciably
    xy_sp = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])) * FPS   # mm/s
    moving = np.where(xy_sp[:tc] > 2.0)[0]
    t_align = int(moving[-1]) + 1 if len(moving) else 0

    seg = z[t_align:tc + 1]
    if len(seg) < 5:
        continue
    z_hover = float(np.median(seg[:max(3, len(seg) // 10)]))

    # deepest point of the excursion between alignment and the close
    i_dip = int(np.argmin(seg))
    z_dip = float(seg[i_dip])
    dip_depth = z_hover - z_dip                       # mm below the hover plateau
    # recovery: how far back up it comes after the dip, before the close
    after = seg[i_dip:]
    i_rec = int(np.argmax(after))
    z_rec = float(after[i_rec])
    lift_back = z_rec - z_dip
    t_dip = t_align + i_dip
    t_rec = t_dip + i_rec

    # final descent: after recovery, does it go down again before/at the close?
    z_close = float(z[tc])
    # lowest point reached in the 1.5 s AFTER the close (the actual grasp depth)
    z_grasp = float(z[tc:min(len(z), tc + int(1.5 * FPS))].min())

    rows.append(dict(
        ep=int(e), n=len(s), t_align=t_align, t_dip=t_dip, t_rec=t_rec, t_close=tc,
        z_hover=z_hover, z_dip=z_dip, z_rec=z_rec, z_close=z_close, z_grasp=z_grasp,
        dip_depth=dip_depth, lift_back=lift_back,
        dwell_after_rec_s=(tc - t_rec) / FPS,
        excursion_s=(t_rec - t_dip) / FPS,
        align_to_close_s=(tc - t_align) / FPS,
        dip_vs_grasp=z_dip - z_grasp,
    ))

dd = np.array([r["dip_depth"] for r in rows])
lb = np.array([r["lift_back"] for r in rows])
print(f"episodes analysed: {len(rows)}")
print("dip below hover (mm)  : mean %.1f  median %.1f  p5 %.1f  p95 %.1f  min %.1f  max %.1f"
      % (dd.mean(), np.median(dd), np.percentile(dd, 5), np.percentile(dd, 95), dd.min(), dd.max()))
print("lift back up (mm)     : mean %.1f  median %.1f  p5 %.1f  p95 %.1f"
      % (lb.mean(), np.median(lb), np.percentile(lb, 5), np.percentile(lb, 95)))
for thr in (5, 10, 20, 25, 30):
    print(f"  dip > {thr:2d} mm AND lift back > {thr:2d} mm : "
          f"{int(((dd > thr) & (lb > thr)).sum())}/{len(rows)} episodes")
for k in ("excursion_s", "dwell_after_rec_s", "align_to_close_s"):
    v = np.array([r[k] for r in rows])
    print(f"{k:20s}: mean {v.mean():.2f}s median {np.median(v):.2f}s p95 {np.percentile(v,95):.2f}s")
gz = np.array([r["z_grasp"] for r in rows]); dz = np.array([r["z_dip"] for r in rows])
print("z at dip bottom  mean %.1f mm | z at true grasp depth mean %.1f mm | difference mean %.1f mm"
      % (dz.mean(), gz.mean(), (dz - gz).mean()))

print("\nep  align  dip  rec  close |  z_hover  z_dip  z_rec z_close z_grasp | dipmm liftmm exc_s dwell_s")
for r in rows[:20]:
    print("%3d %5d %5d %5d %5d | %8.1f %6.1f %6.1f %6.1f %7.1f | %5.1f %6.1f %5.2f %6.2f" % (
        r["ep"], r["t_align"], r["t_dip"], r["t_rec"], r["t_close"], r["z_hover"], r["z_dip"],
        r["z_rec"], r["z_close"], r["z_grasp"], r["dip_depth"], r["lift_back"],
        r["excursion_s"], r["dwell_after_rec_s"]))

json.dump(rows, open(f"{SCRATCH}/dip_rows.json", "w"))
print(f"\nwrote {SCRATCH}/dip_rows.json")
