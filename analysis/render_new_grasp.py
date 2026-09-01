"""Render the grasp window of the rebuilt (nodip) dataset, with the splice frame marked."""
import glob
import json
import sys

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EP = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SCRATCH = "/tmp/claude-1000/-home-zero-matter-imle/1ee9319f-ee95-4f6f-8a2d-25bcbccae2d3/scratchpad"
NEW = "/home/zero/matter/imle/datasets/dice_white_pnp_nodip"
FPS, IMW, IMH, HEAD, PH = 30, 480, 360, 46, 200
SIDE_CROP = (170, 350, 320, 560)
HOVER_ROW = 304

cuts = {int(k): v for k, v in json.load(open(f"{SCRATCH}/cut_points.json")).items()}[EP]
t = pq.read_table(f"{NEW}/data/chunk-000/file-000.parquet")
s = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(float)
epi, fi = t["episode_index"].to_numpy(), t["frame_index"].to_numpy()
m = epi == EP
s = s[m][np.argsort(fi[m])]
z, g, R = s[:, 8], s[:, 12], s[:, 11]

splice = cuts["a"]                                   # last frame before the removed span
t_close = int(np.where((g[:-1] > .5) & (g[1:] <= .5))[0][0]) + 1
t_bot = t_close + int(np.argmin(z[t_close:t_close + 60]))
start, end = max(0, splice - 60), min(len(s) - 1, t_bot + 60)
print(f"ep {EP}: splice at f{splice}, close f{t_close}, bottom f{t_bot}, window {start}-{end}")


def grab(key):
    c = av.open(f"{NEW}/videos/{key}/chunk-000/file-{EP:03d}.mp4")
    st = c.streams.video[0]
    st.thread_type = "AUTO"
    c.seek(int(max(0, start / FPS - 1.0) / st.time_base), stream=st)
    out = {}
    for fr in c.decode(st):
        k = int(round(float(fr.pts * st.time_base) * FPS))
        if start <= k <= end and k not in out:
            out[k] = fr.to_ndarray(format="bgr24")
        if k > end:
            break
    c.close()
    return out


wrist, side = grab("observation.images.gripper"), grab("observation.images.cam1")
W = IMW * 2
ts = np.arange(start, end + 1) / FPS
fig = plt.figure(figsize=(W / 100, PH / 100), dpi=100)
ax = fig.add_axes([0.06, 0.29, 0.92, 0.56])
ax.plot(ts, z[start:end + 1], color="#2d7ff9", lw=2.4, label="tool z")
ax2 = ax.twinx()
ax2.plot(ts, R[start:end + 1], color="#188038", lw=1.6, ls="-", label="tool roll R")
ax2.set_ylabel("roll R (deg)", fontsize=9, color="#188038")
ax2.tick_params(labelsize=8, colors="#188038")
for t_, lab, col in ((splice, "splice (dive removed)", "#9334e6"),
                     (t_close, "gripper CLOSE", "#d93025"),
                     (t_bot, "on the dice", "#188038")):
    ax.axvline(t_ / FPS, color=col, lw=1.5, ls="--", alpha=.9)
    ax.text(t_ / FPS, z[start:end + 1].max() + 3, lab, color=col, fontsize=9, ha="center",
            va="bottom", clip_on=False)
ax.set_ylabel("tool z (mm)", fontsize=10)
ax.set_xlabel("episode time (s)", fontsize=10)
ax.tick_params(labelsize=9)
ax.grid(alpha=.25, lw=.6)
ax.margins(x=0.01)
for sp in ("top",):
    ax.spines[sp].set_visible(False)
fig.canvas.draw()
plot_img = cv2.resize(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR), (W, PH))
px0, px1 = ax.get_position().x0 * W, ax.get_position().x1 * W
plt.close(fig)

y0, y1, cx0, cx1 = SIDE_CROP
frames = []
for k in range(start, end + 1):
    a, b = wrist.get(k), side.get(k)
    if a is None or b is None:
        continue
    canvas = np.full((HEAD + IMH + PH, W, 3), 24, np.uint8)
    canvas[HEAD:HEAD + IMH, :IMW] = cv2.resize(a, (IMW, IMH))
    zoom = cv2.resize(b[y0:y1, cx0:cx1], (IMW, IMH), interpolation=cv2.INTER_CUBIC)
    for xs in range(0, IMW, 22):
        cv2.line(zoom, (xs, HOVER_ROW), (xs + 11, HOVER_ROW), (120, 220, 255), 1, cv2.LINE_AA)
    canvas[HEAD:HEAD + IMH, IMW:] = zoom
    canvas[HEAD + IMH:] = plot_img
    lab, col = (("parked at hover, aligned", (200, 200, 200)) if k < t_close else
                ("gripper CLOSED -> dropping", (37, 70, 217)) if k < t_bot else
                ("grasped, lifting", (56, 128, 24)))
    if k == splice:
        lab, col = "<< splice: the dive-and-return used to be here >>", (230, 100, 230)
    cv2.putText(canvas, f"NODIP  ep {EP}  f{k:4d}  t={k/FPS:5.2f}s  z={z[k]:7.1f}  R={R[k]:6.1f}  "
                        f"grip {'OPEN' if g[k] > .5 else 'CLOSED'}",
                (14, 31), cv2.FONT_HERSHEY_SIMPLEX, .56, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, lab, (620, 31), cv2.FONT_HERSHEY_SIMPLEX, .56, col, 2, cv2.LINE_AA)
    cx = int(px0 + (px1 - px0) * (k - start) / max(1, end - start))
    cv2.line(canvas, (cx, HEAD + IMH + 22), (cx, HEAD + IMH + PH - 44), (60, 60, 220), 2)
    frames.append(canvas)

for path, fps in ((f"{SCRATCH}/ep{EP}_nodip_grasp_slowmo.mp4", 10),):
    c = av.open(path, "w")
    st = c.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = W, frames[0].shape[0], "yuv420p"
    st.options = {"crf": "18", "preset": "medium"}
    for f in frames:
        c.mux(st.encode(av.VideoFrame.from_ndarray(f, format="bgr24")))
    c.mux(st.encode())
    c.close()
    print("wrote", path, len(frames), "frames")
