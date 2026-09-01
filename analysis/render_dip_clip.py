"""Render the pre-grasp lift-and-return window of one episode as an annotated clip.

Panels: wrist cam | overhead cam0 | side cam1 (zoomed on the tool, with the hover-height line).
Below: the tool-z profile with a moving cursor.
"""
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
ROOT = glob.glob("/home/zero/.cache/huggingface/hub/datasets--azorematter--dice_white_pnp_100/snapshots/*")[0]
FPS = 30
Z, GRIP = 8, 12
SIDE_CROP = (170, 350, 320, 560)          # y0, y1, x0, x1 on cam1 (full res 480x640)
HOVER_ROW = 304                            # tool tip row at hover, in the zoomed side panel

# ---------------------------------------------------------------- state
tab = pq.read_table(f"{ROOT}/data/chunk-000/file-000.parquet")
state = np.stack(tab["observation.state"].to_numpy(zero_copy_only=False)).astype(float)
epi, fi = tab["episode_index"].to_numpy(), tab["frame_index"].to_numpy()
m = epi == EP
s = state[m][np.argsort(fi[m])]
z, g = s[:, Z], s[:, GRIP]

r = {row["ep"]: row for row in json.load(open(f"{SCRATCH}/dip_rows.json"))}[EP]
t_align, t_dip, t_rec, t_close = r["t_align"], r["t_dip"], r["t_rec"], r["t_close"]
t_bottom2 = t_close + int(np.argmin(z[t_close:t_close + 60]))
start, end = max(0, t_align - 40), min(len(s) - 1, t_close + 90)
print(f"ep {EP}: align {t_align} dip {t_dip} rec {t_rec} close {t_close} bottom2 {t_bottom2} "
      f"| window {start}-{end} ({(end-start)/FPS:.1f}s)")

meta = pq.read_table(f"{ROOT}/meta/episodes/chunk-000/file-000.parquet").to_pandas()
mrow = meta[meta.episode_index == EP].iloc[0]


def grab(key):
    fidx = int(mrow[f"videos/{key}/file_index"])
    t0 = float(mrow[f"videos/{key}/from_timestamp"])
    c = av.open(f"{ROOT}/videos/{key}/chunk-000/file-{fidx:03d}.mp4")
    st = c.streams.video[0]
    st.thread_type = "AUTO"
    c.seek(int(max(0, t0 + start / FPS - 1.0) / st.time_base), stream=st)
    out = {}
    for fr in c.decode(st):
        k = int(round((float(fr.pts * st.time_base) - t0) * FPS))
        if start <= k <= end and k not in out:
            out[k] = fr.to_ndarray(format="bgr24")
        if k > end:
            break
    c.close()
    print(f"  {key}: {len(out)} frames")
    return out


wrist = grab("observation.images.gripper")
board = grab("observation.images.cam0")
side = grab("observation.images.cam1")

# ---------------------------------------------------------------- strip plot
PW, PH, HEAD, IMH, IMW = 1440, 200, 46, 360, 480
W = IMW * 3
ts = np.arange(start, end + 1) / FPS
fig = plt.figure(figsize=(W / 100, PH / 100), dpi=100)
ax = fig.add_axes([0.04, 0.29, 0.945, 0.58])
ax.plot(ts, z[start:end + 1], color="#2d7ff9", lw=2.4)
zmax = z[start:end + 1].max()
for t_, lab, col, dy, ha in ((t_dip, "1  touch-down", "#e8710a", 3.5, "center"),
                             (t_rec, "2  back up +%.0f mm" % r["lift_back"], "#e8710a", 3.5, "center"),
                             (t_close, "3  gripper CLOSE, still %.0f mm up" % r["lift_back"], "#d93025", 15, "right"),
                             (t_bottom2, "4  drop + grasp", "#188038", 3.5, "left")):
    ax.axvline(t_ / FPS, color=col, lw=1.5, ls="--", alpha=.9)
    ax.text(t_ / FPS + (0.06 if ha == "left" else -0.06 if ha == "right" else 0), zmax + dy, lab,
            color=col, fontsize=9, ha=ha, va="bottom", clip_on=False)
ax.axhline(z[t_rec], color="#9aa0a6", lw=1, ls=":")
ax.set_ylabel("tool z (mm)", fontsize=10)
ax.set_xlabel("episode time (s)", fontsize=10)
ax.tick_params(labelsize=9)
ax.grid(alpha=.25, lw=.6)
ax.margins(x=0.01)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.canvas.draw()
plot_img = cv2.resize(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR), (W, PH))
x0, x1 = ax.get_position().x0 * W, ax.get_position().x1 * W
plt.close(fig)


def phase(k):
    if k < t_align:
        return "approach + XY alignment", (200, 200, 200)
    if k <= t_dip:
        return "1. descending to the dice", (10, 113, 232)
    if k < t_rec:
        return "2. LIFTING BACK UP %.0f mm - no grasp" % r["lift_back"], (10, 113, 232)
    if k < t_close:
        return "3. holding %.0f mm above the dice" % r["lift_back"], (170, 170, 170)
    if k < t_bottom2:
        return "4. gripper CLOSED, dropping onto the dice", (37, 70, 217)
    return "5. grasped - lifting away", (56, 128, 24)


y0, y1, cx0, cx1 = SIDE_CROP
frames = []
for k in range(start, end + 1):
    a, b, c3 = wrist.get(k), board.get(k), side.get(k)
    if a is None or b is None or c3 is None:
        continue
    canvas = np.full((HEAD + IMH + PH, W, 3), 24, np.uint8)
    canvas[HEAD:HEAD + IMH, 0:IMW] = cv2.resize(a, (IMW, IMH))
    canvas[HEAD:HEAD + IMH, IMW:2 * IMW] = cv2.resize(b, (IMW, IMH))
    zoom = cv2.resize(c3[y0:y1, cx0:cx1], (IMW, IMH), interpolation=cv2.INTER_CUBIC)
    for xs in range(0, IMW, 22):            # dashed hover-height reference
        cv2.line(zoom, (xs, HOVER_ROW), (xs + 11, HOVER_ROW), (120, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(zoom, "tool height at hover", (IMW - 232, HOVER_ROW - 8),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (120, 220, 255), 1, cv2.LINE_AA)
    canvas[HEAD:HEAD + IMH, 2 * IMW:] = zoom
    for xs in (IMW, 2 * IMW):
        cv2.line(canvas, (xs, HEAD), (xs, HEAD + IMH), (24, 24, 24), 2)
    canvas[HEAD + IMH:] = plot_img

    lab, col = phase(k)
    grip = "OPEN" if g[k] > .5 else "CLOSED"
    cv2.putText(canvas, f"ep {EP}   f{k:4d}   t={k/FPS:5.2f}s    z={z[k]:7.1f} mm    gripper: {grip}",
                (14, 31), cv2.FONT_HERSHEY_SIMPLEX, .68, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, lab, (760, 31), cv2.FONT_HERSHEY_SIMPLEX, .68, col, 2, cv2.LINE_AA)
    for x_, t_ in ((14, "wrist cam"), (IMW + 14, "overhead cam0"), (2 * IMW + 14, "side cam1 (zoom)")):
        cv2.putText(canvas, t_, (x_, HEAD + 26), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2, cv2.LINE_AA)
    cx = int(x0 + (x1 - x0) * (k - start) / max(1, end - start))
    cv2.line(canvas, (cx, HEAD + IMH + 22), (cx, HEAD + IMH + PH - 44), (60, 60, 220), 2)
    frames.append(canvas)

print(f"  composited {len(frames)} frames")


def encode(path, fps):
    c = av.open(path, "w")
    st = c.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = W, frames[0].shape[0], "yuv420p"
    st.options = {"crf": "18", "preset": "medium"}
    for f in frames:
        c.mux(st.encode(av.VideoFrame.from_ndarray(f, format="bgr24")))
    c.mux(st.encode())
    c.close()
    print("wrote", path)


encode(f"{SCRATCH}/ep{EP}_pregrasp_dip_realtime.mp4", 30)
encode(f"{SCRATCH}/ep{EP}_pregrasp_dip_slowmo.mp4", 10)
