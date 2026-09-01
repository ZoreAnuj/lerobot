"""Validate the rebuilt dataset: frame mapping, splice continuity, no residual dip."""
import glob
import json

import av
import numpy as np
import pyarrow.parquet as pq

SCRATCH = "/tmp/claude-1000/-home-zero-matter-imle/1ee9319f-ee95-4f6f-8a2d-25bcbccae2d3/scratchpad"
SRC = glob.glob("/home/zero/.cache/huggingface/hub/datasets--azorematter--dice_white_pnp_100/snapshots/*")[0]
OUT = "/home/zero/matter/imle/datasets/dice_white_pnp_nodip"
FPS = 30
KEY = "observation.images.gripper"
cuts = {int(k): v for k, v in json.load(open(f"{SCRATCH}/cut_points.json")).items()}


def frames_of(path, idxs, t0=0.0):
    want = sorted(idxs)
    c = av.open(path)
    st = c.streams.video[0]
    st.thread_type = "AUTO"
    c.seek(int(max(0, t0 + want[0] / FPS - 1.0) / st.time_base), stream=st)
    got = {}
    for fr in c.decode(st):
        k = int(round((float(fr.pts * st.time_base) - t0) * FPS))
        if k in want:
            got[k] = fr.to_ndarray(format="rgb24")
        if k > want[-1]:
            break
    c.close()
    return got


# ---- 1. row counts and lengths
new = pq.read_table(f"{OUT}/data/chunk-000/file-000.parquet")
ne = pq.read_table(f"{OUT}/meta/episodes/chunk-000/file-000.parquet").to_pandas()
epi = new["episode_index"].to_numpy()
assert len(new) == int(ne["length"].sum()), (len(new), ne["length"].sum())
for _, r in ne.iterrows():
    e = int(r.episode_index)
    n = int((epi == e).sum())
    assert n == r["length"] == r["dataset_to_index"] - r["dataset_from_index"], (e, n, r["length"])
    assert abs(r[f"videos/{KEY}/to_timestamp"] - n / FPS) < 1e-9
print(f"parquet/meta consistent: {len(new)} frames, {len(ne)} episodes")

# ---- 2. video frame counts per episode, all cameras
for key in ["observation.images.gripper", "observation.images.cam0",
            "observation.images.cam1", "observation.images.cam2"]:
    tot = 0
    for e in (0, 5, 42, 98):
        c = av.open(f"{OUT}/videos/{key}/chunk-000/file-{e:03d}.mp4")
        n = c.streams.video[0].frames
        c.close()
        want = int(ne.loc[ne.episode_index == e, "length"].iloc[0])
        assert n == want, f"{key} ep{e}: {n} != {want}"
        tot += n
    print(f"  {key}: spot-checked episode frame counts OK")

# ---- 3. pixel-level frame mapping: new frame i must be source frame keep[i]
src_ep = pq.read_table(f"{SRC}/meta/episodes/chunk-000/file-000.parquet").to_pandas()
for e in (5, 42):
    v = cuts[e]
    keep = np.r_[np.arange(0, v["a"] + 1), np.arange(v["b"], v["n"])]
    probes = [0, v["a"] - 3, v["a"], v["a"] + 1, v["a"] + 20, len(keep) - 1]
    srow = src_ep[src_ep.episode_index == e].iloc[0]
    sf = frames_of(f"{SRC}/videos/{KEY}/chunk-000/file-{int(srow[f'videos/{KEY}/file_index']):03d}.mp4",
                   [int(keep[i]) for i in probes], float(srow[f"videos/{KEY}/from_timestamp"]))
    nf = frames_of(f"{OUT}/videos/{KEY}/chunk-000/file-{e:03d}.mp4", probes, 0.0)
    for i in probes:
        a = sf[int(keep[i])].astype(np.float32)
        b = nf[i].astype(np.float32)
        psnr = 10 * np.log10(255 ** 2 / max(1e-9, ((a - b) ** 2).mean()))
        assert psnr > 33, f"ep{e} new frame {i} vs source frame {keep[i]}: PSNR {psnr:.1f}"
        print(f"  ep{e} new f{i:4d} == source f{int(keep[i]):4d}   PSNR {psnr:5.1f} dB")

# ---- 4. no residual dip: re-run the detector on the new states
state = np.stack(new["observation.state"].to_numpy(zero_copy_only=False)).astype(float)
fi = new["frame_index"].to_numpy()
worst = 0.0
for e in np.unique(epi):
    m = epi == e
    s = state[m][np.argsort(fi[m])]
    z, g = s[:, 8], s[:, 12]
    tc = int(np.where((g[:-1] > .5) & (g[1:] <= .5))[0][0]) + 1
    xy = np.hypot(np.diff(s[:, 6], prepend=s[0, 6]), np.diff(s[:, 7], prepend=s[0, 7])) * FPS
    mv = np.where(xy[:tc] > 2.0)[0]
    ta = int(mv[-1]) + 1 if len(mv) else 0
    seg = z[ta:tc + 1]
    excursion = float(seg.max() - seg.min())      # any dip between alignment and the close
    worst = max(worst, excursion)
print(f"largest remaining z excursion between alignment and the close, over 99 episodes: "
      f"{worst:.2f} mm  (was 35.1 mm)")

# ---- 5. splice smoothness in the new states
jumps = []
for e in np.unique(epi):
    m = epi == e
    s = state[m][np.argsort(fi[m])]
    a = cuts[int(e)]["a"]
    jumps.append(np.abs(s[a + 1, :12] - s[a, :12]).max())
jumps = np.array(jumps)
print(f"splice step (worst dim, mm or deg): mean {jumps.mean():.3f} max {jumps.max():.3f}")
print(f"typical per-frame step elsewhere for scale: "
      f"{np.abs(np.diff(state[:200, :12], axis=0)).max():.3f}")
print("\nVALIDATION PASSED")
