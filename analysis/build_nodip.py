"""Build dice_white_pnp_nodip: dice_white_pnp_100 minus the pointless pre-grasp dive-and-return.

Per episode we drop frames (a, b) exclusive, where the arm is parked at the hover pose at both
a and b (splice discontinuity < 0.3 mm / 0.07 deg). Everything else — approach, the wrist yaw
alignment that happens at hover, the close, the drop onto the dice, transport, place — is kept.

Videos are re-encoded (libsvtav1, crf 30, g 2) because the source GOP-2 keyframe parity differs
between cameras, so a lossless packet-copy cut cannot use one shared frame set.

Usage: build_nodip.py [data|video|meta|all] [--jobs N]
"""
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRATCH = "/tmp/claude-1000/-home-zero-matter-imle/1ee9319f-ee95-4f6f-8a2d-25bcbccae2d3/scratchpad"
import glob
SRC = glob.glob("/home/zero/.cache/huggingface/hub/datasets--azorematter--dice_white_pnp_100/snapshots/*")[0]
OUT = "/home/zero/matter/imle/datasets/dice_white_pnp_nodip"
FPS = 30
KEYS = ["observation.images.gripper", "observation.images.cam0",
        "observation.images.cam1", "observation.images.cam2"]
ENC_OPTS = {"crf": "30", "preset": "8", "g": "2"}


def keep_indices(n, a, b):
    return np.r_[np.arange(0, a + 1), np.arange(b, n)]


def load_plan():
    cuts = {int(k): v for k, v in json.load(open(f"{SCRATCH}/cut_points.json")).items()}
    src_ep = pq.read_table(f"{SRC}/meta/episodes/chunk-000/file-000.parquet").to_pandas()
    plan = {}
    for e, v in sorted(cuts.items()):
        n = int(src_ep.loc[src_ep.episode_index == e, "length"].iloc[0])
        assert n == v["n"], (e, n, v["n"])
        plan[e] = {**v, "keep": keep_indices(n, v["a"], v["b"]), "n": n}
    return plan, src_ep


# ----------------------------------------------------------------- data parquet
def build_data(plan):
    t = pq.read_table(f"{SRC}/data/chunk-000/file-000.parquet")
    state = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
    epi = t["episode_index"].to_numpy()
    fi = t["frame_index"].to_numpy()
    task = t["task_index"].to_numpy()

    S, A, TS, FR, EP, TK, bounds = [], [], [], [], [], [], {}
    cursor = 0
    for e in sorted(plan):
        m = epi == e
        s = state[m][np.argsort(fi[m])]
        tk = task[m][np.argsort(fi[m])]
        k = plan[e]["keep"]
        s2 = s[k]
        a2 = np.vstack([s2[1:], s2[-1:]])            # action[t] = state[t+1]; last = last state
        S.append(s2)
        A.append(a2)
        TS.append(np.arange(len(k), dtype=np.float32) / FPS)
        FR.append(np.arange(len(k), dtype=np.int64))
        EP.append(np.full(len(k), e, dtype=np.int64))
        TK.append(tk[k].astype(np.int64))
        bounds[e] = (cursor, cursor + len(k))
        cursor += len(k)

    S, A = np.concatenate(S), np.concatenate(A)
    tab = pa.table({
        "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(S.ravel(), pa.float32()), 13),
        "action": pa.FixedSizeListArray.from_arrays(pa.array(A.ravel(), pa.float32()), 13),
        "timestamp": pa.array(np.concatenate(TS), pa.float32()),
        "frame_index": pa.array(np.concatenate(FR), pa.int64()),
        "episode_index": pa.array(np.concatenate(EP), pa.int64()),
        "index": pa.array(np.arange(cursor, dtype=np.int64), pa.int64()),
        "task_index": pa.array(np.concatenate(TK), pa.int64()),
    })
    tab = tab.replace_schema_metadata(pq.ParquetFile(f"{SRC}/data/chunk-000/file-000.parquet")
                                      .schema_arrow.metadata)
    os.makedirs(f"{OUT}/data/chunk-000", exist_ok=True)
    pq.write_table(tab, f"{OUT}/data/chunk-000/file-000.parquet")
    json.dump({str(k): list(v) for k, v in bounds.items()}, open(f"{OUT}/.bounds.json", "w"))
    print(f"data: {cursor} frames (source 75666, -{75666-cursor} = {100*(75666-cursor)/75666:.1f}%)")
    return bounds


# ----------------------------------------------------------------- videos
def episode_video_ok(path, want):
    """True if this episode's video already exists with the expected frame count."""
    if not os.path.exists(path):
        return False
    try:
        c = av.open(path)
        n = c.streams.video[0].frames
        c.close()
        return n == want
    except Exception:
        return False


def build_key(key, plan, src_ep, only_file=None):
    """Re-encode episodes of one camera (optionally one source file), dropping excised frames."""
    os.makedirs(f"{OUT}/videos/{key}/chunk-000", exist_ok=True)
    by_file = {}
    for e in sorted(plan):
        row = src_ep[src_ep.episode_index == e].iloc[0]
        fidx = int(row[f"videos/{key}/file_index"])
        if only_file is not None and fidx != only_file:
            continue
        g0 = int(round(float(row[f"videos/{key}/from_timestamp"]) * FPS))
        by_file.setdefault(fidx, []).append((e, g0, plan[e]["n"]))

    written = {}
    for fidx, eps in sorted(by_file.items()):
        eps.sort(key=lambda x: x[1])
        todo = []
        for e, g0, n in eps:
            p = f"{OUT}/videos/{key}/chunk-000/file-{e:03d}.mp4"
            if episode_video_ok(p, len(plan[e]["keep"])):
                written[e] = len(plan[e]["keep"])
            else:
                if os.path.exists(p):
                    os.remove(p)
                todo.append((e, g0, n))
        if not todo:
            continue
        eps = todo
        path = f"{SRC}/videos/{key}/chunk-000/file-{fidx:03d}.mp4"
        c = av.open(path)
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        it = iter(eps)
        cur = next(it, None)
        oc = ost = None
        keepset, nkept = None, 0
        for gi, frame in enumerate(c.decode(st)):
            while cur is not None and gi >= cur[1] + cur[2]:      # past the end of this episode
                if oc is not None:
                    oc.mux(ost.encode())
                    oc.close()
                    written[cur[0]] = nkept
                    oc = ost = None
                cur = next(it, None)
            if cur is None:
                break
            e, g0, n = cur
            if gi < g0:
                continue
            if oc is None:
                keepset = set(plan[e]["keep"].tolist())
                nkept = 0
                oc = av.open(f"{OUT}/videos/{key}/chunk-000/file-{e:03d}.mp4", "w")
                ost = oc.add_stream("libsvtav1", rate=FPS)
                ost.width, ost.height, ost.pix_fmt = 640, 480, "yuv420p"
                ost.options = dict(ENC_OPTS)
            if (gi - g0) in keepset:
                out_frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
                oc.mux(ost.encode(out_frame))
                nkept += 1
        if oc is not None:
            oc.mux(ost.encode())
            oc.close()
            written[cur[0]] = nkept
        c.close()

    for eps in by_file.values():
        for e, _, _ in eps:
            want = len(plan[e]["keep"])
            got = written.get(e)
            assert got == want, f"{key} ep {e}: wrote {got} frames, expected {want}"
    return (key, only_file), sum(written.values())


def build_videos(plan, src_ep, jobs=8):
    tasks = []
    for k in KEYS:
        for fidx in sorted({int(src_ep.loc[src_ep.episode_index == e, f"videos/{k}/file_index"].iloc[0])
                            for e in plan}):
            tasks.append((k, fidx))
    print(f"video: {len(tasks)} (camera, source-file) tasks on {jobs} workers")
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(build_key, k, plan, src_ep, f) for k, f in tasks]
        for f in futs:
            (k, fidx), n = f.result()
            done += 1
            print(f"  [{done}/{len(tasks)}] {k} file{fidx}: {n} frames", flush=True)


# ----------------------------------------------------------------- meta
def stats_of(x, axis=0):
    x = np.asarray(x, dtype=np.float64)
    return dict(min=np.min(x, axis), max=np.max(x, axis), mean=np.mean(x, axis),
                std=np.std(x, axis), count=np.array([x.shape[0]]),
                q01=np.quantile(x, .01, axis), q10=np.quantile(x, .10, axis),
                q50=np.quantile(x, .50, axis), q90=np.quantile(x, .90, axis),
                q99=np.quantile(x, .99, axis))


def build_meta(plan, src_ep, bounds):
    os.makedirs(f"{OUT}/meta/episodes/chunk-000", exist_ok=True)
    shutil.copy(f"{SRC}/meta/tasks.parquet", f"{OUT}/meta/tasks.parquet")

    t = pq.read_table(f"{OUT}/data/chunk-000/file-000.parquet")
    state = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
    action = np.stack(t["action"].to_numpy(zero_copy_only=False)).astype(np.float64)
    epi = t["episode_index"].to_numpy()
    cols = {c: t[c].to_numpy() for c in ("timestamp", "frame_index", "episode_index", "index", "task_index")}

    # ---- episodes parquet: start from the source rows, overwrite what changed
    import pandas as pd
    df = src_ep.copy()
    # pandas .at squeezes length-1 arrays into scalars, which corrupts every 1-element stat,
    # so collect stat columns and assign them whole at the end.
    stat_cols = {}

    def set_stat(i, col, value, integer=False):
        if col not in stat_cols:
            stat_cols[col] = list(df[col])
        stat_cols[col][i] = np.asarray(value, dtype=np.int64 if integer else np.float64)

    for e in sorted(plan):
        i = df.index[df.episode_index == e][0]
        lo, hi = bounds[e]
        n = hi - lo
        df.at[i, "length"] = n
        df.at[i, "dataset_from_index"] = lo
        df.at[i, "dataset_to_index"] = hi
        df.at[i, "data/chunk_index"] = 0
        df.at[i, "data/file_index"] = 0
        for k in KEYS:
            df.at[i, f"videos/{k}/chunk_index"] = 0
            df.at[i, f"videos/{k}/file_index"] = e
            df.at[i, f"videos/{k}/from_timestamp"] = 0.0
            df.at[i, f"videos/{k}/to_timestamp"] = n / FPS
        m = epi == e
        for name, arr in (("observation.state", state[m]), ("action", action[m])):
            for sk, v in stats_of(arr).items():
                set_stat(i, f"stats/{name}/{sk}", v, integer=(sk == "count"))
        for name in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            v = cols[name][m].astype(np.float64).reshape(-1, 1)
            for sk, val in stats_of(v).items():
                set_stat(i, f"stats/{name}/{sk}", val, integer=(sk == "count"))

    for col, vals in stat_cols.items():
        df[col] = pd.Series(vals, index=df.index, dtype=object)
    bad = {c: {np.asarray(v).shape for v in df[c]} for c in stat_cols}
    for c, shapes in bad.items():
        assert () not in shapes, f"{c} has scalar values: {shapes}"
    src_schema = pq.read_schema(f"{SRC}/meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pandas(df, schema=src_schema, preserve_index=False),
                   f"{OUT}/meta/episodes/chunk-000/file-000.parquet")

    # ---- stats.json (dataset-level), recomputed honestly for the numeric features
    st = json.load(open(f"{SRC}/meta/stats.json"))
    new = dict(st)
    for name, arr in (("observation.state", state), ("action", action)):
        new[name] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stats_of(arr).items()}
    for name in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        v = cols[name].astype(np.float64).reshape(-1, 1)
        new[name] = {k: (val.tolist() if hasattr(val, "tolist") else val) for k, val in stats_of(v).items()}
    json.dump(new, open(f"{OUT}/meta/stats.json", "w"))

    # ---- info.json
    info = json.load(open(f"{SRC}/meta/info.json"))
    info["total_frames"] = int(len(t))
    info["total_episodes"] = len(plan)
    info["splits"] = {"train": f"0:{len(plan)}"}
    json.dump(info, open(f"{OUT}/meta/info.json", "w"), indent=2)
    print(f"meta: {info['total_episodes']} episodes, {info['total_frames']} frames")

    drift = {n: float(np.abs(np.array(new[n]["mean"]) - np.array(st[n]["mean"])).max())
             for n in ("observation.state", "action")}
    print("stats drift vs source (max |Δmean| per dim):", drift)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    plan, src_ep = load_plan()
    os.makedirs(OUT, exist_ok=True)
    if stage in ("data", "all"):
        bounds = build_data(plan)
    else:
        bounds = {int(k): tuple(v) for k, v in json.load(open(f"{OUT}/.bounds.json")).items()}
    if stage in ("video", "all"):
        build_videos(plan, src_ep, jobs=4)
    if stage in ("meta", "all"):
        build_meta(plan, src_ep, bounds)
