#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
from collections.abc import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    """Sampler over episode frames that stores only per-episode boundaries.

    Logical positions map to frame indices on the fly (O(num_episodes) construction memory)
    instead of materializing a Python list of every frame index.

    Each epoch is shuffled with a `torch.randperm` seeded from `(seed, epoch)`, so the data order
    is a pure function of `(seed, epoch)`: it reproduces on every rank without synchronizing the
    global RNG (no `generator` to sync across distributed ranks), and `state_dict` /
    `load_state_dict` resume a run sample-exactly by regenerating the epoch's permutation and
    continuing from the saved offset. Each call to `__iter__` advances the epoch. During a
    resumed epoch, `__len__` still reports the full length.

    Epoch advancement: `__iter__` eagerly advances the epoch, and `set_epoch` / `load_state_dict`
    set it explicitly. Within a single run callers should rely on exactly one of these mechanisms,
    not both: advancing the epoch by hand *and* letting `__iter__` auto-advance over the same
    iterations would skip or repeat epochs. The training loop drives it purely through `__iter__`
    (via `cycle`); `set_epoch` / `load_state_dict` are used only to (re)position before iteration
    starts (e.g. on resume or in tests).
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
    ):
        """
        Args:
            dataset_from_indices: Start index of each episode in the dataset.
            dataset_to_indices: End index of each episode in the dataset.
            episode_indices_to_use: Episode indices to use; None means all.
            drop_n_first_frames: Frames to drop from the start of each episode.
            drop_n_last_frames: Frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            seed: Seed the permutation is derived from (together with the epoch).
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError(
                f"dataset_from_indices and dataset_to_indices must have the same length, "
                f"got {len(from_indices)} and {len(to_indices)}"
            )

        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        starts = from_indices + drop_n_first_frames
        lengths = to_indices - drop_n_last_frames - starts
        for episode_idx in np.flatnonzero(used & (lengths <= 0)):
            logger.warning(
                "Episode %d has %d frames but drop_n_first_frames=%d and "
                "drop_n_last_frames=%d removes all frames. Skipping.",
                episode_idx,
                to_indices[episode_idx] - from_indices[episode_idx],
                drop_n_first_frames,
                drop_n_last_frames,
            )
        used &= lengths > 0
        if not used.any():
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self._starts = starts[used]
        self._cum_lengths = np.cumsum(lengths[used])
        self._num_frames = int(self._cum_lengths[-1])
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._start_index = 0
        self._absolute_to_relative = absolute_to_relative_idx

    @property
    def indices(self) -> list[int]:
        """Materialized frame indices in unshuffled order; O(num_frames), introspection only."""
        return [self._frame_index(k) for k in range(self._num_frames)]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        # Derive a per-epoch seed from (seed, epoch) so the permutation is a pure function of both
        # and reproduces identically on every rank without touching the global RNG.
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(epoch_seed)

    def _frame_index(self, position: int) -> int:
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        absolute_idx = int(self._starts[episode]) + position_in_episode
        if self._absolute_to_relative is not None:
            return self._absolute_to_relative[absolute_idx]
        return absolute_idx

    def __iter__(self) -> Iterator[int]:
        # Advance epoch state eagerly, not on first consumption of the generator.
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        if self.shuffle:
            order = torch.randperm(self._num_frames, generator=self._epoch_generator(epoch))
            for k in range(start, self._num_frames):
                yield self._frame_index(int(order[k]))
        else:
            for k in range(start, self._num_frames):
                yield self._frame_index(k)

    def __len__(self) -> int:
        return self._num_frames


def compute_sampler_state(step: int, num_frames: int, batch_size: int, num_processes: int) -> dict:
    """Map an optimization step to an `EpisodeAwareSampler` state for sample-exact resume.

    Under accelerate's batch sharding, one step consumes `batch_size * num_processes` sampler
    positions and each rank sees `ceil(ceil(num_frames / batch_size) / num_processes)` batches
    per epoch (`even_batches` padding included). The start index provably stays below
    `num_frames`; the `min` is defensive.

    Assumptions (resume is only sample-exact when they hold):
        - `num_processes` and `batch_size` match the run that wrote the checkpoint. Both scale how
          many positions a step consumes, so the epoch/offset are wrong if either changed. The
          caller passes the checkpoint's `num_processes` and `batch_size` and warns on a mismatch.
        - accelerate uses `even_batches=True` (its default). The `ceil(... / num_processes)` term
          mirrors that padding; with `even_batches=False` the per-epoch batch count differs and
          the boundary is off.
    """
    batches_per_epoch = math.ceil(math.ceil(num_frames / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_frames)
    return {"epoch": epoch, "start_index": start_index}


class TransitionOversampler(EpisodeAwareSampler):
    """EpisodeAwareSampler whose epochs repeat a chosen set of frames.

    The position space is extended past the base sampler's frames with `repeats - 1` extra copies
    of each index in `oversample_indices`, and the whole expanded space is shuffled with the same
    (seed, epoch)-pure permutation as the base class — so per-rank determinism, `state_dict` /
    `load_state_dict` resume, and `compute_sampler_state` all keep working unchanged.

    Built for rare-event rebalancing (e.g. frames whose action chunk contains a gripper
    open/close transition), but agnostic to what the indices mean.
    """

    def __init__(self, *args, oversample_indices=None, repeats: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        extra = np.asarray(oversample_indices if oversample_indices is not None else [], dtype=np.int64)
        self._base_num_frames = self._num_frames
        # Only repeat frames the base sampler can emit: anything else (e.g. a frame inside the
        # drop_n_last_frames tail, or from a filtered-out episode) was excluded deliberately, and
        # re-injecting it would hand the model the copy-padded chunks the base sampler avoids.
        valid = np.fromiter((self._frame_index(k) for k in range(self._base_num_frames)), dtype=np.int64)
        dropped = len(extra) - int(np.isin(extra, valid).sum())
        if dropped:
            logger.info(
                "TransitionOversampler: %d oversample indices fall outside the base sampler's "
                "frames (dropped tails or filtered episodes) and are not repeated.",
                dropped,
            )
        extra = extra[np.isin(extra, valid)]
        self._extra = np.repeat(extra, repeats - 1)
        self._num_frames = self._base_num_frames + len(self._extra)

    def _frame_index(self, position: int) -> int:
        if position < self._base_num_frames:
            return super()._frame_index(position)
        return int(self._extra[position - self._base_num_frames])


def _mark_windows(marked: np.ndarray, episodes: np.ndarray, rows: np.ndarray, horizon: int, lead: int):
    """Mark, for each event row, the conditioning frames whose action chunk contains it.

    A frame i's chunk covers rows [i - lead, i - lead + horizon), so the conditioning frames for an
    event at row j are [j - horizon + 1 + lead, j + lead], clipped to j's episode and array bounds.
    """
    for j in rows:
        lo = max(0, j - horizon + 1 + lead)
        hi = min(len(marked) - 1, j + lead)
        while lo < j and episodes[lo] != episodes[j]:
            lo += 1
        while hi > j and episodes[hi] != episodes[j]:
            hi -= 1
        marked[lo : hi + 1] = True


def find_transition_frames(dataset, horizon: int, lead: int = 0, min_dwell: int = 0) -> np.ndarray:
    """Relative indices of frames whose action chunk contains a gripper flip or a motion onset.

    A "flip" is a change of the thresholded (> 0.5) LAST action channel between consecutive frames
    of the same episode — conventionally the gripper open/close bit. When `min_dwell` > 0, a
    "motion onset" is additionally marked: the first row where the action vector changes again
    after being exactly constant for at least `min_dwell` consecutive frames (the dwell-exit
    moments that teach a policy to leave a hover). A frame belongs to the window when the event
    falls inside its `horizon`-step action chunk, which starts `lead` frames in the past
    (`lead = n_obs_steps - 1` for chunked policies whose action window begins at the oldest
    observation).

    `dataset` needs only `.hf_dataset` (rows in relative order with "action" and "episode_index"
    columns), so it works on episode-filtered datasets out of the box. Logs a warning when no
    event is found — e.g. a dataset whose last action channel is not 0/1-coded.
    """
    table = dataset.hf_dataset.data
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    episodes = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    same_ep = episodes[1:] == episodes[:-1]

    grip = actions[:, -1] > 0.5
    flips = np.flatnonzero((grip[1:] != grip[:-1]) & same_ep) + 1

    marked = np.zeros(len(actions), dtype=bool)
    _mark_windows(marked, episodes, flips, horizon, lead)

    if min_dwell > 0:
        still = np.zeros(len(actions), dtype=bool)
        still[1:] = (np.abs(actions[1:] - actions[:-1]).max(axis=1) == 0.0) & same_ep
        run = 0
        onsets = []
        for i in range(1, len(actions)):
            if still[i]:
                run += 1
            else:
                if run >= min_dwell and episodes[i] == episodes[i - 1]:
                    onsets.append(i)
                run = 0
        _mark_windows(marked, episodes, np.asarray(onsets, dtype=np.int64), horizon, lead)

    if not marked.any():
        logger.warning(
            "find_transition_frames found no gripper flips or motion onsets — is the last action "
            "channel really a 0/1 gripper bit? Transition oversampling will be a no-op."
        )
    return np.flatnonzero(marked)
