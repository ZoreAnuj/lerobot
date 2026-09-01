#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the IMLE policy (RS-IMLE loss, one-step generation, trajectory consistency)."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.imle.configuration_imle import IMLEConfig
from lerobot.policies.imle.modeling_imle import IMLEPolicy, _rs_imle_loss
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def make_config(**overrides) -> IMLEConfig:
    kwargs = {
        "n_obs_steps": 2,
        "horizon": 16,
        "n_action_steps": 8,
        # Keep the network tiny and avoid downloading pretrained weights in tests.
        "down_dims": (32, 64),
        "pretrained_backbone_weights": None,
        "use_group_norm": True,
        "spatial_softmax_num_keypoints": 8,
        "n_samples_per_condition": 4,
    }
    kwargs.update(overrides)
    config = IMLEConfig(**kwargs)
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    config.device = "cpu"
    return config


def make_batch(config: IMLEConfig, batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 4),
        OBS_IMAGE: torch.rand(batch_size, config.n_obs_steps, 3, 64, 64),
        ACTION: torch.randn(batch_size, config.horizon, 2),
    }


class TestRsImleLoss:
    def test_nearest_sample_is_selected(self):
        target = torch.zeros(1, 2, 2)
        samples = torch.stack(
            [torch.full((2, 2), 1.0), torch.full((2, 2), 3.0)],
            dim=0,
        ).unsqueeze(0)  # (1, 2, 2, 2)

        loss, _ = _rs_imle_loss(target, samples, epsilon=0.03)

        # Distance to the all-ones sample: sqrt(4 * 1) = 2, to the all-threes sample: sqrt(4 * 9) = 6.
        torch.testing.assert_close(loss, torch.tensor(2.0))

    def test_samples_within_epsilon_are_rejected(self):
        target = torch.zeros(1, 2, 2)
        near = torch.full((2, 2), 0.001)  # distance 0.002, within epsilon
        far = torch.full((2, 2), 1.0)  # distance 2
        samples = torch.stack([near, far], dim=0).unsqueeze(0)

        loss, loss_dict = _rs_imle_loss(target, samples, epsilon=0.03)

        # The near sample is rejected, so the far sample is selected despite being further away.
        torch.testing.assert_close(loss, torch.tensor(2.0))
        assert loss_dict["rejection_rate"] == 0.5

    def test_all_rejected_gives_zero_loss_with_graph(self):
        target = torch.zeros(1, 2, 2)
        samples = torch.full((1, 2, 2, 2), 0.001, requires_grad=True)

        loss, loss_dict = _rs_imle_loss(target, samples, epsilon=0.03)

        assert loss.item() == 0.0
        assert loss.requires_grad
        assert loss_dict["all_rejected_rate"] == 1.0
        loss.backward()  # must not raise

    def test_per_condition_independence(self):
        # First batch element has a valid sample at distance 2, second has all samples rejected.
        target = torch.zeros(2, 2, 2)
        samples = torch.stack(
            [
                torch.stack([torch.full((2, 2), 1.0)]),
                torch.stack([torch.full((2, 2), 0.001)]),
            ]
        )  # (2, 1, 2, 2)

        loss, _ = _rs_imle_loss(target, samples, epsilon=0.03)

        # Only the first element contributes.
        torch.testing.assert_close(loss, torch.tensor(2.0))


class TestIMLEPolicy:
    def test_forward_returns_scalar_loss(self):
        config = make_config()
        policy = IMLEPolicy(config)
        batch = make_batch(config)

        loss, loss_dict = policy.forward(batch)

        assert loss.ndim == 0
        assert loss.requires_grad
        assert "rejection_rate" in loss_dict
        loss.backward()

    def test_select_action_shape_and_queueing(self):
        config = make_config()
        policy = IMLEPolicy(config)
        policy.eval()
        obs = {
            OBS_STATE: torch.randn(2, 4),
            OBS_IMAGE: torch.rand(2, 3, 64, 64),
        }

        action = policy.select_action(obs)
        assert action.shape == (2, 2)
        # The remaining n_action_steps - 1 actions must be queued.
        assert len(policy._queues[ACTION]) == config.n_action_steps - 1

    def test_predict_action_chunk_uses_provided_noise(self):
        config = make_config()
        policy = IMLEPolicy(config)
        policy.eval()
        obs = {
            OBS_STATE: torch.randn(1, config.n_obs_steps, 4),
            OBS_IMAGE: torch.rand(1, config.n_obs_steps, 3, 64, 64),
        }
        noise = torch.randn(1, config.horizon, 2)

        chunk_a = policy.predict_action_chunk(dict(obs), noise=noise)
        chunk_b = policy.predict_action_chunk(dict(obs), noise=noise.clone())

        assert chunk_a.shape == (1, config.n_action_steps, 2)
        torch.testing.assert_close(chunk_a, chunk_b)

    def test_traj_consistency_selects_candidate_and_resets(self):
        config = make_config(use_traj_consistency=True, n_consistency_candidates=4, consistency_reset_every=2)
        policy = IMLEPolicy(config)
        policy.eval()
        obs = {
            OBS_STATE: torch.randn(1, config.n_obs_steps, 4),
            OBS_IMAGE: torch.rand(1, config.n_obs_steps, 3, 64, 64),
        }

        chunk = policy.predict_action_chunk(dict(obs))
        assert chunk.shape == (1, config.n_action_steps, 2)
        # After the first replan the anchor holds the selected full-horizon chunk.
        assert policy.imle._prev_chunk is not None
        assert policy.imle._prev_chunk.shape == (1, config.horizon, 2)

        # The second replan hits `consistency_reset_every` and drops the anchor.
        policy.predict_action_chunk(dict(obs))
        assert policy.imle._prev_chunk is None

        # reset() clears the replan counter.
        policy.reset()
        assert policy.imle._n_replans == 0

    def test_traj_consistency_requires_action_steps_below_horizon(self):
        with pytest.raises(ValueError):
            make_config(use_traj_consistency=True, n_action_steps=16, horizon=16)


class TestGripperRobustness:
    def test_channel_weight_changes_selection(self):
        from lerobot.policies.imle.modeling_imle import _rs_imle_loss

        target = torch.zeros(1, 2, 2)
        # Candidate A: perfect joints, gripper off by 1 on both steps -> unweighted dist sqrt(2).
        cand_a = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        # Candidate B: joints off by 1 everywhere, gripper perfect -> unweighted dist sqrt(2)+eps.
        cand_b = torch.tensor([[1.001, 0.0], [1.001, 0.0]])
        samples = torch.stack([cand_a, cand_b]).unsqueeze(0).requires_grad_(True)

        loss_unweighted, d_unweighted = _rs_imle_loss(target, samples, epsilon=0.0)
        # Unweighted: A is (barely) nearest, so the selected candidate has gripper error 1.
        assert d_unweighted["gripper_err_selected"] > 0.9

        weights = torch.tensor([1.0, 100.0])
        loss_weighted, d_weighted = _rs_imle_loss(target, samples, epsilon=0.0, channel_weights=weights)
        # Weighted: the gripper error makes A far, B is selected (gripper error ~0).
        assert d_weighted["gripper_err_selected"] < 0.1
        assert loss_weighted.requires_grad

    def test_gripper_obs_dropout(self):
        from lerobot.policies.imle.modeling_imle import _apply_gripper_obs_dropout

        torch.manual_seed(0)
        state = torch.randn(16, 2, 7)
        original = state.clone()

        out0 = _apply_gripper_obs_dropout(state, p=0.0)
        torch.testing.assert_close(out0, original)

        out1 = _apply_gripper_obs_dropout(state, p=1.0)
        # Non-gripper dims untouched, input untouched.
        torch.testing.assert_close(out1[..., :-1], original[..., :-1])
        torch.testing.assert_close(state, original)
        # Every corrupted gripper column equals some sample's original column.
        for b in range(16):
            match = (out1[b, :, -1].unsqueeze(0) == original[:, :, -1]).all(dim=1)
            assert match.any()
        # With 16 samples, at least one column must actually have changed.
        assert not torch.equal(out1[..., -1], original[..., -1])

    def test_forward_with_dropout_and_weight(self):
        config = make_config(gripper_obs_dropout=1.0, rs_gripper_weight=5.0)
        policy = IMLEPolicy(config)
        policy.train()
        batch = make_batch(config)
        loss, loss_dict = policy.forward(batch)
        assert loss.requires_grad
        assert "gripper_err_selected" in loss_dict
        loss.backward()

    def test_config_validation(self):
        with pytest.raises(ValueError):
            make_config(rs_gripper_weight=0.0)
        with pytest.raises(ValueError):
            make_config(transition_oversample=0)
        with pytest.raises(ValueError):
            make_config(gripper_obs_dropout=1.5)


class TestTransitionOversampler:
    def _fake_dataset(self, grip_per_episode):
        """Duck-typed dataset: hf_dataset.data with action/episode_index columns."""

        class Col:
            def __init__(self, values):
                self._values = values

            def to_pylist(self):
                return self._values

        class Table:
            def __init__(self, cols):
                self._cols = cols

            def column(self, name):
                return Col(self._cols[name])

        actions, eps = [], []
        for e, grips in enumerate(grip_per_episode):
            for g in grips:
                actions.append([0.0] * 6 + [float(g)])
                eps.append(e)

        class DS:
            pass

        ds = DS()
        ds.hf_dataset = DS()
        ds.hf_dataset.data = Table({"action": actions, "episode_index": eps})
        return ds

    def test_find_transition_frames(self):
        from lerobot.datasets.sampler import find_transition_frames

        # Episode 0: flip at row 5 (1->0). Episode 1 (rows 8..15): no flip.
        ds = self._fake_dataset([[1, 1, 1, 1, 1, 0, 0, 0], [1] * 8])
        idx = find_transition_frames(ds, horizon=3)
        # Window = the 3 chunk-start frames covering the flip row: 3, 4, 5.
        assert idx.tolist() == [3, 4, 5]

        # Flip right at an episode start must not leak into the previous episode.
        ds2 = self._fake_dataset([[1, 1], [1, 0]])
        idx2 = find_transition_frames(ds2, horizon=4)
        assert idx2.tolist() == [2, 3]

    def test_oversampler_counts_and_resume(self):
        from collections import Counter

        from lerobot.datasets.sampler import TransitionOversampler

        kwargs = {"shuffle": True, "seed": 7}
        sampler = TransitionOversampler([0], [10], oversample_indices=[2, 3], repeats=4, **kwargs)
        assert len(sampler) == 10 + 2 * 3
        counts = Counter(iter(sampler))
        assert counts[2] == 4 and counts[3] == 4
        assert counts[0] == 1 and counts[9] == 1

        # Sample-exact resume: consume 5, snapshot, and a fresh sampler continues identically.
        s1 = TransitionOversampler([0], [10], oversample_indices=[2, 3], repeats=4, **kwargs)
        it = iter(s1)
        head = [next(it) for _ in range(5)]
        assert len(head) == 5
        state = {"epoch": 0, "start_index": 5}
        s2 = TransitionOversampler([0], [10], oversample_indices=[2, 3], repeats=4, **kwargs)
        s2.load_state_dict(state)
        assert list(iter(s2)) == list(it)

    def test_repeats_one_matches_base(self):
        from lerobot.datasets.sampler import EpisodeAwareSampler, TransitionOversampler

        base = EpisodeAwareSampler([0], [10], shuffle=True, seed=3)
        over = TransitionOversampler([0], [10], oversample_indices=[2], repeats=1, shuffle=True, seed=3)
        assert list(iter(base)) == list(iter(over))


class TestReviewGaps:
    """Coverage for the failure modes the adversarial review confirmed."""

    def test_weighted_epsilon_lives_in_weighted_space(self):
        from lerobot.policies.imle.modeling_imle import _rs_imle_loss

        target = torch.zeros(1, 1, 2)
        # Distance 0.1 unweighted; with weight 100 on the last channel: sqrt(100*0.01)=1.0.
        cand = torch.tensor([[[0.0, 0.1]]], requires_grad=True).unsqueeze(1)
        _, d_unw = _rs_imle_loss(target, cand, epsilon=0.5)
        assert d_unw["rejection_rate"] == 1.0  # 0.1 < 0.5: rejected unweighted
        _, d_w = _rs_imle_loss(target, cand, epsilon=0.5, channel_weights=torch.tensor([1.0, 100.0]))
        assert d_w["rejection_rate"] == 0.0  # weighted distance 1.0 > 0.5: survives

    def test_weights_of_ones_match_none(self):
        from lerobot.policies.imle.modeling_imle import _rs_imle_loss

        torch.manual_seed(0)
        target = torch.randn(4, 3, 5)
        samples = torch.randn(4, 6, 3, 5)
        l_none, d_none = _rs_imle_loss(target, samples, epsilon=0.03)
        l_ones, d_ones = _rs_imle_loss(target, samples, epsilon=0.03, channel_weights=torch.ones(5))
        torch.testing.assert_close(l_none, l_ones)
        assert d_none["gripper_err_selected"] == pytest.approx(d_ones["gripper_err_selected"])

    def test_gripper_metric_unit_invariant_to_weight(self):
        from lerobot.policies.imle.modeling_imle import _rs_imle_loss

        target = torch.zeros(1, 2, 2)
        cand = torch.tensor([[[0.0, 0.3], [0.0, 0.3]]]).unsqueeze(1)
        _, d1 = _rs_imle_loss(target, cand, epsilon=0.0)
        _, d25 = _rs_imle_loss(target, cand, epsilon=0.0, channel_weights=torch.tensor([1.0, 25.0]))
        assert d1["gripper_err_selected"] == pytest.approx(0.3, abs=1e-5)
        assert d25["gripper_err_selected"] == pytest.approx(0.3, abs=1e-5)

    def test_dropout_never_self_donates_and_eval_mode_off(self):
        from lerobot.policies.imle.modeling_imle import _apply_gripper_obs_dropout

        torch.manual_seed(1)
        state = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1).expand(8, 2, 1).clone()
        out = _apply_gripper_obs_dropout(state, p=1.0)
        # Every sample's gripper column must differ (columns are unique per sample here).
        assert (out[..., -1] != state[..., -1]).all()
        # Batch of one: inert by construction.
        single = torch.randn(1, 2, 7)
        torch.testing.assert_close(_apply_gripper_obs_dropout(single, p=1.0), single)

        # Eval-mode forward leaves the observation untouched (deterministic validation).
        config = make_config(gripper_obs_dropout=1.0)
        policy = IMLEPolicy(config)
        policy.eval()
        batch = make_batch(config)
        state_before = batch["observation.state"].clone()
        with torch.no_grad():
            policy.forward(batch)
        torch.testing.assert_close(batch["observation.state"], state_before)

    def test_oversampler_respects_drop_n_last_frames(self):
        from lerobot.datasets.sampler import TransitionOversampler

        # 30-frame episode, drop_n_last_frames=7 -> base emits 0..22. Oversample indices 20..27:
        # 23..27 must be dropped, 20..22 repeated.
        s = TransitionOversampler(
            [0],
            [30],
            oversample_indices=list(range(20, 28)),
            repeats=3,
            drop_n_last_frames=7,
            shuffle=True,
            seed=1,
        )
        emitted = list(iter(s))
        from collections import Counter

        counts = Counter(emitted)
        for idx in (23, 24, 25, 26, 27):
            assert counts[idx] == 0
        for idx in (20, 21, 22):
            assert counts[idx] == 3

    def test_oversampler_with_episode_filter_and_mapping(self):
        from collections import Counter

        from lerobot.datasets.sampler import TransitionOversampler

        # Episodes: 0 -> rows 0..9, 1 -> rows 10..19; only episode 1 used. Relative mapping:
        # absolute 10..19 -> relative 0..9 (as LeRobotDataset builds for filtered episodes).
        a2r = {abs_i: abs_i - 10 for abs_i in range(10, 20)}
        s = TransitionOversampler(
            [0, 10],
            [10, 20],
            episode_indices_to_use=[1],
            oversample_indices=[3, 4],  # relative indices into the filtered dataset
            repeats=2,
            shuffle=True,
            seed=0,
            absolute_to_relative_idx=a2r,
        )
        counts = Counter(iter(s))
        assert set(counts) <= set(range(10))  # everything lands in relative space
        assert counts[3] == 2 and counts[4] == 2
        assert len(s) == 10 + 2

    def test_find_motion_onsets(self):
        from lerobot.datasets.sampler import find_transition_frames

        class Col:
            def __init__(self, v):
                self._v = v

            def to_pylist(self):
                return self._v

        class Table:
            def __init__(self, cols):
                self._cols = cols

            def column(self, name):
                return Col(self._cols[name])

        class DS:
            pass

        # One episode: move (0..2), dwell rows 3..8 (6 identical), move again at 9.
        acts = [[float(i)] * 6 + [1.0] for i in range(3)]
        acts += [acts[-1][:]] * 6
        acts += [[9.0] * 6 + [1.0], [10.0] * 6 + [1.0]]
        ds = DS()
        ds.hf_dataset = DS()
        ds.hf_dataset.data = Table({"action": acts, "episode_index": [0] * len(acts)})

        # No gripper flip anywhere; onset at row 9 with min_dwell=5.
        none_found = find_transition_frames(ds, horizon=4, lead=1, min_dwell=0)
        assert none_found.tolist() == []
        idx = find_transition_frames(ds, horizon=4, lead=1, min_dwell=5)
        # Window for onset j=9, horizon=4, lead=1: [9-4+1+1, 9+1] = [7, 10].
        assert idx.tolist() == [7, 8, 9, 10]
