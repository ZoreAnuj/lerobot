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
