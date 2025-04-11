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

"""
This script evaluates a pretrained policy from the HuggingFace Hub or a local
training output directory over 5 episodes. It computes cumulative and discounted
returns, averages them, and displays a performance graph.

Requires installation of 'gym_pusht'. Install with:
```bash
pip install --no-binary=av -e ".[pusht]"
```
"""

from pathlib import Path
import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

output_directory = Path("outputs/eval/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

device = "cuda"
pretrained_policy_path = "lerobot/diffusion_pusht"
policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=300,
)

print("Policy input features:", policy.config.input_features)
print("Environment observation space:", env.observation_space)
print("Policy output features:", policy.config.output_features)
print("Environment action space:", env.action_space)

# Evaluation parameters
num_episodes = 5
gamma = 0.99
cumulative_returns = []
discounted_returns = []
steps_per_episode = []
success_flags = []

for episode in range(num_episodes):
    print(f"\n--- Episode {episode + 1} ---")
    policy.reset()
    numpy_observation, info = env.reset(seed=episode + 42)
    rewards = []

    done = False
    step = 0
    success = False

    while not done:
        state = torch.from_numpy(numpy_observation["agent_pos"]).to(torch.float32)
        image = torch.from_numpy(numpy_observation["pixels"]).to(torch.float32) / 255
        image = image.permute(2, 0, 1)

        state = state.to(device).unsqueeze(0)
        image = image.to(device).unsqueeze(0)

        observation = {
            "observation.state": state,
            "observation.image": image,
        }

        with torch.inference_mode():
            action = policy.select_action(observation)

        numpy_action = action.squeeze(0).to("cpu").numpy()
        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)

        rewards.append(reward)
        done = terminated or truncated
        success = terminated
        step += 1

    cumulative = sum(rewards)
    discounted = sum((gamma ** t) * r for t, r in enumerate(rewards))
    cumulative_returns.append(cumulative)
    discounted_returns.append(discounted)
    steps_per_episode.append(step)
    success_flags.append(success)

    print(f"Cumulative Return: {cumulative:.2f}, Discounted Return: {discounted:.2f}, Steps: {step}, Success: {success}")

avg_cumulative = np.mean(cumulative_returns)
avg_discounted = np.mean(discounted_returns)
avg_steps = np.mean(steps_per_episode)
success_rate = np.sum(success_flags) / num_episodes * 100

print("\n========== Evaluation Summary ==========")
print(f"Average Cumulative Return: {avg_cumulative:.2f}")
print(f"Average Discounted Return: {avg_discounted:.2f}")
print(f"Average Episode Length: {avg_steps:.2f} steps")
print(f"Success Rate: {success_rate:.2f}%")
print("=======================================\n")

# Plotting results
plt.figure(figsize=(10, 6))
plt.bar(range(1, num_episodes + 1), cumulative_returns, color='skyblue')
plt.xlabel("Episode")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Return per Episode")
plt.xticks(range(1, num_episodes + 1))
plt.grid(axis='y')
plt.tight_layout()
plt.show()
