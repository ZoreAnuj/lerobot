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
CHANGELOG:
- Added a dynamic FPS check to retrieve the dataset's fps from metadata.
- Calculated the fundamental time interval (dt = 1/fps) and generated suggestions for delta timestamps.
- Provided suggestions for valid past and future time offsets that are multiples of dt.
- Added a check to verify that the camera key (as provided by dataset metadata) exists in the dataset. If not, falls back to a default key.
- The rest of the script remains unchanged from the original version.

This script demonstrates the use of `LeRobotDataset` class for handling and processing robotic datasets from Hugging Face.
It illustrates how to load datasets, manipulate them, and apply transformations suitable for machine learning tasks in PyTorch.

Features included in this script:
- Viewing a dataset's metadata and exploring its properties.
- Loading an existing dataset from the hub or a subset of it.
- Accessing frames by episode number.
- Using advanced dataset features like timestamp-based frame selection.
- Demonstrating compatibility with PyTorch DataLoader for batch processing.
"""

from pprint import pprint
import math

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# -----------------------------------------------------------------------
# List available datasets
# -----------------------------------------------------------------------
print("List of available datasets:")
pprint(lerobot.available_datasets)

hub_api = HfApi()
repo_ids = [info.id for info in hub_api.list_datasets(task_categories="robotics", tags=["LeRobot"])]
pprint(repo_ids)
# Or visit: https://huggingface.co/datasets?other=LeRobot

# -----------------------------------------------------------------------
# Select and inspect a dataset
# -----------------------------------------------------------------------
repo_id = "zero7101/first_data"
ds_meta = LeRobotDatasetMetadata(repo_id)

print(f"Total number of episodes: {ds_meta.total_episodes}")
print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
print(f"Frames per second used during data collection: {ds_meta.fps}")
print(f"Robot type: {ds_meta.robot_type}")
print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

print("Tasks:")
print(ds_meta.tasks)
print("Features:")
pprint(ds_meta.features)

# You can also get a short summary by simply printing the object:
print(ds_meta)

# -----------------------------------------------------------------------
# Load the dataset
# -----------------------------------------------------------------------
# You can load any subset of episodes by specifying them in the constructor; here we load the entire dataset.
dataset = LeRobotDataset(repo_id)
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")
print(dataset.meta)

# LeRobotDataset actually wraps an underlying Hugging Face dataset
# (see https://huggingface.co/docs/datasets for more information).
print(dataset.hf_dataset)

# -----------------------------------------------------------------------
# Check FPS from dataset metadata and suggest valid delta timestamps
# -----------------------------------------------------------------------
fps = dataset.meta.fps  # Retrieve fps from dataset metadata
dt = 1 / fps          # Fundamental time interval (seconds per frame)
print(f"\nDetected FPS: {fps}. Frame duration (1/fps): {dt:.4f} seconds.")

def generate_timestamp_suggestions(dt, window=1.0):
    """
    Generate suggested delta timestamps that are integer multiples of dt
    within a given window (in seconds). Returns two lists:
      - past_suggestions: negative offsets (for past frames)
      - future_suggestions: positive offsets (for future frames)
    """
    # Number of steps in the given window. We use floor to ensure an integer number of steps.
    num_steps = math.floor(window / dt)
    past_suggestions = [round(-i * dt, 4) for i in range(num_steps, 0, -1)]
    future_suggestions = [round(i * dt, 4) for i in range(1, num_steps + 1)]
    return past_suggestions, future_suggestions

# Generate suggestions for up to 1 second in the past and future
past_suggestions, future_suggestions = generate_timestamp_suggestions(dt, window=1.0)
print("Suggested delta timestamps for past frames (multiples of 1/fps):", past_suggestions)
print("Suggested delta timestamps for future frames (multiples of 1/fps):", future_suggestions)

# -----------------------------------------------------------------------
# Define delta_timestamps using a selection of suggested values
# -----------------------------------------------------------------------
# Before using the camera key provided in metadata, check if it exists in the underlying dataset.
if dataset.meta.camera_keys:
    candidate_camera_key = dataset.meta.camera_keys[0]
    if candidate_camera_key not in dataset.hf_dataset.column_names:
        print(f"Warning: Candidate camera key '{candidate_camera_key}' not found in dataset columns.")
        print(f"Available columns: {dataset.hf_dataset.column_names}")
        # Fallback to a default key (e.g., 'observation.state') if the candidate key is missing.
        camera_key = "observation.state"
        print(f"Falling back to key '{camera_key}'.")
    else:
        camera_key = candidate_camera_key
else:
    camera_key = "observation.state"
    print(f"No camera keys provided in metadata, using default key '{camera_key}'.")

# Define delta_timestamps using a selection of the suggested values
delta_timestamps = {
    # For the camera, we'll use 3 past frames and the current frame.
    camera_key: past_suggestions[-3:] + [0],
    # For the robot state observation, we could use 5 past frames plus the current frame.
    "observation.state": past_suggestions[-5:] + [0],
    # For actions, maintain the example using 64 future frames (each being t/fps seconds ahead).
    "action": [t / fps for t in range(64)],
}

# Reload the dataset with the new delta_timestamps settings.
dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

# Check the shape of returned tensors to verify temporal aggregation.
print(f"\n{dataset[0][camera_key].shape=}")          # Expected shape: (number of timestamps, c, h, w)
print(f"{dataset[0]['observation.state'].shape=}")     # Expected shape: (number of timestamps, c)
print(f"{dataset[0]['action'].shape=}\n")               # Expected shape: (64, c)

# -----------------------------------------------------------------------
# Access frames by episode number
# -----------------------------------------------------------------------
# The __getitem__ iterates over the frames of the dataset. Since our datasets are also structured by episodes,
# you can access the frame indices of any episode using the episode_data_index. Here, we access frame indices associated
# with the first episode:
episode_index = 0
from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

# Then we grab all the image frames (or fallback state frames) from the selected camera key:
frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

# The objects returned by the dataset are all torch.Tensors
print(type(frames[0]))
print(frames[0].shape)

# Since we're using PyTorch, the shape is in the channel-first convention (c, h, w).
# We can compare this shape with the information available for that feature.
pprint(dataset.features[camera_key])
# In particular:
print(dataset.features[camera_key]["shape"])
# The shape reported here is in (h, w, c) which is a more universal format.

# -----------------------------------------------------------------------
# PyTorch DataLoader integration
# -----------------------------------------------------------------------
# LeRobot datasets are fully compatible with PyTorch dataloaders and samplers because they are just PyTorch datasets.
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    batch_size=32,
    shuffle=True,
)

for batch in dataloader:
    print(f"{batch[camera_key].shape=}")           # Expected shape: (32, number of timestamps, c, h, w)
    print(f"{batch['observation.state'].shape=}")    # Expected shape: (32, number of timestamps, c)
    print(f"{batch['action'].shape=}")               # Expected shape: (32, 64, c)
    break
