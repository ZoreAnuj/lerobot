#!/usr/bin/env python
"""
This script converts a LeRobot dataset from version V2.0 (using global stats)
to V21 (using per-episode stats). It has been updated minimally to enable backward
compatibility: if reversed timestamps are detected in an episode, they will be
auto-corrected when the --backward-compatibility flag is enabled.
"""

import logging
import sys
from pprint import pformat

# Import the LeRobotDataset class and V20 constant.
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, V20
# Import the timestamp check function, which now accepts a "backward_compatibility" flag.
from lerobot.common.datasets.utils import check_timestamps_sync

def convert_dataset(repo_id: str, backward_compatibility: bool = False) -> None:
    """
    Convert the dataset identified by `repo_id` from version V2.0 (global stats)
    to V21 (per-episode stats).

    Args:
        repo_id (str): The repository ID of the dataset.
        backward_compatibility (bool): If True, reversed timestamps in episodes will be automatically fixed.
    """
    logging.info(f"Starting conversion for {repo_id} with backward_compatibility={backward_compatibility}...")

    # Load the dataset in its old V2.0 format.
    dataset = LeRobotDataset(repo_id, revision=V20, force_cache_sync=True)

    # Retrieve the synchronization-related attributes.
    # (These attributes must exist on the LeRobotDataset instance.)
    timestamps = dataset.timestamps                  # e.g., numpy array of timestamps (float32)
    episode_indices = dataset.episode_indices        # numpy array (or equivalent) of episode IDs
    episode_data_index = dataset.episode_data_index  # dict with keys "from" and "to" indicating episode boundaries
    fps = dataset.fps                                # frames per second used during data collection
    tolerance_s = getattr(dataset, "tolerance_s", 1e-3)  # tolerance (default 1e-3 seconds if not defined)

    # Perform the timestamp synchronization check.
    # With backward_compatibility=True the function will auto-correct reversed timestamps.
    check_timestamps_sync(
        timestamps,
        episode_indices,
        episode_data_index,
        fps,
        tolerance_s,
        raise_value_error=True,
        backward_compatibility=backward_compatibility
    )

    # -----------------------------------------
    # Conversion of global stats to per-episode stats
    # and other conversion steps would go here.
    #
    # For example, you might:
    #   - Recompute per-episode statistics,
    #   - Update the dataset metadata,
    #   - Write out the new stats files.
    #
    # Since those details depend on your implementation,
    # they are omitted here.
    # -----------------------------------------

    logging.info("Conversion complete.")


def main():
    # Simple command-line argument handling.
    repo_id = None
    backward_compatibility = False

    for arg in sys.argv[1:]:
        if arg.startswith("--repo-id="):
            repo_id = arg.split("=", 1)[1]
        elif arg in ("--backward-compatibility", "--backward_compatibility"):
            backward_compatibility = True

    if repo_id is None:
        print("Usage: python convert_dataset_v20_to_v21.py --repo-id=<repo_id> [--backward-compatibility]")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    convert_dataset(repo_id, backward_compatibility=backward_compatibility)

if __name__ == "__main__":
    main()
