#!/usr/bin/env python3
"""
A simple script to fix timestamp issues in the therarelab/so100_pick_place dataset.
This script directly modifies the problematic timestamp arrays in the repo.
"""

import os
import json
import numpy as np
from huggingface_hub import hf_hub_download, upload_file

# Target repository
REPO_ID = "therarelab/so100_pick_place"
# Problematic episodes from the error message
PROBLEMATIC_EPISODES = ["10", "11", "12", "13"]  # Note: Using strings as keys

def main():
    print(f"Fixing timestamp issues in dataset: {REPO_ID}")
    
    # Download the episodes JSON file
    try:
        episodes_file = hf_hub_download(
            repo_id=REPO_ID,
            filename="episodes.json",
            repo_type="dataset"
        )
        print(f"Downloaded episodes.json to {episodes_file}")
    except Exception as e:
        print(f"Failed to download episodes.json: {e}")
        return False
        
    # Load the episodes data
    with open(episodes_file, 'r') as f:
        episodes_data = json.load(f)
    
    # Make sure episodes are in the expected format
    if not any(ep_id in episodes_data for ep_id in PROBLEMATIC_EPISODES):
        print("WARNING: Episode structure doesn't match expectations!")
        print("First-level keys:", list(episodes_data.keys())[:5])
        # Try to check if episodes might be nested differently
        if len(episodes_data.keys()) > 0:
            first_key = list(episodes_data.keys())[0]
            print(f"Structure of first item ({first_key}):", list(episodes_data[first_key].keys()))
        return False
    
    # Fix the problematic episodes
    print("Fixing problematic episodes...")
    fixed_count = 0
    fps = 30  # From the dataset metadata
    
    for ep_id in PROBLEMATIC_EPISODES:
        if ep_id in episodes_data:
            episode = episodes_data[ep_id]
            
            # Check if timestamps exist
            if "timestamps" in episode:
                old_timestamps = episode["timestamps"]
                num_frames = len(old_timestamps)
                
                # Generate new uniformly spaced timestamps
                dt = 1.0 / fps
                new_timestamps = [float(i * dt) for i in range(num_frames)]
                
                # Replace the timestamps
                episodes_data[ep_id]["timestamps"] = new_timestamps
                
                print(f"Episode {ep_id}: Replaced {num_frames} timestamps. Original first/last: {old_timestamps[0]}/{old_timestamps[-1]} → New: {new_timestamps[0]}/{new_timestamps[-1]}")
                fixed_count += 1
            else:
                print(f"Episode {ep_id}: No timestamps found, skipping.")
        else:
            print(f"Episode {ep_id}: Not found in episodes data, skipping.")
    
    if fixed_count == 0:
        print("No episodes were fixed.")
        return False
        
    # Save the modified episodes file
    with open(episodes_file, 'w') as f:
        json.dump(episodes_data, f)
    print(f"Saved modified episodes.json with fixed timestamps")
    
    # Upload the fixed file back to Hugging Face
    try:
        upload_file(
            path_or_fileobj=episodes_file,
            path_in_repo="episodes.json",
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        print(f"Successfully uploaded fixed episodes.json to {REPO_ID}")
    except Exception as e:
        print(f"Failed to upload fixed file: {e}")
        print(f"The fixed file is saved locally at: {episodes_file}")
        print("You may need to run 'huggingface-cli login' first")
        
        # Provide instructions for manual fix
        print("\nTo manually fix this:")
        print(f"1. Upload the fixed file from {episodes_file} to the dataset repository")
        print("2. Or run the script again after logging in with 'huggingface-cli login'")
        return False
    
    print("\nTimestamp fix completed. Now run your original script to convert the dataset.")
    return True

if __name__ == "__main__":
    main()