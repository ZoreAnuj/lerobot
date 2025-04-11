#!/usr/bin/env python

from pathlib import Path
import torch
import time
import random
import logging
import torchvision.transforms as T
from torch.amp import GradScaler
from contextlib import nullcontext
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainACTPolicy():
    def __init__(self, repo_id, device=None, n_steps=5000, log_freq=10, 
                 save_freq=500, batch_size=32, max_retries=5, retry_delay=10,
                 add_version_tag=False, hf_token=None):
        self.repo_id = repo_id
        self.output_directory = Path(f"outputs/train/act/{repo_id}")
        self.add_version_tag = add_version_tag
        self.hf_token = hf_token
        
        # Handle device properly
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
        else:
            self.device = device
            
        self.training_steps = n_steps
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dataset_metadata = None
        self.features = None
        self.output_features = None
        self.input_features = None
        self.config = None
        self.policy = None
        self.use_amp = True if self.device.type == 'cuda' else False
        self.image_keys = []  # Store the original image keys

    def add_missing_version_tag(self):
        """Add version tag to dataset if needed and requested"""
        if not self.add_version_tag:
            logger.warning("Dataset is missing version tag, but auto-tagging is disabled.")
            logger.warning("Run the following command manually to add a version tag:")
            logger.warning(f"python -c \"from huggingface_hub import HfApi; HfApi().create_tag('{self.repo_id}', tag='1.0.0', repo_type='dataset')\"")
            return False

        if not self.hf_token:
            logger.error("HF token is required to add version tag to dataset")
            return False

        logger.info(f"Adding version tag to dataset {self.repo_id}")
        try:
            hub_api = HfApi(token=self.hf_token)
            # Use a standard semantic version as the tag
            hub_api.create_tag(self.repo_id, tag="1.0.0", repo_type="dataset")
            logger.info(f"Successfully added version tag '1.0.0' to {self.repo_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add version tag: {e}")
            return False
        
    def _with_retry(self, fn, operation_name):
        """Execute a function with retry logic for handling rate limits and version errors"""
        retries = 0
        while retries < self.max_retries:
            try:
                logger.info(f"Attempting {operation_name}...")
                result = fn()
                logger.info(f"{operation_name} successful!")
                return result
            except RevisionNotFoundError as e:
                logger.error(f"Version tag missing: {e}")
                # Try to add the version tag if requested
                if self.add_version_tag and self.add_missing_version_tag():
                    logger.info("Added version tag, retrying operation...")
                    continue
                else:
                    raise
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    retries += 1
                    wait_time = self.retry_delay * (1 + random.random())
                    logger.warning(f"Rate limit hit. Retry {retries}/{self.max_retries} after {wait_time:.1f}s")
                    if retries < self.max_retries:
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Max retries reached for {operation_name}")
                        raise
                else:
                    logger.error(f"Error in {operation_name}: {e}")
                    raise
    
    class ResizedImageDatasetWrapper(torch.utils.data.Dataset):
        """Dataset wrapper that resizes all images to a consistent shape"""
        def __init__(self, dataset, image_keys, target_size=(84, 84)):
            self.dataset = dataset
            self.image_keys = image_keys
            self.target_size = target_size
            self.resize_transform = T.Resize(target_size, antialias=True)
            logger.info(f"Creating resized dataset wrapper with target size {target_size}")
            logger.info(f"Will resize the following keys: {image_keys}")
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            
            # Resize all image inputs to the target size
            for key in self.image_keys:
                if key in item and isinstance(item[key], torch.Tensor):
                    # Handle different possible image formats
                    if len(item[key].shape) == 3:  # Single image (C, H, W)
                        item[key] = self.resize_transform(item[key])
                    elif len(item[key].shape) == 4:  # Sequence of images (S, C, H, W)
                        s, c, h, w = item[key].shape
                        # Process each image in the sequence separately
                        resized = torch.zeros((s, c, *self.target_size), dtype=item[key].dtype, device=item[key].device)
                        for i in range(s):
                            resized[i] = self.resize_transform(item[key][i])
                        item[key] = resized
            
            return item
        
    def dataprep(self):
        # Make sure the output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Get dataset metadata
        self.dataset_metadata = self._with_retry(
            lambda: LeRobotDatasetMetadata(self.repo_id),
            "Getting dataset metadata"
        )
        
        # Extract features
        self.features = dataset_to_policy_features(self.dataset_metadata.features)
        self.output_features = {key: ft for key, ft in self.features.items() if ft.type is FeatureType.ACTION}
        
        # Find all image features and state features
        all_image_keys = [key for key in self.features if key.startswith("observation.images")]
        self.image_keys = all_image_keys  # Store for later use
        logger.info(f"All available image sources: {all_image_keys}")
        
        # Get a reference to the robot state feature if it exists
        robot_state_key = next((key for key in self.features if key == "observation.state"), None)
        env_state_key = next((key for key in self.features if key == "observation.environment_state"), None)
        
        # Create input features dictionary
        # For ACT, we need at least one image feature or environment state
        self.input_features = {}
        
        # Add robot state if it exists
        if robot_state_key:
            self.input_features[robot_state_key] = self.features[robot_state_key]
            
        # Add environment state if it exists
        if env_state_key:
            self.input_features[env_state_key] = self.features[env_state_key]
            
        # Add all image features
        for key in all_image_keys:
            self.input_features[key] = self.features[key]
        
        logger.info(f"Input features for config: {list(self.input_features.keys())}")
        logger.info(f"Output features: {list(self.output_features.keys())}")
        
        # Validate that we have the required inputs for ACT
        if not any(key.startswith("observation.images") for key in self.input_features) and env_state_key not in self.input_features:
            raise ValueError("ACT policy requires at least one image input or environment state")
        
        # Create ACT config with parameters from the reference file
        self.config = ACTConfig(
            input_features=self.input_features,
            output_features=self.output_features,
            
            # ACT specific parameters matching configuration_act.py
            n_obs_steps=1,
            chunk_size=100,
            n_action_steps=100,
            
            # Architecture parameters
            vision_backbone="resnet18",
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            pre_norm=False,
            dim_model=512,
            n_heads=8,
            dim_feedforward=3200,
            feedforward_activation="relu",
            n_encoder_layers=4,
            n_decoder_layers=1,
            
            # VAE parameters
            use_vae=True,
            latent_dim=32,
            n_vae_encoder_layers=4,
            
            # Training parameters
            dropout=0.1,
            kl_weight=10.0,
            
            # Optimizer parameters
            optimizer_lr=1e-5,
            optimizer_weight_decay=1e-4,
            optimizer_lr_backbone=1e-5,
            
            # Set device
            device=str(self.device)
        )
        
        # Create delta timestamps for ALL features
        self.delta_timestamps = {}
        
        # Add observation timestamps for all input features (observations)
        for key in self.input_features:
            # For ACT, we only use the current observation (n_obs_steps=1)
            self.delta_timestamps[key] = [0.0]
        
        # Add action timestamps for the chunk size
        self.delta_timestamps["action"] = [
            i / self.dataset_metadata.fps for i in range(self.config.chunk_size)
        ]
        
        logger.info(f"Delta timestamps: {self.delta_timestamps}")
        
        # Create the dataset
        original_dataset = self._with_retry(
            lambda: LeRobotDataset(self.repo_id, delta_timestamps=self.delta_timestamps),
            "Creating dataset"
        )
        
        # Create the resized dataset wrapper if we have image inputs
        if all_image_keys:
            self.dataset = self.ResizedImageDatasetWrapper(
                original_dataset, 
                all_image_keys,
                target_size=(84, 84)  # Standard size for many robot vision models
            )
        else:
            self.dataset = original_dataset
        
    def train(self):
        # Ensure data is prepared
        if self.dataset_metadata is None:
            self.dataprep()
        
        # Create a custom ACT policy subclass that won't try to extract image features
        # from specific keys but instead will use what we provide
        class CustomACTPolicy(ACTPolicy):
            def forward(self, batch):
                # Don't modify the batch - assume it already has observation.images 
                # formed correctly
                return super().forward(batch)
        
        # Create and configure the policy directly
        logger.info("Creating ACT policy")
        self.policy = CustomACTPolicy(
            config=self.config,
            dataset_stats=self.dataset_metadata.stats
        )
        self.policy.train()
        self.policy.to(self.device)
        
        # Create optimizer - use the ACT-specific parameter groups
        logger.info("Creating optimizer")
        optimizer_params = self.policy.get_optim_params()
        optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.config.optimizer_lr,
            weight_decay=self.config.optimizer_weight_decay,
        )
        
        # Set up mixed precision training if using CUDA
        grad_scaler = GradScaler(enabled=self.use_amp)
        
        # Create dataloader with appropriate handling for action sequences
        def collate_fn(batch):
            """Custom collation function to handle dynamic keys in batches"""
            # First, identify all unique keys across all batch items
            all_keys = set()
            for item in batch:
                all_keys.update(item.keys())
            
            # Now create the collated batch with all found keys
            collated = {}
            for key in all_keys:
                # Skip keys that are explicitly for padding flags
                if key.endswith("_is_pad"):
                    continue
                
                # Find all items that have this key
                items_with_key = [item for item in batch if key in item]
                
                # Skip if no items have this key (shouldn't happen, but for safety)
                if not items_with_key:
                    continue
                
                # Get the values for this key
                values = [item[key] for item in items_with_key]
                
                # Handle tensor data - check first value
                if isinstance(values[0], torch.Tensor):
                    # For tensors, we need to handle potential shape mismatches
                    try:
                        # Try standard stacking first
                        collated[key] = torch.stack(values)
                    except RuntimeError as e:
                        logger.warning(f"Cannot stack tensors for key '{key}': {e}")
                        # If stacking fails, store as a list (will be handled in preprocessing)
                        collated[key] = values
                else:
                    # For non-tensor data, just store as a list
                    collated[key] = values
            
            # For ACT, we need to create an action_is_pad tensor
            # This indicates which actions in the sequence are valid (not padding)
            if "action" in collated:
                # Default: all actions are valid (not padding)
                collated["action_is_pad"] = torch.zeros(
                    (len(batch), self.config.chunk_size), 
                    dtype=torch.bool
                )
            
            return collated
        
        logger.info("Creating dataloader")
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=2,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.device.type != "cpu",
            drop_last=False,
            collate_fn=collate_fn
        )
        
        # Debug the first batch
        try:
            first_batch = next(iter(dataloader))
            logger.info("First batch keys:")
            for key, value in first_batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    logger.info(f"  {key}: list of {len(value)} tensors, first shape={value[0].shape}, dtype={value[0].dtype}")
                else:
                    logger.info(f"  {key}: {type(value)}")
        except Exception as e:
            logger.error(f"Error examining first batch: {e}")
            raise
        
        # Process batch for ACT model
        def preprocess_batch(batch):
            """Convert batch format to what ACT expects, handling dynamic key structures"""
            processed = {}
            
            # Original image keys from the dataset
            image_keys = self.image_keys
            
            # Also prepare individual image keys for the ACT policy
            # Map all original image keys to their corresponding processed tensors
            image_tensors = []
            for key in image_keys:
                if key in batch:
                    img = batch[key]
                    # Standardize the shape to (B, C, H, W)
                    if len(img.shape) == 4 and img.shape[1] == 1:  # (B, S=1, C, H, W)
                        img = img[:, 0]  # Take first frame
                    
                    # Also add the individual image tensors to the processed batch
                    # This ensures both approaches (list and individual tensors) will work
                    processed[key] = img
                    image_tensors.append(img)
            
            # Set the image list for ACT
            processed["observation.images"] = image_tensors
            
            # Add robot state if it exists
            if "observation.state" in batch:
                state = batch["observation.state"]
                # If state has sequence dimension, take first frame
                if len(state.shape) == 3 and state.shape[1] == 1:  # (B, S=1, D)
                    state = state[:, 0]  # -> (B, D)
                processed["observation.state"] = state
            
            # Add environment state if it exists
            if "observation.environment_state" in batch:
                env_state = batch["observation.environment_state"]
                # If env_state has sequence dimension, take first frame
                if len(env_state.shape) == 3 and env_state.shape[1] == 1:  # (B, S=1, D)
                    env_state = env_state[:, 0]  # -> (B, D)
                processed["observation.environment_state"] = env_state
            
            # Add action and padding mask
            if "action" in batch:
                processed["action"] = batch["action"]
                processed["action_is_pad"] = batch["action_is_pad"]
            
            return processed
        
        # Training metrics
        train_metrics = {
            "loss": 0.0,
            "l1_loss": 0.0,
            "kld_loss": 0.0,
            "grad_norm": 0.0,
        }
        
        # Run training loop
        logger.info("Starting training loop")
        step = 0
        done = False
        
        try:
            while not done:
                for batch in dataloader:
                    # Move batch to device
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(self.device, non_blocking=True)
                        elif isinstance(batch[key], list) and all(isinstance(item, torch.Tensor) for item in batch[key]):
                            batch[key] = [item.to(self.device, non_blocking=True) for item in batch[key]]
                    
                    try:
                        # Preprocess batch for ACT
                        processed_batch = preprocess_batch(batch)
                        
                        # Training step (similar to update_policy in train.py)
                        with torch.autocast(device_type=self.device.type) if self.use_amp else nullcontext():
                            loss, output_dict = self.policy.forward(processed_batch)
                        
                        # Scale loss and backpropagate
                        grad_scaler.scale(loss).backward()
                        
                        # Unscale gradients for clipping
                        grad_scaler.unscale_(optimizer)
                        
                        # Clip gradients (as in train.py)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(),
                            1.0,  # Clip at 1.0 as a sane default
                            error_if_nonfinite=False
                        )
                        
                        # Update weights
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        optimizer.zero_grad()
                        
                        # Update metrics
                        train_metrics["loss"] = loss.item()
                        train_metrics["grad_norm"] = grad_norm.item()
                        
                        for k, v in output_dict.items():
                            if k in train_metrics:
                                train_metrics[k] = v
                        
                        # Logging
                        if step % self.log_freq == 0:
                            log_str = f"Step: {step}/{self.training_steps} | Loss: {train_metrics['loss']:.4f}"
                            if "l1_loss" in output_dict:
                                log_str += f" | L1: {train_metrics['l1_loss']:.4f}"
                            if "kld_loss" in output_dict:
                                log_str += f" | KLD: {train_metrics['kld_loss']:.4f}"
                            log_str += f" | Grad: {train_metrics['grad_norm']:.4f}"
                            logger.info(log_str)
                        
                        # Save checkpoint
                        if self.save_freq > 0 and (step % self.save_freq == 0 or step == self.training_steps) and step > 0:
                            checkpoint_path = self.output_directory / f"checkpoint_{step}"
                            logger.info(f"Saving checkpoint at step {step}")
                            self.policy.save_pretrained(checkpoint_path)
                            logger.info(f"Saved checkpoint to {checkpoint_path}")
                        
                        step += 1
                        if step >= self.training_steps:
                            done = True
                            break
                            
                    except Exception as e:
                        logger.error(f"Error in training step {step}: {e}")
                        
                        # Print batch shape information for debugging
                        logger.info("Batch structure that caused the error:")
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                                logger.info(f"  {key}: list of {len(value)} tensors, first shape={value[0].shape}")
                            else:
                                logger.info(f"  {key}: {type(value)}")
                        
                        # If processed batch was created, log it too
                        if 'processed_batch' in locals():
                            logger.info("Processed batch structure:")
                            for key, value in processed_batch.items():
                                if isinstance(value, torch.Tensor):
                                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                                    logger.info(f"  {key}: list of {len(value)} tensors, first shape={value[0].shape}")
                                else:
                                    logger.info(f"  {key}: {type(value)}")
                        
                        # If this is the first step, something fundamental is wrong, so raise
                        if step == 0:
                            raise
                        else:
                            logger.warning("Continuing despite error...")
                            continue
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            if step > 0:
                self.policy.save_pretrained(self.output_directory)
                logger.info(f"Partial policy saved to {self.output_directory}")
        
        # Save the final policy checkpoint
        if done:
            self.policy.save_pretrained(self.output_directory)
            logger.info(f"Final policy saved to {self.output_directory}")
        
    def run(self):
        """Main method to execute the full training pipeline"""
        try:
            self.dataprep()
            self.train()
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise


# Example usage:
def main():
    # Replace with your actual dataset repo id
    repo_id = "abhisb/so100_51_ep"  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set to your Hugging Face token if you want to auto-tag the dataset
    hf_token = ""  # Replace with "hf_..."
    
    # Initialize with more retries and ability to add version tag
    trainer = TrainACTPolicy(
        repo_id=repo_id, 
        device=device,
        n_steps=5000,
        batch_size=32,
        max_retries=10, 
        retry_delay=30,
        add_version_tag=True,  # Set to True if you want to automatically add version tag
        hf_token=hf_token  # Required if add_version_tag is True
    )
    trainer.run()

if __name__ == "__main__":
    main()
