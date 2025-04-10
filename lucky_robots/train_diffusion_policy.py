
from pathlib import Path
import torch
import time
import random
import logging
import torchvision.transforms as T
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainPolicy():
    def __init__(self, repo_id, device=None, n_steps=5000, log_freq=1, max_retries=5, retry_delay=10):
        self.repo_id = repo_id
        self.output_directory = Path(f"outputs/train/{repo_id}")
        
        # Handle device properly
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
        else:
            self.device = device
            
        self.training_steps = n_steps
        self.log_freq = log_freq
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dataset_metadata = None
        self.features = None
        self.output_features = None
        self.input_features = None
        self.config = None
        self.policy = None
        
    def _with_retry(self, fn, operation_name):
        """Execute a function with retry logic for handling rate limits"""
        retries = 0
        while retries < self.max_retries:
            try:
                logger.info(f"Attempting {operation_name}...")
                result = fn()
                logger.info(f"{operation_name} successful!")
                return result
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
        logger.info(f"All available image sources: {all_image_keys}")
        
        # Choose one reference image key for our configuration
        reference_image_key = all_image_keys[0] if all_image_keys else None
        
        # Create input features dictionary with only one image source for config
        # (We'll still use all images in training through our wrapper)
        if reference_image_key:
            self.input_features = {
                key: self.features[key] for key in self.features 
                if (key not in self.output_features and 
                    (not key.startswith("observation.images") or key == reference_image_key))
            }
        else:
            self.input_features = {
                key: self.features[key] for key in self.features if key not in self.output_features
            }
        
        logger.info(f"Input features for config: {list(self.input_features.keys())}")
        logger.info(f"Output features: {list(self.output_features.keys())}")
        
        # Create the configuration using our filtered inputs
        self.config = DiffusionConfig(
            input_features=self.input_features,
            output_features=self.output_features,
            # Ensure crop_shape is set appropriately
            crop_shape=(84, 84)
        )
        
        # Create delta timestamps for ALL features (including all image sources)
        self.delta_timestamps = {}
        
        # Get all available keys from the original features
        all_feature_keys = list(self.features.keys())
        
        # Add observation timestamps for all input features
        for key in all_feature_keys:
            if key not in self.output_features:
                self.delta_timestamps[key] = [
                    i / self.dataset_metadata.fps for i in self.config.observation_delta_indices
                ]
        
        # Add action timestamps
        self.delta_timestamps["action"] = [
            i / self.dataset_metadata.fps for i in self.config.action_delta_indices
        ]
        
        logger.info(f"Delta timestamps: {self.delta_timestamps}")
        
        # Create the dataset
        original_dataset = self._with_retry(
            lambda: LeRobotDataset(self.repo_id, delta_timestamps=self.delta_timestamps),
            "Creating dataset"
        )
        
        # Create the resized dataset wrapper
        self.dataset = self.ResizedImageDatasetWrapper(
            original_dataset, 
            all_image_keys,
            target_size=(84, 84)
        )
        
    def train(self):
        # Ensure data is prepared
        if self.dataset_metadata is None:
            self.dataprep()
        
        # Create and configure the policy
        self.policy = DiffusionPolicy(self.config, dataset_stats=self.dataset_metadata.stats)
        self.policy.train()
        self.policy.to(self.device)
        
        # Create optimizer and dataloader
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=1,
            batch_size=64,
            shuffle=True,
            pin_memory=self.device.type != "cpu",
            drop_last=True,
        )
        
        # Debug the first batch
        try:
            first_batch = next(iter(dataloader))
            logger.info("First batch keys:")
            for key, value in first_batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logger.info(f"  {key}: {type(value)}")
        except Exception as e:
            logger.error(f"Error examining first batch: {e}")
        
        # Run training loop
        step = 0
        done = False
        
        try:
            while not done:
                for batch in dataloader:
                    batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                    
                    try:
                        loss, _ = self.policy.forward(batch)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        if step % self.log_freq == 0:
                            logger.info(f"step: {step} loss: {loss.item():.3f}")
                        
                        step += 1
                        if step >= self.training_steps:
                            done = True
                            break
                    except Exception as e:
                        logger.error(f"Error in training step {step}: {e}")
                        
                        # Print batch shape information for debugging
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                            
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
        
        # Save the policy checkpoint
        if done:
            self.policy.save_pretrained(self.output_directory)
            logger.info(f"Policy saved to {self.output_directory}")
        
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
    repo_id = "abhisb/so100_51_ep"  # Changed to match the error message
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize with more retries and longer delay between retries
    trainer = TrainPolicy(
        repo_id, 
        device, 
        max_retries=10, 
        retry_delay=30
    )
    trainer.run()

if __name__ == "__main__":
    main()