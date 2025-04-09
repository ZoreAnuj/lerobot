from pathlib import Path
import torch
import time
import random
import logging
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
        
        # Extract all input features for the policy
        self.input_features = {key: ft for key, ft in self.features.items() if key not in self.output_features}
        
        # Log the features we found
        image_keys = [key for key in self.input_features if key.startswith("observation.images")]
        state_keys = [key for key in self.input_features if key.startswith("observation.state")]
        
        logger.info(f"Input features: {list(self.input_features.keys())}")
        logger.info(f"Image features: {image_keys}")
        logger.info(f"State features: {state_keys}")
        logger.info(f"Output features: {list(self.output_features.keys())}")
        
        # Create the configuration with appropriate settings based on dataset
        self.config = DiffusionConfig(
            input_features=self.input_features,
            output_features=self.output_features,
            # Configure for multiple cameras if needed
            use_separate_rgb_encoder_per_camera=(len(image_keys) > 1)
        )
        
        # Create the delta timestamps dictionary
        self.delta_timestamps = {}
        
        # Add timestamps for observation state
        if state_keys:
            self.delta_timestamps["observation.state"] = [
                i / self.dataset_metadata.fps for i in self.config.observation_delta_indices
            ]
        
        # Add timestamps for each image feature
        for key in image_keys:
            self.delta_timestamps[key] = [
                i / self.dataset_metadata.fps for i in self.config.observation_delta_indices
            ]
        
        # Add action timestamps
        self.delta_timestamps["action"] = [
            i / self.dataset_metadata.fps for i in self.config.action_delta_indices
        ]
        
        logger.info(f"Delta timestamps: {self.delta_timestamps}")
        
        # Create the dataset
        self.dataset = self._with_retry(
            lambda: LeRobotDataset(self.repo_id, delta_timestamps=self.delta_timestamps),
            "Creating dataset"
        )
        
    def train(self):
        # Ensure data is prepared
        if self.dataset_metadata is None:
            self.dataprep()
        
        # Create and configure the policy
        self.policy = DiffusionPolicy(self.config, dataset_stats=self.dataset_metadata.stats)
        self.policy.train()
        self.policy.to(self.device)
        
        # Log the policy structure
        logger.info(f"Policy created with config: {self.config.__class__.__name__}")
        
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
        first_batch = next(iter(dataloader))
        logger.info("First batch keys:")
        for key, value in first_batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                logger.info(f"  {key}: {type(value)}")
        
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
                        # For debugging purposes, print tensor shapes
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                logger.info(f"  {key}: shape={value.shape}")
                        
                        # Re-raise if this is the first step
                        if step == 0:
                            raise
                        else:
                            logger.warning("Skipping this batch and continuing...")
                            continue
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Try to save model
            if step > 0:
                try:
                    logger.info("Saving partial model...")
                    self.policy.save_pretrained(self.output_directory)
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")
        
        # Save the policy checkpoint
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
    repo_id = "jchun/so100_pickplace_small_20250323_120056"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize with more retries and longer delay between retries
    trainer = TrainPolicy(repo_id, device, max_retries=10, retry_delay=30)
    trainer.run()

if __name__ == "__main__":
    main()