from pathlib import Path
import torch
import time
import random
import logging
import argparse
import torchvision.transforms as T
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType
from contextlib import nullcontext
from torch.amp import GradScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolicyTrainer:
    """Flexible policy trainer that supports multiple policy types"""
    
    def __init__(self, 
                 repo_id, 
                 policy_type="diffusion",
                 device=None, 
                 n_steps=5000, 
                 batch_size=64,
                 learning_rate=1e-4,
                 grad_clip_norm=1.0,
                 use_amp=False,
                 log_freq=10, 
                 max_retries=5, 
                 retry_delay=10):
        """
        Initialize the policy trainer
        
        Args:
            repo_id: The repository ID for the dataset
            policy_type: Type of policy to train (diffusion, bc, etc.)
            device: Training device
            n_steps: Number of training steps
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            grad_clip_norm: Gradient clipping norm
            use_amp: Whether to use automatic mixed precision
            log_freq: How often to log training progress
            max_retries: Maximum number of retries for API rate limits
            retry_delay: Delay between retries
        """
        self.repo_id = repo_id
        self.policy_type = policy_type
        self.output_directory = Path(f"outputs/train/{policy_type}/{repo_id}")
        
        # Handle device properly
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
        else:
            self.device = device
            
        self.training_steps = n_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.grad_clip_norm = grad_clip_norm
        self.use_amp = use_amp
        self.log_freq = log_freq
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize placeholder variables
        self.dataset_metadata = None
        self.features = None
        self.output_features = None
        self.input_features = None
        self.config = None
        self.policy = None
        self.grad_scaler = None
        
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
        """Prepare the dataset for training"""
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
        
        # Find all image features
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
        
        # Create the configuration based on policy type
        self._create_policy_config()
        
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
    
    def _create_policy_config(self):
        """Create policy configuration based on the selected policy type"""
        if self.policy_type == "diffusion":
            self.config = DiffusionConfig(
                input_features=self.input_features,
                output_features=self.output_features,
                crop_shape=(84, 84)
            )
        # Add configurations for other policy types here
        elif self.policy_type == "bc":
            # Example for a Behavior Cloning policy configuration
            # This would need to be implemented with your specific BC config
            from lerobot.common.policies.bc.configuration_bc import BCConfig
            self.config = BCConfig(
                input_features=self.input_features,
                output_features=self.output_features,
                crop_shape=(84, 84)
            )
        else:
            raise ValueError(f"Unsupported policy type: {self.policy_type}")
    
    def _create_policy(self):
        """Create the policy model based on configuration"""
        # Create policy based on the type
        if self.policy_type == "diffusion":
            self.policy = DiffusionPolicy(self.config, dataset_stats=self.dataset_metadata.stats)
        # Add initializations for other policy types here
        elif self.policy_type == "bc":
            # Example for a Behavior Cloning policy
            from lerobot.common.policies.bc.modeling_bc import BCPolicy
            self.policy = BCPolicy(self.config, dataset_stats=self.dataset_metadata.stats)
        else:
            # Try to use the factory method from the reference code
            try:
                self.policy = make_policy(
                    cfg=self.config,
                    ds_meta=self.dataset_metadata,
                )
            except Exception as e:
                logger.error(f"Failed to create policy using factory: {e}")
                raise ValueError(f"Unsupported policy type: {self.policy_type}")
                
        self.policy.train()
        self.policy.to(self.device)
        
    def train(self):
        """Train the policy"""
        # Ensure data is prepared
        if self.dataset_metadata is None:
            self.dataprep()
        
        # Create and configure the policy
        self._create_policy()
        
        # Initialize grad scaler for mixed precision training
        self.grad_scaler = GradScaler(device_type=self.device.type, enabled=self.use_amp)
        
        # Create optimizer and dataloader
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=1,
            batch_size=self.batch_size,
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
                        # Forward pass with mixed precision if enabled
                        with torch.autocast(device_type=self.device.type) if self.use_amp else nullcontext():
                            loss, output_dict = self.policy.forward(batch)
                        
                        # Backward pass with gradient scaling
                        self.grad_scaler.scale(loss).backward()
                        
                        # Unscale for gradient clipping
                        self.grad_scaler.unscale_(optimizer)
                        
                        # Clip gradients
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(),
                            self.grad_clip_norm,
                            error_if_nonfinite=False
                        )
                        
                        # Step optimizer with scaling
                        self.grad_scaler.step(optimizer)
                        self.grad_scaler.update()
                        optimizer.zero_grad()
                        
                        if step % self.log_freq == 0:
                            logger.info(f"step: {step} loss: {loss.item():.3f} grad_norm: {grad_norm.item():.3f}")
                        
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


def main():
    """Parse command line arguments and run the trainer"""
    parser = argparse.ArgumentParser(description="Train a robotic policy")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID for the dataset")
    parser.add_argument("--policy_type", type=str, default="diffusion", 
                        choices=["diffusion", "bc"], 
                        help="Type of policy to train")
    parser.add_argument("--device", type=str, default=None, 
                        help="Training device (default: use CUDA if available)")
    parser.add_argument("--n_steps", type=int, default=5000, 
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Learning rate for optimizer")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, 
                        help="Gradient clipping norm")
    parser.add_argument("--use_amp", action="store_true", 
                        help="Use automatic mixed precision training")
    parser.add_argument("--log_freq", type=int, default=10, 
                        help="How often to log training progress")
    parser.add_argument("--max_retries", type=int, default=10, 
                        help="Maximum number of retries for API rate limits")
    parser.add_argument("--retry_delay", type=int, default=30, 
                        help="Delay between retries in seconds")
    
    args = parser.parse_args()
    
    # Set device from args or use CUDA if available
    device = None
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize trainer with parsed arguments
    trainer = PolicyTrainer(
        repo_id=args.repo_id,
        policy_type=args.policy_type,
        device=device,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
        use_amp=args.use_amp,
        log_freq=args.log_freq,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay
    )
    trainer.run()

if __name__ == "__main__":
    main()