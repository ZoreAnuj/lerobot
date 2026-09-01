#!/usr/bin/env python

# Copyright 2025 QUT Centre for Robotics and The HuggingFace Inc. team.
# All rights reserved.
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
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import CosineAnnealingWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("imle")
@dataclass
class IMLEConfig(PreTrainedConfig):
    """Configuration class for IMLEPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations,
    matching the reference implementation of "IMLE Policy: Fast and Sample Efficient Visuomotor Policy
    Learning via Implicit Maximum Likelihood Estimation" (https://huggingface.co/papers/2502.12371).

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Size of the action chunk generated in one forward pass of the generator. See
            `IMLEPolicy.select_action` for more details.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            See `IMLEPolicy.select_action` for more details.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        resize_shape: (H, W) shape to resize images to as a preprocessing step for the vision
            backbone. If None, no resizing is done and the original image resolution is used.
        crop_ratio: Ratio in (0, 1] used to derive the crop size from resize_shape
            (crop_h = int(resize_shape[0] * crop_ratio), likewise for width).
            Set to 1.0 to disable cropping. Only takes effect when resize_shape is not None.
        crop_shape: (H, W) shape to crop images to. When resize_shape is set and crop_ratio < 1.0,
            this is computed automatically. Can also be set directly for legacy configs that use
            crop-only (without resize). If None and no derivation applies, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center
            crop in eval mode).
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
            The group sizes are set to be about 16 (to be precise, feature_dim // 16).
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoder_per_camera: Whether to use a separate RGB encoder for each camera view.
        down_dims: Feature dimension for each stage of temporal downsampling in the generator Unet.
            You may provide a variable number of dimensions, therefore also controlling the degree of
            downsampling.
        kernel_size: The convolutional kernel size of the generator Unet.
        n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
        use_film_scale_modulation: FiLM (https://huggingface.co/papers/1709.07871) is used for the Unet
            conditioning. Bias modulation is used by default, while this parameter indicates whether to also
            use scale modulation.
        gradient_checkpointing: Whether to checkpoint the Unet residual blocks during training. This reduces
            activation memory at the cost of recomputing those blocks during the backward pass.
        n_samples_per_condition: Number of latent samples drawn per conditioning input during training. The
            RS-IMLE loss pulls the nearest (non-rejected) generated sample towards the ground-truth action
            chunk, so more samples give a better estimate of the nearest mode at the cost of memory/compute.
        rs_epsilon: Rejection sampling radius of RS-IMLE (in normalized action space): generated samples
            closer than this to the ground truth are discarded before the nearest-neighbour selection, which
            prevents the generator from collapsing samples onto individual data points.
        use_traj_consistency: Whether to use inference-time trajectory consistency: sample
            `n_consistency_candidates` action chunks in a single batch and select the one whose start is
            closest to the tail of the previously executed chunk. Helps with mode switching between replans
            on strongly multimodal tasks.
        n_consistency_candidates: Number of candidate chunks to sample when `use_traj_consistency` is set.
        consistency_reset_every: Reset the "previous chunk" anchor to random noise every this many policy
            replans, to avoid locking into a single mode. Set to 0 to never reset.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True
    # Unet.
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    use_film_scale_modulation: bool = True
    gradient_checkpointing: bool = False

    # RS-IMLE training.
    n_samples_per_condition: int = 20
    rs_epsilon: float = 0.03
    # Gripper-transition robustness. Close/open transitions are a tiny fraction of frames, and an
    # unweighted chunk distance lets the generator treat the gripper channel (conventionally the
    # LAST action/state dimension) as an afterthought — or worse, copy the observed gripper bit.
    # `rs_gripper_weight` multiplies the last action dimension's squared error inside the RS-IMLE
    # distance, affecting candidate selection, the loss, and the epsilon-rejection metric alike.
    # `transition_oversample` makes every frame whose next `horizon` actions contain a flip of the
    # last action dimension appear that many times per epoch (integer; 1 disables).
    # `gripper_obs_dropout` is the per-sample probability, during training only, of replacing the
    # observed gripper value with one taken from another sample in the batch — destroying the
    # input bit's predictive value so the policy must ground the transition visually.
    rs_gripper_weight: float = 1.0
    transition_oversample: int = 1
    gripper_obs_dropout: float = 0.0
    # When > 0, `transition_oversample` also repeats motion-onset frames: the moments where the
    # action vector changes again after being exactly constant for at least this many consecutive
    # frames (dwell exits, e.g. descending after a stationary hover). 0 disables onset detection.
    motion_onset_min_dwell: int = 0

    # Inference.
    use_traj_consistency: bool = False
    n_consistency_candidates: int = 32
    consistency_reset_every: int = 5

    # Optimization.
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        if self.n_samples_per_condition < 1:
            raise ValueError(
                f"`n_samples_per_condition` must be a positive integer. Got {self.n_samples_per_condition}."
            )
        if self.rs_epsilon < 0:
            raise ValueError(f"`rs_epsilon` must be non-negative. Got {self.rs_epsilon}.")
        if self.rs_gripper_weight <= 0:
            raise ValueError(f"`rs_gripper_weight` must be positive. Got {self.rs_gripper_weight}.")
        if self.transition_oversample < 1:
            raise ValueError(
                f"`transition_oversample` must be a positive integer. Got {self.transition_oversample}."
            )
        if self.motion_onset_min_dwell < 0:
            raise ValueError(
                f"`motion_onset_min_dwell` must be non-negative. Got {self.motion_onset_min_dwell}."
            )
        if not (0.0 <= self.gripper_obs_dropout <= 1.0):
            raise ValueError(f"`gripper_obs_dropout` must be in [0, 1]. Got {self.gripper_obs_dropout}.")
        if self.n_consistency_candidates < 1:
            raise ValueError(
                f"`n_consistency_candidates` must be a positive integer. Got {self.n_consistency_candidates}."
            )
        if self.use_traj_consistency and self.n_action_steps >= self.horizon:
            raise ValueError(
                "Trajectory consistency compares the tail of the previous chunk (of length "
                "`horizon - n_action_steps`) against candidate starts, so it requires "
                f"`n_action_steps < horizon`. Got {self.n_action_steps=} and {self.horizon=}."
            )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                # Explicitly disable cropping for resize+ratio path when crop_ratio == 1.0.
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineAnnealingWithWarmupSchedulerConfig:
        # Linear warmup then cosine annealing to zero: the same schedule as the reference implementation's
        # diffusers `get_scheduler("cosine", ...)`, without the diffusers dependency.
        return CosineAnnealingWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
