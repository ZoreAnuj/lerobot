# IMLE Policy

One-step generative visuomotor policy trained with Rejection-Sampling Implicit Maximum Likelihood
Estimation (RS-IMLE). The policy maps a trajectory-shaped Gaussian latent plus observation features to a
full action chunk in a single forward pass (no iterative denoising), which makes inference roughly an order
of magnitude faster than Diffusion Policy while remaining multimodal and notably sample-efficient in
low-data regimes.

## Paper

- Project page: https://imle-policy.github.io/
- Paper: https://huggingface.co/papers/2502.12371 (RSS 2025)
- Reference implementation: https://github.com/krishanrana/imle_policy

## Method summary

During training, for every conditioning input the generator produces `n_samples_per_condition` candidate
chunks from independent latents. Candidates within `rs_epsilon` (L2 in normalized action space) of the
ground-truth chunk are rejected, and the loss pulls the nearest surviving candidate towards the ground
truth. Rejection prevents the generator from collapsing samples onto individual data points, which is what
makes the method work with very few demonstrations.

At inference, one forward pass yields the chunk. For strongly multimodal tasks, enable
`--policy.use_traj_consistency=true`: the policy then samples `n_consistency_candidates` chunks in a single
batch and executes the one whose start best matches the tail of the previously executed chunk, avoiding
mode switching between replans.

## Gripper-transition robustness

Binary gripper transitions are a tiny fraction of frames (often <1%), and with an unweighted chunk
distance the loss-cheapest strategy for the gripper channel is to copy the observed gripper bit —
which deadlocks at deployment when the observed bit is rebuilt from the policy's own last command.
Three flags counter this (the gripper is assumed to be the LAST state/action dimension):

```bash
lerobot-train --policy.type=imle \
  --policy.rs_gripper_weight=5.0 \      # weight the gripper channel inside the IMLE distance
  --policy.transition_oversample=10 \   # transition-window frames appear 10x per epoch
  --policy.gripper_obs_dropout=0.2 \    # randomly swap the observed gripper bit between samples
  ...
```

`rs_gripper_weight` reshapes candidate selection, the loss, and the epsilon-rejection metric
consistently. `transition_oversample` rebalances sampling toward chunks that contain a flip.
`gripper_obs_dropout` destroys the input bit's predictive value so open/close must be grounded
visually. Watch `gripper_err_selected` in the training logs — plain loss and step-level gripper
accuracy provably cannot see the copy-shortcut failure (a copy baseline scores ~99%); evaluate
transition recall (does the policy flip the gripper near demonstrated transitions?) instead.

## Training

The reference implementation maintains an exponential moving average (EMA) of the policy weights during
training and evaluates the EMA weights. To reproduce this behavior, enable the trainer's EMA shadow:

```bash
lerobot-train \
  --policy.type=imle \
  --dataset.repo_id=lerobot/pusht \
  --env.type=pusht \
  --ema.enable=true \
  --ema.power=0.75
```

Checkpoints then contain a directly loadable copy of the EMA weights next to the live ones, e.g. for
evaluation:

```bash
lerobot-eval --policy.path=outputs/train/.../checkpoints/last/pretrained_model_ema ...
```

Note on memory: each training step runs the U-Net on `batch_size * n_samples_per_condition` latents (the
observation encoder runs once per conditioning input). Lower `--policy.n_samples_per_condition` or
`--batch_size` if you run out of VRAM.

## Citation

```bibtex
@inproceedings{rana2025imle,
	author = {Krishan Rana and Robert Lee and David Pershouse and Niko Suenderhauf},
	title = {IMLE Policy: Fast and Sample Efficient Visuomotor Policy Learning via Implicit Maximum Likelihood Estimation},
	booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
	year = {2025},
}
```
