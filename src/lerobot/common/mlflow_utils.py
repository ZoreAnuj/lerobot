#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""MLflow experiment tracking, mirroring the interface of `lerobot.common.wandb_utils.WandBLogger` so the
training script can use either backend interchangeably."""

import logging
from pathlib import Path

from termcolor import colored

from lerobot.common.wandb_utils import cfg_to_group
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.import_utils import require_package


class MLflowLogger:
    """A helper class to log objects using MLflow.

    Exposes the same methods as `WandBLogger` (`log_dict`, `log_policy`, `log_video`) so the two are
    interchangeable in the training loop. The tracking URI defaults to a local file store under the run's
    output directory; point it at an MLflow tracking server with `--mlflow.tracking_uri=http://host:port`.
    """

    def __init__(self, cfg: TrainPipelineConfig):
        require_package("mlflow", extra="mlflow")
        import mlflow

        self.cfg = cfg.mlflow
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        if self.cfg.tracking_uri:
            tracking_uri = self.cfg.tracking_uri
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.cfg.experiment)
        else:
            # Local run store: SQLite backend (MLflow >= 3 deprecates the plain file store) with artifacts
            # next to it. Browse later with `mlflow ui --backend-store-uri sqlite:///<path>/mlflow.db`.
            mlflow_dir = Path(self.log_dir).resolve() / "mlflow"
            mlflow_dir.mkdir(parents=True, exist_ok=True)
            tracking_uri = f"sqlite:///{mlflow_dir / 'mlflow.db'}"
            mlflow.set_tracking_uri(tracking_uri)
            if mlflow.get_experiment_by_name(self.cfg.experiment) is None:
                mlflow.create_experiment(
                    self.cfg.experiment, artifact_location=f"file:{mlflow_dir / 'artifacts'}"
                )
            mlflow.set_experiment(self.cfg.experiment)

        run_id = self.cfg.run_id if (self.cfg.run_id and cfg.resume) else None
        run = mlflow.start_run(run_id=run_id, run_name=self.job_name, log_system_metrics=True)
        # Persist the run id in the config so checkpoints can resume into the same MLflow run.
        cfg.mlflow.run_id = run.info.run_id

        if not cfg.resume:
            mlflow.set_tags({"group": self._group})
            params = _flatten_dict(cfg.to_dict())
            # MLflow rejects values longer than 6000 chars; truncate defensively.
            mlflow.log_params({k: str(v)[:6000] for k, v in params.items()})

        logging.info(colored("Logs will be synced with MLflow.", "blue", attrs=["bold"]))
        logging.info(
            f"Track this run --> {colored(f'{tracking_uri} (run {run.info.run_id})', 'yellow', attrs=['bold'])}"
        )
        self._mlflow = mlflow

    def log_policy(self, checkpoint_dir: Path):
        """Upload the policy checkpoint to the MLflow artifact store."""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
        if not pretrained_model_dir.exists():
            logging.warning(f"No {pretrained_model_dir} found. Skipping model artifact upload to MLflow.")
            return
        self._mlflow.log_artifacts(str(pretrained_model_dir), artifact_path=f"checkpoints/{step_id}")

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")
        if step is None:
            step = int(d[custom_step_key])

        metrics = {}
        for k, v in d.items():
            if isinstance(v, bool) or not isinstance(v, (int | float)):
                # MLflow metrics are numeric-only; strings (and bools) go to tags instead of metrics.
                continue
            metrics[f"{mode}/{k}"] = float(v)

        if metrics:
            self._mlflow.log_metrics(metrics, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        self._mlflow.log_artifact(video_path, artifact_path=f"{mode}/videos/step_{step:09d}")


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict into {'a.b.c': value} form for MLflow params."""
    items: dict = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key, sep=sep))
        else:
            items[key] = v
    return items
