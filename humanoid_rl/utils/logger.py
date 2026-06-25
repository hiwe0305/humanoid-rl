"""
humanoid_rl/utils/logger.py
Unified logger: WandB + TensorBoard + console.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


class Logger:
    """
    Logs metrics to WandB and TensorBoard simultaneously.
    Gracefully falls back if either is not configured.
    """

    def __init__(self, log_dir: str, project: str = "humanoid-rl", run_name: str | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer  = None
        self._wandb_run  = None

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(str(self.log_dir))
            print(f"[Logger] TensorBoard: {self.log_dir}")
        except Exception as e:
            print(f"[Logger] TensorBoard unavailable: {e}")

        # WandB (only if WANDB_API_KEY is set or user is logged in)
        try:
            import wandb
            if os.environ.get("WANDB_API_KEY") or wandb.api.api_key:
                self._wandb_run = wandb.init(
                    project=project,
                    name=run_name,
                    dir=str(self.log_dir),
                )
                print(f"[Logger] WandB: {project}/{run_name}")
        except Exception as e:
            print(f"[Logger] WandB offline (set WANDB_API_KEY to enable): {e}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    self._tb_writer.add_scalar(k, float(v), global_step=step)

        if self._wandb_run is not None:
            self._wandb_run.log(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                step=step
            )

    def log_video(self, tag: str, frames: list, step: int, fps: int = 30) -> None:
        """Log a video (list of RGB numpy arrays)."""
        if self._tb_writer is not None:
            import torch
            # TensorBoard expects [T, C, H, W] uint8
            video = np.stack(frames, axis=0)
            video_t = torch.as_tensor(video).permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, C, H, W]
            self._tb_writer.add_video(tag, video_t, global_step=step, fps=fps)

        if self._wandb_run is not None:
            import wandb
            video = np.stack(frames, axis=0)  # [T, H, W, C]
            self._wandb_run.log({tag: wandb.Video(video.transpose(0, 3, 1, 2), fps=fps)}, step=step)

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
