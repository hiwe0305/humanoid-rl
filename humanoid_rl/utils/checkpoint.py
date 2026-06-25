"""
humanoid_rl/utils/checkpoint.py
Model checkpointing utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class Checkpoint:
    """Save and load model checkpoints."""

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, state: dict[str, Any], tag: str = "latest") -> Path:
        path = self.directory / f"checkpoint_{tag}.pt"
        torch.save(state, path)
        # Always also write to 'latest'
        latest = self.directory / "checkpoint_latest.pt"
        torch.save(state, latest)
        print(f"[Checkpoint] Saved: {path}")
        return path

    def load(self, path: str | None = None) -> dict[str, Any]:
        if path is None:
            path = self.directory / "checkpoint_latest.pt"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu")
        print(f"[Checkpoint] Loaded: {path}")
        return state

    def list_checkpoints(self) -> list[Path]:
        return sorted(self.directory.glob("checkpoint_*.pt"))
