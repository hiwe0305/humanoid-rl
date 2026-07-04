"""
humanoid_rl/utils/video_recorder.py
VideoRecorder — tự động quay video training progress.

Periodically renders environment frames and saves as mp4.
Helps monitor training progress visually without display.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import mujoco


class VideoRecorder:
    """
    Records training video periodically.
    
    Args:
        env_fn:         callable that creates env (for rendering)
        log_dir:        directory to save videos
        record_every:   record every N PPO updates
        fps:            video framerate
        robot_name:     "h1" or "humanoid" (for model paths)
        duration:       video duration in env steps (default 200 = ~4s)
    """
    def __init__(
        self,
        env_fn,
        log_dir: str = "runs/ppo",
        record_every: int = 50,
        fps: int = 50,
        robot_name: str = "h1",
        duration: int = 200,
    ):
        self.env_fn = env_fn
        self.log_dir = Path(log_dir) / "videos"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.record_every = record_every
        self.fps = fps
        self.robot_name = robot_name
        self.duration = duration
        
        # Create single env for rendering
        self.env = env_fn()
        
        # Setup renderer
        try:
            self.renderer = mujoco.Renderer(self.env.model, 480, 640)
            self._render_ok = True
        except Exception:
            print(f"[VideoRecorder] Warning: cannot create renderer (EGL?)")
            self._render_ok = False
        
        # Camera setup
        self._cam = mujoco.MjvCamera()
        self._cam.type = 0  # free
        self._cam.azimuth, self._cam.elevation, self._cam.distance = 135, -20, 4.5
        self._cam.lookat[:] = [0, 0, 0.9]
        
        self._last_recorded = -self.record_every  # record on first eligible

    def step(self, update_count: int, metrics: dict) -> None:
        """Check if we should record this update."""
        if not self._render_ok:
            return
        if update_count - self._last_recorded < self.record_every:
            return
        self._last_recorded = update_count
        
        # Record video
        self._record(
            tag=f"update_{update_count}",
            ep_return=metrics.get("mean_ep_return", 0.0),
        )

    def _record(self, tag: str, ep_return: float) -> None:
        """Render and save video."""
        import cv2
        
        try:
            obs, _ = self.env.reset(seed=42)
            frames = []
            
            for step in range(self.duration):
                # Use zero action (PD holds default pose)
                action = np.zeros(self.env.action_space.shape, dtype=np.float32)
                obs, reward, term, trunc, info = self.env.step(action)
                
                # Render
                if hasattr(self.env, 'model'):
                    self._cam.lookat[:] = [
                        float(self.env.data.qpos[0]),
                        float(self.env.data.qpos[1]),
                        0.9,
                    ]
                    self.renderer.update_scene(self.env.data, camera=self._cam)
                    frame = self.renderer.render().copy()
                    frames.append(frame)
                
                if term or trunc:
                    break
            
            # Save video
            if frames:
                save_path = self.log_dir / f"{tag}_return_{ep_return:.0f}.mp4"
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(save_path), fourcc, self.fps, (w, h))
                for f in frames:
                    writer.write(f)
                writer.release()
                # Remove old videos if too many
                self._cleanup(max_videos=10)
            
        except Exception as e:
            print(f"[VideoRecorder] Error recording: {e}")

    def _cleanup(self, max_videos: int = 10):
        """Keep only the N most recent videos."""
        videos = sorted(self.log_dir.glob("*.mp4"))
        while len(videos) > max_videos:
            videos[0].unlink()
            videos = videos[1:]

    def close(self):
        """Cleanup resources."""
        if hasattr(self, 'renderer') and self.renderer:
            self.renderer.close()
        if hasattr(self, 'env') and self.env:
            self.env.close()
