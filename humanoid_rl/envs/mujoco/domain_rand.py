"""
humanoid_rl/envs/mujoco/domain_rand.py
Domain Randomization — randomizes physics parameters each episode.

Improves sim-to-real transfer by training on a distribution of physics,
not a single fixed set of parameters.

Randomizes:
  - Link masses (±20%)
  - Floor friction
  - Joint damping
  - Actuator strength (gear ratio)
  - External perturbation pushes (optional)
"""
from __future__ import annotations

import numpy as np


class DomainRandomizer:
    """
    Randomizes MuJoCo model parameters at the start of each episode.

    Usage:
        rand = DomainRandomizer(model, cfg)
        # Call at each reset:
        rand.randomize(model, rng)

    Args:
        model:  mujoco.MjModel (modified in-place, original values stored)
        cfg:    randomization config dict
    """

    def __init__(self, model, cfg: dict | None = None):
        cfg = cfg or {}

        # Mass randomization
        self.mass_ratio     = cfg.get("mass_ratio",     0.2)   # ±20%
        self.mass_enabled   = cfg.get("mass_enabled",   True)

        # Friction randomization
        self.friction_ratio = cfg.get("friction_ratio", 0.3)   # ±30%
        self.friction_enabled = cfg.get("friction_enabled", True)

        # Damping randomization
        self.damping_ratio  = cfg.get("damping_ratio",  0.2)
        self.damping_enabled = cfg.get("damping_enabled", True)

        # Gear (actuator strength) randomization
        self.gear_ratio     = cfg.get("gear_ratio",     0.1)   # ±10%
        self.gear_enabled   = cfg.get("gear_enabled",   True)

        # Store original values to reset each episode
        self._orig_mass    = model.body_mass.copy()
        self._orig_friction = model.geom_friction.copy()
        self._orig_damping  = model.dof_damping.copy()
        self._orig_gear     = model.actuator_gear[:, 0].copy()

    def randomize(self, model, rng: np.random.Generator | None = None) -> None:
        """
        Apply randomization to model in-place.
        Call at the start of each episode (before mj_forward).
        """
        if rng is None:
            rng = np.random.default_rng()

        # 1. Mass: skip world body (body 0)
        if self.mass_enabled:
            n = len(self._orig_mass)
            factors = rng.uniform(1 - self.mass_ratio, 1 + self.mass_ratio, n)
            model.body_mass[:] = self._orig_mass * factors
            model.body_mass[0] = self._orig_mass[0]  # keep world body fixed

        # 2. Floor friction (geom_friction: [slide, torsion, rolling])
        if self.friction_enabled:
            n = len(self._orig_friction)
            factors = rng.uniform(1 - self.friction_ratio, 1 + self.friction_ratio, (n, 1))
            model.geom_friction[:] = np.clip(self._orig_friction * factors, 0.01, None)

        # 3. Joint damping
        if self.damping_enabled:
            n = len(self._orig_damping)
            factors = rng.uniform(1 - self.damping_ratio, 1 + self.damping_ratio, n)
            model.dof_damping[:] = np.clip(self._orig_damping * factors, 0.0, None)

        # 4. Actuator gear (effective torque scaling)
        if self.gear_enabled:
            n = len(self._orig_gear)
            factors = rng.uniform(1 - self.gear_ratio, 1 + self.gear_ratio, n)
            model.actuator_gear[:, 0] = self._orig_gear * factors

    def reset(self, model) -> None:
        """Restore original model parameters (call to undo randomization)."""
        model.body_mass[:]        = self._orig_mass
        model.geom_friction[:]    = self._orig_friction
        model.dof_damping[:]      = self._orig_damping
        model.actuator_gear[:, 0] = self._orig_gear


class PushRandomizer:
    """
    Applies random external force pushes to the robot torso.
    Simulates unexpected perturbations (wind, collisions).

    Args:
        push_prob:      probability of a push per step
        push_force_max: max force magnitude (N)
        push_duration:  how many steps the push lasts
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.push_prob      = cfg.get("push_prob",       0.005)  # 0.5% per step
        self.force_max      = cfg.get("push_force_max",  80.0)   # N
        self.duration       = cfg.get("push_duration",   5)      # steps
        self._countdown     = 0
        self._push_force    = np.zeros(3)

    def step(self, model, data, rng: np.random.Generator) -> None:
        """Apply push if active or triggered. Call every env step."""
        if self._countdown > 0:
            # Apply existing push
            # Apply as xfrc_applied on torso (body 1 typically)
            data.xfrc_applied[1, :3] = self._push_force
            self._countdown -= 1
        else:
            data.xfrc_applied[1, :3] = 0.0  # clear
            if rng.random() < self.push_prob:
                # Random horizontal push
                angle = rng.uniform(0, 2 * np.pi)
                mag   = rng.uniform(self.force_max * 0.3, self.force_max)
                self._push_force = np.array([
                    mag * np.cos(angle),
                    mag * np.sin(angle),
                    0.0,
                ])
                self._countdown = self.duration

    def reset(self, data) -> None:
        self._countdown = 0
        self._push_force[:] = 0.0
        data.xfrc_applied[:] = 0.0
