"""
humanoid_rl/envs/mujoco/terrain.py
Procedural terrain generator for MuJoCo XML.

Generates terrain snippets that can be injected into the base MJCF XML
before loading the model. Each terrain is a string of <body>/<geom> elements
to be inserted inside <worldbody>.

Supported terrains:
    flat        — default, no obstacles (just floor)
    stair       — ascending staircase in front of robot
    rough       — random height blocks
    slope       — inclined plane
    jump_gap    — gaps in the floor (jump task)
    hurdles     — hurdle bars at random intervals
"""
from __future__ import annotations

import numpy as np


# ─── Terrain Generators ───────────────────────────────────────────────────────

def flat_terrain() -> str:
    """No extra geometry — just the default floor."""
    return ""


def stair_terrain(
    n_steps: int      = 6,
    step_height: float = 0.15,
    step_depth: float  = 0.40,
    step_width: float  = 2.0,
    start_x: float     = 2.0,
    rgba: str          = "0.65 0.45 0.25 1",
) -> str:
    """
    Ascending staircase starting at start_x in front of robot.
    Each step is a box that is cumulatively taller.
    """
    bodies = []
    for i in range(n_steps):
        x    = start_x + i * step_depth + step_depth / 2
        h    = (i + 1) * step_height
        z    = h / 2
        bodies.append(
            f'    <body name="stair_{i}" pos="{x:.3f} 0 {z:.3f}">\n'
            f'      <geom type="box" size="{step_depth/2:.3f} {step_width/2:.3f} {h/2:.3f}" '
            f'rgba="{rgba}" contype="1" conaffinity="1" friction="0.8 0.1 0.1"/>\n'
            f'    </body>'
        )
    return "\n".join(bodies)


def rough_terrain(
    n_blocks: int       = 30,
    block_size: float   = 0.3,
    max_height: float   = 0.12,
    start_x: float      = 1.5,
    length: float       = 8.0,
    width: float        = 1.5,
    seed: int           = 42,
    rgba: str           = "0.50 0.50 0.55 1",
) -> str:
    """Random height blocks — for rough terrain walking."""
    rng = np.random.default_rng(seed)
    bodies = []
    for i in range(n_blocks):
        x = rng.uniform(start_x, start_x + length)
        y = rng.uniform(-width / 2, width / 2)
        h = rng.uniform(0.02, max_height)
        bodies.append(
            f'    <body name="block_{i}" pos="{x:.3f} {y:.3f} {h/2:.3f}">\n'
            f'      <geom type="box" size="{block_size/2:.3f} {block_size/2:.3f} {h/2:.3f}" '
            f'rgba="{rgba}" contype="1" conaffinity="1" friction="0.6 0.1 0.1"/>\n'
            f'    </body>'
        )
    return "\n".join(bodies)


def slope_terrain(
    angle_deg: float = 15.0,
    length: float    = 4.0,
    width: float     = 3.0,
    start_x: float   = 2.0,
    rgba: str        = "0.40 0.60 0.30 1",
) -> str:
    """Inclined slope (ascending then flat)."""
    import math
    rad = math.radians(angle_deg)
    # The slope center is offset
    cx = start_x + (length / 2) * math.cos(rad)
    cz = (length / 2) * math.sin(rad)
    bodies = [
        f'    <body name="slope" pos="{cx:.3f} 0 {cz:.3f}" euler="0 {-angle_deg:.1f} 0">\n'
        f'      <geom type="box" size="{length/2:.3f} {width/2:.3f} 0.04" '
        f'rgba="{rgba}" contype="1" conaffinity="1" friction="0.7 0.1 0.1"/>\n'
        f'    </body>'
    ]
    return "\n".join(bodies)


def hurdle_terrain(
    n_hurdles: int      = 5,
    hurdle_height: float = 0.25,
    hurdle_width: float  = 2.0,
    spacing: float       = 2.5,
    start_x: float       = 2.0,
    rgba: str            = "0.80 0.30 0.20 1",
) -> str:
    """Hurdle bars to jump over."""
    bodies = []
    for i in range(n_hurdles):
        x = start_x + i * spacing
        bodies.append(
            f'    <body name="hurdle_{i}" pos="{x:.3f} 0 {hurdle_height/2:.3f}">\n'
            f'      <geom type="box" size="0.06 {hurdle_width/2:.3f} {hurdle_height/2:.3f}" '
            f'rgba="{rgba}" contype="1" conaffinity="1"/>\n'
            f'    </body>'
        )
    return "\n".join(bodies)


def gap_terrain(
    n_gaps: int        = 3,
    gap_width: float   = 0.5,
    platform_len: float = 1.5,
    start_x: float     = 2.0,
    rgba: str          = "0.30 0.30 0.80 1",
) -> str:
    """Platforms separated by gaps — robot must jump across."""
    bodies = []
    x = start_x
    for i in range(n_gaps + 1):
        cx = x + platform_len / 2
        bodies.append(
            f'    <body name="platform_{i}" pos="{cx:.3f} 0 -0.05">\n'
            f'      <geom type="box" size="{platform_len/2:.3f} 1.5 0.05" '
            f'rgba="{rgba}" contype="1" conaffinity="1"/>\n'
            f'    </body>'
        )
        x += platform_len + gap_width
    return "\n".join(bodies)


# ─── Registry ─────────────────────────────────────────────────────────────────

TERRAIN_REGISTRY: dict[str, callable] = {
    "flat":    flat_terrain,
    "stair":   stair_terrain,
    "rough":   rough_terrain,
    "slope":   slope_terrain,
    "hurdle":  hurdle_terrain,
    "gap":     gap_terrain,
}


def get_terrain_xml(terrain_type: str, **kwargs) -> str:
    """
    Get terrain XML snippet by name.

    Args:
        terrain_type: one of "flat", "stair", "rough", "slope", "hurdle", "gap"
        **kwargs:     forwarded to the terrain function

    Returns:
        XML string to inject into <worldbody>
    """
    if terrain_type not in TERRAIN_REGISTRY:
        raise ValueError(
            f"Unknown terrain: '{terrain_type}'. "
            f"Available: {list(TERRAIN_REGISTRY.keys())}"
        )
    return TERRAIN_REGISTRY[terrain_type](**kwargs)


def inject_terrain(base_xml: str, terrain_type: str, **kwargs) -> str:
    """
    Inject terrain XML into base MJCF string.

    Args:
        base_xml:     full MJCF XML string
        terrain_type: terrain name
        **kwargs:     terrain parameters

    Returns:
        Modified XML string with terrain injected before </worldbody>
    """
    terrain_xml = get_terrain_xml(terrain_type, **kwargs)
    if not terrain_xml:
        return base_xml
    return base_xml.replace("</worldbody>", f"{terrain_xml}\n  </worldbody>")
