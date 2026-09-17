"""Wheelie environment configs for SvanM2."""

import math
from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from xterra_mjlab.tasks.velocity.config.svanm2.env_cfgs import svanm2_flat_env_cfg
from xterra_mjlab.tasks.wheelie import mdp as wheelie_mdp
from xterra_mjlab.tasks.wheelie.mdp.rewards import base_height_too_low

def svanm2_front_wheelie_cfg(play: bool = False):
    cfg = svanm2_flat_env_cfg(play=play)

    # No velocity commands for wheelie
    cfg.commands.clear()
    cfg.curriculum.clear()
    
    cfg.observations["actor"].terms.pop("command", None)
    cfg.observations["critic"].terms.pop("command", None)

    # Remove velocity tracking rewards
    for key in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "pose",
        "upright",
        "foot_slip",
    ]:
        cfg.rewards.pop(key, None)

    # Wheelie looks like falling - remove fell_over
    cfg.terminations.pop("fell_over", None)

    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=base_height_too_low,
        params={"min_height": 0.25}, # If the torso drops below 25cm, terminate!
    )

    # Add wheelie rewards
    # In svanm2_front_wheelie_cfg replace the three reward terms with:
    cfg.rewards["front_wheelie"] = RewardTermCfg(
        func=wheelie_mdp.front_wheelie_reward,
        weight=10.0,
        params={
            "sensor_name": "feet_ground_contact",
            "target_pitch": 0.5,
            "min_pitch": 0.3,
        },
    )

    # Only terminate on sideways roll, not pitch
    cfg.terminations["sideways_fall"] = TerminationTermCfg(
        func=wheelie_mdp.excessive_roll,
        params={"max_roll": math.radians(45.0)},
    )

    return cfg


def svanm2_rear_wheelie_cfg(play: bool = False):
    cfg = svanm2_front_wheelie_cfg(play=play)

    cfg.rewards.pop("front_wheelie", None)

    # In svanm2_rear_wheelie_cfg replace with:
    cfg.rewards["rear_wheelie"] = RewardTermCfg(
        func=wheelie_mdp.rear_wheelie_reward,
        weight=10.0,
        params={
            "sensor_name": "feet_ground_contact",
            "target_pitch": 0.5,
        },
    )

    return cfg
