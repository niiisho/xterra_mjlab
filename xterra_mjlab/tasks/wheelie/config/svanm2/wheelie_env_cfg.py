"""Wheelie environment configs for SvanM2."""

import math
from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch

from xterra_mjlab.tasks.velocity.config.svanm2.env_cfgs import svanm2_flat_env_cfg
from xterra_mjlab.tasks.wheelie import mdp as wheelie_mdp
from mjlab.managers.event_manager import EventTermCfg


def svanm2_front_wheelie_cfg(play: bool = False):
    cfg = svanm2_flat_env_cfg(play=play)

    # 1. REMOVE ALL CUSTOM SPAWNS
    # By popping these, the robot naturally spawns perfectly flat and stable
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events.pop("push_robot", None)
    # cfg.events.pop("reset_robot_state", None)

    cfg.events["reset_robot_state"] = EventTermCfg(
        func=wheelie_mdp.reset_to_wheelie_pose,
        mode="reset",
        params={
            "pitch_angle": -0.6,  # THE FIX: Negative pitches backward onto rear legs!
            "base_height": 0.45,
        },
    )

    # 2. SENSORS (Only checking Thighs and Trunk against any ground surface)
    thigh_ground_cfg = ContactSensorCfg(
        name="thigh_ground_touch",
        primary=ContactMatch(mode="body", entity="robot", pattern="(FL|FR|RL|RR)_thigh_link"),
        secondary=ContactMatch(mode="body", pattern="terrain"), 
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_touch",
        primary=ContactMatch(mode="body", entity="robot", pattern="base"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (thigh_ground_cfg, trunk_ground_cfg)

    # 3. CLEANUP LOCOMOTION TASKS
    cfg.commands.clear()
    cfg.curriculum.clear()
    cfg.observations["actor"].terms.pop("command", None)
    cfg.observations["critic"].terms.pop("command", None)

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

    cfg.terminations.pop("fell_over", None)

    # 4. STRICT TERMINATIONS
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=wheelie_mdp.illegal_contact_fall,
        params={
            "thigh_sensor_name": "thigh_ground_touch",
            "trunk_sensor_name": "trunk_ground_touch",
        },
    )
    
    # 0.25m gives it enough room to do a deep squat before jumping, but kills it if it completely collapses
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=wheelie_mdp.base_height_too_low,
        params={"min_height": 0.25},
    )
    
    cfg.terminations["sideways_fall"] = TerminationTermCfg(
        func=wheelie_mdp.excessive_roll,
        params={"max_roll": math.radians(30.0)},
    )

    # 5. THE HOP REWARD
    cfg.rewards["front_hop"] = RewardTermCfg(
        func=wheelie_mdp.front_wheelie_reward,
        weight=10.0,
        params={
            "sensor_name": "feet_ground_contact",
            "target_pitch": 0.5,
        },
    )

    cfg.rewards["continuous_air"] = RewardTermCfg(
        func=wheelie_mdp.front_continuous_air_reward,
        weight=10.0,
        params={"sensor_name": "feet_ground_contact"},
    )

    cfg.rewards["front_contact_penalty"] = RewardTermCfg(
        func=wheelie_mdp.front_contact_penalty,
        weight=-2.0,
        params={"sensor_name": "feet_ground_contact"},
    )

    return cfg


def svanm2_front_wheelie_fromflat_cfg(play: bool = False):
    # Placeholder to prevent the __init__.py import crash
    return svanm2_front_wheelie_cfg(play=play)

def svanm2_rear_wheelie_cfg(play: bool = False):
    # Placeholder to prevent the __init__.py import crash
    return svanm2_front_wheelie_cfg(play=play)
