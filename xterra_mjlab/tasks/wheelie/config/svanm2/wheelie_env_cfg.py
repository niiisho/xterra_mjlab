"""Wheelie environment configs for SvanM2."""

import math
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch

from xterra_mjlab.tasks.velocity.config.svanm2.env_cfgs import svanm2_flat_env_cfg
from xterra_mjlab.tasks.wheelie import mdp as wheelie_mdp


def svanm2_front_wheelie_cfg(play: bool = False):
    cfg = svanm2_flat_env_cfg(play=play)

    # Remove events that override our spawn
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events.pop("push_robot", None)

    # Spawn in wheelie position
    cfg.events["reset_robot_state"] = EventTermCfg(
        func=wheelie_mdp.reset_to_wheelie_pose,
        mode="reset",
        params={
            "pitch_angle": 0.6,
            "base_height": 0.45,
        },
    )

    # Add contact sensors for non-foot body parts
    shank_ground_cfg = ContactSensorCfg(
        name="shank_ground_touch",
        primary=ContactMatch(mode="body", entity="robot", pattern="(FL|FR|RL|RR)_shank_link"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
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

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        shank_ground_cfg,
        thigh_ground_cfg,
        trunk_ground_cfg,
    )

    # No velocity commands for wheelie
    cfg.commands.clear()
    cfg.curriculum.clear()
    cfg.observations["actor"].terms.pop("command", None)
    cfg.observations["critic"].terms.pop("command", None)

    # Remove locomotion rewards
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

    # Remove locomotion terminations
    cfg.terminations.pop("fell_over", None)

    # Wheelie terminations
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=wheelie_mdp.base_height_too_low,
        params={"min_height": 0.25},
    )
    cfg.terminations["sideways_fall"] = TerminationTermCfg(
        func=wheelie_mdp.excessive_roll,
        params={"max_roll": math.radians(45.0)},
    )

    # Wheelie rewards
    cfg.rewards["front_wheelie"] = RewardTermCfg(
        func=wheelie_mdp.front_wheelie_reward,
        weight=10.0,
        params={
            "sensor_name": "feet_ground_contact",
            "target_pitch": 0.5,
        },
    )
    cfg.rewards["non_foot_contact_penalty"] = RewardTermCfg(
        func=wheelie_mdp.non_foot_contact_penalty,
        weight=-5.0,
        params={
            "shank_sensor_name": "shank_ground_touch",
            "thigh_sensor_name": "thigh_ground_touch",
            "trunk_sensor_name": "trunk_ground_touch",
        },
    )

    return cfg


def svanm2_front_wheelie_fromflat_cfg(play: bool = False):
    cfg = svanm2_front_wheelie_cfg(play=play)

    cfg.events["reset_robot_state"] = EventTermCfg(
        func=wheelie_mdp.reset_to_wheelie_pose,
        mode="reset",
        params={
            "pitch_angle": 0.2,
            "base_height": 0.45,
        },
    )

    return cfg


def svanm2_rear_wheelie_cfg(play: bool = False):
    cfg = svanm2_front_wheelie_cfg(play=play)

    cfg.rewards.pop("front_wheelie", None)

    cfg.rewards["rear_wheelie"] = RewardTermCfg(
        func=wheelie_mdp.rear_wheelie_reward,
        weight=10.0,
        params={
            "sensor_name": "feet_ground_contact",
            "target_pitch": 0.5,
        },
    )

    return cfg
