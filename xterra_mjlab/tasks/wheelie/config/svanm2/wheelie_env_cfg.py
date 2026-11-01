"""Standing wheelie and stair environment configs for SvanM2."""

import math
from dataclasses import dataclass
import numpy as np
import mujoco

from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor.contact_sensor import ContactSensorCfg, ContactMatch
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensorCfg
from mjlab.sensor.raycast_sensor import GridPatternCfg
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.terrains.terrain_generator import (
    TerrainGeneratorCfg,
    SubTerrainCfg,
    TerrainOutput,
    TerrainGeometry,
)
from mjlab.envs.mdp.observations import height_scan

from xterra_mjlab.tasks.velocity.config.svanm2.env_cfgs import svanm2_flat_env_cfg
from xterra_mjlab.tasks.wheelie import mdp as wheelie_mdp
from xterra_mjlab.tasks.wheelie.mdp import rewards as wheelie_rewards


@dataclass(kw_only=True)
class PrimitiveStairsCfg(SubTerrainCfg):
    step_height_min: float = 0.05
    step_height_max: float = 0.12
    step_run_min: float = 0.3
    step_run_max: float = 0.6
    num_steps: int = 10

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        geometries = []
        size_x, size_y = self.size[0], self.size[1]
        cx, cy = size_x / 2.0, size_y / 2.0

        # 1. THE FLOOR
        base_geom = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[cx, cy, 0.5],
            pos=[cx, cy, -0.5],
            rgba=[0.3, 0.5, 0.3, 1.0],
        )
        geometries.append(TerrainGeometry(geom=base_geom))

        # 2. RANDOMIZED STAIRS (Placed ahead of cx so spawn has a flat standing pad)
        norm_diff = difficulty / 10.0 if difficulty > 1.0 else difficulty
        target_h = self.step_height_min + norm_diff * (self.step_height_max - self.step_height_min)

        current_x = cx + 1.5
        current_z = 0.0

        for _ in range(1, self.num_steps + 1):
            step_h = max(0.02, target_h + rng.uniform(-0.02, 0.02))
            current_z += step_h

            step_r = rng.uniform(self.step_run_min, self.step_run_max)
            if current_x + step_r > size_x - 0.5:
                break

            half_x = step_r / 2.0
            half_z = current_z / 2.0
            box_cx = current_x + half_x

            box_geom = spec.body("terrain").add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_x, cy, half_z],
                pos=[box_cx, cy, half_z],
                rgba=[0.6, 0.6, 0.6, 1.0],
            )
            geometries.append(TerrainGeometry(geom=box_geom))
            current_x += step_r

        # 3. TOP LANDING PAD
        safe_end_x = size_x - 0.5
        pad_length = safe_end_x - current_x
        if pad_length > 0:
            pad_half_x = pad_length / 2.0
            pad_half_z = max(0.02, current_z / 2.0)
            pad_cx = current_x + pad_half_x

            pad_geom = spec.body("terrain").add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[pad_half_x, cy, pad_half_z],
                pos=[pad_cx, cy, pad_half_z],
                rgba=[0.6, 0.6, 0.6, 1.0],
            )
            geometries.append(TerrainGeometry(geom=pad_geom))

        # 4. SPAWN ORIGIN (On flat floor at center cx, cy)
        origin = np.array([cx, cy, 0.45])
        return TerrainOutput(origin=origin, geometries=geometries, flat_patches=None)


def _apply_standing_wheelie_mdp(cfg):
    """Shared helper: configures 0-velocity commands, removes walking rewards, keeps posture & 270-obs policy."""
    # 1. SENSORS
    height_scanner_cfg = TerrainHeightSensorCfg(
        name="base_height_scan",
        frame=ObjRef(type="body", name="base", entity="robot"),
        pattern=GridPatternCfg(
            size=(1.5, 1.5),  # Matches 270-actor / privileged-critic checkpoint shapes
            resolution=0.1,
            direction=(0.0, 0.0, -1.0),
        ),
        max_distance=2.0,
        debug_vis=True,
    )

    thigh_ground_cfg = ContactSensorCfg(
        name="thigh_ground_touch",
        primary=ContactMatch(mode="body", entity="robot", pattern="(FL|FR)_hip_link|(FL|FR|RL|RR)_thigh_link"),
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

    shank_ground_cfg = ContactSensorCfg(
        name="shank_ground_touch",
        primary=ContactMatch(mode="body", entity="robot", pattern="(FL|FR|RL|RR)_shank_link"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        thigh_ground_cfg,
        trunk_ground_cfg,
        height_scanner_cfg,
        shank_ground_cfg,
    )

    # 2. REMOVE CUSTOM SPAWN PERTURBATIONS
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events.pop("push_robot", None)
    cfg.events.pop("reset_robot_state", None)

    # 3. KEEP POLICY SHAPE (270 OBS) & FORCE 0.0 VELOCITY COMMAND
    cfg.curriculum.pop("commands_vel", None)
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
        func=height_scan,
        params={"sensor_name": "base_height_scan"},
    )

    cmd_name = "velocity" if "velocity" in cfg.commands else "base_velocity"
    if cmd_name in cfg.commands:
        cfg.commands[cmd_name].ranges.lin_vel_x = (0.0, 0.0)
        cfg.commands[cmd_name].ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands[cmd_name].ranges.ang_vel_z = (0.0, 0.0)
        if hasattr(cfg.commands[cmd_name], "heading_command"):
            cfg.commands[cmd_name].heading_command = False
        if hasattr(cfg.commands[cmd_name], "rel_standing_envs"):
            cfg.commands[cmd_name].rel_standing_envs = 1.0

    # 4. REMOVE WALKING GAIT & 4-LEG FLAT REWARDS
    for key in [
        "air_time",
        "foot_clearance",
        "pose",
        "upright",
        "foot_slip",
        "forward_drive",
        "min_velocity",
        "reached_the_top",
        "smart_clearance",
    ]:
        cfg.rewards.pop(key, None)

    cfg.terminations.pop("fell_over", None)
    cfg.terminations.pop("success_reached_goal", None)

    # Gentle tracking of the 0.0 command (bell curve stays flat near 0 so tiny balance steps aren't punished)
    if "track_linear_velocity" in cfg.rewards:
        cfg.rewards["track_linear_velocity"].weight = 1.5
    if "track_angular_velocity" in cfg.rewards:
        cfg.rewards["track_angular_velocity"].weight = 1.0

    # Deadband penalty: 0 penalty for <= 0.20 m/s balancing shuffles, penalizes actual walking
    cfg.rewards["balance_drift_limit"] = RewardTermCfg(
        func=wheelie_rewards.balance_velocity_penalty,
        weight=-3.0,
        params={"deadband": 0.20},
    )

    # 5. TERMINATIONS
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=wheelie_mdp.illegal_contact_fall,
        params={"thigh_sensor_name": "thigh_ground_touch", "trunk_sensor_name": "trunk_ground_touch"},
    )
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=wheelie_mdp.base_height_too_low,
        params={"min_height": 0.25, "grace_period": 30},
    )
    cfg.terminations["sideways_fall"] = TerminationTermCfg(
        func=wheelie_mdp.excessive_roll,
        params={"max_roll": math.radians(35.0)},
    )
    cfg.terminations["front_contact_penalty"] = TerminationTermCfg(
        func=wheelie_mdp.front_contact_penalty,
        params={"sensor_name": "feet_ground_contact", "grace_period": 22},
    )
    cfg.terminations["abduction_limit_crossed"] = TerminationTermCfg(
        func=wheelie_mdp.rear_startup_contact_termination,
        params={"joint_indices": [6, 9], "lower_limit": -0.8, "upper_limit": 0.8},
    )
    cfg.terminations["knee_touch_fall"] = TerminationTermCfg(
        func=wheelie_mdp.isolated_knee_termination,
        params={"shank_sensor": "shank_ground_touch", "foot_sensor": "feet_ground_contact"},
    )

    # 6. WHEELIE POSTURE & SIM-TO-REAL REGULARIZATION REWARDS
    cfg.rewards["front_hop"] = RewardTermCfg(
        func=wheelie_mdp.front_wheelie_reward,
        weight=10.0,
        params={"sensor_name": "feet_ground_contact", "target_pitch": 0.5},
    )
    cfg.rewards["squat_penalty"] = RewardTermCfg(
        func=wheelie_mdp.base_height_penalty,
        weight=-10.0,
        params={"penalty_threshold": 0.32},
    )
    cfg.rewards["front_air_height"] = RewardTermCfg(
        func=wheelie_mdp.front_air_height_reward,
        weight=5.0,
        params={"sensor_name": "feet_ground_contact", "min_height": 0.40},
    )
    cfg.rewards["front_symmetry"] = RewardTermCfg(
        func=wheelie_mdp.front_symmetry_penalty,
        weight=-0.5,
    )
    cfg.rewards["rear_knee_posture"] = RewardTermCfg(
        func=wheelie_mdp.rear_knee_posture_penalty,
        weight=-2.0,
        params={"target_calf": -2},
    )
    cfg.rewards["front_leg_direction"] = RewardTermCfg(
        func=wheelie_mdp.front_leg_direction_penalty,
        weight=-2.0,
        params={"min_thigh": 0.0, "grace_period": 30},
    )
    cfg.rewards["action_rate"] = RewardTermCfg(
        func=wheelie_mdp.action_rate_penalty,
        weight=-0.01,
    )
    cfg.rewards["mechanical_work"] = RewardTermCfg(
        func=wheelie_mdp.mechanical_work_penalty,
        weight=-0.005,
    )

    default_joint_angles = [0.0, 0.5, -1.0, 0.0, 0.5, -1.0, 0.0, 0.5, -1.0, 0.0, 0.5, -1.0]
    cfg.rewards["nominal_posture"] = RewardTermCfg(
        func=wheelie_mdp.nominal_posture_penalty,
        weight=-0.1,
        params={"nominal_positions": default_joint_angles},
    )

    straitjacket_indices = [1, 4, 6, 9]
    straitjacket_targets = [0.5, 0.5, 0.0, 0.0]
    cfg.rewards["strict_belly_thigh_vel"] = RewardTermCfg(
        func=wheelie_mdp.targeted_joint_velocity_penalty,
        weight=-0.05,
        params={"joint_indices": straitjacket_indices},
    )
    cfg.rewards["strict_belly_thigh_pos"] = RewardTermCfg(
        func=wheelie_mdp.targeted_joint_pos_penalty,
        weight=-0.5,
        params={"joint_indices": straitjacket_indices, "target_positions": straitjacket_targets},
    )

    return cfg


def svanm2_front_wheelie_cfg(play: bool = False):
    cfg = svanm2_flat_env_cfg(play=play)
    cfg = _apply_standing_wheelie_mdp(cfg)
    cfg.terminations["tunnel_boundary"] = TerminationTermCfg(
        func=wheelie_mdp.lateral_out_of_bounds,
        params={"max_drift": 1.0},
    )
    return cfg


def svanm2_front_wheelie_fromflat_cfg(play: bool = False):
    return svanm2_front_wheelie_cfg(play=play)


def svanm2_rear_wheelie_cfg(play: bool = False):
    return svanm2_front_wheelie_cfg(play=play)


def svanm2_wheelie_stairs_cfg(play: bool = False):
    cfg = svanm2_flat_env_cfg(play=play)
    cfg.sim.nconmax = 200
    cfg.sim.njmax = 500

    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        curriculum=False,
        size=(10.0, 10.0),
        num_rows=10,
        num_cols=20,
        sub_terrains={
            "primitive_stairs": PrimitiveStairsCfg(
                step_height_min=0.05,
                step_height_max=0.12,
                step_run_min=0.3,
                step_run_max=0.6,
                num_steps=10,
            )
        },
    )

    return _apply_standing_wheelie_mdp(cfg)
