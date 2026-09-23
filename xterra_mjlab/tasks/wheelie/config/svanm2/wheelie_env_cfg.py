"""Stair climbing wheelie environment configs for SvanM2."""

import math
from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

# 1. New imports based on your discovered files
from mjlab.managers.observation_manager import ObservationTermCfg #[cite: 8]
from mjlab.sensor.contact_sensor import ContactSensorCfg, ContactMatch #[cite: 3]
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensorCfg #[cite: 6]
from mjlab.sensor.raycast_sensor import GridPatternCfg #[cite: 4]
from mjlab.sensor.builtin_sensor import ObjRef #[cite: 1]
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg #[cite: 10]
from mjlab.terrains.heightfield_terrains import HfDiscreteObstaclesTerrainCfg #[cite: 9]

from xterra_mjlab.tasks.velocity.config.svanm2.env_cfgs import svanm2_flat_env_cfg
from xterra_mjlab.tasks.wheelie import mdp as wheelie_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.terrains.heightfield_terrains import HfPyramidSlopedTerrainCfg
from mjlab.envs.mdp.observations import height_scan

def svanm2_front_wheelie_cfg(play: bool = False):
    cfg = svanm2_flat_env_cfg(play=play)

    # 1. REMOVE ALL CUSTOM SPAWNS
    # By popping these, the robot naturally spawns perfectly flat and stable
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events.pop("push_robot", None)
    cfg.events.pop("reset_robot_state", None)

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
        params={
            "min_height": 0.2,
            "grace_period": 30
        },
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

    cfg.rewards["front_air_height"] = RewardTermCfg(
        func=wheelie_mdp.front_air_height_reward,
        weight=5.0,  # A flat 5.0 points per frame if it holds the high wheelie
        params={
            "sensor_name": "feet_ground_contact",
            "min_height": 0.25
        },
    )

    cfg.terminations["front_contact_penalty"] = TerminationTermCfg(
        func=wheelie_mdp.front_contact_penalty,
        params={"sensor_name": "feet_ground_contact", "grace_period": 22},
    )

    cfg.rewards["front_symmetry"] = RewardTermCfg(
        func=wheelie_mdp.front_symmetry_penalty,
        weight=-0.5,  
    )

    cfg.rewards["forward_drive"] = RewardTermCfg(
        func=wheelie_mdp.forward_velocity_reward,
        weight=5,  
    )

    cfg.rewards["min_velocity"] = RewardTermCfg(
        func=wheelie_mdp.min_velocity_penalty,
        weight=-3.0,
        params={
            "min_vel": 0.25,
            "grace_period": 50,
        },
    )

    cfg.terminations["tunnel_boundary"] = TerminationTermCfg(
        func=wheelie_mdp.lateral_out_of_bounds,
        params={"max_drift": 1.0}, 
    )

    cfg.rewards["rear_knee_posture"] = RewardTermCfg(
        func=wheelie_mdp.rear_knee_posture_penalty,
        weight=-2.0,
        params={
            "target_calf": -0.5, 
            "grace_period": 30  # Gives it 100 frames to stand up penalty-free
        }, 
    )

    cfg.rewards["front_leg_direction"] = RewardTermCfg(
        func=wheelie_mdp.front_leg_direction_penalty,
        weight=-2.0,
        params={
            "min_thigh": 0.0,
            "grace_period": 30,
        },
    )
    
    return cfg


def svanm2_front_wheelie_fromflat_cfg(play: bool = False):
    # Placeholder to prevent the __init__.py import crash
    return svanm2_front_wheelie_cfg(play=play)

def svanm2_rear_wheelie_cfg(play: bool = False):
    # Placeholder to prevent the __init__.py import crash
    return svanm2_front_wheelie_cfg(play=play)

def svanm2_wheelie_stairs_cfg(play: bool = False):
    cfg = svanm2_flat_env_cfg(play=play)

    # 1. TERRAIN GENERATOR: "Easy Start" Wide Platforms
    cfg.scene.terrain.terrain_type = "generator"
    
    # Assign the generator to the specific attribute, do not overwrite 'cfg.scene.terrain'
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg( #[cite: 10]
        curriculum=True, 
        size=(12.0, 12.0), #[cite: 10]
        sub_terrains={
            "easy_platforms": HfDiscreteObstaclesTerrainCfg( #[cite: 9]
                obstacle_height_mode="fixed", #[cite: 9]
                obstacle_height_range=(0.04, 0.08), #[cite: 9]
                obstacle_width_range=(1.0, 2.0), #[cite: 9]
                num_obstacles=60, #[cite: 9]
                platform_width=3.0, #[cite: 9]
                square_obstacles=False, #[cite: 9]
                origin_z_offset=0.05 #[cite: 9]
            )
        }
    )

    # 2. EXTEROCEPTION (Height Scanners) with Visualization enabled
    height_scanner_cfg = TerrainHeightSensorCfg(
        name="base_height_scan", #[cite: 6]
        # CRITICAL FIX: Ensure entity="robot" is included here
        frame=ObjRef(type="body", name="base", entity="robot"), #[cite: 1, 4]
        pattern=GridPatternCfg(
            size=(1.5, 1.5), #[cite: 4]
            resolution=0.1,  #[cite: 4]
            direction=(0.0, 0.0, -1.0) #[cite: 4]
        ),
        max_distance=2.0, #[cite: 4]
        debug_vis=True #[cite: 4]
    )

    # 3. CONTACT SENSORS (With Calf/Shank Regex)[cite: 3]
    thigh_ground_cfg = ContactSensorCfg(
        name="thigh_ground_touch",
        primary=ContactMatch(mode="body", entity="robot", pattern="(FL|FR|RL|RR)_(thigh|calf|shank)_link"), #[cite: 3]
        secondary=ContactMatch(mode="body", pattern="terrain"), #[cite: 3]
        fields=("found",), #[cite: 3]
        reduce="none", #[cite: 3]
        num_slots=1, #[cite: 3]
    )
    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_touch",
        primary=ContactMatch(mode="body", entity="robot", pattern="base"), #[cite: 3]
        secondary=ContactMatch(mode="body", pattern="terrain"), #[cite: 3]
        fields=("found",), #[cite: 3]
        reduce="none", #[cite: 3]
        num_slots=1, #[cite: 3]
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (thigh_ground_cfg, trunk_ground_cfg, height_scanner_cfg)

    # 4. REMOVE CUSTOM SPAWNS
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events.pop("push_robot", None)
    cfg.events.pop("reset_robot_state", None)

    # 5. BLIND COMMAND MANAGER & ADD HEIGHT OBSERVATIONS
    cfg.commands.clear()
    cfg.curriculum.clear()
    cfg.observations["actor"].terms.pop("command", None)
    cfg.observations["critic"].terms.pop("command", None)

    # Use the height_scan MDP function to read the base_height_scan sensor
    cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
        func=height_scan,
        params={"sensor_name": "base_height_scan"}
    )
    cfg.observations["critic"].terms["height_scan"] = cfg.observations["actor"].terms["height_scan"]

    # Pop default tracking rewards since we are hard-coding forward drive
    for key in ["track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance", "pose", "upright", "foot_slip"]:
        cfg.rewards.pop(key, None)
    cfg.terminations.pop("fell_over", None)

    # 6. RELAXED TERMINATIONS FOR JUMPING
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=wheelie_mdp.illegal_contact_fall,
        params={"thigh_sensor_name": "thigh_ground_touch", "trunk_sensor_name": "trunk_ground_touch"},
    )
    
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=wheelie_mdp.base_height_too_low,
        params={"min_height": 0.15, "grace_period": 30},
    )
    
    cfg.terminations["sideways_fall"] = TerminationTermCfg(
        func=wheelie_mdp.excessive_roll,
        params={"max_roll": math.radians(35.0)},
    )

    # 7. WHEELIE POSTURE REWARDS
    cfg.rewards["front_hop"] = RewardTermCfg(
        func=wheelie_mdp.front_wheelie_reward,
        weight=10.0,
        params={"sensor_name": "feet_ground_contact", "target_pitch": 0.5},
    )

    cfg.rewards["squat_penalty"] = RewardTermCfg(
        func=wheelie_mdp.base_height_penalty,
        weight=-2.0,
        params={"penalty_threshold": 0.20},
    )

    cfg.rewards["front_air_height"] = RewardTermCfg(
        func=wheelie_mdp.front_air_height_reward,
        weight=5.0,
        params={"sensor_name": "feet_ground_contact", "min_height": 0.4},
    )

    cfg.terminations["front_contact_penalty"] = TerminationTermCfg(
        func=wheelie_mdp.front_contact_penalty,
        params={"sensor_name": "feet_ground_contact", "grace_period": 22},
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

    # 8. MAX-EFFORT FORWARD DRIVE
    cfg.rewards["forward_drive"] = RewardTermCfg(
        func=wheelie_mdp.forward_velocity_reward,
        weight=10.0,  
    )
    
    cfg.rewards["min_velocity"] = RewardTermCfg(
        func=wheelie_mdp.min_velocity_penalty,
        weight=-3.0,
        params={"min_vel": 0.1, "grace_period": 50},
    )

    return cfg
