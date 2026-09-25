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

import numpy as np
import mujoco
from dataclasses import dataclass
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainOutput, TerrainGeometry

import numpy as np
import mujoco
from dataclasses import dataclass
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainOutput, TerrainGeometry

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
        
        # EXACT CENTER of the 40x40 tile (20.0, 20.0)
        cx, cy = size_x / 2.0, size_y / 2.0
        
        # 1. THE FLOOR
        base_geom = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[cx, cy, 0.5], 
            pos=[cx, cy, -0.5], 
            rgba=[0.3, 0.5, 0.3, 1.0] 
        )
        geometries.append(TerrainGeometry(geom=base_geom))
        
        # 2. RANDOMIZED STAIRS
        # 2. RANDOMIZED STAIRS
        norm_diff = difficulty / 10.0 if difficulty > 1.0 else difficulty
        target_h = self.step_height_min + norm_diff * (self.step_height_max - self.step_height_min)
        
        # Randomize the starting runway between 2.0 and 4.0 meters!
        runway_length = rng.uniform(2.0, 4.0)
        current_x = cx + runway_length
        current_z = 0.0
        
        for i in range(1, self.num_steps + 1):
            step_h = target_h + rng.uniform(-0.02, 0.02)
            step_h = max(0.02, step_h) 
            current_z += step_h
            
            step_r = rng.uniform(self.step_run_min, self.step_run_max)
            half_x = step_r / 2.0
            half_z = current_z / 2.0
            box_cx = current_x + half_x
            
            box_geom = spec.body("terrain").add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_x, cy, half_z],
                pos=[box_cx, cy, half_z],
                rgba=[0.6, 0.6, 0.6, 1.0] 
            )
            geometries.append(TerrainGeometry(geom=box_geom))
            current_x += step_r
            
        # 3. TOP LANDING PAD
        safe_end_x = size_x - 0.5 
        pad_length = safe_end_x - current_x
        
        if pad_length > 0:
            pad_half_x = pad_length / 2.0
            pad_half_z = current_z / 2.0
            pad_cx = current_x + pad_half_x
            
            pad_geom = spec.body("terrain").add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[pad_half_x, cy, pad_half_z],
                pos=[pad_cx, cy, pad_half_z],
                rgba=[0.6, 0.6, 0.6, 1.0]
            )
            geometries.append(TerrainGeometry(geom=pad_geom))
            
        # 4. SPAWN ORIGIN
        # We must return the exact center (cx) so the framework's tracking starts distance at 0.0!
        origin = np.array([cx, cy, 1.0])
        return TerrainOutput(origin=origin, geometries=geometries, flat_patches=None)

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
            "min_height": 0.27,
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

    cfg.rewards["squat_penalty"] = RewardTermCfg(
        func=wheelie_mdp.base_height_penalty,
        weight=-2.0,
        params={
            "penalty_threshold": 0.32,
        },
    )

    cfg.rewards["front_air_height"] = RewardTermCfg(
        func=wheelie_mdp.front_air_height_reward,
        weight=5.0,  # A flat 5.0 points per frame if it holds the high wheelie
        params={
            "sensor_name": "feet_ground_contact",
            "min_height": 0.36
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
        params={"target_calf": -2}, # Adjust sign if your URDF bends the other way 
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
    
    cfg.sim.nconmax = 200  # Increase maximum allowed contacts
    cfg.sim.njmax = 500

    cfg.scene.terrain.terrain_type = "generator"
    
    # 1. TERRAIN GENERATOR: Primitive Solid Stairs
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        curriculum=True, 
        
        # 1. Shrink the individual tile to a compact runway
        size=(40.0, 40.0), 
        
        # 2. Build the Grid!
        num_rows=10,  # 10 levels of increasing difficulty
        num_cols=20,  # 20 completely different, randomized variations per level
        
        sub_terrains={
            "primitive_stairs": PrimitiveStairsCfg(
                step_height_min=0.05,  
                step_height_max=0.12,  
                step_run_min=0.3,      
                step_run_max=0.6,      
                num_steps=10
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
        # Added (FL|FR)_hip_link to the front of the regex pattern
        primary=ContactMatch(mode="body", entity="robot", pattern="(FL|FR)_hip_link|(FL|FR|RL|RR)_thigh_link"),
        secondary=ContactMatch(mode="body", pattern="terrain"), 
        fields=("found",),
        reduce="none",
        num_slots=1,
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
    
    # HUGE reward for crossing the finish line (teaches it that reaching the top is good)
    cfg.rewards["reached_the_top"] = RewardTermCfg(
        func=wheelie_mdp.reached_goal_distance,
        weight=1000.0,  # Massive bonus payout
        params={"target_distance": 5}, 
    )
    
    # Clean reset when it successfully clears the 6.5m mark
    cfg.terminations["success_reached_goal"] = TerminationTermCfg(
        func=wheelie_mdp.reached_goal_distance,
        params={"target_distance": 5}, 
    )


    return cfg
