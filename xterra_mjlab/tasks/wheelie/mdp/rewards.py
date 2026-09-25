"""Reward and termination functions for wheelie tasks."""

import torch

def front_wheelie_reward(
    env,
    sensor_name: str,
    target_pitch: float = 0.5,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found.squeeze(-1)

    fl_off = contact[:, 0] < 0.5
    fr_off = contact[:, 1] < 0.5
    rl_on = contact[:, 2] >= 0.5
    rr_on = contact[:, 3] >= 0.5

    # REQUIREMENT: Both rear legs MUST be glued to the ground to get any points
    rear_planted = (rl_on | rr_on).float()

    # BONUS: Did it successfully lift the front legs?
    front_lifted = (fl_off & fr_off).float()

    gravity = env.scene["robot"].data.projected_gravity_b
    pitch_raw = gravity[:, 0].clamp(min=0.0, max=target_pitch) / target_pitch

    # The AI gets breadcrumb points for tilting back, and a massive +1.0 bonus for taking the front feet off the floor
    return rear_planted * (pitch_raw + front_lifted)


def excessive_roll(env, max_roll: float = 0.785) -> torch.Tensor:
    gravity = env.scene["robot"].data.projected_gravity_b
    roll_signal = gravity[:, 1].abs()
    return roll_signal > max_roll


def base_height_too_low(env, min_height: float, grace_period: int = 50):
    is_low = env.scene["robot"].data.body_com_pos_w[:, 0, 2] < min_height
    past_grace = env.episode_length_buf > grace_period
    
    return is_low & past_grace


def illegal_contact_fall(
    env,
    thigh_sensor_name: str,
    trunk_sensor_name: str,
):
    # Notice we removed the shank sensor completely to prevent foot false-positives
    thigh_contact = env.scene.sensors[thigh_sensor_name].data.found
    trunk_contact = env.scene.sensors[trunk_sensor_name].data.found

    # Bulletproof reduction that will never crash the Reward Manager
    thigh_any = thigh_contact.view(env.num_envs, -1).any(dim=1)
    trunk_any = trunk_contact.view(env.num_envs, -1).any(dim=1)

    return thigh_any | trunk_any


def front_air_height_reward(
    env, 
    sensor_name: str, 
    min_height: float = 0.42
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found.squeeze(-1)

    # Front in air, rear on ground
    fl_off = contact[:, 0] < 0.5
    fr_off = contact[:, 1] < 0.5
    rl_on = contact[:, 2] >= 0.5
    rr_on = contact[:, 3] >= 0.5

    valid_feet = fl_off & fr_off & (rl_on | rr_on)
    
    # Check if the hips are high enough
    height = env.scene["robot"].data.body_com_pos_w[:, 0, 2]
    valid_height = height > min_height

    forward_vel = env.scene["robot"].data.root_link_vel_w[:, 0]
    vel_multiplier = torch.clamp(forward_vel / 0.3, min=0.0, max=1.0)

    return (valid_feet & valid_height).float() * vel_multiplier


def front_contact_penalty(env, sensor_name: str, grace_period: int = 50) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found.squeeze(-1)
    
    fl_on = contact[:, 0] >= 0.5
    fr_on = contact[:, 1] >= 0.5
    
    past_grace = env.episode_length_buf > grace_period
    return (fl_on | fr_on) & past_grace

def base_height_penalty(env, penalty_threshold: float) -> torch.Tensor:
    height = env.scene["robot"].data.body_com_pos_w[:, 0, 2]
    
    # Check if the robot's hips have dropped into the warning zone
    is_low = height < penalty_threshold
    return (is_low).float()

def yaw_rate_penalty(env) -> torch.Tensor:
    # Shape: [num_envs, num_geometries, 3]
    ang_vel = env.scene["robot"].data.geom_ang_vel_w
    yaw_rate = ang_vel[:, 0, 2]
    return yaw_rate.abs()

def front_symmetry_penalty(env) -> torch.Tensor:
    joint_pos = env.scene["robot"].data.joint_pos
    fl_joints = joint_pos[:, 0:3]
    fr_joints = joint_pos[:, 3:6]
    
    symmetry_error = torch.sum(torch.square(fl_joints - fr_joints), dim=1)
    
    return symmetry_error

def forward_velocity_reward(env) -> torch.Tensor:
    forward_vel = env.scene["robot"].data.root_link_vel_w[:, 0]
    return forward_vel

def lateral_out_of_bounds(env, max_drift: float) -> torch.Tensor:
    y_pos = env.scene["robot"].data.body_com_pos_w[:, 0, 1]
    return y_pos.abs() > max_drift

def min_velocity_penalty(env, min_vel: float = 0.1, grace_period: int = 50) -> torch.Tensor:
    forward_vel = env.scene["robot"].data.root_link_vel_w[:, 0]
    too_slow = forward_vel < min_vel
    past_grace = env.episode_length_buf > grace_period
    return (too_slow & past_grace).float()

def rear_knee_posture_penalty(env, target_calf: float = 1.0, grace_period: int = 30) -> torch.Tensor:
    joint_pos = env.scene["robot"].data.joint_pos
    
    rl_calf = joint_pos[:, 8]
    rr_calf = joint_pos[:, 11]
    
    rl_error = torch.square(rl_calf - target_calf)
    rr_error = torch.square(rr_calf - target_calf)
    
    return rl_error + rr_error


def front_leg_direction_penalty(
    env,
    min_thigh: float = 0.0,
    grace_period: int = 50
) -> torch.Tensor:
    joint_pos = env.scene["robot"].data.joint_pos

    fl_thigh = joint_pos[:, 1]  # FL_thigh_joint
    fr_thigh = joint_pos[:, 4]  # FR_thigh_joint

    # Penalize only when thigh goes below min_thigh
    # clamp means no penalty when above threshold, penalty scales with how far below
    fl_wrong = torch.clamp(min_thigh - fl_thigh, min=0.0)
    fr_wrong = torch.clamp(min_thigh - fr_thigh, min=0.0)

    past_grace = (env.episode_length_buf > grace_period).float()

    return (fl_wrong + fr_wrong) * past_grace


def reached_goal_absolute(env, target_x: float) -> torch.Tensor:
    """Checks if the robot's absolute X coordinate has crossed the finish line."""
    # body_com_pos_w is the absolute position in the world. No origin subtraction needed!
    root_x = env.scene["robot"].data.body_com_pos_w[:, 0, 0]
    
    # Returns True (1) if it crossed the absolute X line, False (0) otherwise
    return root_x > target_x
