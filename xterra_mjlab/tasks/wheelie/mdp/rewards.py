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
    rear_planted = (rl_on & rr_on).float()

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

    valid_feet = fl_off & fr_off & rl_on & rr_on
    
    # Check if the hips are high enough
    height = env.scene["robot"].data.body_com_pos_w[:, 0, 2]
    valid_height = height > min_height

    # Returns a flat 1.0 per frame only if both conditions are true
    return (valid_feet & valid_height).float()


def front_contact_penalty(env, sensor_name: str, grace_period: int = 50) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found.squeeze(-1)
    
    fl_on = contact[:, 0] >= 0.5
    fr_on = contact[:, 1] >= 0.5
    
    # Returns 1.0 if either front foot touches the ground
    return (fl_on | fr_on).float()

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

def forward_velocity_reward(env) -> torch.Tensor:!
    forward_vel = env.scene["robot"].data.root_lin_vel_w[:, 0]
    return forward_vel
