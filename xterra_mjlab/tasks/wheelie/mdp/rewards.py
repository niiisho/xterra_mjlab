"""Reward and termination functions for wheelie tasks."""

import torch


def front_wheelie_reward(env, sensor_name, target_pitch=0.5):
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found.squeeze(-1)

    fl_off = contact[:, 0] < 0.5
    fr_off = contact[:, 1] < 0.5
    rl_on = contact[:, 2] >= 0.5
    rr_on = contact[:, 3] >= 0.5

    correct_config = (fl_off & fr_off & (rl_on | rr_on)).float()

    gravity = env.scene["robot"].data.projected_gravity_b
    pitch_raw = gravity[:, 0].clamp(min=0.0, max=target_pitch) / target_pitch

    # Full reward for correct wheelie config
    wheelie_reward = correct_config * pitch_raw

    # Small pitch signal always active - gives gradient from flat ground
    # Weight 10.0 * 0.05 = 0.5 max, much less than full wheelie reward of 10.0
    pitch_signal = pitch_raw * 0.05

    return wheelie_reward + pitch_signal


def rear_wheelie_reward(
    env,
    sensor_name: str,
    target_pitch: float = 0.5
) -> torch.Tensor:
    """
    Rear wheelie: front feet grounded, rear feet airborne, body pitched forward.
    Mirror of front_wheelie_reward.
    """
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found.squeeze(-1)

    fl_on = contact[:, 0] >= 0.5
    fr_on = contact[:, 1] >= 0.5
    rl_off = contact[:, 2] < 0.5
    rr_off = contact[:, 3] < 0.5

    correct_config = (fl_off & fr_off & (rl_on | rr_on)).float()

    # Pitch forward = gravity x-component negative in body frame
    gravity = env.scene["robot"].data.projected_gravity_b
    pitch = (-gravity[:, 0]).clamp(min=0.0, max=target_pitch) / target_pitch

    return correct_config * pitch


def excessive_roll(env, max_roll: float = 0.785) -> torch.Tensor:
    """
    Terminate when sideways roll exceeds max_roll radians.
    Returns bool tensor as required by mjlab termination manager.
    Gravity y-component in body frame indicates roll.
    """
    gravity = env.scene["robot"].data.projected_gravity_b
    roll_signal = gravity[:, 1].abs()
    return roll_signal > max_roll  # bool tensor, no .float()

def base_height_too_low(env, min_height: float):
    # Check if the body dropped below the limit
    is_low = env.scene["robot"].data.body_com_pos_w[:, 0, 2] < min_height    
    return is_low

def front_foot_ground_penalty(env, sensor_name: str) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found.squeeze(-1)
    fl_on = contact[:, 0] >= 0.5
    fr_on = contact[:, 1] >= 0.5
    return (fl_on | fr_on).float()

def non_foot_illegal_contact(
    env,
    shank_sensor_name: str,
    thigh_sensor_name: str,
    trunk_sensor_name: str,
) -> torch.Tensor:
    shank_any = env.scene.sensors[shank_sensor_name].data.found.any(dim=-1).any(dim=-1)
    thigh_any = env.scene.sensors[thigh_sensor_name].data.found.any(dim=-1).any(dim=-1)
    trunk_any = env.scene.sensors[trunk_sensor_name].data.found.any(dim=-1).any(dim=-1)
    return (shank_any | thigh_any | trunk_any).view(env.num_envs)
