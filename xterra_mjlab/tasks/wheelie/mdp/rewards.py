"""Reward and termination functions for wheelie tasks."""

import torch


def front_wheelie_reward(
    env,
    sensor_name: str,
    target_pitch: float = 0.5,
    min_pitch: float = 0.3,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found.squeeze(-1)

    fl_off = contact[:, 0] < 0.5
    fr_off = contact[:, 1] < 0.5
    rl_on = contact[:, 2] >= 0.5
    rr_on = contact[:, 3] >= 0.5

    correct_config = (fl_off & fr_off & rl_on & rr_on).float()

    gravity = env.scene["robot"].data.projected_gravity_b
    pitch_raw = gravity[:, 0]

    # Zero reward if pitch below minimum threshold
    # This kills the knee-tripod exploit - body isn't pitched enough
    above_minimum = (pitch_raw >= min_pitch).float()

    pitch = pitch_raw.clamp(min=0.0, max=target_pitch) / target_pitch

    return correct_config * above_minimum * pitch


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

    correct_config = (fl_on & fr_on & rl_off & rr_off).float()

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
    # body_com_pos_w = Body Center of Mass Position in World frame
    return env.scene["robot"].data.body_com_pos_w[:, 0, 2] < min_height
