"""Reward and termination functions for wheelie tasks."""

import torch


def front_wheelie_reward(
    env,
    sensor_name: str,
    target_pitch: float = 0.5
) -> torch.Tensor:
    """
    Front wheelie: rear feet grounded, front feet airborne, body pitched back.
    All three conditions required simultaneously - prevents dead bug exploit.
    target_pitch normalizes the pitch component so reward peaks at target angle.
    """
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found.squeeze(-1)

    fl_off = contact[:, 0] < 0.5
    fr_off = contact[:, 1] < 0.5
    rl_on = contact[:, 2] >= 0.5
    rr_on = contact[:, 3] >= 0.5

    # All four conditions must be true at the same time
    # If robot is on its back, rl_on and rr_on are False -> reward is zero
    correct_config = (fl_off & fr_off & rl_on & rr_on).float()

    # Pitch reward normalized by target_pitch
    # gravity[:, 0] positive when body tilts backward in body frame
    # Clamp to target_pitch so reward doesn't keep growing past target angle
    gravity = env.scene["robot"].data.projected_gravity_b
    pitch = gravity[:, 0].clamp(min=0.0, max=target_pitch) / target_pitch

    # Multiply: zero reward if contact config is wrong regardless of pitch
    return correct_config * pitch


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
