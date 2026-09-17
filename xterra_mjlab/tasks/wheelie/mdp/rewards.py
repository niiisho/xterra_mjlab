"""Reward and termination functions for wheelie tasks."""

import torch


def front_wheelie_reward(
    env,
    sensor_name: str,
    target_pitch: float = 0.5,
    grace_steps: int = 150,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found.squeeze(-1)

    fl_off = contact[:, 0] < 0.5
    fr_off = contact[:, 1] < 0.5
    rl_on = contact[:, 2] >= 0.5
    rr_on = contact[:, 3] >= 0.5

    correct_config = (fl_off & fr_off & rl_on & rr_on).float()

    gravity = env.scene["robot"].data.projected_gravity_b
    pitch_raw = gravity[:, 0].clamp(min=0.0, max=target_pitch) / target_pitch

    wheelie_reward = correct_config * pitch_raw
    pitch_only_reward = pitch_raw * 0.1

    # Block all wheelie reward for first grace_steps
    # Gives robot time to land and stabilize before trying to wheelie
    past_grace = (env.episode_length_buf >= grace_steps).float()

    return past_grace * (wheelie_reward + pitch_only_reward)


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

def non_foot_contact_penalty(
    env,
    shank_sensor_name: str,
    thigh_sensor_name: str,
    trunk_sensor_name: str,
    grace_steps: int = 150,
) -> torch.Tensor:
    shank_contact = env.scene.sensors[shank_sensor_name].data.found
    thigh_contact = env.scene.sensors[thigh_sensor_name].data.found
    trunk_contact = env.scene.sensors[trunk_sensor_name].data.found

    shank_any = shank_contact.any(dim=-1).any(dim=-1)
    thigh_any = thigh_contact.any(dim=-1).any(dim=-1)
    trunk_any = trunk_contact.squeeze(-1).squeeze(-1).bool()

    illegal_contact = shank_any | thigh_any | trunk_any

    past_grace = (env.episode_length_buf >= grace_steps).float()

    return illegal_contact.float() * past_grace
