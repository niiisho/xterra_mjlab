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

    correct_config = (fl_off & fr_off & (rl_on | rr_on)).float()

    gravity = env.scene["robot"].data.projected_gravity_b
    pitch_raw = gravity[:, 0].clamp(min=0.0, max=target_pitch) / target_pitch

    wheelie_reward = correct_config * pitch_raw

    # Very small pitch signal - gives gradient when robot falls flat
    # 0.03 weight means max contribution is 0.3 vs full wheelie of 10.0
    # Not profitable enough to exploit with 4-leg tilt
    pitch_signal = pitch_raw * 0.03

    return wheelie_reward + pitch_signal


def rear_wheelie_reward(
    env,
    sensor_name: str,
    target_pitch: float = 0.5,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found.squeeze(-1)

    fl_on = contact[:, 0] >= 0.5
    fr_on = contact[:, 1] >= 0.5
    rl_off = contact[:, 2] < 0.5
    rr_off = contact[:, 3] < 0.5

    # Fixed - using correct variables defined in this function
    correct_config = ((fl_on | fr_on) & rl_off & rr_off).float()

    gravity = env.scene["robot"].data.projected_gravity_b
    pitch_raw = (-gravity[:, 0]).clamp(min=0.0, max=target_pitch) / target_pitch

    wheelie_reward = correct_config * pitch_raw
    pitch_signal = pitch_raw * 0.03

    return wheelie_reward + pitch_signal


def excessive_roll(env, max_roll: float = 0.785) -> torch.Tensor:
    gravity = env.scene["robot"].data.projected_gravity_b
    roll_signal = gravity[:, 1].abs()
    return roll_signal > max_roll


def base_height_too_low(env, min_height: float):
    return env.scene["robot"].data.body_com_pos_w[:, 0, 2] < min_height


def non_foot_contact_penalty(
    env,
    shank_sensor_name: str,
    thigh_sensor_name: str,
    trunk_sensor_name: str,
) -> torch.Tensor:
    shank_contact = env.scene.sensors[shank_sensor_name].data.found
    thigh_contact = env.scene.sensors[thigh_sensor_name].data.found
    trunk_contact = env.scene.sensors[trunk_sensor_name].data.found

    shank_any = shank_contact.any(dim=-1).any(dim=-1)
    thigh_any = thigh_contact.any(dim=-1).any(dim=-1)
    trunk_any = trunk_contact.any(dim=-1).any(dim=-1)

    illegal_contact = shank_any | thigh_any | trunk_any

    return illegal_contact.float().view(env.num_envs)
