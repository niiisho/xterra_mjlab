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


def base_height_too_low(env, min_height: float):
    is_low = env.scene["robot"].data.body_com_pos_w[:, 0, 2] < min_height
    return is_low


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


def front_continuous_air_reward(env, sensor_name: str) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found.squeeze(-1)

    fl_off = contact[:, 0] < 0.5
    fr_off = contact[:, 1] < 0.5
    rl_on = contact[:, 2] >= 0.5
    rr_on = contact[:, 3] >= 0.5

    # We only count air time if the rear feet are safely planted!
    valid_hop = (fl_off & fr_off & rl_on & rr_on)

    # 1. Initialize the custom timer if it doesn't exist yet
    device = contact.device
    if not hasattr(env, "_front_air_steps"):
        env._front_air_steps = torch.zeros(env.num_envs, dtype=torch.float, device=device)
    
    # 2. Reset the timer for any environments that died on the last frame
    reset_buf = getattr(env, "reset_buf", torch.zeros(env.num_envs, dtype=torch.bool, device=device))
    env._front_air_steps = torch.where(
        reset_buf, 
        torch.zeros_like(env._front_air_steps), 
        env._front_air_steps
    )

    # 3. Increment the timer if hopping, otherwise slam it back to 0
    env._front_air_steps = torch.where(
        valid_hop,
        env._front_air_steps + 1.0,
        torch.zeros_like(env._front_air_steps)
    )

    # 4. Exponential Reward: Square the time held
    # We divide by 100 first so a 100-step hold equals 1.0 points, 200 steps = 4.0 points, etc.
    # This prevents the numbers from exploding to infinity.
    return (env._front_air_steps / 100.0) ** 2


def front_contact_penalty(env, sensor_name: str) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found.squeeze(-1)
    
    fl_on = contact[:, 0] >= 0.5
    fr_on = contact[:, 1] >= 0.5
    
    # Returns 1.0 if either front foot touches the ground
    return (fl_on | fr_on).float()
    
