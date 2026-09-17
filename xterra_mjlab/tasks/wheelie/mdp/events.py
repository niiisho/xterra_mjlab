"""Reset events for wheelie tasks."""

import torch
import math


def reset_to_wheelie_pose(
    env,
    env_ids: torch.Tensor,
    pitch_angle: float = 0.6,
    base_height: float = 0.45,
):
    robot = env.scene["robot"]

    root_state = robot.data.default_root_state[env_ids].clone()

    root_state[:, 2] = base_height

    half = pitch_angle / 2.0
    root_state[:, 3] = math.cos(half)
    root_state[:, 4] = 0.0
    root_state[:, 5] = math.sin(half)
    root_state[:, 6] = 0.0
    root_state[:, 7:] = 0.0

    robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)
