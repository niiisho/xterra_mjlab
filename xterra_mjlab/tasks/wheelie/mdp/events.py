"""Reset events for wheelie tasks."""

import torch
import math


def reset_to_wheelie_pose(
    env,
    env_ids: torch.Tensor,
    pitch_angle: float = 0.7,
    base_height: float = 0.45,
):
    """
    Reset selected environments with robot already in wheelie position.
    pitch_angle: backward tilt in radians. 0.7 = ~40 degrees.
    base_height: height of base center above ground.
    """
    robot = env.scene["robot"]

    # Start from default root state
    root_state = robot.data.default_root_state[env_ids].clone()

    # Set height
    root_state[:, 2] = base_height

    # Set backward pitch quaternion (rotation around Y axis)
    # MuJoCo convention: (w, x, y, z)
    half = pitch_angle / 2.0
    root_state[:, 3] = math.cos(half)   # w
    root_state[:, 4] = 0.0               # x
    root_state[:, 5] = math.sin(half)   # y
    root_state[:, 6] = 0.0               # z

    # Zero velocity so robot doesn't spawn with momentum
    root_state[:, 7:] = 0.0

    robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    # Use default joint positions
    # Robot will figure out leg placement from there
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_pos_to_sim(joint_pos, env_ids=env_ids)
    robot.write_joint_vel_to_sim(joint_vel, env_ids=env_ids)
