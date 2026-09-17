"""Reset events for wheelie tasks."""

import torch
import math


def reset_to_wheelie_pose(
    env,
    env_ids: torch.Tensor,
    pitch_angle: float = 0.8,
    base_height: float = 0.5,
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

    # Custom joint positions for wheelie stance
    joint_pos = robot.data.default_joint_pos[env_ids].clone()

    # Get joint indices - names from the XML
    # Rear legs: extend hips back, bend knees to push weight up
    # Front legs: retract up so feet don't drag
    # These are approximate - may need tuning after first visual check
    joint_names = [n for n in robot.data.joint_names]

    for i, name in enumerate(joint_names):
        if "RL_hip" in name or "RR_hip" in name:
            joint_pos[env_ids, i] = -0.3   # rear hips back
        elif "RL_thigh" in name or "RR_thigh" in name:
            joint_pos[env_ids, i] = 0.8    # rear thighs push down
        elif "RL_calf" in name or "RR_calf" in name:
            joint_pos[env_ids, i] = -1.4   # rear calves extend
        elif "FL_hip" in name or "FR_hip" in name:
            joint_pos[env_ids, i] = 0.3    # front hips forward
        elif "FL_thigh" in name or "FR_thigh" in name:
            joint_pos[env_ids, i] = -0.8   # front thighs retract up
        elif "FL_calf" in name or "FR_calf" in name:
            joint_pos[env_ids, i] = 1.4    # front calves fold up

    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)
