import torch
import matplotlib.pyplot as plt
from rsl_rl.runners import OnPolicyRunner
# Import your env setup functions here (same as play.py)

def record_and_plot_torques(env, runner, num_steps=1000):
    obs, _ = env.reset()
    policy = runner.get_inference_policy(device=env.device)
    
    # Storage for the 4 rear joints (assuming indexes: RL_hip, RL_thigh, RL_calf, etc.)
    # We will just record all joint torques for the first environment (index 0)
    torque_history = []

    print("Simulating and recording torques...")
    for _ in range(num_steps):
        actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)
        
        # Read the applied torques directly from the MuJoCo robot data
        # shape is usually (num_envs, num_joints). We grab env 0.
        current_torques = env.scene["robot"].data.applied_torque[0].detach().cpu().numpy()
        torque_history.append(current_torques)
    
    # Convert list to a numpy array for easy plotting: Shape (1000, num_joints)
    import numpy as np
    torque_history = np.array(torque_history)
    
    # Plotting the Rear Legs (Adjust the column indexes based on your robot's exact joint order)
    # Assuming standard order: FL, FR, RL, RR (3 joints each)
    plt.figure(figsize=(12, 6))
    
    # Example: Plotting Rear Left and Rear Right Knee/Calf torques
    plt.plot(torque_history[:, 8], label="Rear Left Knee Torque", alpha=0.8)
    plt.plot(torque_history[:, 11], label="Rear Right Knee Torque", alpha=0.8)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Joint Torques During Rear-Leg Balance (1000 Steps)")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.grid(True)
    plt.savefig("torque_plot.png")
    print("Plot saved as torque_plot.png")
