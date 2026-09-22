import torch

# The exact path from your error log
old_ckpt_path = "/workspace/xterra_mjlab/logs/rsl_rl/svanm2_wheelie/2026-9-11_12-00-00/model_11650.pt"
new_ckpt_path = "/workspace/xterra_mjlab/logs/rsl_rl/svanm2_wheelie/2026-9-11_12-00-00/patched_model.pt"

print("Loading checkpoint...")
checkpoint = torch.load(old_ckpt_path, map_location="cpu")

# 1. Pad the Actor and Critic networks
for model_type in ["actor_state_dict", "critic_state_dict"]:
    if model_type in checkpoint:
        sd = checkpoint[model_type]
        old_w = sd.get("mlp.0.weight")
        
        # If we find the layer that expects 252 inputs
        if old_w is not None and old_w.shape[1] == 252:
            # Create a new blank layer expecting 270 inputs (filled with zeros)
            new_w = torch.zeros((old_w.shape[0], 270), dtype=old_w.dtype, device=old_w.device)
            
            # Copy the old 252 learned weights into the beginning of the new layer
            new_w[:, :252] = old_w
            
            # Replace the layer in the dictionary
            sd["mlp.0.weight"] = new_w
            print(f"Successfully patched {model_type} from {old_w.shape} to {new_w.shape}")

# 2. Delete the old Optimizer state 
# (Since we changed the network shape, the old Adam optimizer will crash. 
# Deleting it forces PyTorch to generate a fresh one, which is perfect for transfer learning).
if "optimizer_state_dict" in checkpoint:
    del checkpoint["optimizer_state_dict"]
    print("Cleared old optimizer state.")

# Save the rescued model
torch.save(checkpoint, new_ckpt_path)
print(f"\nSaved rescued model to: {new_ckpt_path}")
