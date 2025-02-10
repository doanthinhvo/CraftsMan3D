import os
import torch
import trimesh
import numpy as np
from craftsman.utils.config import load_config, parse_structured
from omegaconf import OmegaConf
import craftsman
from dataclasses import fields
import matplotlib.pyplot as plt

# After separating the VAE checkpoints from CraftsMan, I tested them here.

device: str = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "ckpts/craftsman_vae"
# 1. Load model
config_path = os.path.join(model_path, "config.yaml")
ckpt_path = os.path.join(model_path, "model.ckpt")
cfg = load_config(config_path)
system = craftsman.find(cfg.system_type)(cfg.system)

# 2. Load checkpoint
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
system.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt, strict=False)
system = system.to(device).eval()
print(f"Loaded checkpoint from {ckpt_path}")

point_normal_list = np.load("/home/thinhvd/workspace/occupancy_networks/ship_v2/pointcloud/objaverse_model_82_watertight.npz")

# TODO: Fix this
surface = torch.from_numpy(point_normal_list['salient_points']).unsqueeze(0).float()
feats = torch.from_numpy(point_normal_list['salient_normals']).unsqueeze(0).float()
surface = torch.cat([surface, feats], dim=-1).to(device)

shape_latents, kl_embed, posterior = system.encode(
    surface=surface,
    sample_posterior=True
)

latents = system.decode(kl_embed)

mesh_outputs, has_surface = system.extract_geometry(
    latents
)

output_dir = "reconstruction_vae"
os.makedirs(output_dir, exist_ok=True)

if has_surface[0]:
    reconstructed_mesh = trimesh.Trimesh(
        vertices=mesh_outputs[0][0],
        faces=mesh_outputs[0][1]
    )
    
    # base_name = os.path.splitext(os.path.basename(input_mesh_path))[0]
    reconstructed_mesh.export(os.path.join(output_dir, f"meomeo_reconstructed.obj"))
    
    # 5. Print some statistics
    # print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Reconstructed mesh: {len(reconstructed_mesh.vertices)} vertices, {len(reconstructed_mesh.faces)} faces")
    
else:
    print("Failed to extract surface from the latent representation")

    

    