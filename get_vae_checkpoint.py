import os
import torch
import trimesh
import numpy as np
from craftsman.utils.config import load_config, parse_structured
from omegaconf import OmegaConf
import craftsman
from dataclasses import fields
import matplotlib.pyplot as plt


def filter_config(cfg_dict, target_class):
    """Filter config to only include fields defined in the target class."""
    valid_fields = {field.name for field in fields(target_class)}
    return {k: v for k, v in cfg_dict.items() if k in valid_fields}


def save_vae_checkpoint(
    model_path: str
):
    """
    Test the reconstruction ability of ShapeAutoEncoderSystem
    """
    # 1. Load the model
    config_path = os.path.join(model_path, "config.yaml")
    ckpt_path = os.path.join(model_path, "model.ckpt")
    
    cfg = load_config(config_path)    
    
    # Get the system class
    system = craftsman.find(cfg.system_type)(cfg.system)
    # 

    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

    
    system.load_state_dict(
        ckpt["state_dict"] if "state_dict" in ckpt else ckpt,
        strict=False
    )

    

    # Extract shape_model state dict
    shape_model_state = {k.replace('shape_model.', ''): v 
                        for k, v in system.state_dict().items() 
                        if k.startswith('shape_model.')}

    # Save shape model checkpoint
    shape_model_path = os.path.join("ckpts/craftsman_vae", "model.ckpt")
    print(f"Saving shape model checkpoint to {shape_model_path}")
    torch.save(shape_model_state, shape_model_path)

def load_pretrained_vae(
    model_path: str
):
    """
    Load the pretrained vae checkpoint
    """

    # 1. Load model
    config_path = os.path.join(model_path, "config.yaml")
    ckpt_path = os.path.join(model_path, "model.ckpt")
    cfg = load_config(config_path)
    system = craftsman.find(cfg.system_type)(cfg.system)

    # 2. Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    system.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt, strict=False)

    point_normal_list = np.load("dora_code_debug/objaverse_model_82_watertight.npz")

    # TODO: Fix this
    surface = torch.from_numpy(point_normal_list['salient_points']).unsqueeze(0).float()
    feats = torch.from_numpy(point_normal_list['salient_normals']).unsqueeze(0).float()
    surface = torch.cat([surface, feats], dim=-1)

    
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
        
        return reconstructed_mesh
    else:
        print("Failed to extract surface from the latent representation")
        return None
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False, 
                       help="Path to model checkpoint directory containing config.yaml and model.ckpt")
    args = parser.parse_args()
    
    save_vae_checkpoint(
        "ckpts/craftsman"
    )

    # load_pretrained_vae(
    #     "ckpts/craftsman_vae"
    # )
