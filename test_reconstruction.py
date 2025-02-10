import os
import torch
import trimesh
import numpy as np
from craftsman.utils.config import load_config, parse_structured
from omegaconf import OmegaConf
import craftsman
from dataclasses import fields

def filter_config(cfg_dict, target_class):
    """Filter config to only include fields defined in the target class."""
    valid_fields = {field.name for field in fields(target_class)}
    return {k: v for k, v in cfg_dict.items() if k in valid_fields}

def test_autoencoder_reconstruction(
    model_path: str,
    input_npz_path: str,
    n_surface_points: int = 10240,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
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
    
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    system.load_state_dict(
        ckpt["state_dict"] if "state_dict" in ckpt else ckpt,
    )
    system = system.to(device).eval()

    # 2. Load surface data.
    surface_data = np.load(input_npz_path)
    uniform_points = torch.from_numpy(surface_data['uniform_points']).float().unsqueeze(0).to(device)
    salient_points = torch.from_numpy(surface_data['salient_points']).float().unsqueeze(0).to(device)
    uniform_normals = torch.from_numpy(surface_data['uniform_normals']).float().unsqueeze(0).to(device)
    salient_normals = torch.from_numpy(surface_data['salient_normals']).float().unsqueeze(0).to(device)
    
    surface = {
        'uniform_pc': uniform_points,
        'salient_pc': salient_points,
        'uniform_feats': uniform_normals,
        'salient_feats': salient_normals
    }

    # torch.Size([1, 10240, 6])
    

    shape_latents, kl_embed, posterior = system.shape_model.encode(
        surface = surface, 
        sample_posterior=True
        )

    latents = system.shape_model.decode(kl_embed)     

    # Extract geometry from latents
    mesh_outputs, has_surface = system.shape_model.extract_geometry(
        latents
    )                                       

    # 4. Save reconstructed mesh
    output_dir = "reconstruction_results"
    os.makedirs(output_dir, exist_ok=True)
    
    if has_surface[0]:
        reconstructed_mesh = trimesh.Trimesh(
            vertices=mesh_outputs[0][0],
            faces=mesh_outputs[0][1]
        )
        
        base_name = os.path.splitext(os.path.basename(input_npz_path))[0]
        reconstructed_mesh.export(os.path.join(output_dir, f"{base_name}_reconstructed.obj"))
        
        # 5. Print some statistics
        print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"Reconstructed mesh: {len(reconstructed_mesh.vertices)} vertices, {len(reconstructed_mesh.faces)} faces")
        
        return reconstructed_mesh
    else:
        print("Failed to extract surface from the latent representation")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to model checkpoint directory containing config.yaml and model.ckpt")
    parser.add_argument("--input_mesh", type=str, required=False, 
                       help="Path to input mesh file (obj or glb)")
    parser.add_argument("--input_npz", type=str, required=True, 
                       help="Path to input npz file")
    parser.add_argument("--n_points", type=int, default=10240, 
                       help="Number of surface points to sample")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    reconstructed = test_autoencoder_reconstruction(
        args.model_path,
        args.input_npz,
        args.n_points,
        args.device
    )