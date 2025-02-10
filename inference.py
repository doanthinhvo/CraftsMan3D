from craftsman import CraftsManPipeline
import torch

# # load from local ckpt
pipeline = CraftsManPipeline.from_pretrained("./ckpts/craftsman", device="cuda:0", torch_dtype=torch.float32) 

image_file = "val_data/images/rgba_monster.png"
obj_file = "monster.glb" # output obj or glb file
# inference
mesh = pipeline(image_file, seed=42).meshes[0]
mesh.export(obj_file)
