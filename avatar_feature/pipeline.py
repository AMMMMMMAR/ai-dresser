import os
import torch
import trimesh
import pickle
from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator
from huggingface_hub import login

# Authenticate with Hugging Face to access the gated Meta models
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
elif "HF_TOKEN" not in os.environ:
    print("Warning: No HF_TOKEN environment variable found. Gated models may fail to download.")

class AvatarExtractorPipeline:
    def __init__(self):
        # Gracefully fallback to CPU if GPU is unavailable
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Initializing AvatarExtractorPipeline on {self.device}...")
        
        # Load the SAM-3D-Body model
        self.model, self.model_cfg = load_sam_3d_body_hf("facebook/sam-3d-body-vith", device=self.device)
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.model,
            model_cfg=self.model_cfg,
            human_detector=None, 
            human_segmentor=None,
            fov_estimator=None
        )

    def extract_avatar(self, image_path):
        """
        Process the image and return the path to the 3D .obj mesh file.
        """
        outputs = self.estimator.process_one_image(str(image_path), inference_type="body")
        
        if not outputs or len(outputs) == 0:
            raise ValueError("No human could be tracked/processed in the image.")
            
        output = outputs[0]
        
        # Build raw mesh. Use process=False to preserve raw SAM-3D vertex indexing.
        vertices = output["pred_vertices"]
        faces = self.estimator.faces
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Save mesh to an output folder
        out_dir = os.path.dirname(image_path)
        out_mesh_path = os.path.join(out_dir, "avatar.obj")
        mesh.export(out_mesh_path)
        
        # Safely convert PyTorch tensors to Numpy arrays for universal reading
        def to_numpy(data):
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            elif isinstance(data, dict):
                return {k: to_numpy(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [to_numpy(v) for v in data]
            else:
                return data
                
        safe_output = to_numpy(output)
        
        # Export skeleton and structural data
        out_pkl_path = os.path.join(out_dir, "avatar_data.pkl")
        with open(out_pkl_path, "wb") as f:
            pickle.dump(safe_output, f)
        
        return out_mesh_path, out_pkl_path
