"""
Avatar Service
Loads SAM 3D Body model once.
Processes image bytes in RAM — exports .obj as bytes, no disk.
"""

import gc
import io
import base64
import torch
import trimesh
import numpy as np
import cv2
from PIL import Image

_pipeline = None


def load_avatar_model():
    """Load SAM 3D Body model once at startup."""
    global _pipeline
    if _pipeline is not None:
        return
    try:
        from avatar.sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator
        import os
        from huggingface_hub import login

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_cfg = load_sam_3d_body_hf("facebook/sam-3d-body-vith", device=device)
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        _pipeline = {"estimator": estimator, "device": device}
        print(f"✅ Avatar model (SAM 3D Body) loaded on {device}")
    except Exception as e:
        print(f"⚠️  Avatar model failed to load: {e}. Avatar feature will be unavailable.")
        _pipeline = None


def generate_avatar(image_bytes: bytes, skin_tone: str = None) -> str:
    """
    Process image bytes → 3D mesh → return base64 .obj string.
    All processing done in RAM.

    Args:
        image_bytes: raw bytes of the uploaded image
        skin_tone:   optional skin tone string (logged, for future coloring)

    Returns:
        base64-encoded .obj mesh string
    """
    if _pipeline is None:
        raise RuntimeError("Avatar model is not loaded.")

    # Decode image bytes → numpy (RAM only)
    arr     = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image bytes.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    estimator = _pipeline["estimator"]

    # Run SAM 3D Body estimator
    # Pass PIL image directly to avoid temp file writes
    outputs = estimator.process_one_image(pil_img, inference_type="body")

    if not outputs or len(outputs) == 0:
        raise ValueError("No human detected in the image.")

    output   = outputs[0]
    vertices = output["pred_vertices"]
    faces    = estimator.faces

    # Handle tensor conversion
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    # Build mesh in RAM
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Export to bytes buffer — no disk write
    obj_buffer = io.BytesIO()
    mesh.export(obj_buffer, file_type="obj")
    obj_bytes = obj_buffer.getvalue()

    # Encode to base64 for JSON transport
    mesh_b64 = base64.b64encode(obj_bytes).decode("utf-8")

    # Cleanup
    del arr, img_bgr, img_rgb, pil_img, outputs, output, vertices, faces, mesh, obj_buffer, obj_bytes
    gc.collect()
    if _pipeline["device"].type == "cuda":
        torch.cuda.empty_cache()

    return mesh_b64
