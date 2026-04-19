# SAM 3D Body - High-Level Walkthrough

This document provides a high-level summary and walkthrough of the **[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)** repository by Facebook/Meta Research.

## 1. Overview
**SAM 3D Body** is a state-of-the-art foundation model for full-body 3D Human Mesh Recovery (HMR) from a single image. It estimates the human pose of the body, feet, and hands using a parametric representation called the **Momentum Human Rig (MHR)**. MHR decouples skeletal structure from surface shapes, providing highly accurate and interpretable body representations. 

Inspired by the SAM (Segment Anything) family, it employs an encoder-decoder architecture and accepts user-guided prompts (like 2D keypoints and bounding boxes/masks) to refine the 3D mesh reconstruction.

## 2. Project Structure
After cloning, the repository has the following key directories and files:

*   **`sam_3d_body/`**: The core package.
    *   `sam_3d_body_estimator.py`: The wrapper class `SAM3DBodyEstimator`, which orchestrates the entire inference pipeline (object detection, segmentation, field-of-view estimation, and the core mesh recovery).
    *   **`models/`**: The PyTorch model unet/encoder-decoder definitions for 3D body and hand recovery.
    *   **`data/`**: Utilities for reading images, handling geometry, transforming/cropping bounding boxes, and data loaders.
*   **`tools/`**: Helper modules to initialize external dependency models:
    *   `build_detector.py`: Sets up bounding-box human detection (e.g., using `vitdet` or `sam3`).
    *   `build_sam.py`: Sets up a segmenter (SAM2) to segment humans.
    *   `build_fov_estimator.py`: Generates the camera Field-of-View handling via external models like `moge2`.
    *   `vis_utils.py`: Used for rendering outputs back onto original images.
*   **`demo.py` & `notebook/demo_human.ipynb`**: Entry points demonstrating how to use the model on single images or a directory of images.

## 3. High-Level Execution Pipeline
If you run `demo.py`, the code goes through the following high-level flow (see `sam_3d_body_estimator.py:process_one_image`):

1.  **Image Input**: Load an RGB image.
2.  **Human Detection**: If bounded boxes are not provided, it uses a top-down approach by running a human detector (like ViTDet or SAM3) to detect human subjects.
3.  **Prompting / Masking**: If specified (`use_mask=True`), it runs SAM2 across the bounding boxes to extract human masks. This focuses the mesh recovery.
4.  **Batching & Cropping**: Crops the detected regions, applies affine transforms, and packages them into a batch.
5.  **Camera Intrinsics / FOV**: If intrinsics aren't given, it runs an FOV Estimator (e.g., MoGe) to approximate the focal length and depth.
6.  **SAM 3D Body Inference**: Passes the tensors into the SAM 3D model.
    *   The model can run in `"full"` mode (running both body and hand decoders) or separate modes.
    *   Outputs include: `pred_keypoints_3d`/`2d`, `pred_vertices`, `cam_t` (translation), `body_pose`/`hand_pose` parameters, and `shape` parameters.
7.  **Rendering**: Uses pyrender in `tools/vis_utils.py` to project the 3D vertices/faces onto the 2D image plane to visualize the result.

## 4. Setup & Running the Demo

> [!WARNING]
> SAM 3D Body relies on large pre-trained checkpoints (like `DINOv3` and `ViT-H`) which are exclusively hosted on Hugging Face. You must go to the [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) repository and accept the user agreement to gain access.

**Quick Demo Command Structure**:
```bash
python demo.py \
    --image_folder <path_to_images> \
    --output_folder <path_to_output> \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
    --detector_name vitdet \
    --use_mask
```

## 5. Main Takeaway
Think of SAM 3D Body as a "pipeline manager" that chains together multiple state-of-the-art vision models:
*   A **Detector** to find people.
*   A **Segmenter** to isolate them (optional but helpful).
*   A **Depth/FOV analyzer** to understand the camera layout.
*   **The Main SAM 3D Model** to orchestrate the translation of the 2D crops into precise 3D human vertex coordinates and parametric MHR coordinates.
