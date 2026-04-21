# inference.py
# Contains the full prediction pipeline
# Input  → raw image bytes
# Output → prediction dict with class, confidence, hex colors

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model_loader import DEVICE, CLASS_NAMES

# ── Fitzpatrick scale mappings ────────────────────────────────────────────────
FITZPATRICK_MAP = {
    "fair":    "Type I   (Very Fair)",
    "light":   "Type II  (Light)",
    "medium":  "Type III (Medium)",
    "Tan":     "Type IV  (Olive/Tan)",
    "caramel": "Type V   (Brown)",
    "cocoa":   "Type VI  (Black)",
}

FITZ_REFERENCE_HEX = {
    "fair":    "#F6D3B7",
    "light":   "#EFBF9A",
    "medium":  "#C68642",
    "Tan":     "#A0522D",
    "caramel": "#7B4A2D",
    "cocoa":   "#3B1E08",
}

# ── Image transforms (must match val_transforms from training) ────────────────
transforms_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def extract_avg_hex(img_bgr: np.ndarray) -> str:
    """
    Samples the center region of the image
    and returns the average skin color as a hex string.
    Ignores black pixels (masked areas).
    """
    h, w     = img_bgr.shape[:2]
    cx1, cx2 = int(w * 0.3), int(w * 0.7)
    cy1, cy2 = int(h * 0.2), int(h * 0.7)
    roi      = img_bgr[cy1:cy2, cx1:cx2]

    mask    = np.any(roi != [0, 0, 0], axis=2)
    avg_bgr = roi[mask].mean(axis=0) if mask.sum() > 0 else roi.mean(axis=(0, 1))

    r, g, b = int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0])
    return f"#{r:02X}{g:02X}{b:02X}"


def predict(image_bytes: bytes, model) -> dict:
    """
    Full prediction pipeline.

    Args:
        image_bytes : raw bytes of the uploaded image file
        model       : loaded EfficientNetB0 model from model_loader.py

    Returns:
        dict with prediction results
    """

    # ── Step 1: Decode image bytes → numpy array ──────────────────────────────
    np_arr  = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Could not decode image. Make sure it's a valid jpg/png.")

    # ── Step 2: Convert BGR → RGB → PIL ──────────────────────────────────────
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # ── Step 3: Apply transforms → tensor ────────────────────────────────────
    input_tensor = transforms_pipeline(pil_img).unsqueeze(0).to(DEVICE)

    # ── Step 4: Run model ─────────────────────────────────────────────────────
    with torch.no_grad():
        logits     = model(input_tensor)
        probs      = torch.softmax(logits, dim=1)[0]
        pred_idx   = probs.argmax().item()
        confidence = probs[pred_idx].item() * 100

    # ── Step 5: Build result ──────────────────────────────────────────────────
    pred_class    = CLASS_NAMES[pred_idx]
    fitzpatrick   = FITZPATRICK_MAP.get(pred_class,   "Unknown")
    reference_hex = FITZ_REFERENCE_HEX.get(pred_class, "#888888")
    sampled_hex   = extract_avg_hex(img_bgr)

    # All class probabilities (useful for debugging)
    all_probs = {
        CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "skin_tone":     pred_class,
         "sampled_hex":   sampled_hex,
         "reference_hex": reference_hex,
    }