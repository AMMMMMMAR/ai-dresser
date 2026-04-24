"""
Skin Tone Service
Loads EfficientNetB0 model once into RAM.
Runs inference on raw image bytes — no disk writes.
"""

import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
import os
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "model", "best_skintone_model.pth")
NUM_CLASSES = 6
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["caramel", "cocoa", "fair", "light", "medium", "Tan"]

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

_model = None

TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_skin_tone_model():
    """Load model once at startup."""
    global _model
    if _model is not None:
        return
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    _model = model
    print(f"✅ Skin tone model loaded on {DEVICE}")


def _extract_avg_hex(img_bgr: np.ndarray) -> str:
    """Sample center region and return average skin color as hex."""
    h, w = img_bgr.shape[:2]
    roi  = img_bgr[int(h*0.2):int(h*0.7), int(w*0.3):int(w*0.7)]
    mask = np.any(roi != [0, 0, 0], axis=2)
    avg  = roi[mask].mean(axis=0) if mask.sum() > 0 else roi.mean(axis=(0, 1))
    r, g, b = int(avg[2]), int(avg[1]), int(avg[0])
    return f"#{r:02X}{g:02X}{b:02X}"


def predict_skin_tone(image_bytes: bytes) -> dict:
    """
    Full inference pipeline — RAM only.
    Input:  raw image bytes
    Output: dict with skin_tone, reference_hex, sampled_hex
    """
    if _model is None:
        raise RuntimeError("Skin tone model is not loaded.")

    # Decode bytes → numpy
    arr     = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image bytes.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Run model
    tensor = TRANSFORMS(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits   = _model(tensor)
        probs    = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    pred_class    = CLASS_NAMES[pred_idx]
    reference_hex = FITZ_REFERENCE_HEX.get(pred_class, "#888888")
    sampled_hex   = _extract_avg_hex(img_bgr)

    # Free tensors
    del tensor, logits, probs
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "skin_tone":     pred_class,
        "reference_hex": reference_hex,
        "sampled_hex":   sampled_hex,
    }
