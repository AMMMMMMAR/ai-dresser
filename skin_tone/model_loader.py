# model_loader.py
# Loads the EfficientNetB0 model once when the server starts
# Think of this as the "setup" file — runs once, stays in memory

import torch
import torch.nn as nn
from torchvision import models

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "model/best_skintone_model.pth"
NUM_CLASSES = 6
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Class names (must match your training folder order exactly) ───────────────
CLASS_NAMES = ["caramel", "cocoa", "fair", "light", "medium", "Tan"]

def load_model():
    """
    Rebuilds EfficientNetB0 architecture and loads trained weights.
    Returns the model ready for inference.
    """
    model = models.efficientnet_b0(weights=None)

    # Replace classifier head — must match exactly what you trained
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES)
    )

    # Load saved weights
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model = model.to(DEVICE)
    model.eval()   # inference mode — disables dropout

    print(f"✅ Model loaded on {DEVICE}")
    return model