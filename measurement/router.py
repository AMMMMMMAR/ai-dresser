"""
Measurement Router
INPUT:  front_image (UploadFile), side_image (UploadFile), height_cm (float)
OUTPUT: MeasurementResponse JSON → frontend + recommendation feature
"""

import gc
import numpy as np
import cv2
from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from schemas import MeasurementResponse
from measurement.service import extract_measurements

router = APIRouter(prefix="/measurements", tags=["Body Measurements"])


def decode_image(file_bytes: bytes, label: str) -> np.ndarray:
    """Decode image bytes to numpy RGB array — RAM only, no disk."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode {label}. Please upload a valid JPG or PNG image."
        )
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


@router.post("/extract", response_model=MeasurementResponse)
async def extract_body_measurements(
    front_image: UploadFile = File(..., description="Front-facing full body photo"),
    side_image:  UploadFile = File(..., description="Side-profile full body photo"),
    height_cm:   float      = Form(..., description="User height in centimeters"),
):
    """
    Extracts body measurements from a front and side photo.

    - Input: two images (multipart/form-data) + height in cm
    - Output: shoulder, chest, waist, hip, arm length, inseam in cm
    """

    # ── Validate height ───────────────────────────────────────────────────────
    if height_cm < 100 or height_cm > 250:
        raise HTTPException(
            status_code=400,
            detail="height_cm must be between 100 and 250 cm."
        )

    # ── Validate content types ────────────────────────────────────────────────
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if front_image.content_type not in allowed:
        raise HTTPException(status_code=400, detail="front_image must be a JPEG, PNG, or WebP image.")
    if side_image.content_type not in allowed:
        raise HTTPException(status_code=400, detail="side_image must be a JPEG, PNG, or WebP image.")

    # ── Read bytes into RAM ───────────────────────────────────────────────────
    front_bytes = await front_image.read()
    side_bytes  = await side_image.read()

    front_img = decode_image(front_bytes, "front_image")
    side_img  = decode_image(side_bytes,  "side_image")

    # ── Free raw bytes immediately ────────────────────────────────────────────
    del front_bytes, side_bytes

    # ── Run YOLO measurement pipeline ─────────────────────────────────────────
    try:
        result = extract_measurements(front_img, side_img, height_cm)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Measurement pipeline failed unexpectedly: {str(e)}"
        )
    finally:
        # Free image arrays from RAM after processing
        del front_img, side_img
        gc.collect()

    # ── Handle pipeline-level errors ─────────────────────────────────────────
    if "error" in result:
        return MeasurementResponse(
            success=False,
            error=result["error"],
            diagnostics=result.get("diagnostics", {}),
            user_message=(
                "We could not detect your full body in one or both photos. "
                "Please stand straight, ensure your full body is visible, "
                "use good lighting, and try again."
            )
        )

    return MeasurementResponse(
        success=True,
        measurements=result["measurements"],
        diagnostics=result["diagnostics"],
    )
