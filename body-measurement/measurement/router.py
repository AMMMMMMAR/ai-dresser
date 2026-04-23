from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2

from .yolo_service import extract_measurements
from .schemas import MeasurementResponse

router = APIRouter(prefix="/measurements", tags=["Body Measurements"])


def decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


@router.post("/extract", response_model=MeasurementResponse)
async def extract_body_measurements(
    front_image: UploadFile = File(..., description="Front-facing full body photo"),
    side_image:  UploadFile = File(..., description="Side-profile full body photo"),
    height_cm:   float      = Form(..., description="User height in centimeters"),
):
    """
    Takes a front image, side image, and user height.
    Returns body measurements: shoulder, chest, waist, hip, arm, inseam.
    """
    if height_cm < 100 or height_cm > 250:
        raise HTTPException(status_code=400, detail="height_cm must be between 100 and 250.")

    front_bytes = await front_image.read()
    side_bytes  = await side_image.read()

    front_img = decode_image(front_bytes)
    side_img  = decode_image(side_bytes)

    result = extract_measurements(front_img, side_img, height_cm)

    if "error" in result:
        return MeasurementResponse(
            error=result["error"],
            diagnostics=result.get("diagnostics", {})
        )

    return MeasurementResponse(
        measurements=result["measurements"],
        diagnostics=result["diagnostics"]
    )