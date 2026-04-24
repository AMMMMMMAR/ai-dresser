"""
Skin Tone Router
INPUT:  person_image (UploadFile)
OUTPUT: SkinToneResponse JSON → frontend + recommendation feature + avatar feature
"""

import gc
from fastapi import APIRouter, File, UploadFile, HTTPException
from schemas import SkinToneResponse, SkinToneResult
from skin_tone.service import predict_skin_tone

router = APIRouter(prefix="/skin-tone", tags=["Skin Tone Detection"])


@router.post("/detect", response_model=SkinToneResponse)
async def detect_skin_tone(
    person_image: UploadFile = File(..., description="Photo of the person"),
):
    """
    Detects the skin tone of the person in the uploaded image.

    - Input:  one image (multipart/form-data)
    - Output: skin_tone class, reference hex color, sampled hex color
    - This output feeds into: recommendation feature, avatar feature, frontend
    """

    allowed = {"image/jpeg", "image/png", "image/webp"}
    if person_image.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail="person_image must be a JPEG, PNG, or WebP image."
        )

    image_bytes = await person_image.read()

    try:
        result = predict_skin_tone(image_bytes)
    except RuntimeError as e:
        return SkinToneResponse(
            success=False,
            error=str(e),
            user_message=(
                "The skin tone detection service is currently unavailable. "
                "Other features are still working. Please try again shortly."
            )
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        return SkinToneResponse(
            success=False,
            error=f"Unexpected error: {str(e)}",
            user_message=(
                "Skin tone detection failed. Please ensure your photo is "
                "clear, well-lit, and shows your face or skin area."
            )
        )
    finally:
        del image_bytes
        gc.collect()

    return SkinToneResponse(
        success=True,
        skin_tone=SkinToneResult(**result)
    )
