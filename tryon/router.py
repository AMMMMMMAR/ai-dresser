"""
Virtual Try-On Router
INPUT:  person_image (UploadFile), garment_image (UploadFile), number_of_images (int)
OUTPUT: TryOnResponse JSON with base64 result images → frontend
"""

import gc
import base64
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from schemas import TryOnResponse
from tryon.service import run_virtual_tryon

router = APIRouter(prefix="/tryon", tags=["Virtual Try-On"])


@router.post("/generate", response_model=TryOnResponse)
async def virtual_try_on(
    person_image:     UploadFile = File(..., description="Full-body photo of the person"),
    garment_image:    UploadFile = File(..., description="Flat-lay product photo of the garment"),
    number_of_images: int        = Form(default=1, ge=1, le=4, description="Number of result variations (1-4)"),
):
    """
    Generates a virtual try-on image of the person wearing the selected garment.

    - Input:  person image + garment image (multipart/form-data)
    - Output: base64-encoded result image(s) for rendering in frontend
    """

    allowed = {"image/jpeg", "image/png", "image/webp"}
    if person_image.content_type not in allowed:
        raise HTTPException(status_code=400, detail="person_image must be JPEG, PNG, or WebP.")
    if garment_image.content_type not in allowed:
        raise HTTPException(status_code=400, detail="garment_image must be JPEG, PNG, or WebP.")

    person_bytes  = await person_image.read()
    garment_bytes = await garment_image.read()
    output_mime   = person_image.content_type or "image/jpeg"

    try:
        result_images = run_virtual_tryon(
            person_image_bytes=person_bytes,
            garment_image_bytes=garment_bytes,
            number_of_images=number_of_images,
            output_mime_type=output_mime,
        )
    except RuntimeError as e:
        return TryOnResponse(
            success=False,
            error=str(e),
            user_message=(
                "The virtual try-on service is currently unavailable. "
                "Please check your API credentials and try again."
            )
        )
    except Exception as e:
        return TryOnResponse(
            success=False,
            error=f"Try-on API error: {str(e)}",
            user_message=(
                "Virtual try-on failed. For best results use a clear "
                "front-facing photo and a flat-lay garment image on a white background."
            )
        )
    finally:
        del person_bytes, garment_bytes
        gc.collect()

    images_b64 = [base64.b64encode(img).decode("utf-8") for img in result_images]

    return TryOnResponse(
        success=True,
        images_base64=images_b64,
        count=len(images_b64),
        mime_type=output_mime,
    )
