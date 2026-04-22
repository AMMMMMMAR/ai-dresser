import base64
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from .service import run_virtual_tryon
from .schemas import TryOnResponse

router = APIRouter(prefix="/tryon", tags=["Virtual Try-On"])


@router.post("/generate", response_model=TryOnResponse)
async def virtual_try_on(
    person_image: UploadFile = File(..., description="Full-body photo of the person"),
    garment_image: UploadFile = File(..., description="Flat-lay product photo of the garment"),
    number_of_images: int = Form(default=1, ge=1, le=4, description="Number of result images to generate (1-4)"),
):
    """
    Takes a person image and a garment image.
    Returns base64-encoded try-on result images.
    """
    # Validate file types
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if person_image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"person_image must be jpeg, png, or webp.")
    if garment_image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"garment_image must be jpeg, png, or webp.")

    person_bytes  = await person_image.read()
    garment_bytes = await garment_image.read()

    # Determine output mime type from person image
    output_mime = person_image.content_type or "image/jpeg"

    try:
        result_images = run_virtual_tryon(
            person_image_bytes=person_bytes,
            garment_image_bytes=garment_bytes,
            number_of_images=number_of_images,
            output_mime_type=output_mime,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"API call failed: {str(e)}")

    images_b64 = [base64.b64encode(img).decode("utf-8") for img in result_images]

    return TryOnResponse(
        images_base64=images_b64,
        count=len(images_b64),
        mime_type=output_mime,
    )