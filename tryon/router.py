"""
Virtual Try-On Router
INPUT:  person_image (UploadFile), garment_image (UploadFile), number_of_images (int)
        OR for full outfit: person_image, upper_garment_image, lower_garment_image
OUTPUT: TryOnResponse JSON with base64 result images → frontend
"""

import gc
import base64
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from schemas import TryOnResponse
from schemas import MAX_IMAGE_BYTES
from tryon.service import run_virtual_tryon

router = APIRouter(prefix="/tryon", tags=["Virtual Try-On"])


@router.post("/generate", response_model=TryOnResponse)
async def virtual_try_on(
    person_image:     UploadFile = File(..., description="Full-body photo of the person"),
    garment_image:    UploadFile = File(..., description="Flat-lay product photo of the garment"),
    number_of_images: int        = Form(default=1, ge=1, le=4, description="Number of result variations (1-4)"),
):
    """
    Generates a virtual try-on image of the person wearing one selected garment.

    - Input:  person image + garment image (multipart/form-data)
    - Output: base64-encoded result image(s) for rendering in frontend
    """

    allowed = {"image/jpeg", "image/png", "image/webp"}
    if person_image.content_type not in allowed:
        raise HTTPException(status_code=400, detail="person_image must be JPEG, PNG, or WebP.")
    if garment_image.content_type not in allowed:
        raise HTTPException(status_code=400, detail="garment_image must be JPEG, PNG, or WebP.")

    person_bytes  = await person_image.read()
    if len(person_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="person_image exceeds the 10 MB size limit.")

    garment_bytes = await garment_image.read()
    if len(garment_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="garment_image exceeds the 10 MB size limit.")
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


@router.post("/generate-outfit", response_model=TryOnResponse)
async def virtual_try_on_outfit(
    person_image:        UploadFile = File(..., description="Full-body photo of the person"),
    upper_garment_image: UploadFile = File(..., description="Upper body garment flat-lay photo"),
    lower_garment_image: UploadFile = File(..., description="Lower body garment flat-lay photo"),
    number_of_images:    int        = Form(default=1, ge=1, le=4, description="Number of result variations (1-4)"),
):
    """
    Full outfit try-on — applies upper then lower garment sequentially.
    Step 1: person + upper garment → intermediate result
    Step 2: intermediate result + lower garment → final result

    - Input:  person image + upper garment + lower garment (multipart/form-data)
    - Output: base64-encoded result image(s) showing full outfit
    """

    allowed = {"image/jpeg", "image/png", "image/webp"}
    if person_image.content_type not in allowed:
        raise HTTPException(status_code=400, detail="person_image must be JPEG, PNG, or WebP.")
    if upper_garment_image.content_type not in allowed:
        raise HTTPException(status_code=400, detail="upper_garment_image must be JPEG, PNG, or WebP.")
    if lower_garment_image.content_type not in allowed:
        raise HTTPException(status_code=400, detail="lower_garment_image must be JPEG, PNG, or WebP.")

    person_bytes = await person_image.read()
    if len(person_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="person_image exceeds the 10 MB size limit.")

    upper_bytes  = await upper_garment_image.read()
    if len(upper_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="upper_garment_image exceeds the 10 MB size limit.")

    lower_bytes  = await lower_garment_image.read()
    if len(lower_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="lower_garment_image exceeds the 10 MB size limit.")
    output_mime  = person_image.content_type or "image/jpeg"

    try:
        # Step 1 — apply upper garment to person
        intermediate = run_virtual_tryon(
            person_image_bytes=person_bytes,
            garment_image_bytes=upper_bytes,
            number_of_images=1,
            output_mime_type=output_mime,
        )

        # Step 2 — apply lower garment to the result of step 1
        final = run_virtual_tryon(
            person_image_bytes=intermediate[0],
            garment_image_bytes=lower_bytes,
            number_of_images=number_of_images,
            output_mime_type=output_mime,
        )

    except RuntimeError as e:
        return TryOnResponse(
            success=False,
            error=str(e),
            user_message=(
                "The virtual try-on service is currently unavailable. "
                "Please try again."
            )
        )
    except Exception as e:
        return TryOnResponse(
            success=False,
            error=f"Try-on API error: {str(e)}",
            user_message=(
                "Full outfit try-on failed. Please try again with clear "
                "flat-lay garment images on a white background."
            )
        )
    finally:
        del person_bytes, upper_bytes, lower_bytes
        gc.collect()

    images_b64 = [base64.b64encode(img).decode("utf-8") for img in final]

    return TryOnResponse(
        success=True,
        images_base64=images_b64,
        count=len(images_b64),
        mime_type=output_mime,
    )