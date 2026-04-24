"""
Avatar Router
INPUT:  person_image (UploadFile), skin_tone (str, optional Form field)
OUTPUT: AvatarResponse JSON with base64 .obj mesh → frontend (Three.js)
"""

import gc
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from schemas import AvatarResponse
from avatar.service import generate_avatar

router = APIRouter(prefix="/avatar", tags=["3D Avatar"])


@router.post("/generate", response_model=AvatarResponse)
async def generate_3d_avatar(
    person_image: UploadFile = File(..., description="Photo of the person"),
    skin_tone:    Optional[str] = Form(None, description="Skin tone from skin tone feature (optional)"),
):
    """
    Generates a 3D body avatar from a person photo.

    - Input:  one image (multipart/form-data) + optional skin_tone string
    - Output: base64-encoded .obj mesh file for Three.js rendering in frontend
    """

    allowed = {"image/jpeg", "image/png", "image/webp"}
    if person_image.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail="person_image must be a JPEG, PNG, or WebP image."
        )

    image_bytes = await person_image.read()

    try:
        mesh_b64 = generate_avatar(image_bytes, skin_tone=skin_tone)
    except RuntimeError as e:
        return AvatarResponse(
            success=False,
            error=str(e),
            user_message=(
                "The 3D avatar feature is currently unavailable. "
                "Your size and color recommendations are still ready. "
                "Please try the avatar again later."
            )
        )
    except ValueError as e:
        return AvatarResponse(
            success=False,
            error=str(e),
            user_message=(
                "We could not detect a full body in your photo for the 3D avatar. "
                "Please use a clear, full-body photo and try again."
            )
        )
    except Exception as e:
        return AvatarResponse(
            success=False,
            error=f"Unexpected error: {str(e)}",
            user_message="Avatar generation failed. Please try again."
        )
    finally:
        del image_bytes
        gc.collect()

    return AvatarResponse(
        success=True,
        mesh_base64=mesh_b64,
    )
