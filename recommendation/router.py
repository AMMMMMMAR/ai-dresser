"""
Recommendation Router
INPUT:  RecommendationRequest JSON (measurements + skin_tone)
OUTPUT: RecommendationResponse JSON → frontend
"""

from fastapi import APIRouter, HTTPException
from schemas import RecommendationRequest, RecommendationResponse, RecommendationResult, ColorResult
from recommendation.service import get_recommendation, COLOR_HEX_MAP

router = APIRouter(prefix="/recommendation", tags=["LLM Recommendation"])


def map_colors_to_hex(color_list):
    """Helper to convert list of color strings to list of ColorResult objects."""
    results = []
    for color_name in color_list:
        hex_code = COLOR_HEX_MAP.get(color_name, "#888888") # Default gray if not found
        results.append(ColorResult(name=color_name, hex=hex_code))
    return results


@router.post("/get", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    """
    Takes body measurements and skin tone from the previous pipeline steps.
    Returns recommended clothing size and color palette.

    - Input:  JSON body with chest_cm, waist_cm, hip_cm, inseam_cm, skin_tone
    - Output: recommended_size, recommended_colors, reasoning
    """

    try:
        result = get_recommendation(
            chest_cm=request.chest_cm,
            waist_cm=request.waist_cm,
            hip_cm=request.hip_cm,
            inseam_cm=request.inseam_cm,
            skin_tone=request.skin_tone,
        )
    except RuntimeError as e:
        return RecommendationResponse(
            success=False,
            error=str(e),
            user_message=(
                "The recommendation service is temporarily unavailable. "
                "Your measurements were recorded successfully. "
                "Please try the recommendation again in a moment."
            )
        )
    except ValueError as e:
        return RecommendationResponse(
            success=False,
            error=str(e),
            user_message=(
                "The AI model returned an unexpected response. Please try again."
            )
        )
    except Exception as e:
        return RecommendationResponse(
            success=False,
            error=f"Unexpected error: {str(e)}",
            user_message="Recommendation failed unexpectedly. Please try again."
        )

    # Validate required keys in LLM response
    required_keys = ["shirt_size", "pants_size", "recommended_colors", "avoid_colors"]
    if not all(k in result for k in required_keys):
        return RecommendationResponse(
            success=False,
            error="LLM response missing required fields.",
            user_message="The AI returned an incomplete recommendation. Please try again."
        )

    return RecommendationResponse(
        success=True,
        recommendation=RecommendationResult(
            shirt_size=result.get("shirt_size", "Unknown"),
            pants_size=result.get("pants_size", "Unknown"),
            recommended_colors=map_colors_to_hex(result.get("recommended_colors", [])),
            avoid_colors=map_colors_to_hex(result.get("avoid_colors", []))
        )
    )
