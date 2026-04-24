"""
Recommendation Router
INPUT:  RecommendationRequest JSON (measurements + skin_tone)
OUTPUT: RecommendationResponse JSON → frontend
"""

from fastapi import APIRouter, HTTPException
from schemas import RecommendationRequest, RecommendationResponse, RecommendationResult
from recommendation.service import get_recommendation

router = APIRouter(prefix="/recommendation", tags=["LLM Recommendation"])


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
    if "recommended_size" not in result or "recommended_colors" not in result:
        return RecommendationResponse(
            success=False,
            error="LLM response missing required fields.",
            user_message="The AI returned an incomplete recommendation. Please try again."
        )

    return RecommendationResponse(
        success=True,
        recommendation=RecommendationResult(
            recommended_size=result.get("recommended_size", "M"),
            recommended_colors=result.get("recommended_colors", []),
            size_reasoning=result.get("size_reasoning"),
            color_reasoning=result.get("color_reasoning"),
        )
    )
