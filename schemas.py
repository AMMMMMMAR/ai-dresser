"""
Shared Pydantic schemas used across all VDS V2 features.
Import from here to avoid duplication.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List


# ── Measurements ──────────────────────────────────────────────────────────────

class MeasurementResult(BaseModel):
    shoulder_width_cm:      float = Field(description="Shoulder width in cm")
    chest_circumference_cm: float = Field(description="Chest circumference in cm")
    waist_circumference_cm: float = Field(description="Waist circumference in cm")
    hip_circumference_cm:   float = Field(description="Hip circumference in cm")
    arm_length_cm:          float = Field(description="Arm length in cm")
    inseam_cm:              float = Field(description="Inseam length in cm")

class MeasurementResponse(BaseModel):
    success:      bool                         = True
    measurements: Optional[MeasurementResult] = None
    diagnostics:  Dict[str, str]               = {}
    error:        Optional[str]                = None
    user_message: Optional[str]                = None


# ── Skin Tone ─────────────────────────────────────────────────────────────────

class SkinToneResult(BaseModel):
    skin_tone:     str = Field(description="Detected skin tone class e.g. fair, medium, cocoa")
    reference_hex: str = Field(description="Reference hex color for the Fitzpatrick type")
    sampled_hex:   str = Field(description="Average hex color sampled from the image center")

class SkinToneResponse(BaseModel):
    success:      bool                      = True
    skin_tone:    Optional[SkinToneResult]  = None
    error:        Optional[str]             = None
    user_message: Optional[str]             = None


# ── Recommendation ────────────────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    chest_cm:   float = Field(description="Chest circumference in cm")
    waist_cm:   float = Field(description="Waist circumference in cm")
    hip_cm:     float = Field(description="Hip circumference in cm")
    inseam_cm:  float = Field(description="Inseam length in cm")
    skin_tone:  str   = Field(description="Skin tone class from skin tone feature")

class ColorResult(BaseModel):
    name: str = Field(description="Name of the color")
    hex: str  = Field(description="Hex code of the color")

class RecommendationResult(BaseModel):
    shirt_size:         str        = Field(description="Recommended shirt size")
    pants_size:         str        = Field(description="Recommended pants size (W*L)")
    recommended_colors: List[ColorResult]  = Field(description="List of recommended clothing colors")
    avoid_colors:       List[ColorResult]  = Field(description="List of colors to avoid")

class RecommendationResponse(BaseModel):
    success:        bool                           = True
    recommendation: Optional[RecommendationResult] = None
    error:          Optional[str]                  = None
    user_message:   Optional[str]                  = None


# ── Avatar ────────────────────────────────────────────────────────────────────

class AvatarResponse(BaseModel):
    success:      bool            = True
    mesh_base64:  Optional[str]   = Field(None, description="Base64-encoded .obj mesh file")
    error:        Optional[str]   = None
    user_message: Optional[str]   = None


# ── Virtual Try-On ────────────────────────────────────────────────────────────

class TryOnResponse(BaseModel):
    success:       bool       = True
    images_base64: List[str]  = Field(default=[], description="Base64-encoded result images")
    count:         int        = Field(default=0, description="Number of images returned")
    mime_type:     str        = Field(default="image/jpeg")
    error:         Optional[str]  = None
    user_message:  Optional[str]  = None
