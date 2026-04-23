from pydantic import BaseModel, Field
from typing import Optional, Dict

class MeasurementResult(BaseModel):
    shoulder_width_cm:      float
    chest_circumference_cm: float
    waist_circumference_cm: float
    hip_circumference_cm:   float
    arm_length_cm:          float
    inseam_cm:              float

class MeasurementResponse(BaseModel):
    measurements: Optional[MeasurementResult] = None
    diagnostics:  Dict[str, str] = {}
    error:        Optional[str] = None