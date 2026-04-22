from pydantic import BaseModel, Field
from typing import List, Optional


class TryOnResponse(BaseModel):
    images_base64: List[str] = Field(
        description="List of base64-encoded result images"
    )
    count: int = Field(description="Number of images returned")
    mime_type: str = Field(description="MIME type of the result images")


class TryOnErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None