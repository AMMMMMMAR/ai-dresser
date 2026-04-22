import os
import base64
from google import genai
from google.genai.types import (
    Image,
    ProductImage,
    RecontextImageConfig,
    RecontextImageSource,
)

VIRTUAL_TRYON_MODEL = "virtual-try-on-001"

_client = None

def get_client() -> genai.Client:
    global _client
    if _client is None:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable not set.")
        _client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
    return _client


def run_virtual_tryon(
    person_image_bytes: bytes,
    garment_image_bytes: bytes,
    number_of_images: int = 1,
    output_mime_type: str = "image/jpeg",
    safety_filter_level: str = "BLOCK_LOW_AND_ABOVE",
) -> list[bytes]:
    """
    Calls the virtual try-on API with person and garment images.
    Returns a list of result images as raw bytes.
    """
    client = get_client()

    response = client.models.recontext_image(
        model=VIRTUAL_TRYON_MODEL,
        source=RecontextImageSource(
            person_image=Image(image_bytes=person_image_bytes),
            product_images=[
                ProductImage(
                    product_image=Image(image_bytes=garment_image_bytes)
                )
            ],
        ),
        config=RecontextImageConfig(
            output_mime_type=output_mime_type,
            number_of_images=number_of_images,
            safety_filter_level=safety_filter_level,
        ),
    )

    return [img.image.image_bytes for img in response.generated_images]