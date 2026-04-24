"""
Recommendation Service
Sends measurements + skin tone to local Ollama LLM.
Returns recommended size and colors as structured JSON.
"""

import os
import json
import requests

OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "smartfit")

SYSTEM_PROMPT = """
You are a professional fashion sizing and color expert.
Given body measurements and a skin tone, you return ONLY a valid JSON object — no extra text.

The JSON must have exactly these keys:
{
  "recommended_size": "S" | "M" | "L" | "XL" | "XXL",
  "recommended_colors": ["color1", "color2", "color3"],
  "size_reasoning": "one sentence explanation",
  "color_reasoning": "one sentence explanation"
}
"""


def get_recommendation(
    chest_cm:  float,
    waist_cm:  float,
    hip_cm:    float,
    inseam_cm: float,
    skin_tone: str,
) -> dict:
    """
    Calls local Ollama LLM and returns a structured recommendation dict.
    Raises RuntimeError if Ollama is unavailable.
    Raises ValueError if LLM response is not valid JSON.
    """

    prompt = (
        f"Body measurements — "
        f"Chest: {chest_cm}cm, Waist: {waist_cm}cm, "
        f"Hip: {hip_cm}cm, Inseam: {inseam_cm}cm. "
        f"Skin tone: {skin_tone}. "
        f"Return only the JSON object."
    )

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to the recommendation service (Ollama). "
            "Make sure Ollama is running."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Recommendation service timed out. Please try again."
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Recommendation service error: {str(e)}")

    data         = response.json()
    raw_response = data.get("response", "")

    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError:
        raise ValueError(
            f"LLM returned invalid JSON. Raw response: {raw_response[:200]}"
        )

    return result
