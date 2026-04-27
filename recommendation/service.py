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


COLOR_HEX_MAP = {
    "Emerald Green": "#50C878", "Royal Blue": "#4169E1", "Ruby Red": "#9B111E", 
    "Silver": "#C0C0C0", "Crisp White": "#FFFFFF", "Navy": "#000080", 
    "Hunter Green": "#355E3B", "Burgundy": "#800020", "Dusty Rose": "#DCAE96", 
    "Lavender": "#E6E6FA", "Olive": "#808000", "Plum": "#8E4585", 
    "Deep Teal": "#003636", "Metallic Bronze": "#A57164", "Forest Green": "#228B22", 
    "White": "#FFFFFF", "Mustard": "#FFDB58", "Earthy Brown": "#8B4513", 
    "Terracotta": "#E2725B", "Coral": "#FF7F50", "Cobalt Blue": "#0047AB", 
    "Bright Orange": "#FFAC1C", "Turquoise": "#40E0D0", "Vibrant Yellow": "#FFDA03", 
    "Magenta": "#FF00FF", "Gold": "#FFD700", "Pastel Pink": "#FFD1DC", 
    "Sky Blue": "#87CEEB", "Mint Green": "#98FF98", "Royal Purple": "#7851A9",
    "Off-White": "#FAF9F6", "Beige": "#F5F5DC", "Camel": "#C19A6B", 
    "Peachy Pink": "#FFDAB9", "Soft Yellow": "#FDFD96", "Taupe": "#483C32", 
    "Light Coral": "#F08080", "Pastel Yellow": "#FDFD96", "Pale Green": "#98FB98", 
    "Caramel": "#FFD59A", "Light Gray": "#D3D3D3", "Warm Orange": "#FF8C00", 
    "Rust": "#B7410E", "Pale Blue": "#AFEEEE", "Neon Pink": "#FF10F0", 
    "Dark Orange": "#FF8C00", "Deep Red": "#850101", "Dark Brown": "#654321", 
    "Olive Green": "#BAB86C", "Deep Orange": "#FF8C00", "Warm Red": "#FF4500", 
    "Tan": "#D2B48C", "Charcoal": "#36454F", "Navy Blue": "#000080", 
    "Bright Yellow": "#FFEA00", "Black": "#000000"
}


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
