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
    # ----- RECOMMENDED -----
    "Soft Navy": "#2C3E50",
    "Dusty Rose": "#D8A7B1",
    "Cool Gray": "#BDC3C7",
    "Lavender": "#C8A2C8",

    "Teal": "#1ABC9C",
    "Soft Blue": "#5DADE2",
    "Mauve": "#AF7AC5",
    "Charcoal Blue": "#34495E",

    "Olive": "#808000",
    "Coral": "#FF7F50",
    "Cream": "#FFFDD0",
    "Turquoise": "#40E0D0",

    "Emerald": "#2ECC71",
    "Mustard": "#D4AC0D",
    "Burgundy": "#800020",

    "Burnt Orange": "#CC5500",
    "Deep Purple": "#6A0DAD",
    "Gold": "#FFD700",
    "Forest Green": "#228B22",

    "Royal Blue": "#4169E1",
    "Crimson": "#DC143C",
    "White": "#FFFFFF",
    "Fuchsia": "#FF00FF",

    # ----- AVOID -----
    "Pure Black": "#000000",
    "Neon Green": "#39FF14",
    "Bright Orange": "#FF5F1F",
    "Warm Beige": "#F5DEB3",

    "Neon Yellow": "#FFFF33",
    "Lime Green": "#32CD32",
    "Bright Red": "#FF0000",

    "Icy Blue": "#E0FFFF",
    "Cool Purple": "#800080",
    "Silver": "#C0C0C0",
    "Neon Pink": "#FF1493",

    "Ash Gray": "#B2BEB5",
    "Pale Yellow": "#FFFFE0",
    "Mint Green": "#98FF98",
    "Light Blue": "#ADD8E6",

    "Dusty Beige": "#D7CCC8",
    "Dark Gray": "#A9A9A9",
    "Pastel Pink": "#FFD1DC",
    "Pale Blue": "#B0E0E6",

    "Mud Brown": "#5C4033",
    "Olive Drab": "#6B8E23",
    "Muted Gray": "#808080",
    "Light Beige": "#F5F5DC"
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
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
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
