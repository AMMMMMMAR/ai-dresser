"""
Main FastAPI server for the SmartFit AI Engine.
Acts as a gateway between external requests and the local Ollama LLM service.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json

# Initialize FastAPI application
app = FastAPI(title="SmartFit AI Engine API")

# Configure CORS (Cross-Origin Resource Sharing)
# This allows frontends like Netlify to make requests to this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, restrict this to your Netlify URL in production.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Ollama local endpoint
# The start.sh script runs Ollama on localhost port 11434 inside the same container.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")

# Define the expected JSON payload from the user
class MeasurementRequest(BaseModel):
    chest_cm: float
    waist_cm: float
    hip_cm: float
    inseam_cm: float
    skin_tone: str

# Health check endpoint for Hugging Face Spaces
@app.get("/health")
def health_check():
    """Simple endpoint to verify the API is running."""
    return {"status": "ok"}

# Main recommendation endpoint
@app.post("/recommend")
def get_recommendation(request: MeasurementRequest):
    """
    Takes user measurements, formats them into a prompt, 
    sends it to the local Ollama Llama 3.2 3B model, and returns the response.
    """
    # Construct the prompt for the Ollama model
    prompt = (
        f"Input measurements: Chest: {request.chest_cm}cm, "
        f"Waist: {request.waist_cm}cm, Hip: {request.hip_cm}cm, "
        f"Inseam: {request.inseam_cm}cm, Skin Tone: {request.skin_tone}"
    )

    # Prepare payload for Ollama
    payload = {
        "model": "smartfit", # The custom model built in start.sh
        "prompt": prompt,
        "stream": False,     # We want the complete JSON response, not a stream
        "format": "json"     # Enforce JSON formatting from the LLM
    }

    try:
        # Forward request to local Ollama service
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        # Parse output from Ollama
        data = response.json()
        raw_response = data.get("response", "")
        
        # We assume the response is valid JSON because of the system prompt and format flag
        result = json.loads(raw_response)
        return result
        
    except requests.exceptions.RequestException as e:
        # Handle connection errors with Ollama
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}")
    except json.JSONDecodeError:
        # Handle cases where LLM failed to output valid JSON
        raise HTTPException(status_code=500, detail="LLM response was not valid JSON")

if __name__ == "__main__":
    # Provides ability to run the file directly via python main.py for debugging
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
