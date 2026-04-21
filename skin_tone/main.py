# main.py
# The FastAPI application
# Defines the API endpoints and handles incoming requests

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from model_loader import load_model
from inference import predict

# ── Lifespan: load model once when server starts ──────────────────────────────
# This runs BEFORE the server accepts any requests
# Avoids reloading the model on every single request (would be very slow)
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting server — loading model...")
    ml_models["skintone"] = load_model()
    print("✅ Server ready")
    yield
    ml_models.clear()

# ── Create FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="Skin Tone Detection API",
    description="Detects Fitzpatrick skin tone from a face image",
    version="1.0.0",
    lifespan=lifespan
)

# ── CORS — allows frontend apps to call this API ──────────────────────────────
# Without this, browsers block requests from other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # change to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check endpoint ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Skin Tone Detection API is live"
    }

# ── Main prediction endpoint ──────────────────────────────────────────────────
@app.post("/predict")
async def predict_skin_tone(file: UploadFile = File(...)):
    """
    Accepts an image file and returns skin tone prediction.

    - Input  : image file (jpg, jpeg, png)
    - Output : predicted class, fitzpatrick type, confidence, hex colors
    """

    # ── Validate file type ────────────────────────────────────────────────────
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only jpg and png are accepted."
        )

    # ── Read image bytes ──────────────────────────────────────────────────────
    image_bytes = await file.read()

    # ── Run prediction ────────────────────────────────────────────────────────
    try:
        result = predict(image_bytes, ml_models["skintone"])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result