"""
VDS V2 — Virtual Dressing System Backend
Single unified FastAPI application.
All features: measurements, skin tone, recommendation, avatar, tryon.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from schemas import MAX_REQUEST_BYTES

from measurement.router import router as measurement_router
from skin_tone.router  import router as skin_tone_router
from recommendation.router import router as recommendation_router
from avatar.router     import router as avatar_router
from tryon.router      import router as tryon_router

# ── Startup: load all heavy models once into RAM ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Measurement — YOLO loads on first request (lazy), no action needed here
    # Skin tone — load neural network
    from skin_tone.service import load_skin_tone_model
    load_skin_tone_model()

    # Avatar — load SAM 3D Body
    from avatar.service import load_avatar_model
    load_avatar_model()

    print("✅ All models loaded. VDS V2 backend ready.")
    yield
    print("VDS V2 backend shutting down.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VDS V2 API",
    description="Virtual Dressing System V2 — Full Pipeline Backend",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global request size guard (40 MB) ────────────────────────────────────────
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": "Total request body too large. Maximum allowed request size is 40 MB."},
        )
    return await call_next(request)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(measurement_router)
app.include_router(skin_tone_router)
app.include_router(recommendation_router)
app.include_router(avatar_router)
app.include_router(tryon_router)

# ── Home / Redirect ───────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def index():
    return {
        "message": "Welcome to VDS V2 API",
        "docs": "/docs",
        "health": "/health",
        "status": "Ready"
    }

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": "2.0.0"}
