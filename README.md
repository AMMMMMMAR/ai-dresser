---
title: VDS V2 Backend
emoji: 👗
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# VDS V2 — Virtual Dressing System Backend

FastAPI backend for the Virtual Dressing System. Deployed on Hugging Face Spaces via Docker.

## Features

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/measurements/extract` | POST | Body measurements from front + side photos (YOLO pose) |
| `/skin-tone/detect` | POST | Skin tone detection from face selfie |
| `/recommendation/get` | POST | Size + color recommendation via LLM (Ollama) |
| `/avatar/generate` | POST | 3D body avatar as base64-encoded `.obj` (SAM 3D Body) |
| `/tryon/generate` | POST | Virtual try-on via Vertex AI (person + garment image) |

Full API docs available at `/docs` when the Space is running.

## Input / Output Summary

### `/measurements/extract`
- **Input:** `front_image` (file), `side_image` (file), `height_cm` (float)
- **Output:** `shoulder_width_cm`, `chest_circumference_cm`, `waist_circumference_cm`, `hip_circumference_cm`, `arm_length_cm`, `inseam_cm`

### `/skin-tone/detect`
- **Input:** `person_image` (file — face selfie)
- **Output:** `skin_tone` (label), `reference_hex`, `sampled_hex`

### `/recommendation/get`
- **Input:** JSON — `chest_cm`, `waist_cm`, `hip_cm`, `inseam_cm`, `skin_tone`
- **Output:** `shirt_size`, `pants_size`, `recommended_colors [{name, hex}]`, `avoid_colors [{name, hex}]`

### `/avatar/generate`
- **Input:** `person_image` (file — front photo)
- **Output:** `mesh_base64` (base64-encoded `.obj` mesh)

### `/tryon/generate`
- **Input:** `person_image` (file), `garment_image` (file), `number_of_images` (int, default 1)
- **Output:** `images_base64` (array of base64-encoded result images)

## Environment Variables

Set these in HF Spaces → Settings → Repository secrets:

| Variable | Description |
|---|---|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID for Vertex AI try-on |
| `GOOGLE_CLOUD_LOCATION` | GCP region (default: `us-central1`) |
| `HF_TOKEN` | HuggingFace token for SAM 3D Body model download |
| `OLLAMA_URL` | Ollama endpoint (default: `http://127.0.0.1:11434/api/generate`) |
| `OLLAMA_MODEL` | Ollama model name (default: `smartfit`) |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins — **required for frontend connection** |

### ALLOWED_ORIGINS examples

```
# Local development only
ALLOWED_ORIGINS=http://localhost:5174

# Local + Netlify production
ALLOWED_ORIGINS=http://localhost:5174,https://your-app.netlify.app
```

## Failure Handling

Every endpoint returns a `success` boolean. On failure, `user_message` contains a human-readable explanation. The frontend handles failures per-feature — critical features (measurements, skin tone, recommendation) trigger a retry prompt; the avatar is non-critical.

## Structure

```
ai-dresser/
├── main.py              FastAPI app + CORS + router registration
├── schemas.py           Shared Pydantic models
├── measurement/         YOLO pose-based body measurement pipeline
├── skin_tone/           Neural network skin tone classifier
├── recommendation/      Ollama LLM sizing + color recommendation
├── avatar/              SAM 3D Body mesh generation
└── tryon/               Vertex AI virtual try-on
```
