---
title: VDS V2 Backend
emoji: 👗
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# VDS V2 — Virtual Dressing System Backend

FastAPI backend for the Virtual Dressing System V2.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/measurements/extract` | Extract body measurements from two images |
| POST | `/skin-tone/detect` | Detect skin tone from one image |
| POST | `/recommendation/get` | Get size + color recommendation from LLM |
| POST | `/avatar/generate` | Generate 3D body avatar |
| POST | `/tryon/generate` | Virtual try-on |

## Environment Variables (set in HF Spaces secrets)

| Variable | Description |
|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID for virtual try-on |
| `GOOGLE_CLOUD_LOCATION` | GCP region (default: us-central1) |
| `HF_TOKEN` | HuggingFace token for SAM 3D Body model |
| `OLLAMA_URL` | Ollama endpoint (default: http://127.0.0.1:11434/api/generate) |
| `ALLOWED_ORIGINS` | Comma-separated allowed CORS origins e.g. https://your-app.netlify.app |
