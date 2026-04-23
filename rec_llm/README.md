# AI Dresser - Recommendation LLM Service

This microservice provides an AI-powered sizing and color recommendation engine for the AI Dresser project. It encapsulates a local Large Language Model (Llama 3.2 3B) within an Ollama environment, exposed through a FastAPI gateway.

## Features

- **Size Recommendations**: Calculates the optimal shirt size (XS to XXL) taking into account Chest, Waist, and Hip measurements.
- **Pants Size Calculation**: Dynamically computes pants width and length measurements (e.g., "32*34") from waist and inseam inputs.
- **Color Recommendations**: Recommends exactly three colors best suited to the user's skin tone.
- **FastAPI Gateway**: Provides a robust CORS-enabled HTTP endpoint.
- **Dockerized**: Fully containerized using Docker for seamless, single-container deployment.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running.

## Running the Service

The entire environment (Ollama + Model + FastAPI) is managed within a single Docker container.

1. **Build the Docker Image:**
   ```bash
   docker build -t ai-dresser-rec-llm .
   ```

2. **Run the Container:**
   ```bash
   docker run -d -p 8000:8000 --name rec-llm-service ai-dresser-rec-llm
   ```

   *Note: On the first run, the container will automatically download the Llama 3.2 3B model, which may take a few minutes.*

## API Endpoints

### `POST /api/recommend`

Generates sizing and color recommendations.

**Request Body (JSON):**
```json
{
  "chest_cm": 100.5,
  "waist_cm": 85.0,
  "hip_cm": 98.0,
  "inseam_cm": 82.0,
  "skin_tone": "Medium"
}
```

*Supported skin tones: Fair, Light, Medium, Tan, Coco, Deep.*

**Response (JSON):**
```json
{
  "status": "success",
  "shirt_size": "M",
  "pants_size": "33*32",
  "recommended_colors": ["Olive", "Plum", "Deep Teal"]
}
```

## Architecture Notes
- `Modelfile`: Contains the system instructions and rules provided to the local LLM.
- `main.py`: The FastAPI application bridging HTTP requests to the Ollama engine.
- `start.sh`: The entrypoint script that manages launching both Ollama natively and the FastAPI server.
