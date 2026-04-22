#!/bin/bash
# start.sh - Orchestrates exactly what Hugging Face Spaces needs
# It starts Ollama in the background, pulls/builds the model, and then starts FastAPI.

echo "Starting Ollama server in background..."
ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama API to be ready..."
# Keep checking if we get a response from Ollama
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    echo "Waiting for Ollama to wake up..."
    sleep 2
done

echo "Ollama is ready. Pulling base model llama3.2:3b..."
ollama pull llama3.2:3b

echo "Building custom SmartFit model from Modelfile..."
ollama create smartfit -f Modelfile

echo "Setup complete. Local SmartFit model is ready."

echo "Starting FastAPI server on port 7860 (Hugging Face standard port)..."
# We run uvicorn in the foreground so the container stays alive.
# Host 0.0.0.0 and port 7860 are required by Hugging Face Spaces.
uvicorn main:app --host 0.0.0.0 --port 7860

# Wait for background process to keep script tied to it
wait $OLLAMA_PID
