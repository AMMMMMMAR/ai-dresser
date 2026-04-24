#!/bin/bash
set -e

echo "========================================"
echo "  VDS V2 Backend Starting..."
echo "========================================"

# ── Step 1: Start Ollama in background ───────────────────────────────────────
echo "▶ Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# ── Step 2: Wait for Ollama to be ready ──────────────────────────────────────
echo "⏳ Waiting for Ollama to be ready..."
MAX_RETRIES=30
COUNT=0
until curl -s http://127.0.0.1:11434 > /dev/null 2>&1; do
    COUNT=$((COUNT + 1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Ollama failed to start after ${MAX_RETRIES} retries."
        echo "   Recommendation feature will be unavailable."
        echo "   All other features will still work."
        break
    fi
    echo "   Still waiting... (${COUNT}/${MAX_RETRIES})"
    sleep 2
done

if curl -s http://127.0.0.1:11434 > /dev/null 2>&1; then
    echo "✅ Ollama is running"

    # ── Step 3: Pull base LLM model ──────────────────────────────────────────
    echo "▶ Pulling llama3.2:3b base model (this may take a few minutes)..."
    ollama pull llama3.2:3b
    echo "✅ Base model ready"

    # ── Step 4: Build the custom smartfit model from Modelfile ───────────────
    if [ -f "/app/rec_llm/Modelfile" ]; then
        echo "▶ Building smartfit custom model..."
        ollama create smartfit -f /app/rec_llm/Modelfile
        echo "✅ smartfit model ready"
    else
        echo "⚠️  rec_llm/Modelfile not found — smartfit model not created."
        echo "   Recommendation feature may use base model instead."
    fi
else
    echo "⚠️  Ollama did not start. Continuing without recommendation feature."
fi

# ── Step 5: Start FastAPI backend ────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Starting FastAPI on port 7860..."
echo "========================================"
exec uvicorn main:app --host 0.0.0.0 --port 7860
