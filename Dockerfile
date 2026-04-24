FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/avatar

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    zstd \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Install Ollama binary ─────────────────────────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Upgrade pip + fix setuptools for Cython extensions ───────────────────────
RUN pip install --upgrade pip "setuptools<70" wheel cython "numpy<2.0.0"

# ── Upgrade PyTorch >= 2.4.0 (SAM requires UInt32Storage) ────────────────────
RUN pip install "torch>=2.4.0" "torchvision>=0.19.0" --index-url https://download.pytorch.org/whl/cu121

# ── Install detectron2 from source (required by SAM internals) ────────────────
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY . .

# ── Make startup script executable ───────────────────────────────────────────
RUN chmod +x start.sh

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["./start.sh"]
