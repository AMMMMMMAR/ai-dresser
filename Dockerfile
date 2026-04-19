FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Fix potential torch/detectron2 issues, install cython early for xtcocotools
RUN pip install --upgrade pip setuptools wheel cython numpy

WORKDIR /app

# Install detectron2 first from source (requires devel image and ninja)
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port for HuggingFace / FastApi
EXPOSE 7860

# CMD config for streamlit testing default
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
