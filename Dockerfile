FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Disable Streamlit CORS and XSRF to allow file uploads through Hugging Face's reverse proxy
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip "setuptools<70" wheel cython "numpy<2.0.0"
# Upgrade PyTorch to >= 2.4.0 to support torch.UInt32Storage (which SAM models require)
RUN pip install "torch>=2.4.0" "torchvision>=0.19.0" --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app

# IMPORTANT: Build from source using git because no pre-compiled Python 3.11 wheels exist for PyTorch 2.1
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
