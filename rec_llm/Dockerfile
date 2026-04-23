# Use an official Python runtime as a secure base image
FROM python:3.10-slim

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
# Set root directory for Hugging Face space
WORKDIR /app

# Install system dependencies: curl for checking API, and Ollama installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Note: In Hugging Face Spaces, the container runs as a non-root user (user name 'user', ID 1000).
# We must create this user and give it ownership.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Change ownership of app directory before switching users
RUN chown -R user:user /app

# Switch to the non-root user
USER user

# Copy Python dependencies
COPY --chown=user:user requirements.txt .

# Install Python packages (FastAPI, Uvicorn, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user:user main.py .
COPY --chown=user:user Modelfile .
COPY --chown=user:user start.sh .

# Ensure the bash script is executable
RUN chmod +x start.sh

# Expose port 7860 to route requests from Hugging Face outer proxy
EXPOSE 7860

# Specify the launch sequence
CMD ["./start.sh"]
