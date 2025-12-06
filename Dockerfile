FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# -------------------------
# OS dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# -------------------------
# Install CUDA-enabled PyTorch (ONCE)
# -------------------------
RUN pip install --upgrade pip && \
    pip install \
      torch==2.1.0 \
      torchvision==0.16.0 \
      torchaudio==2.1.0 \
      --index-url https://download.pytorch.org/whl/cu118

# -------------------------
# Install remaining Python deps
# -------------------------
COPY requirements.txt .
RUN pip install -r requirements.txt

# -------------------------
# Preload FastPitch model
# (prevents serverless cold-start crash)
# -------------------------
RUN python - << 'EOF'
from TTS.api import TTS
TTS("tts_models/en/ljspeech/fast_pitch")
print("FastPitch model cached successfully")
EOF

# -------------------------
# Copy handler
# -------------------------
COPY handler.py .

# -------------------------
# Start RunPod serverless
# -------------------------
CMD ["python", "handler.py"]
