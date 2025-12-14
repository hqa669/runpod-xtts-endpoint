FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Accept Coqui TOS non-interactively
ENV COQUI_TOS_AGREED=1

WORKDIR /app
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    espeak-ng \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip && \
    pip install \
      torch==2.1.0 \
      torchvision==0.16.0 \
      torchaudio==2.1.0 \
      --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python - <<EOF
from TTS.api import TTS
TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("XTTS v2 model cached successfully")
EOF

COPY handler.py .

CMD ["python", "handler.py"]
