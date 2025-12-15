import runpod
import torch
import io
import base64
import soundfile as sf
from TTS.api import TTS
import os

# -------------------------
# Device setup
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# -------------------------
# Load XTTS v2 model ONCE
# -------------------------
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=False,
    gpu=DEVICE == "cuda",
)

# CUDA tuning (safe)
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# -------------------------
# Default speaker WAV (REQUIRED for XTTS)
# -------------------------
DEFAULT_SPEAKER_WAV = "/app/voices/default.wav"

if not os.path.exists(DEFAULT_SPEAKER_WAV):
    raise RuntimeError(f"Default speaker wav not found: {DEFAULT_SPEAKER_WAV}")

# -------------------------
# Warm up model (XTTS requires speaker_wav)
# -------------------------
_ = tts.tts(
    text="warmup",
    speaker_wav=DEFAULT_SPEAKER_WAV,
    language="en",
)

if DEVICE == "cuda":
    torch.cuda.synchronize()

print("XTTS warmup complete")

# -------------------------
# RunPod handler
# -------------------------
def handler(event):
    """
    Supported input:

    {
      "input": {
        "text": "Hello world",
        "language": "en",
        "speaker_wav": "/optional/path/to.wav"
      }
    }
    """

    input_data = event.get("input", {})

    text = input_data.get("text") or input_data.get("prompt")
    language = input_data.get("language", "en")
    speaker_wav = input_data.get("speaker_wav") or DEFAULT_SPEAKER_WAV

    if not text or not isinstance(text, str):
        return {"error": "Missing or invalid 'text' or 'prompt' field."}

    if not os.path.exists(speaker_wav):
        return {"error": f"Speaker wav not found: {speaker_wav}"}

    wav = tts.tts(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
    )

    buffer = io.BytesIO()
    sf.write(buffer, wav, samplerate=24000, format="WAV")

    return {
        "audio_base64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        "sample_rate": 24000,
        "format": "wav",
        "model": "xtts_v2",
        "speaker_wav": speaker_wav,
    }

# -------------------------
# Start RunPod serverless
# -------------------------
runpod.serverless.start({"handler": handler})
