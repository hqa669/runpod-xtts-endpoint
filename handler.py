import runpod
import torch
import io
import base64
import soundfile as sf
from TTS.api import TTS

# -------------------------
# Device setup
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# -------------------------
# Load FastPitch model ONCE
# -------------------------
tts = TTS("tts_models/en/ljspeech/fast_pitch").to(DEVICE)

# -------------------------
# RunPod handler
# -------------------------
def handler(event):
    """
    Supported request formats:

    {
      "input": {
        "text": "Hello world"
      }
    }

    OR

    {
      "input": {
        "prompt": "Hello world"
      }
    }
    """

    input_data = event.get("input", {})

    # ✅ Accept BOTH `text` and `prompt`
    text = input_data.get("text") or input_data.get("prompt")

    if not text or not isinstance(text, str):
        return {
            "error": "Missing or invalid 'text' or 'prompt' field in input."
        }

    # ✅ Generate speech
    wav = tts.tts(text)

    # ✅ Encode WAV to Base64 in memory
    buffer = io.BytesIO()
    sf.write(buffer, wav, samplerate=22050, format="WAV")

    return {
        "audio_base64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        "sample_rate": 22050,
        "format": "wav",
        "model": "fast_pitch"
    }

# -------------------------
# IMPORTANT: block forever
# -------------------------
runpod.serverless.start({"handler": handler})
