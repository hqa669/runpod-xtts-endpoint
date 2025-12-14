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
# Load XTTS v2 model ONCE
# -------------------------
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

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

    # ✅ XTTS v2 requires language
    wav = tts.tts(
        text=text,
        language="en"
    )

    # ✅ Encode WAV to Base64 in memory (XTTS = 24kHz)
    buffer = io.BytesIO()
    sf.write(buffer, wav, samplerate=24000, format="WAV")

    return {
        "audio_base64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        "sample_rate": 24000,
        "format": "wav",
        "model": "xtts_v2"
    }

# -------------------------
# IMPORTANT: block forever
# -------------------------
runpod.serverless.start({"handler": handler})
