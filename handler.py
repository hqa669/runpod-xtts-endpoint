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
    Expected input:
    {
        "input": {
            "text": "Hello world"
        }
    }
    """
    text = event.get("input", {}).get("text", "Hello from FastPitch")

    # Generate waveform (numpy array)
    wav = tts.tts(text)

    # Encode WAV in memory
    buffer = io.BytesIO()
    sf.write(buffer, wav, samplerate=22050, format="WAV")

    return {
        "audio_base64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        "sample_rate": 22050,
        "model": "fast_pitch",
        "device": DEVICE
    }

# -------------------------
# IMPORTANT: block forever
# -------------------------
runpod.serverless.start({"handler": handler})
