import runpod
import torch
import io
import base64
import soundfile as sf
from TTS.api import TTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

tts = TTS("tts_models/en/ljspeech/fast_pitch").to(DEVICE)

def handler(event):
    text = event.get("input", {}).get("text", "Hello from FastPitch")
    wav = tts.tts(text)
    buffer = io.BytesIO()
    sf.write(buffer, wav, samplerate=22050, format="WAV")
    return {
        "audio_base64": base64.b64encode(buffer.getvalue()).decode()
    }

runpod.serverless.start({"handler": handler})
