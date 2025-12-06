import io
import base64
import runpod
import torch
import soundfile as sf

from TTS.api import TTS

# ------------------------
# Model Load (ONCE)
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=False,
    gpu=(DEVICE == "cuda")
)

# ------------------------
# Handler
# ------------------------
def handler(event):
    text = event.get("input", {}).get("text", "Hello, this is XTTS on RunPod")

    # Generate waveform
    wav = tts.tts(text=text)

    # Write WAV to memory
    buffer = io.BytesIO()
    sf.write(buffer, wav, samplerate=24000, format="WAV")

    audio_bytes = buffer.getvalue()

    return {
        "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
        "sample_rate": 24000
    }

runpod.serverless.start({"handler": handler})
