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

# Optional FP16 for speed (safe on modern GPUs)
if DEVICE == "cuda":
    tts.tts_model = tts.tts_model.half()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# -------------------------
# Resolve default speaker (built-in)
# -------------------------
speaker_manager = tts.synthesizer.tts_model.speaker_manager
speaker_keys = list(speaker_manager.name_to_id.keys())
DEFAULT_SPEAKER = speaker_keys[0] if speaker_keys else None

print("Resolved default speaker:", DEFAULT_SPEAKER)

# Warm up model once (important for serverless latency)
if DEFAULT_SPEAKER:
    _ = tts.tts(
        text="warmup",
        speaker=DEFAULT_SPEAKER,
        language="en"
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()

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
    speaker_wav = input_data.get("speaker_wav")  # optional

    if not text or not isinstance(text, str):
        return {
            "error": "Missing or invalid 'text' or 'prompt' field."
        }

    # -------------------------
    # Speaker selection logic
    # Priority:
    #   1) speaker_wav (voice cloning)
    #   2) built-in default speaker
    # -------------------------
    if speaker_wav:
        wav = tts.tts(
            text=text,
            language=language,
            speaker_wav=speaker_wav
        )
        speaker_used = "speaker_wav"
    else:
        wav = tts.tts(
            text=text,
            language=language,
            speaker=DEFAULT_SPEAKER
        )
        speaker_used = DEFAULT_SPEAKER

    # -------------------------
    # Encode WAV → Base64
    # -------------------------
    buffer = io.BytesIO()
    sf.write(buffer, wav, samplerate=24000, format="WAV")

    return {
        "audio_base64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        "sample_rate": 24000,
        "format": "wav",
        "model": "xtts_v2",
        "speaker": speaker_used
    }

# -------------------------
# IMPORTANT: block forever
# -------------------------
runpod.serverless.start({"handler": handler})
