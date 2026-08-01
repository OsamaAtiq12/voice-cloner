"""OmniVoice model loader and generation helpers."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "OmniVoice"
ASR_DIR = ROOT / "models" / "whisper-large-v3-turbo"
VOICES_DIR = ROOT / "voices"
OUTPUTS_DIR = ROOT / "outputs"
UPLOADS_DIR = ROOT / "uploads"

SAMPLE_RATE = 24000

_model = None
_lock = threading.Lock()
_load_error: Optional[str] = None
_status = "idle"


def ensure_dirs() -> None:
    for path in (VOICES_DIR, OUTPUTS_DIR, UPLOADS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def status() -> dict[str, Any]:
    device = get_device()
    gpu_name = None
    vram = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram = round(props.total_memory / (1024**3), 1)
    return {
        "ready": _model is not None,
        "status": _status,
        "error": _load_error,
        "device": device,
        "gpu": gpu_name,
        "vram_gb": vram,
        "model_path": str(MODEL_DIR),
    }


def load_model() -> None:
    global _model, _load_error, _status
    if _model is not None:
        return

    with _lock:
        if _model is not None:
            return
        _status = "loading"
        _load_error = None
        try:
            from omnivoice import OmniVoice

            device = get_device()
            # Prefer local weights; fall back to Hugging Face ids for fresh installs.
            model_id = str(MODEL_DIR) if MODEL_DIR.exists() else "k2-fsa/OmniVoice"
            asr_kwargs: dict[str, Any] = {"load_asr": True}
            if ASR_DIR.exists():
                asr_kwargs["asr_model_name"] = str(ASR_DIR)
            else:
                asr_kwargs["asr_model_name"] = "openai/whisper-large-v3-turbo"
            logger.info("Loading OmniVoice from %s on %s", model_id, device)
            _model = OmniVoice.from_pretrained(
                model_id,
                device_map=device,
                dtype=torch.float16 if device != "cpu" else torch.float32,
                **asr_kwargs,
            )
            _status = "ready"
            logger.info("OmniVoice ready")
        except Exception as exc:
            _status = "error"
            _load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to load OmniVoice")
            raise


def get_model():
    if _model is None:
        load_model()
    return _model


def languages() -> list[str]:
    try:
        from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

        return ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)
    except Exception:
        return ["Auto"]


def build_instruct(
    gender: str | None = None,
    age: str | None = None,
    pitch: str | None = None,
    style: str | None = None,
    accent: str | None = None,
    dialect: str | None = None,
    custom: str | None = None,
) -> str | None:
    parts: list[str] = []
    for value in (gender, age, pitch, style, accent, dialect):
        if value and value.strip() and value.strip().lower() != "auto":
            parts.append(value.strip())
    if custom and custom.strip():
        parts.append(custom.strip())
    return ", ".join(parts) if parts else None


def generate(
    text: str,
    *,
    mode: str = "clone",
    ref_audio: str | None = None,
    voice_prompt_path: str | None = None,
    instruct: str | None = None,
    language: str | None = None,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    denoise: bool = True,
    t_shift: float = 0.1,
    position_temperature: float = 5.0,
    class_temperature: float = 0.0,
    layer_penalty_factor: float = 5.0,
    speed: float | None = 1.0,
    duration: float | None = None,
    preprocess_prompt: bool = True,
    postprocess_output: bool = True,
    pad_duration: float = 0.1,
    fade_duration: float = 0.1,
    audio_chunk_duration: float = 15.0,
    audio_chunk_threshold: float = 30.0,
    normalize_text: bool = False,
) -> tuple[np.ndarray, int]:
    from omnivoice import OmniVoiceGenerationConfig, VoiceClonePrompt

    model = get_model()
    if not text or not text.strip():
        raise ValueError("Please enter text to speak.")

    config = OmniVoiceGenerationConfig(
        num_step=int(num_step),
        guidance_scale=float(guidance_scale),
        denoise=bool(denoise),
        t_shift=float(t_shift),
        position_temperature=float(position_temperature),
        class_temperature=float(class_temperature),
        layer_penalty_factor=float(layer_penalty_factor),
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
        pad_duration=float(pad_duration),
        fade_duration=float(fade_duration),
        audio_chunk_duration=float(audio_chunk_duration),
        audio_chunk_threshold=float(audio_chunk_threshold),
    )

    lang = language if language and language != "Auto" else None
    kw: dict[str, Any] = {
        "text": text.strip(),
        "language": lang,
        "generation_config": config,
    }

    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)
    elif speed is not None:
        kw["speed"] = float(speed)

    if normalize_text:
        kw["normalize_text"] = True

    if mode == "clone":
        if voice_prompt_path:
            prompt = VoiceClonePrompt.load(voice_prompt_path)
            kw["voice_clone_prompt"] = prompt
        else:
            if not ref_audio:
                raise ValueError("Upload a reference audio clip to clone a voice.")
            # Omit transcript so OmniVoice reads the clip automatically (audio-only flow).
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=None,
            )
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()
    elif mode == "design":
        if not instruct or not instruct.strip():
            raise ValueError("Choose at least one voice attribute for Voice Design.")
        kw["instruct"] = instruct.strip()
    elif mode == "auto":
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    with _lock:
        audio = model.generate(**kw)

    waveform = np.asarray(audio[0], dtype=np.float32)
    return waveform, getattr(model, "sampling_rate", SAMPLE_RATE)


def save_voice_prompt(name: str, ref_audio: str) -> Path:
    ensure_dirs()
    safe = "".join(c for c in name.strip() if c.isalnum() or c in ("-", "_", " ")).strip()
    if not safe:
        raise ValueError("Enter a valid voice name.")
    out = VOICES_DIR / f"{safe}.pt"
    model = get_model()
    prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=None,
    )
    prompt.save(str(out))
    return out


def list_saved_voices() -> list[dict[str, str]]:
    ensure_dirs()
    voices = []
    for path in sorted(VOICES_DIR.glob("*.pt")):
        voices.append({"id": path.stem, "name": path.stem, "path": str(path)})
    return voices
