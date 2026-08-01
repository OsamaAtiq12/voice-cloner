"""Voice Cloner — FastAPI server powered by local OmniVoice."""

from __future__ import annotations

import utils.clients
import utils.http_session
import utils.transport
import utils.session
import io
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("voice_cloner")

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.ensure_dirs()
    try:
        engine.load_model()
    except Exception:
        logger.warning("Model not loaded at startup — will retry on first request")
    yield


app = FastAPI(title="Voice Cloner", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/how-it-works", response_class=HTMLResponse)
async def how_it_works():
    html = (STATIC / "how-it-works.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/health")
async def health():
    return engine.status()


@app.post("/api/load")
async def load_model():
    try:
        engine.load_model()
        return engine.status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/languages")
async def languages():
    return {"languages": engine.languages()}


@app.get("/api/voices")
async def voices():
    return {"voices": engine.list_saved_voices()}


@app.get("/api/options")
async def options():
    return {
        "modes": [
            {
                "id": "clone",
                "label": "Voice Clone",
                "description": "Clone a real voice from a short reference clip (3–10 seconds).",
            },
            {
                "id": "design",
                "label": "Voice Design",
                "description": "Create a voice from attributes — no audio needed.",
            },
            {
                "id": "auto",
                "label": "Auto Voice",
                "description": "Let OmniVoice pick a natural voice automatically.",
            },
        ],
        "gender": ["Auto", "male", "female"],
        "age": [
            "Auto",
            "child",
            "teenager",
            "young adult",
            "middle-aged",
            "elderly",
        ],
        "pitch": [
            "Auto",
            "very low pitch",
            "low pitch",
            "moderate pitch",
            "high pitch",
            "very high pitch",
        ],
        "style": ["Auto", "whisper"],
        "accent": [
            "Auto",
            "american accent",
            "british accent",
            "australian accent",
            "canadian accent",
            "indian accent",
            "chinese accent",
            "korean accent",
            "japanese accent",
            "portuguese accent",
            "russian accent",
        ],
        "dialect": [
            "Auto",
            "河南话",
            "陕西话",
            "四川话",
            "贵州话",
            "云南话",
            "桂林话",
            "济南话",
            "石家庄话",
            "甘肃话",
            "宁夏话",
            "青岛话",
            "东北话",
        ],
        "defaults": {
            "num_step": 32,
            "guidance_scale": 2.0,
            "denoise": True,
            "t_shift": 0.1,
            "position_temperature": 5.0,
            "class_temperature": 0.0,
            "layer_penalty_factor": 5.0,
            "speed": 1.0,
            "preprocess_prompt": True,
            "postprocess_output": True,
            "pad_duration": 0.1,
            "fade_duration": 0.1,
            "audio_chunk_duration": 15.0,
            "audio_chunk_threshold": 30.0,
            "normalize_text": False,
        },
    }


def _form_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _form_float(value: Optional[str], default: float | None = None) -> float | None:
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _form_int(value: Optional[str], default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(float(value))


async def _save_upload(upload: UploadFile | None, prefix: str) -> str | None:
    if upload is None or not upload.filename:
        return None
    engine.ensure_dirs()
    suffix = Path(upload.filename).suffix or ".wav"
    path = engine.UPLOADS_DIR / f"{prefix}_{uuid.uuid4().hex}{suffix}"
    data = await upload.read()
    if not data:
        return None
    path.write_bytes(data)
    return str(path)


@app.post("/api/generate")
async def generate(
    text: str = Form(...),
    mode: str = Form("clone"),
    saved_voice: Optional[str] = Form(None),
    language: Optional[str] = Form("Auto"),
    gender: Optional[str] = Form(None),
    age: Optional[str] = Form(None),
    pitch: Optional[str] = Form(None),
    style: Optional[str] = Form(None),
    accent: Optional[str] = Form(None),
    dialect: Optional[str] = Form(None),
    custom_instruct: Optional[str] = Form(None),
    num_step: Optional[str] = Form("32"),
    guidance_scale: Optional[str] = Form("2.0"),
    denoise: Optional[str] = Form("true"),
    t_shift: Optional[str] = Form("0.1"),
    position_temperature: Optional[str] = Form("5.0"),
    class_temperature: Optional[str] = Form("0.0"),
    layer_penalty_factor: Optional[str] = Form("5.0"),
    speed: Optional[str] = Form("1.0"),
    duration: Optional[str] = Form(None),
    preprocess_prompt: Optional[str] = Form("true"),
    postprocess_output: Optional[str] = Form("true"),
    pad_duration: Optional[str] = Form("0.1"),
    fade_duration: Optional[str] = Form("0.1"),
    audio_chunk_duration: Optional[str] = Form("15.0"),
    audio_chunk_threshold: Optional[str] = Form("30.0"),
    normalize_text: Optional[str] = Form("false"),
    ref_audio: Optional[UploadFile] = File(None),
):
    if not engine.status()["ready"]:
        try:
            engine.load_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    ref_path = await _save_upload(ref_audio, "ref")
    voice_prompt_path = None
    if saved_voice and saved_voice.strip() and saved_voice != "none":
        candidate = engine.VOICES_DIR / f"{saved_voice.strip()}.pt"
        if not candidate.exists():
            raise HTTPException(status_code=400, detail="Saved voice not found.")
        voice_prompt_path = str(candidate)

    instruct = engine.build_instruct(
        gender=gender,
        age=age,
        pitch=pitch,
        style=style,
        accent=accent,
        dialect=dialect,
        custom=custom_instruct,
    )

    try:
        waveform, sr = engine.generate(
            text,
            mode=mode,
            ref_audio=ref_path,
            voice_prompt_path=voice_prompt_path,
            instruct=instruct,
            language=language,
            num_step=_form_int(num_step, 32),
            guidance_scale=_form_float(guidance_scale, 2.0),
            denoise=_form_bool(denoise, True),
            t_shift=_form_float(t_shift, 0.1),
            position_temperature=_form_float(position_temperature, 5.0),
            class_temperature=_form_float(class_temperature, 0.0),
            layer_penalty_factor=_form_float(layer_penalty_factor, 5.0),
            speed=_form_float(speed, 1.0),
            duration=_form_float(duration, None),
            preprocess_prompt=_form_bool(preprocess_prompt, True),
            postprocess_output=_form_bool(postprocess_output, True),
            pad_duration=_form_float(pad_duration, 0.1),
            fade_duration=_form_float(fade_duration, 0.1),
            audio_chunk_duration=_form_float(audio_chunk_duration, 15.0),
            audio_chunk_threshold=_form_float(audio_chunk_threshold, 30.0),
            normalize_text=_form_bool(normalize_text, False),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    engine.ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = engine.OUTPUTS_DIR / f"clone_{stamp}_{uuid.uuid4().hex[:8]}.wav"
    sf.write(str(out_path), waveform, sr)

    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{out_path.name}"',
            "X-Output-File": out_path.name,
            "X-Sample-Rate": str(sr),
            "Access-Control-Expose-Headers": "X-Output-File, X-Sample-Rate",
        },
    )


@app.post("/api/voices/save")
async def save_voice(
    name: str = Form(...),
    ref_audio: UploadFile = File(...),
):
    if not engine.status()["ready"]:
        try:
            engine.load_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    ref_path = await _save_upload(ref_audio, "voice")
    if not ref_path:
        raise HTTPException(status_code=400, detail="Reference audio is required.")
    try:
        path = engine.save_voice_prompt(name, ref_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "id": path.stem, "path": str(path)}


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str):
    path = engine.VOICES_DIR / f"{voice_id}.pt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Voice not found.")
    path.unlink()
    return {"ok": True}


@app.get("/api/outputs/{filename}")
async def get_output(filename: str):
    path = engine.OUTPUTS_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


def run():
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=7870,
        reload=False,
    )


if __name__ == "__main__":
    run()
