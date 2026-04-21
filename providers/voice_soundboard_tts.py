from __future__ import annotations

import base64
import importlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


VOICE_SOUNDBOARD_BACKEND = os.getenv("VOICE_SOUNDBOARD_BACKEND")
VOICE_SOUNDBOARD_VOICE = os.getenv("VOICE_SOUNDBOARD_VOICE", "af_bella")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


@dataclass
class VoiceSoundboardResult:
    provider: str
    model: str
    voice: str
    audio_format: str
    instructions: str
    audio_path: Optional[str]
    audio_base64: Optional[str]
    mime_type: str


def _default_model_dir() -> Path:
    env_path = os.getenv("VOICE_SOUNDBOARD_MODELS")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "models").resolve()


def _load_voice_soundboard():
    try:
        return importlib.import_module("voice_soundboard")
    except ModuleNotFoundError as exc:
        soundboard_spec = importlib.util.find_spec("Soundboard")
        if soundboard_spec is not None:
            raise RuntimeError(
                "Found the package 'Soundboard', but this project needs the TTS package "
                "'voice-soundboard' (import name: voice_soundboard). "
                "Please install the correct package into the active .venv."
            ) from exc
        raise RuntimeError(
            "voice_soundboard is not installed in the active .venv. "
            "Install the TTS package 'voice-soundboard' in this environment first."
        ) from exc


def _resolve_engine():
    module = _load_voice_soundboard()
    VoiceEngine = getattr(module, "VoiceEngine", None)
    Config = getattr(module, "Config", None)
    if VoiceEngine is None:
        raise RuntimeError("voice_soundboard is installed, but VoiceEngine is not available.")

    model_dir = _default_model_dir()
    os.environ.setdefault("VOICE_SOUNDBOARD_MODELS", str(model_dir))

    if VOICE_SOUNDBOARD_BACKEND and Config is not None:
        return VoiceEngine(Config(backend=VOICE_SOUNDBOARD_BACKEND, model_dir=str(model_dir)))
    return VoiceEngine()


def is_available() -> bool:
    try:
        _resolve_engine()
        return True
    except Exception:
        return False


def availability_error() -> Optional[str]:
    try:
        _resolve_engine()
        return None
    except Exception as exc:
        return str(exc)


def _emotion_to_voice_soundboard(emotion: str) -> tuple[str, str, str]:
    mapping = {
        "Anger": (
            "angry",
            "intense but controlled, with tighter rhythm and firmer stress",
            "announcer",
        ),
        "Disgust": (
            "disgusted",
            "cool, dry, slightly distant, with restrained breath",
            "narrator",
        ),
        "Fear": (
            "nervous",
            "careful, soft, and slightly hesitant, with smaller pauses before key words",
            "whisper",
        ),
        "Happy": (
            "happy",
            "warmly and lightly, with a brighter smile in the voice",
            "assistant",
        ),
        "Neutral": (
            "calm",
            "steady, close, natural, and conversational",
            "assistant",
        ),
        "Sad": (
            "sad",
            "soft, slower, and gentler, with a little sinking weight at the end of phrases",
            "whisper",
        ),
        "Surprise": (
            "excited",
            "alert, fresh, and a little quicker, with upward motion in emphasis",
            "announcer",
        ),
    }
    return mapping.get(emotion, ("calm", "natural, close, and conversational", "assistant"))


def _emotion_to_voice_id(emotion: str) -> str:
    voice_map = {
        "Anger": "am_michael",
        "Disgust": "bm_george",
        "Fear": "af_nicole",
        "Happy": "af_bella",
        "Neutral": "af_bella",
        "Sad": "af_nicole",
        "Surprise": "af_bella",
    }
    return voice_map.get(emotion, VOICE_SOUNDBOARD_VOICE)


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _ensure_supported_text(text: str) -> None:
    if _contains_cjk(text):
        raise RuntimeError(
            "Voice Soundboard 当前接的是 Kokoro 英文声库。"
            "这个后端的 voice 列表是英美音色，底层 tokenizer 默认按 en-us/en-gb 走，"
            "所以不适合直接朗读中文。请改用 OpenAI TTS，或配置带中文模型的 Piper/其他中文 TTS。"
        )


def _friendly_error(exc: Exception) -> RuntimeError:
    message = str(exc)
    model_dir = _default_model_dir()
    if "voices-v1.0.bin" in message:
        return RuntimeError(
            f"Voice Soundboard 缺少语音资源文件：{model_dir / 'voices-v1.0.bin'}。"
            "请把 voices-v1.0.bin 放到项目 models 目录后再试。"
        )
    if "kokoro-v1.0.onnx" in message:
        return RuntimeError(
            f"Voice Soundboard 缺少模型文件：{model_dir / 'kokoro-v1.0.onnx'}。"
            "请把 kokoro-v1.0.onnx 放到项目 models 目录后再试。"
        )
    if "No module named" in message:
        return RuntimeError("Voice Soundboard 依赖未安装完整，请先在 .venv311 里安装语音依赖。")
    return RuntimeError(f"Voice Soundboard 合成失败：{message}")


def synthesize(
    *,
    text: str,
    emotion: str,
    request_id: str,
    output_dir: Path,
    include_audio_base64: bool,
    save_audio_to_file: bool,
    preferred_voice: Optional[str] = None,
) -> VoiceSoundboardResult:
    _ensure_supported_text(text)
    engine = _resolve_engine()
    vs_emotion, style, preset = _emotion_to_voice_soundboard(emotion)
    voice = preferred_voice or _emotion_to_voice_id(emotion)

    try:
        result = engine.speak(
            text,
            voice=voice,
            emotion=vs_emotion,
            preset=preset,
            style=style,
        )
    except Exception as exc:
        raise _friendly_error(exc) from exc

    raw_audio_path = getattr(result, "audio_path", None)
    if not raw_audio_path:
        raise RuntimeError("voice_soundboard returned no audio_path")

    generated_path = Path(raw_audio_path).resolve()
    if not generated_path.exists():
        raise RuntimeError(f"voice_soundboard output file not found: {generated_path}")

    suffix = generated_path.suffix.lower() or ".wav"
    audio_format = suffix.lstrip(".") or "wav"
    mime_type = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "opus": "audio/ogg",
    }.get(audio_format, f"audio/{audio_format}")

    final_path: Optional[Path] = None
    if save_audio_to_file:
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"{request_id}{suffix}"
        if generated_path != final_path:
            shutil.copyfile(generated_path, final_path)
    audio_bytes = (final_path or generated_path).read_bytes()

    return VoiceSoundboardResult(
        provider="voice_soundboard",
        model=f"voice-soundboard[{VOICE_SOUNDBOARD_BACKEND or 'auto'}]",
        voice=voice,
        audio_format=audio_format,
        instructions=(
            f"voice_soundboard emotion={vs_emotion}, preset={preset}, style={style}, voice={voice}"
        ),
        audio_path=str(final_path.resolve()) if final_path is not None else None,
        audio_base64=base64.b64encode(audio_bytes).decode("ascii") if include_audio_base64 else None,
        mime_type=mime_type,
    )
