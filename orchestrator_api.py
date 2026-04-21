import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from infer_seed_vii_emotion import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    find_feature_path,
    load_manifest,
    load_model,
    predict_feature,
)
from providers import doubao_natural_llm, emotivoice_tts, voice_soundboard_tts


CHECKPOINT_PATH = DEFAULT_CHECKPOINT
MANIFEST_PATH = DEFAULT_MANIFEST
DEFAULT_STRIDE = 4
PROTOCOL_VERSION = "emotion-puppet.orchestrator.v1"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts" / "orchestrator"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "coral")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
PIPER_EXE = os.getenv("PIPER_EXE")
PIPER_MODEL = os.getenv("PIPER_MODEL")
EDGE_TTS_ZH_VOICE = os.getenv("EDGE_TTS_ZH_VOICE", "zh-CN-XiaoxiaoNeural")
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

EMOTION_AFFECT = {
    "Anger": {"valence": -0.80, "arousal": 0.85},
    "Disgust": {"valence": -0.75, "arousal": 0.55},
    "Fear": {"valence": -0.90, "arousal": 0.95},
    "Happy": {"valence": 0.95, "arousal": 0.70},
    "Neutral": {"valence": 0.00, "arousal": 0.15},
    "Sad": {"valence": -0.85, "arousal": 0.25},
    "Surprise": {"valence": 0.20, "arousal": 0.90},
}


class OrchestrateRequest(BaseModel):
    feature_path: Optional[str] = Field(default=None, description="Absolute or relative .npy feature path")
    subject_id: Optional[int] = Field(default=None, description="Subject id from the cached manifest")
    trial_id: Optional[int] = Field(default=None, description="Trial id from the cached manifest")
    stride: int = Field(default=DEFAULT_STRIDE, ge=1, description="Sliding stride on the feature sequence")
    user_text: Optional[str] = Field(default=None, description="Latest user utterance")
    session_id: Optional[str] = Field(default=None, description="Frontend session id")
    client_timestamp: Optional[str] = Field(default=None, description="Frontend event timestamp")
    llm_provider: str = Field(default="stub", description="stub, openai, doubao, or none")
    tts_provider: str = Field(default="stub", description="stub, pyttsx3, edge_tts, sapi, piper, emotivoice, voice_soundboard, openai, or none")
    include_audio_base64: bool = Field(default=False, description="Embed audio as base64 in response")
    save_audio_to_file: bool = Field(default=True, description="Persist generated audio to artifacts/orchestrator")
    openai_text_model: Optional[str] = Field(default=None, description="Override text model")
    openai_tts_model: Optional[str] = Field(default=None, description="Override TTS model")
    openai_voice: Optional[str] = Field(default=None, description="Override TTS voice")
    tts_format: str = Field(default="mp3", description="mp3, wav, pcm, opus, flac, or aac")


class RealtimeTokenRequest(BaseModel):
    model: str = Field(default=OPENAI_REALTIME_MODEL)
    voice: str = Field(default="marin")
    instructions: Optional[str] = Field(default=None)


class ProbabilityItem(BaseModel):
    emotion: str
    probability: float


class MetaInfo(BaseModel):
    protocol_version: str
    request_id: str
    server_timestamp: str
    model_checkpoint: str
    window_len: int
    stride: int
    window_count: int


class SourceInfo(BaseModel):
    feature_path: str
    subject_id: Optional[int] = None
    trial_id: Optional[int] = None
    session_id: Optional[str] = None
    client_timestamp: Optional[str] = None


class EmotionInfo(BaseModel):
    label_id: int
    dominant_emotion: str
    top_3_emotions: List[ProbabilityItem]
    probabilities: List[ProbabilityItem]
    valence: float
    arousal: float
    confidence: float
    intensity: float
    stability: float


class AvatarInfo(BaseModel):
    expression: str
    animation_state: str
    motion_scale: float
    blink_rate: float
    head_motion: float
    body_motion: float
    breathing_rate: float
    color_hint: str


class LLMInfo(BaseModel):
    provider: str
    status: str
    model: Optional[str] = None
    system_mood: str
    response_style: str
    prompt_hint: str
    system_prompt: str
    user_prompt: str
    response_text: Optional[str] = None
    provider_response_id: Optional[str] = None


class TTSInfo(BaseModel):
    provider: str
    status: str
    model: Optional[str] = None
    voice: Optional[str] = None
    audio_format: str
    voice_style: str
    speaking_rate: float
    pitch: float
    energy: float
    pause_scale: float
    pronunciation_style: str
    instructions: str
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    mime_type: Optional[str] = None


class FrontendInfo(BaseModel):
    text: Optional[str] = None
    should_render_avatar: bool
    should_play_audio: bool
    next_actions: List[str]


class OrchestrateResponse(BaseModel):
    meta: MetaInfo
    source: SourceInfo
    emotion: EmotionInfo
    avatar: AvatarInfo
    llm: LLMInfo
    tts: TTSInfo
    frontend: FrontendInfo


app = FastAPI(
    title="Emotion Puppet Orchestrator API",
    version="0.1.0",
    description="Orchestrates EEG emotion inference, LLM text generation, and TTS planning for Emotion Puppet.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_manifest_rows: List[Dict[str, str]] = []
_checkpoint: Dict[str, object] = {}
_model: Optional[torch.nn.Module] = None
_device = torch.device("cpu")
_window_len = 0


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or os.getenv("ORCHESTRATOR_OPENAI_API_KEY")


def _optional_import_pyttsx3():
    try:
        import pyttsx3  # type: ignore

        return pyttsx3
    except Exception:
        return None


def _optional_import_edge_tts():
    try:
        import edge_tts  # type: ignore

        return edge_tts
    except Exception:
        return None


def _resolve_piper_exe() -> Optional[str]:
    if PIPER_EXE:
        return PIPER_EXE
    return shutil.which("piper") or shutil.which("piper.exe")


def _resolve_powershell_exe() -> Optional[str]:
    candidates = [
        os.getenv("POWERSHELL_EXE"),
        shutil.which("powershell"),
        shutil.which("powershell.exe"),
        shutil.which("pwsh"),
        shutil.which("pwsh.exe"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def _audio_dir() -> Path:
    path = ARTIFACTS_DIR / "audio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _audio_url_from_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        relative = path.resolve().relative_to(ARTIFACTS_DIR.resolve()).as_posix()
    except Exception:
        return None
    return f"/media/{relative}"


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _cors_preflight_headers(request: Request) -> Dict[str, str]:
    origin = request.headers.get("origin", "*")
    requested_headers = request.headers.get("access-control-request-headers", "content-type")
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": requested_headers,
        "Access-Control-Max-Age": "86400",
        "Vary": "Origin",
    }


def _json_request(url: str, body: Dict[str, object], api_key: str) -> Dict[str, object]:
    data = json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {payload}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc


def _binary_request(url: str, body: Dict[str, object], api_key: str) -> bytes:
    data = json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            return response.read()
    except HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {payload}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc


def _extract_response_text(payload: Dict[str, object]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output", [])
    if isinstance(output, list):
        fragments: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if isinstance(content.get("text"), str):
                    fragments.append(content["text"])
        text = "\n".join(part.strip() for part in fragments if part.strip()).strip()
        if text:
            return text

    raise RuntimeError("Could not extract text from OpenAI response payload")


def _project_affect(probabilities: List[Dict[str, float]]) -> Dict[str, float]:
    valence = 0.0
    arousal = 0.0
    for item in probabilities:
        affect = EMOTION_AFFECT[item["emotion"]]
        valence += item["probability"] * affect["valence"]
        arousal += item["probability"] * affect["arousal"]
    return {"valence": float(valence), "arousal": float(arousal)}


def _build_dialogue_style(emotion: str, valence: float, arousal: float) -> str:
    if emotion == "Happy":
        return "温暖、明亮、鼓励式、带一点轻盈玩心"
    if emotion == "Sad":
        return "轻柔、放慢、低声、带安抚感"
    if emotion == "Anger":
        return "紧绷、短句、有力度，但仍然克制"
    if emotion == "Fear":
        return "迟疑、警觉、轻微发紧、带一点不安"
    if emotion == "Disgust":
        return "克制、拉开距离、怀疑感、更冷一点"
    if emotion == "Surprise":
        return "灵动、偏快、警醒、反应感强"
    if valence >= 0.2 and arousal < 0.4:
        return "平稳、安定、让人放松"
    return "自然、专注、可对话"


def _build_avatar_info(emotion: str, valence: float, arousal: float, confidence: float) -> AvatarInfo:
    expression_map = {
        "Anger": ("angry", "tense_idle", "#d1495b"),
        "Disgust": ("disgust", "reserved_idle", "#6c757d"),
        "Fear": ("fear", "alert_idle", "#7b2cbf"),
        "Happy": ("smile", "buoyant_idle", "#ffb703"),
        "Neutral": ("neutral", "calm_idle", "#8ecae6"),
        "Sad": ("sad", "slow_idle", "#577590"),
        "Surprise": ("surprised", "reactive_idle", "#f4a261"),
    }
    expression, animation_state, color_hint = expression_map[emotion]
    return AvatarInfo(
        expression=expression,
        animation_state=animation_state,
        motion_scale=_clamp(0.35 + arousal * 0.65, 0.2, 1.0),
        blink_rate=_clamp(0.18 + (1.0 - confidence) * 0.25 + arousal * 0.15, 0.12, 0.5),
        head_motion=_clamp(0.15 + arousal * 0.75, 0.1, 1.0),
        body_motion=_clamp(0.10 + arousal * 0.85, 0.08, 1.0),
        breathing_rate=_clamp(0.20 + arousal * 0.60 - valence * 0.05, 0.12, 0.9),
        color_hint=color_hint,
    )


def _build_tts_style(emotion: str, valence: float, arousal: float) -> Dict[str, object]:
    speaking_rate = _clamp(0.90 + arousal * 0.30 + max(valence, 0.0) * 0.05, 0.75, 1.35)
    pitch = _clamp(0.95 + valence * 0.12 + (arousal - 0.5) * 0.10, 0.75, 1.25)
    energy = _clamp(0.75 + arousal * 0.45, 0.55, 1.35)
    pause_scale = _clamp(1.15 - arousal * 0.35 + max(-valence, 0.0) * 0.15, 0.75, 1.35)
    style_map = {
        "Anger": ("firm_intense", "clear, clipped, low warmth"),
        "Disgust": ("cool_distant", "restrained, slightly dry"),
        "Fear": ("uneasy_breathing", "lighter voice, cautious phrasing"),
        "Happy": ("warm_bright", "smiling tone, upbeat cadence"),
        "Neutral": ("natural_conversational", "balanced and steady"),
        "Sad": ("soft_gentle", "low energy, longer tails"),
        "Surprise": ("animated_reactive", "quick emphasis, alert tone"),
    }
    voice_style, pronunciation_style = style_map[emotion]
    return {
        "voice_style": voice_style,
        "speaking_rate": speaking_rate,
        "pitch": pitch,
        "energy": energy,
        "pause_scale": pause_scale,
        "pronunciation_style": pronunciation_style,
    }


def _build_llm_prompt(emotion: str, dialogue_style: str, user_text: Optional[str]) -> Dict[str, str]:
    mood_map = {
        "Anger": "压着火气，但还在控制",
        "Disgust": "有点后撤，也带着怀疑",
        "Fear": "发紧、警觉、小心翼翼",
        "Happy": "敞开、愉快、愿意靠近",
        "Neutral": "平稳、安静、在观察",
        "Sad": "柔软、低落、心口有点沉",
        "Surprise": "被点亮了一下，注意力迅速聚拢",
    }
    system_mood = mood_map[emotion]
    system_prompt = (
        "You are the inner voice of a digital being called Qingxu Puppet. "
        f"Your current internal affect is {emotion.lower()} and your mood is {system_mood}. "
        "Reply only in natural spoken Simplified Chinese. "
        "Do not switch to English unless the user explicitly asks for English. "
        f"Speak in a {dialogue_style} way. "
        "Reflect the mood in wording, rhythm, and emotional framing without explicitly naming an emotion label "
        "unless the user asks. Keep replies concise, vivid, and natural for spoken delivery."
    )
    user_prompt = user_text or (
        "Please reply with 1-3 short lines of natural spoken Simplified Chinese that reveal your current inner state."
    )
    prompt_hint = (
        f"以“{system_mood}”的内在状态回应。"
        f"整体口吻保持“{dialogue_style}”，并且只使用自然的简体中文口语。"
    )
    return {
        "system_mood": system_mood,
        "response_style": dialogue_style,
        "prompt_hint": prompt_hint,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def _resolve_feature_path(request: OrchestrateRequest) -> Path:
    if request.feature_path:
        path = Path(request.feature_path)
    elif request.subject_id is not None and request.trial_id is not None:
        path = find_feature_path(_manifest_rows, request.subject_id, request.trial_id)
    else:
        raise ValueError("Provide feature_path or subject_id + trial_id")

    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    return path


def _predict_emotion(feature_path: Path, stride: int) -> Dict[str, object]:
    feature = np.load(feature_path).astype(np.float32)
    result = predict_feature(_model, feature, window_len=_window_len, stride=stride, device=_device)
    affect = _project_affect(result["probabilities"])
    probabilities = [ProbabilityItem(**item) for item in result["probabilities"]]
    confidence = probabilities[0].probability
    runner_up = probabilities[1].probability if len(probabilities) > 1 else 0.0
    intensity = _clamp((abs(affect["valence"]) + affect["arousal"]) / 2.0, 0.0, 1.0)
    stability = _clamp(confidence - runner_up, 0.0, 1.0)
    dominant_emotion = result["predicted_emotion"]
    dialogue_style = _build_dialogue_style(dominant_emotion, affect["valence"], affect["arousal"])
    avatar = _build_avatar_info(dominant_emotion, affect["valence"], affect["arousal"], confidence)
    tts_style = _build_tts_style(dominant_emotion, affect["valence"], affect["arousal"])
    llm_prompt = _build_llm_prompt(dominant_emotion, dialogue_style, None)

    return {
        "emotion": EmotionInfo(
            label_id=result["predicted_label_id"],
            dominant_emotion=dominant_emotion,
            top_3_emotions=probabilities[:3],
            probabilities=probabilities,
            valence=affect["valence"],
            arousal=affect["arousal"],
            confidence=confidence,
            intensity=intensity,
            stability=stability,
        ),
        "avatar": avatar,
        "tts_style": tts_style,
        "llm_prompt": llm_prompt,
        "window_count": result["window_count"],
    }


def _stub_llm(prompt_bundle: Dict[str, str], emotion: EmotionInfo) -> LLMInfo:
    text_templates = {
        "Happy": "我现在像被暖光轻轻托起来，想更靠近你一点，听你继续说下去。",
        "Sad": "我有一点往里收，像把声音放轻了些，但我还在认真地陪着你。",
        "Anger": "我胸口有股发紧的力道，话会更直一点，但我还是在控制自己不刺伤你。",
        "Fear": "我有点绷着，像在先确认周围是否安全，所以会更小心地回应你。",
        "Disgust": "我会本能地后撤半步，语气也更冷静克制，不过我仍然在看着这件事。",
        "Surprise": "我像被突然点亮了一下，注意力一下就聚拢过来了。",
        "Neutral": "我现在比较平稳，像把情绪放在一边，安静地接住你这句话。",
    }
    return LLMInfo(
        provider="stub",
        status="ready",
        model="stub-emotion-prompt",
        system_mood=prompt_bundle["system_mood"],
        response_style=prompt_bundle["response_style"],
        prompt_hint=prompt_bundle["prompt_hint"],
        system_prompt=prompt_bundle["system_prompt"],
        user_prompt=prompt_bundle["user_prompt"],
        response_text=text_templates[emotion.dominant_emotion],
        provider_response_id=None,
    )


def _openai_llm(prompt_bundle: Dict[str, str], model_override: Optional[str]) -> LLMInfo:
    api_key = _openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model_name = model_override or OPENAI_TEXT_MODEL
    body = {
        "model": model_name,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": prompt_bundle["system_prompt"]}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt_bundle["user_prompt"]}],
            },
        ],
    }
    payload = _json_request(f"{OPENAI_BASE_URL}/v1/responses", body, api_key)
    response_text = _extract_response_text(payload)
    return LLMInfo(
        provider="openai",
        status="ready",
        model=model_name,
        system_mood=prompt_bundle["system_mood"],
        response_style=prompt_bundle["response_style"],
        prompt_hint=prompt_bundle["prompt_hint"],
        system_prompt=prompt_bundle["system_prompt"],
        user_prompt=prompt_bundle["user_prompt"],
        response_text=response_text,
        provider_response_id=str(payload.get("id")) if payload.get("id") is not None else None,
    )


def _doubao_llm(prompt_bundle: Dict[str, str], emotion: EmotionInfo, user_text: Optional[str]) -> LLMInfo:
    reply = doubao_natural_llm.generate_emotion_reply(
        emotion=emotion.dominant_emotion,
        system_mood=prompt_bundle["system_mood"],
        dialogue_style=prompt_bundle["response_style"],
        valence=emotion.valence,
        arousal=emotion.arousal,
        user_text=user_text,
    )
    return LLMInfo(
        provider="doubao",
        status="ready",
        model=reply.model,
        system_mood=prompt_bundle["system_mood"],
        response_style=prompt_bundle["response_style"],
        prompt_hint=prompt_bundle["prompt_hint"],
        system_prompt=reply.system_prompt,
        user_prompt=reply.user_prompt,
        response_text=reply.text,
        provider_response_id=reply.response_id,
    )


def _none_llm(prompt_bundle: Dict[str, str]) -> LLMInfo:
    return LLMInfo(
        provider="none",
        status="skipped",
        model=None,
        system_mood=prompt_bundle["system_mood"],
        response_style=prompt_bundle["response_style"],
        prompt_hint=prompt_bundle["prompt_hint"],
        system_prompt=prompt_bundle["system_prompt"],
        user_prompt=prompt_bundle["user_prompt"],
        response_text=None,
        provider_response_id=None,
    )


def _build_tts_instructions(emotion: EmotionInfo, tts_style: Dict[str, object]) -> str:
    return (
        f"Speak with a {tts_style['voice_style']} delivery. "
        f"The emotional center is {emotion.dominant_emotion.lower()}. "
        f"Target speaking rate about {tts_style['speaking_rate']:.2f}x, "
        f"pitch about {tts_style['pitch']:.2f}, "
        f"energy about {tts_style['energy']:.2f}, "
        f"and pause scale about {tts_style['pause_scale']:.2f}. "
        f"Pronunciation style: {tts_style['pronunciation_style']}."
    )


def _stub_tts(tts_style: Dict[str, object], audio_format: str) -> TTSInfo:
    instructions = (
        f"voice_style={tts_style['voice_style']}; rate={tts_style['speaking_rate']:.2f}; "
        f"pitch={tts_style['pitch']:.2f}; energy={tts_style['energy']:.2f}"
    )
    return TTSInfo(
        provider="stub",
        status="planned",
        model="stub-tts-plan",
        voice="coral",
        audio_format=audio_format,
        voice_style=str(tts_style["voice_style"]),
        speaking_rate=float(tts_style["speaking_rate"]),
        pitch=float(tts_style["pitch"]),
        energy=float(tts_style["energy"]),
        pause_scale=float(tts_style["pause_scale"]),
        pronunciation_style=str(tts_style["pronunciation_style"]),
        instructions=instructions,
        audio_path=None,
        audio_url=None,
        audio_base64=None,
        mime_type=f"audio/{audio_format}",
    )


def _load_audio_base64(path: Path, include_audio_base64: bool) -> Optional[str]:
    if not include_audio_base64:
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _list_windows_sapi_voices() -> List[Dict[str, str]]:
    powershell_exe = _resolve_powershell_exe()
    if not powershell_exe:
        return []

    script = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $voices = $s.GetInstalledVoices() | ForEach-Object {
    $v = $_.VoiceInfo
    [PSCustomObject]@{
      Name = $v.Name
      Culture = $v.Culture.Name
      Gender = [string]$v.Gender
      Description = $v.Description
    }
  }
  $voices | ConvertTo-Json -Compress
} finally {
  $s.Dispose()
}
""".strip()

    completed = subprocess.run(
        [powershell_exe, "-NoProfile", "-Command", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        encoding="utf-8",
        errors="ignore",
    )
    if completed.returncode != 0:
        return []

    payload = completed.stdout.strip()
    if not payload:
        return []

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _pick_windows_sapi_voice(text: str) -> Optional[str]:
    voices = _list_windows_sapi_voices()
    if not voices:
        return None

    if _contains_cjk(text):
        for voice in voices:
            if str(voice.get("Culture", "")).lower().startswith("zh"):
                return str(voice.get("Name"))

    for voice in voices:
        if str(voice.get("Culture", "")).lower().startswith("en"):
            return str(voice.get("Name"))

    return str(voices[0].get("Name")) if voices[0].get("Name") else None


def _pyttsx3_tts(
    text: str,
    tts_style: Dict[str, object],
    request_id: str,
    include_audio_base64: bool,
    save_audio_to_file: bool,
) -> TTSInfo:
    pyttsx3 = _optional_import_pyttsx3()
    if pyttsx3 is None:
        raise RuntimeError("pyttsx3 is not installed in the active environment")

    audio_format = "wav"
    audio_path_obj = _audio_dir() / f"{request_id}.wav"
    engine = pyttsx3.init()
    try:
        default_rate = engine.getProperty("rate") or 200
        default_volume = engine.getProperty("volume") or 1.0
        engine.setProperty("rate", int(round(default_rate * float(tts_style["speaking_rate"]))))
        engine.setProperty("volume", _clamp(default_volume * float(tts_style["energy"]), 0.2, 1.0))
        try:
            voices = engine.getProperty("voices")
            if voices:
                target_voice = "female" if tts_style["voice_style"] in {"warm_bright", "soft_gentle"} else None
                if target_voice:
                    for voice in voices:
                        voice_name = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
                        if target_voice in voice_name:
                            engine.setProperty("voice", voice.id)
                            break
        except Exception:
            pass

        engine.save_to_file(text, str(audio_path_obj))
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass

    if not audio_path_obj.exists():
        raise RuntimeError("pyttsx3 did not produce an audio file")

    audio_path = str(audio_path_obj.resolve()) if save_audio_to_file else None
    if not save_audio_to_file:
        temp_bytes = audio_path_obj.read_bytes()
        audio_path_obj.unlink()
        if include_audio_base64:
            audio_base64 = base64.b64encode(temp_bytes).decode("ascii")
        else:
            audio_base64 = None
    else:
        audio_base64 = _load_audio_base64(audio_path_obj, include_audio_base64)

    instructions = (
        f"Local pyttsx3 synthesis with voice_style={tts_style['voice_style']}, "
        f"rate_scale={float(tts_style['speaking_rate']):.2f}, "
        f"energy_scale={float(tts_style['energy']):.2f}"
    )
    return TTSInfo(
        provider="pyttsx3",
        status="ready",
        model="pyttsx3-local",
        voice=None,
        audio_format=audio_format,
        voice_style=str(tts_style["voice_style"]),
        speaking_rate=float(tts_style["speaking_rate"]),
        pitch=float(tts_style["pitch"]),
        energy=float(tts_style["energy"]),
        pause_scale=float(tts_style["pause_scale"]),
        pronunciation_style=str(tts_style["pronunciation_style"]),
        instructions=instructions,
        audio_path=audio_path,
        audio_url=_audio_url_from_path(audio_path_obj) if save_audio_to_file else None,
        audio_base64=audio_base64,
        mime_type="audio/wav",
    )


def _windows_sapi_tts(
    text: str,
    tts_style: Dict[str, object],
    request_id: str,
    include_audio_base64: bool,
    save_audio_to_file: bool,
) -> TTSInfo:
    powershell_exe = _resolve_powershell_exe()
    if not powershell_exe:
        raise RuntimeError("PowerShell executable not found for Windows SAPI TTS.")

    voice_name = _pick_windows_sapi_voice(text)
    if not voice_name:
        raise RuntimeError("No installed Windows SAPI voice is available.")

    audio_path_obj = _audio_dir() / f"{request_id}.wav"
    rate_value = int(round((_clamp(float(tts_style["speaking_rate"]), 0.75, 1.35) - 1.0) * 12))
    volume_value = int(round(_clamp(float(tts_style["energy"]), 0.55, 1.35) / 1.35 * 100))

    script = """
Add-Type -AssemblyName System.Speech
$text = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($env:ORCH_TTS_TEXT_B64))
$output = $env:ORCH_TTS_OUTPUT
$voiceName = $env:ORCH_TTS_VOICE
$rate = [int]$env:ORCH_TTS_RATE
$volume = [int]$env:ORCH_TTS_VOLUME
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  if ($voiceName) {
    $synth.SelectVoice($voiceName)
  }
  $synth.Rate = $rate
  $synth.Volume = $volume
  $synth.SetOutputToWaveFile($output)
  $synth.Speak($text)
} finally {
  $synth.Dispose()
}
""".strip()

    env = os.environ.copy()
    env["ORCH_TTS_TEXT_B64"] = base64.b64encode(text.encode("utf-8")).decode("ascii")
    env["ORCH_TTS_OUTPUT"] = str(audio_path_obj)
    env["ORCH_TTS_VOICE"] = voice_name
    env["ORCH_TTS_RATE"] = str(rate_value)
    env["ORCH_TTS_VOLUME"] = str(max(0, min(volume_value, 100)))

    completed = subprocess.run(
        [powershell_exe, "-NoProfile", "-Command", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
        encoding="utf-8",
        errors="ignore",
    )
    if completed.returncode != 0:
        stderr_text = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Windows SAPI synthesis failed: {stderr_text}")
    if not audio_path_obj.exists():
        raise RuntimeError("Windows SAPI did not produce an audio file")

    audio_path = str(audio_path_obj.resolve()) if save_audio_to_file else None
    if not save_audio_to_file:
        temp_bytes = audio_path_obj.read_bytes()
        audio_path_obj.unlink()
        audio_base64 = base64.b64encode(temp_bytes).decode("ascii") if include_audio_base64 else None
    else:
        audio_base64 = _load_audio_base64(audio_path_obj, include_audio_base64)

    instructions = (
        f"Windows SAPI synthesis with voice={voice_name}, "
        f"rate={rate_value}, volume={max(0, min(volume_value, 100))}"
    )
    return TTSInfo(
        provider="sapi",
        status="ready",
        model="windows-sapi",
        voice=voice_name,
        audio_format="wav",
        voice_style=str(tts_style["voice_style"]),
        speaking_rate=float(tts_style["speaking_rate"]),
        pitch=float(tts_style["pitch"]),
        energy=float(tts_style["energy"]),
        pause_scale=float(tts_style["pause_scale"]),
        pronunciation_style=str(tts_style["pronunciation_style"]),
        instructions=instructions,
        audio_path=audio_path,
        audio_url=_audio_url_from_path(audio_path_obj) if save_audio_to_file else None,
        audio_base64=audio_base64,
        mime_type="audio/wav",
    )


def _edge_tts_voice_for_emotion(emotion: str) -> str:
    voice_map = {
        "Anger": os.getenv("EDGE_TTS_ANGER_VOICE", "zh-CN-YunxiNeural"),
        "Disgust": os.getenv("EDGE_TTS_DISGUST_VOICE", "zh-CN-YunxiNeural"),
        "Fear": os.getenv("EDGE_TTS_FEAR_VOICE", EDGE_TTS_ZH_VOICE),
        "Happy": os.getenv("EDGE_TTS_HAPPY_VOICE", EDGE_TTS_ZH_VOICE),
        "Neutral": os.getenv("EDGE_TTS_NEUTRAL_VOICE", EDGE_TTS_ZH_VOICE),
        "Sad": os.getenv("EDGE_TTS_SAD_VOICE", EDGE_TTS_ZH_VOICE),
        "Surprise": os.getenv("EDGE_TTS_SURPRISE_VOICE", "zh-CN-YunxiNeural"),
    }
    return voice_map.get(emotion, EDGE_TTS_ZH_VOICE)


def _edge_tts_tts(
    text: str,
    emotion: EmotionInfo,
    tts_style: Dict[str, object],
    request_id: str,
    include_audio_base64: bool,
    save_audio_to_file: bool,
) -> TTSInfo:
    edge_tts = _optional_import_edge_tts()
    if edge_tts is None:
        raise RuntimeError("edge-tts is not installed in the active environment")

    audio_path_obj = _audio_dir() / f"{request_id}.mp3"
    rate_value = int(round((float(tts_style["speaking_rate"]) - 1.0) * 100))
    volume_value = int(round((float(tts_style["energy"]) - 1.0) * 40))
    pitch_value = int(round((float(tts_style["pitch"]) - 1.0) * 40))
    rate = f"{rate_value:+d}%"
    volume = f"{volume_value:+d}%"
    pitch = f"{pitch_value:+d}Hz"
    voice_name = _edge_tts_voice_for_emotion(emotion.dominant_emotion)

    async def _save_audio() -> None:
        communicator = edge_tts.Communicate(
            text=text,
            voice=voice_name,
            rate=rate,
            volume=volume,
            pitch=pitch,
        )
        await communicator.save(str(audio_path_obj))

    try:
        asyncio.run(_save_audio())
    except RuntimeError as exc:
        if "asyncio.run() cannot be called from a running event loop" in str(exc):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_save_audio())
            finally:
                loop.close()
        else:
            raise RuntimeError(f"Edge TTS synthesis failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Edge TTS synthesis failed: {exc}") from exc

    if not audio_path_obj.exists():
        raise RuntimeError("Edge TTS did not produce an audio file")

    audio_path = str(audio_path_obj.resolve()) if save_audio_to_file else None
    if not save_audio_to_file:
        temp_bytes = audio_path_obj.read_bytes()
        audio_path_obj.unlink()
        audio_base64 = base64.b64encode(temp_bytes).decode("ascii") if include_audio_base64 else None
    else:
        audio_base64 = _load_audio_base64(audio_path_obj, include_audio_base64)

    instructions = (
        f"Edge TTS synthesis with voice={voice_name}, rate={rate}, volume={volume}, pitch={pitch}"
    )
    return TTSInfo(
        provider="edge_tts",
        status="ready",
        model="edge-tts",
        voice=voice_name,
        audio_format="mp3",
        voice_style=str(tts_style["voice_style"]),
        speaking_rate=float(tts_style["speaking_rate"]),
        pitch=float(tts_style["pitch"]),
        energy=float(tts_style["energy"]),
        pause_scale=float(tts_style["pause_scale"]),
        pronunciation_style=str(tts_style["pronunciation_style"]),
        instructions=instructions,
        audio_path=audio_path,
        audio_url=_audio_url_from_path(audio_path_obj) if save_audio_to_file else None,
        audio_base64=audio_base64,
        mime_type="audio/mpeg",
    )


def _piper_tts(
    text: str,
    tts_style: Dict[str, object],
    request_id: str,
    include_audio_base64: bool,
    save_audio_to_file: bool,
) -> TTSInfo:
    piper_exe = _resolve_piper_exe()
    if not piper_exe:
        raise RuntimeError("Piper executable not found. Set PIPER_EXE or add piper to PATH.")
    if not PIPER_MODEL:
        raise RuntimeError("PIPER_MODEL is not set.")

    audio_path_obj = _audio_dir() / f"{request_id}.wav"
    command = [
        piper_exe,
        "--model",
        PIPER_MODEL,
        "--output_file",
        str(audio_path_obj),
    ]

    completed = subprocess.run(
        command,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        stderr_text = completed.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Piper synthesis failed: {stderr_text}")
    if not audio_path_obj.exists():
        raise RuntimeError("Piper did not produce an audio file")

    audio_path = str(audio_path_obj.resolve()) if save_audio_to_file else None
    if not save_audio_to_file:
        temp_bytes = audio_path_obj.read_bytes()
        audio_path_obj.unlink()
        if include_audio_base64:
            audio_base64 = base64.b64encode(temp_bytes).decode("ascii")
        else:
            audio_base64 = None
    else:
        audio_base64 = _load_audio_base64(audio_path_obj, include_audio_base64)

    instructions = (
        f"Local Piper synthesis with model={PIPER_MODEL}, voice_style={tts_style['voice_style']}, "
        f"target_rate_scale={float(tts_style['speaking_rate']):.2f}"
    )
    return TTSInfo(
        provider="piper",
        status="ready",
        model=PIPER_MODEL,
        voice=None,
        audio_format="wav",
        voice_style=str(tts_style["voice_style"]),
        speaking_rate=float(tts_style["speaking_rate"]),
        pitch=float(tts_style["pitch"]),
        energy=float(tts_style["energy"]),
        pause_scale=float(tts_style["pause_scale"]),
        pronunciation_style=str(tts_style["pronunciation_style"]),
        instructions=instructions,
        audio_path=audio_path,
        audio_url=_audio_url_from_path(audio_path_obj) if save_audio_to_file else None,
        audio_base64=audio_base64,
        mime_type="audio/wav",
    )


def _openai_tts(
    text: str,
    emotion: EmotionInfo,
    tts_style: Dict[str, object],
    request_id: str,
    audio_format: str,
    include_audio_base64: bool,
    save_audio_to_file: bool,
    model_override: Optional[str],
    voice_override: Optional[str],
) -> TTSInfo:
    api_key = _openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model_name = model_override or OPENAI_TTS_MODEL
    voice_name = voice_override or OPENAI_TTS_VOICE
    instructions = _build_tts_instructions(emotion, tts_style)
    body = {
        "model": model_name,
        "voice": voice_name,
        "input": text,
        "instructions": instructions,
        "response_format": audio_format,
    }
    audio_bytes = _binary_request(f"{OPENAI_BASE_URL}/v1/audio/speech", body, api_key)

    audio_path = None
    if save_audio_to_file:
        audio_dir = ARTIFACTS_DIR / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        path = audio_dir / f"{request_id}.{audio_format}"
        path.write_bytes(audio_bytes)
        audio_path = str(path.resolve())

    audio_base64 = None
    if include_audio_base64:
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")

    mime_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "pcm": "audio/L16",
        "opus": "audio/ogg",
        "flac": "audio/flac",
        "aac": "audio/aac",
    }

    return TTSInfo(
        provider="openai",
        status="ready",
        model=model_name,
        voice=voice_name,
        audio_format=audio_format,
        voice_style=str(tts_style["voice_style"]),
        speaking_rate=float(tts_style["speaking_rate"]),
        pitch=float(tts_style["pitch"]),
        energy=float(tts_style["energy"]),
        pause_scale=float(tts_style["pause_scale"]),
        pronunciation_style=str(tts_style["pronunciation_style"]),
        instructions=instructions,
        audio_path=audio_path,
        audio_url=_audio_url_from_path(path) if save_audio_to_file else None,
        audio_base64=audio_base64,
        mime_type=mime_map.get(audio_format, f"audio/{audio_format}"),
    )


def _emotivoice_tts(
    text: str,
    emotion: EmotionInfo,
    tts_style: Dict[str, object],
    request_id: str,
    include_audio_base64: bool,
    save_audio_to_file: bool,
) -> TTSInfo:
    result = emotivoice_tts.synthesize(
        text=text,
        emotion=emotion.dominant_emotion,
        request_id=request_id,
        output_dir=_audio_dir(),
        include_audio_base64=include_audio_base64,
        save_audio_to_file=save_audio_to_file,
    )
    audio_path_obj = Path(result.audio_path) if result.audio_path else None
    return TTSInfo(
        provider="emotivoice",
        status="ready",
        model=result.model,
        voice=result.voice,
        audio_format=result.audio_format,
        voice_style=str(tts_style["voice_style"]),
        speaking_rate=float(tts_style["speaking_rate"]),
        pitch=float(tts_style["pitch"]),
        energy=float(tts_style["energy"]),
        pause_scale=float(tts_style["pause_scale"]),
        pronunciation_style=str(tts_style["pronunciation_style"]),
        instructions=result.instructions,
        audio_path=result.audio_path,
        audio_url=_audio_url_from_path(audio_path_obj) if audio_path_obj is not None else None,
        audio_base64=result.audio_base64,
        mime_type=result.mime_type,
    )


def _voice_soundboard_tts(
    text: str,
    emotion: EmotionInfo,
    tts_style: Dict[str, object],
    request_id: str,
    include_audio_base64: bool,
    save_audio_to_file: bool,
) -> TTSInfo:
    result = voice_soundboard_tts.synthesize(
        text=text,
        emotion=emotion.dominant_emotion,
        request_id=request_id,
        output_dir=_audio_dir(),
        include_audio_base64=include_audio_base64,
        save_audio_to_file=save_audio_to_file,
    )
    audio_path_obj = Path(result.audio_path) if result.audio_path else None
    return TTSInfo(
        provider="voice_soundboard",
        status="ready",
        model=result.model,
        voice=result.voice,
        audio_format=result.audio_format,
        voice_style=str(tts_style["voice_style"]),
        speaking_rate=float(tts_style["speaking_rate"]),
        pitch=float(tts_style["pitch"]),
        energy=float(tts_style["energy"]),
        pause_scale=float(tts_style["pause_scale"]),
        pronunciation_style=str(tts_style["pronunciation_style"]),
        instructions=result.instructions,
        audio_path=result.audio_path,
        audio_url=_audio_url_from_path(audio_path_obj) if audio_path_obj is not None else None,
        audio_base64=result.audio_base64,
        mime_type=result.mime_type,
    )


def _none_tts(tts_style: Dict[str, object], audio_format: str) -> TTSInfo:
    return TTSInfo(
        provider="none",
        status="skipped",
        model=None,
        voice=None,
        audio_format=audio_format,
        voice_style=str(tts_style["voice_style"]),
        speaking_rate=float(tts_style["speaking_rate"]),
        pitch=float(tts_style["pitch"]),
        energy=float(tts_style["energy"]),
        pause_scale=float(tts_style["pause_scale"]),
        pronunciation_style=str(tts_style["pronunciation_style"]),
        instructions="TTS disabled for this request.",
        audio_path=None,
        audio_url=None,
        audio_base64=None,
        mime_type=None,
    )


def _provider_status() -> Dict[str, object]:
    api_key_ready = bool(_openai_api_key())
    pyttsx3_ready = _optional_import_pyttsx3() is not None
    edge_tts_ready = _optional_import_edge_tts() is not None
    piper_exe = _resolve_piper_exe()
    windows_sapi_voices = _list_windows_sapi_voices()
    windows_sapi_zh_voice = next(
        (voice.get("Name") for voice in windows_sapi_voices if str(voice.get("Culture", "")).lower().startswith("zh")),
        None,
    )
    return {
        "openai_api_key_present": api_key_ready,
        "text_model": OPENAI_TEXT_MODEL,
        "doubao_api_key_present": bool(doubao_natural_llm.DOUBAO_API_KEY),
        "doubao_model": doubao_natural_llm.DOUBAO_MODEL,
        "doubao_ready": doubao_natural_llm.is_available(),
        "emotivoice_ready": emotivoice_tts.is_available(),
        "emotivoice_error": emotivoice_tts.availability_error(),
        "emotivoice_root": str(emotivoice_tts.EMOTIVOICE_ROOT),
        "emotivoice_voice": emotivoice_tts.EMOTIVOICE_SPEAKER,
        "voice_soundboard_ready": voice_soundboard_tts.is_available(),
        "voice_soundboard_error": voice_soundboard_tts.availability_error(),
        "tts_model": OPENAI_TTS_MODEL,
        "tts_voice": OPENAI_TTS_VOICE,
        "realtime_model": OPENAI_REALTIME_MODEL,
        "local_tts": {
            "pyttsx3_available": pyttsx3_ready,
            "edge_tts_available": edge_tts_ready,
            "edge_tts_voice": EDGE_TTS_ZH_VOICE,
            "piper_executable": piper_exe,
            "piper_model": PIPER_MODEL,
            "windows_sapi_available": bool(windows_sapi_voices),
            "windows_sapi_voices": windows_sapi_voices,
            "windows_sapi_zh_voice": windows_sapi_zh_voice,
        },
    }


@app.on_event("startup")
def startup_event() -> None:
    global _manifest_rows, _checkpoint, _model, _window_len

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"Manifest not found: {MANIFEST_PATH}")

    _manifest_rows = load_manifest(MANIFEST_PATH)
    _checkpoint, _model = load_model(CHECKPOINT_PATH, _device)
    _window_len = int(_checkpoint["window_len"])
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


app.mount("/media", StaticFiles(directory=str(ARTIFACTS_DIR.resolve())), name="media")
app.mount("/live2d", StaticFiles(directory=str(FRONTEND_DIR.parent / "live2d")), name="live2d")


@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "ok": True,
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint": str(CHECKPOINT_PATH.resolve()),
        "manifest": str(MANIFEST_PATH.resolve()),
        "window_len": _window_len,
        "device": str(_device),
        "providers": _provider_status(),
    }


@app.get("/providers")
def providers() -> Dict[str, object]:
    return _provider_status()


@app.options("/{rest_of_path:path}", include_in_schema=False)
def cors_preflight(rest_of_path: str, request: Request) -> Response:
    return Response(status_code=204, headers=_cors_preflight_headers(request))


@app.post("/providers/openai/realtime/client-secret")
def create_openai_realtime_client_secret(request: RealtimeTokenRequest) -> Dict[str, object]:
    api_key = _openai_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set")

    body = {
        "session": {
            "type": "realtime",
            "model": request.model,
            "audio": {
                "output": {
                    "voice": request.voice,
                }
            },
        }
    }
    if request.instructions:
        body["session"]["instructions"] = request.instructions

    try:
        payload = _json_request(f"{OPENAI_BASE_URL}/v1/realtime/client_secrets", body, api_key)
        return payload
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/orchestrate", response_model=OrchestrateResponse)
def orchestrate(request: OrchestrateRequest) -> OrchestrateResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Emotion model is not loaded")

    request_id = uuid4().hex
    try:
        feature_path = _resolve_feature_path(request)
        emotion_bundle = _predict_emotion(feature_path, request.stride)
        emotion_info: EmotionInfo = emotion_bundle["emotion"]
        avatar_info: AvatarInfo = emotion_bundle["avatar"]
        tts_style = emotion_bundle["tts_style"]
        llm_prompt = _build_llm_prompt(
            emotion_info.dominant_emotion,
            emotion_bundle["llm_prompt"]["response_style"],
            request.user_text,
        )

        llm_provider = request.llm_provider.lower()
        if llm_provider == "openai":
            llm_info = _openai_llm(llm_prompt, request.openai_text_model)
        elif llm_provider == "doubao":
            llm_info = _doubao_llm(llm_prompt, emotion_info, request.user_text)
        elif llm_provider == "none":
            llm_info = _none_llm(llm_prompt)
        else:
            llm_info = _stub_llm(llm_prompt, emotion_info)

        tts_provider = request.tts_provider.lower()
        if tts_provider == "openai" and llm_info.response_text:
            tts_info = _openai_tts(
                text=llm_info.response_text,
                emotion=emotion_info,
                tts_style=tts_style,
                request_id=request_id,
                audio_format=request.tts_format,
                include_audio_base64=request.include_audio_base64,
                save_audio_to_file=request.save_audio_to_file,
                model_override=request.openai_tts_model,
                voice_override=request.openai_voice,
            )
        elif tts_provider == "pyttsx3" and llm_info.response_text:
            tts_info = _pyttsx3_tts(
                text=llm_info.response_text,
                tts_style=tts_style,
                request_id=request_id,
                include_audio_base64=request.include_audio_base64,
                save_audio_to_file=request.save_audio_to_file,
            )
        elif tts_provider == "edge_tts" and llm_info.response_text:
            tts_info = _edge_tts_tts(
                text=llm_info.response_text,
                emotion=emotion_info,
                tts_style=tts_style,
                request_id=request_id,
                include_audio_base64=request.include_audio_base64,
                save_audio_to_file=request.save_audio_to_file,
            )
        elif tts_provider == "sapi" and llm_info.response_text:
            tts_info = _windows_sapi_tts(
                text=llm_info.response_text,
                tts_style=tts_style,
                request_id=request_id,
                include_audio_base64=request.include_audio_base64,
                save_audio_to_file=request.save_audio_to_file,
            )
        elif tts_provider == "piper" and llm_info.response_text:
            tts_info = _piper_tts(
                text=llm_info.response_text,
                tts_style=tts_style,
                request_id=request_id,
                include_audio_base64=request.include_audio_base64,
                save_audio_to_file=request.save_audio_to_file,
            )
        elif tts_provider == "emotivoice" and llm_info.response_text:
            tts_info = _emotivoice_tts(
                text=llm_info.response_text,
                emotion=emotion_info,
                tts_style=tts_style,
                request_id=request_id,
                include_audio_base64=request.include_audio_base64,
                save_audio_to_file=request.save_audio_to_file,
            )
        elif tts_provider == "voice_soundboard" and llm_info.response_text:
            tts_info = _voice_soundboard_tts(
                text=llm_info.response_text,
                emotion=emotion_info,
                tts_style=tts_style,
                request_id=request_id,
                include_audio_base64=request.include_audio_base64,
                save_audio_to_file=request.save_audio_to_file,
            )
        elif tts_provider == "none":
            tts_info = _none_tts(tts_style, request.tts_format)
        else:
            tts_info = _stub_tts(tts_style, request.tts_format)

        next_actions = ["render_avatar"]
        if llm_info.response_text:
            next_actions.append("display_text")
        if tts_info.status == "ready":
            next_actions.append("play_audio")
        elif tts_info.status == "planned":
            next_actions.append("request_tts")

        return OrchestrateResponse(
            meta=MetaInfo(
                protocol_version=PROTOCOL_VERSION,
                request_id=request_id,
                server_timestamp=_utc_now(),
                model_checkpoint=str(CHECKPOINT_PATH.resolve()),
                window_len=_window_len,
                stride=request.stride,
                window_count=int(emotion_bundle["window_count"]),
            ),
            source=SourceInfo(
                feature_path=str(feature_path.resolve()),
                subject_id=request.subject_id,
                trial_id=request.trial_id,
                session_id=request.session_id,
                client_timestamp=request.client_timestamp,
            ),
            emotion=emotion_info,
            avatar=avatar_info,
            llm=llm_info,
            tts=tts_info,
            frontend=FrontendInfo(
                text=llm_info.response_text,
                should_render_avatar=True,
                should_play_audio=tts_info.status == "ready",
                next_actions=next_actions,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
