from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from infer_seed_vii_emotion import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    find_feature_path,
    load_manifest,
    load_model,
    predict_feature,
)


CHECKPOINT_PATH = DEFAULT_CHECKPOINT
MANIFEST_PATH = DEFAULT_MANIFEST
DEFAULT_STRIDE = 4

# A simple affective projection for downstream avatar / dialogue control.
EMOTION_AFFECT = {
    "Anger": {"valence": -0.80, "arousal": 0.85},
    "Disgust": {"valence": -0.75, "arousal": 0.55},
    "Fear": {"valence": -0.90, "arousal": 0.95},
    "Happy": {"valence": 0.95, "arousal": 0.70},
    "Neutral": {"valence": 0.00, "arousal": 0.15},
    "Sad": {"valence": -0.85, "arousal": 0.25},
    "Surprise": {"valence": 0.20, "arousal": 0.90},
}


class PredictRequest(BaseModel):
    feature_path: Optional[str] = Field(default=None, description="Absolute or relative .npy feature path")
    subject_id: Optional[int] = Field(default=None, description="Subject id from the cached manifest")
    trial_id: Optional[int] = Field(default=None, description="Trial id from the cached manifest")
    stride: int = Field(default=DEFAULT_STRIDE, ge=1, description="Sliding stride on the feature sequence")
    user_text: Optional[str] = Field(default=None, description="Optional latest user utterance for downstream LLM/TTS")
    session_id: Optional[str] = Field(default=None, description="Frontend session id")
    client_timestamp: Optional[str] = Field(default=None, description="Frontend event timestamp")


class EmotionProbability(BaseModel):
    emotion: str
    probability: float


class SourceInfo(BaseModel):
    feature_path: str
    subject_id: Optional[int] = None
    trial_id: Optional[int] = None
    session_id: Optional[str] = None
    client_timestamp: Optional[str] = None


class EmotionState(BaseModel):
    label_id: int
    dominant_emotion: str
    top_3_emotions: List[EmotionProbability]
    probabilities: List[EmotionProbability]
    valence: float
    arousal: float
    confidence: float
    intensity: float
    stability: float


class AvatarState(BaseModel):
    expression: str
    animation_state: str
    motion_scale: float
    blink_rate: float
    head_motion: float
    body_motion: float
    breathing_rate: float
    color_hint: str


class TTSState(BaseModel):
    voice_style: str
    speaking_rate: float
    pitch: float
    energy: float
    pause_scale: float
    pronunciation_style: str


class LLMState(BaseModel):
    system_mood: str
    response_style: str
    prompt_hint: str
    user_text: Optional[str] = None


class MetaState(BaseModel):
    protocol_version: str
    request_id: str
    server_timestamp: str
    model_checkpoint: str
    window_len: int
    stride: int
    window_count: int


class PredictResponse(BaseModel):
    meta: MetaState
    source: SourceInfo
    emotion: EmotionState
    avatar: AvatarState
    tts: TTSState
    llm: LLMState


app = FastAPI(
    title="Emotion Puppet EEG API",
    version="0.2.0",
    description="SEED-VII feature-based emotion inference service for the Emotion Puppet prototype.",
)


_manifest_rows: List[Dict[str, str]] = []
_checkpoint: Dict[str, object] = {}
_model: Optional[torch.nn.Module] = None
_device = torch.device("cpu")
_window_len = 0
PROTOCOL_VERSION = "emotion-puppet.v1"


def _build_dialogue_style(emotion: str, valence: float, arousal: float) -> str:
    if emotion == "Happy":
        return "warm, bright, encouraging, lightly playful"
    if emotion == "Sad":
        return "gentle, slow, soft, emotionally validating"
    if emotion == "Anger":
        return "tense, clipped, intense, forceful but controlled"
    if emotion == "Fear":
        return "hesitant, vigilant, breathy, uneasy"
    if emotion == "Disgust":
        return "restrained, distancing, skeptical, slightly cold"
    if emotion == "Surprise":
        return "animated, quick, alert, reactive"
    if valence >= 0.2 and arousal < 0.4:
        return "calm, reassuring, steady"
    return "neutral, attentive, conversational"


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _build_avatar_state(emotion: str, valence: float, arousal: float, confidence: float) -> AvatarState:
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
    motion_scale = _clamp(0.35 + arousal * 0.65, 0.2, 1.0)
    blink_rate = _clamp(0.18 + (1.0 - confidence) * 0.25 + arousal * 0.15, 0.12, 0.5)
    head_motion = _clamp(0.15 + arousal * 0.75, 0.1, 1.0)
    body_motion = _clamp(0.10 + arousal * 0.85, 0.08, 1.0)
    breathing_rate = _clamp(0.20 + arousal * 0.60 - valence * 0.05, 0.12, 0.9)
    return AvatarState(
        expression=expression,
        animation_state=animation_state,
        motion_scale=motion_scale,
        blink_rate=blink_rate,
        head_motion=head_motion,
        body_motion=body_motion,
        breathing_rate=breathing_rate,
        color_hint=color_hint,
    )


def _build_tts_state(emotion: str, valence: float, arousal: float) -> TTSState:
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
    return TTSState(
        voice_style=voice_style,
        speaking_rate=speaking_rate,
        pitch=pitch,
        energy=energy,
        pause_scale=pause_scale,
        pronunciation_style=pronunciation_style,
    )


def _build_llm_state(emotion: str, dialogue_style: str, user_text: Optional[str]) -> LLMState:
    mood_map = {
        "Anger": "frustrated but contained",
        "Disgust": "withdrawn and skeptical",
        "Fear": "uneasy and vigilant",
        "Happy": "open and delighted",
        "Neutral": "calm and observant",
        "Sad": "tender and heavy-hearted",
        "Surprise": "alert and intrigued",
    }
    system_mood = mood_map[emotion]
    prompt_hint = (
        f"Respond as a digital being whose internal affect is {emotion.lower()}. "
        f"Keep the language {dialogue_style}. Reflect the emotion in wording, pacing, and emotional framing "
        f"without explicitly naming an emotion label unless the user asks."
    )
    return LLMState(
        system_mood=system_mood,
        response_style=dialogue_style,
        prompt_hint=prompt_hint,
        user_text=user_text,
    )


def _project_affect(probabilities: List[Dict[str, float]]) -> Dict[str, float]:
    valence = 0.0
    arousal = 0.0
    for item in probabilities:
        affect = EMOTION_AFFECT[item["emotion"]]
        valence += item["probability"] * affect["valence"]
        arousal += item["probability"] * affect["arousal"]
    return {"valence": float(valence), "arousal": float(arousal)}


def _build_protocol_example() -> Dict[str, object]:
    return {
        "meta": {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": "req_example",
            "server_timestamp": "2026-04-19T08:00:00Z",
            "model_checkpoint": str(CHECKPOINT_PATH.resolve()),
            "window_len": 12,
            "stride": 4,
            "window_count": 3,
        },
        "source": {
            "feature_path": "F:/SEED_EEG_3D/artifacts/seed_vii/feature_cache/de_LDS/subject_01/trial_01.npy",
            "subject_id": 1,
            "trial_id": 1,
            "session_id": "demo-session",
            "client_timestamp": "2026-04-19T16:00:00+08:00",
        },
        "emotion": {
            "label_id": 3,
            "dominant_emotion": "Happy",
            "top_3_emotions": [
                {"emotion": "Happy", "probability": 0.61},
                {"emotion": "Surprise", "probability": 0.18},
                {"emotion": "Neutral", "probability": 0.09},
            ],
            "probabilities": [
                {"emotion": "Happy", "probability": 0.61},
                {"emotion": "Surprise", "probability": 0.18},
                {"emotion": "Neutral", "probability": 0.09},
                {"emotion": "Sad", "probability": 0.05},
                {"emotion": "Anger", "probability": 0.03},
                {"emotion": "Disgust", "probability": 0.02},
                {"emotion": "Fear", "probability": 0.02},
            ],
            "valence": 0.68,
            "arousal": 0.62,
            "confidence": 0.61,
            "intensity": 0.64,
            "stability": 0.43,
        },
        "avatar": {
            "expression": "smile",
            "animation_state": "buoyant_idle",
            "motion_scale": 0.75,
            "blink_rate": 0.24,
            "head_motion": 0.61,
            "body_motion": 0.58,
            "breathing_rate": 0.52,
            "color_hint": "#ffb703",
        },
        "tts": {
            "voice_style": "warm_bright",
            "speaking_rate": 1.12,
            "pitch": 1.08,
            "energy": 1.03,
            "pause_scale": 0.91,
            "pronunciation_style": "smiling tone, upbeat cadence",
        },
        "llm": {
            "system_mood": "open and delighted",
            "response_style": "warm, bright, encouraging, lightly playful",
            "prompt_hint": "Respond as a digital being whose internal affect is happy.",
            "user_text": "今天过得怎么样？",
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


@app.get("/")
def root() -> Dict[str, object]:
    return {
        "name": "Emotion Puppet EEG API",
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint": str(CHECKPOINT_PATH.resolve()),
        "window_len": _window_len,
        "predict_endpoint": "/predict",
        "protocol_endpoint": "/protocol",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "ok": True,
        "checkpoint": str(CHECKPOINT_PATH.resolve()),
        "manifest": str(MANIFEST_PATH.resolve()),
        "protocol_version": PROTOCOL_VERSION,
        "window_len": _window_len,
        "device": str(_device),
    }


@app.get("/protocol")
def protocol() -> Dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_fields": PredictRequest.model_json_schema(),
        "response_example": _build_protocol_example(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        if request.feature_path:
            feature_path = Path(request.feature_path)
        elif request.subject_id is not None and request.trial_id is not None:
            feature_path = find_feature_path(_manifest_rows, request.subject_id, request.trial_id)
        else:
            raise ValueError("Provide feature_path or subject_id + trial_id")

        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        feature = np.load(feature_path).astype(np.float32)
        result = predict_feature(_model, feature, window_len=_window_len, stride=request.stride, device=_device)
        affect = _project_affect(result["probabilities"])
        probabilities = [EmotionProbability(**item) for item in result["probabilities"]]
        confidence = probabilities[0].probability
        runner_up = probabilities[1].probability if len(probabilities) > 1 else 0.0
        intensity = _clamp((abs(affect["valence"]) + affect["arousal"]) / 2.0, 0.0, 1.0)
        stability = _clamp(confidence - runner_up, 0.0, 1.0)
        dialogue_style = _build_dialogue_style(
            result["predicted_emotion"],
            affect["valence"],
            affect["arousal"],
        )
        avatar_state = _build_avatar_state(result["predicted_emotion"], affect["valence"], affect["arousal"], confidence)
        tts_state = _build_tts_state(result["predicted_emotion"], affect["valence"], affect["arousal"])
        llm_state = _build_llm_state(result["predicted_emotion"], dialogue_style, request.user_text)
        request_id = uuid4().hex
        server_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        return PredictResponse(
            meta=MetaState(
                protocol_version=PROTOCOL_VERSION,
                request_id=request_id,
                server_timestamp=server_timestamp,
                model_checkpoint=str(CHECKPOINT_PATH.resolve()),
                window_len=_window_len,
                stride=request.stride,
                window_count=result["window_count"],
            ),
            source=SourceInfo(
                feature_path=str(feature_path.resolve()),
                subject_id=request.subject_id,
                trial_id=request.trial_id,
                session_id=request.session_id,
                client_timestamp=request.client_timestamp,
            ),
            emotion=EmotionState(
                label_id=result["predicted_label_id"],
                dominant_emotion=result["predicted_emotion"],
                top_3_emotions=probabilities[:3],
                probabilities=probabilities,
                valence=affect["valence"],
                arousal=affect["arousal"],
                confidence=confidence,
                intensity=intensity,
                stability=stability,
            ),
            avatar=avatar_state,
            tts=tts_state,
            llm=llm_state,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/manifest/sample")
def manifest_sample() -> Dict[str, object]:
    sample_rows = _manifest_rows[:5]
    return {
        "count": len(_manifest_rows),
        "sample": sample_rows,
    }
