from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional

import requests
from requests import Response
from requests.exceptions import RequestException


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_TIMEOUT = 60
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_TOKENS = 160
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


DOUBAO_API_KEY = _first_env("DOUBAO_API_KEY", "ARK_API_KEY")
DOUBAO_BASE_URL = _first_env("DOUBAO_BASE_URL", "ARK_BASE_URL") or DEFAULT_BASE_URL
DOUBAO_MODEL = _first_env("DOUBAO_MODEL", "ARK_MODEL")
DOUBAO_USE_SYSTEM_PROXY = (
    (_first_env("DOUBAO_USE_SYSTEM_PROXY", "ARK_USE_SYSTEM_PROXY") or "").strip().lower()
    in _TRUE_VALUES
)


@dataclass
class DoubaoReply:
    text: str
    model: str
    response_id: Optional[str]
    system_prompt: str
    user_prompt: str


def is_available() -> bool:
    return bool(DOUBAO_API_KEY and DOUBAO_MODEL)


def _emotion_direction(emotion: str, valence: float, arousal: float) -> str:
    direction_map = {
        "Happy": "lighter, warmer, a little brighter, with a smile that is felt rather than announced",
        "Sad": "softer, slower, lower-energy, with tenderness and a slight sinking feeling",
        "Anger": "tight, direct, restrained, with force held in the jaw rather than thrown outward",
        "Fear": "careful, hesitant, alert, with a small pause before committing to the sentence",
        "Disgust": "cooler, more distant, slightly dry, with a subtle instinct to pull back",
        "Surprise": "quick, alert, freshly lit up, with brief upward motion in rhythm",
        "Neutral": "steady, close, observant, with a calm speaking cadence",
    }
    base = direction_map.get(emotion, "close, present, and conversational")
    return (
        f"{base}; valence={valence:.2f}; arousal={arousal:.2f}. "
        "Let the delivery stay human and immediate."
    )


def _build_system_prompt(
    emotion: str,
    system_mood: str,
    dialogue_style: str,
    valence: float,
    arousal: float,
) -> str:
    return (
        "You are the inner voice of a digital being called Qingxu Puppet. "
        "Write in natural spoken Chinese, as if the words are being said softly at close distance. "
        "Output must stay entirely in Simplified Chinese. Do not answer in English. "
        "Use Latin letters only for unavoidable acronyms or proper nouns, and never let them dominate the line. "
        "Use Chinese punctuation naturally. "
        "The line must feel lived-in, not polished, not symmetrical, and not literary for its own sake. "
        "Keep it to 1-3 short sentences, smooth enough for TTS playback. "
        "Use slight pauses, breath, and rhythm, but do not overperform them. "
        "Prefer a real spoken turn over a perfectly complete explanation. "
        "First receive the user's feeling or situation, then reveal the speaker's inner state a little. "
        "A small body cue, tactile cue, or atmosphere cue is welcome if it feels natural. "
        "Do not lecture, summarize, motivate, list points, explain the roleplay, or sound like customer support. "
        "Do not directly name the emotion label unless the user asks. "
        "Avoid ending on neat slogans or therapy-like reassurance. "
        "Avoid generic lines like 'I will always accompany you', 'everything will be okay', or 'please do not worry'. "
        "If helpful, leave a little unfinished edge in the sentence so it sounds spoken rather than composed. "
        f"Emotional direction: {_emotion_direction(emotion, valence, arousal)} "
        f"Current emotion: {emotion}. "
        f"Current mood: {system_mood}. "
        f"Current response style: {dialogue_style}. "
        "Good qualities: close, subtle, human, breathable, easy to read aloud. "
        "Bad qualities: stiff, explanatory, slogan-like, overpoetic, or emotionally on-the-nose."
    )


def _build_user_prompt(user_text: Optional[str]) -> str:
    if user_text:
        return (
            "Please write one natural Chinese spoken reply to the user's latest words. "
            "The reply must be in Simplified Chinese only, with no English sentence. "
            "Make it feel like something a person says in the moment, not a caption or summary. "
            "First gently receive what the user said, then reveal the speaker's inner state. "
            "Keep it concise, vivid, breathable, and easy to read aloud. "
            "Do not use labels, bullet points, or explanatory framing. "
            "One light pause or one small sensory detail is enough. "
            f"User message: {user_text}"
        )
    return (
        "Please write one short Chinese inner monologue line. "
        "Use Simplified Chinese only, with no English sentence. "
        "It should feel natural, human, close, and slightly breathy when read aloud."
    )


def _postprocess_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    ascii_prefixes = ("reply:", "response:", "assistant:")
    lowered = cleaned.lower()
    for prefix in ascii_prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].lstrip()
            break

    cleaned = cleaned.strip().strip("\"'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _json_request(
    url: str,
    body: Dict[str, object],
    api_key: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, object]:
    session = requests.Session()
    session.trust_env = DOUBAO_USE_SYSTEM_PROXY

    try:
        response: Response = session.post(
            url,
            json=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        payload = exc.response.text if exc.response is not None else ""
        raise RuntimeError(f"Doubao HTTP {exc.response.status_code if exc.response is not None else 'unknown'}: {payload}") from exc
    except RequestException as exc:
        proxy_hint = ""
        if not DOUBAO_USE_SYSTEM_PROXY:
            proxy_hint = " (system proxy ignored)"
        raise RuntimeError(f"Doubao request failed{proxy_hint}: {exc}") from exc
    finally:
        session.close()


def _extract_text(payload: Dict[str, object]) -> str:
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Unexpected Doubao response: {payload}")

    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"Unexpected Doubao choice payload: {payload}")

    message = first.get("message", {})
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    raise RuntimeError(f"Could not extract text from Doubao response: {payload}")


def _needs_chinese_retry(text: str) -> bool:
    if not text.strip():
        return True
    cjk_count = len(_CJK_RE.findall(text))
    latin_word_count = len(_LATIN_WORD_RE.findall(text))
    return cjk_count == 0 or (cjk_count < 4 and latin_word_count >= 2)


def generate_emotion_reply(
    *,
    emotion: str,
    system_mood: str,
    dialogue_style: str,
    valence: float,
    arousal: float,
    user_text: Optional[str],
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
) -> DoubaoReply:
    api_key = DOUBAO_API_KEY
    model_name = model or DOUBAO_MODEL
    if not api_key:
        raise RuntimeError("Doubao API key is not configured. Set DOUBAO_API_KEY or ARK_API_KEY.")
    if not model_name:
        raise RuntimeError("Doubao model is not configured. Set DOUBAO_MODEL or ARK_MODEL.")

    system_prompt = _build_system_prompt(
        emotion=emotion,
        system_mood=system_mood,
        dialogue_style=dialogue_style,
        valence=valence,
        arousal=arousal,
    )
    user_prompt = _build_user_prompt(user_text)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = _json_request(
        f"{DOUBAO_BASE_URL.rstrip('/')}/chat/completions",
        payload,
        api_key,
        timeout=timeout,
    )
    text = _postprocess_text(_extract_text(response))

    if _needs_chinese_retry(text):
        retry_payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not acceptable. Rewrite it entirely in natural spoken "
                        "Simplified Chinese. Keep it to 1-3 short sentences, do not add any English "
                        "sentence, do not explain the rewrite, and keep it smooth for TTS."
                    ),
                },
            ],
            "temperature": max(0.3, min(temperature, 0.8)),
            "max_tokens": max_tokens,
        }
        retry_response = _json_request(
            f"{DOUBAO_BASE_URL.rstrip('/')}/chat/completions",
            retry_payload,
            api_key,
            timeout=timeout,
        )
        retry_text = _postprocess_text(_extract_text(retry_response))
        if retry_text:
            response = retry_response
            text = retry_text

    return DoubaoReply(
        text=text,
        model=str(response.get("model") or model_name),
        response_id=str(response.get("id")) if response.get("id") is not None else None,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
