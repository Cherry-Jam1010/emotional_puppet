from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


EMOTIVOICE_ROOT = Path(
    os.getenv("EMOTIVOICE_ROOT", str(Path(__file__).resolve().parent.parent / "third_party" / "EmotiVoice"))
).resolve()

# 优先查找 EmotiVoice conda 虚拟环境，再查找系统 PATH 中的 python
def _resolve_python() -> str:
    import shutil as _sh

    env_path = os.getenv("EMOTIVOICE_PYTHON")
    if env_path:
        return env_path
    candidates = [
        r"C:\Users\Think\anaconda3\envs\EmotiVoice\python.exe",
        r"C:\ProgramData\Anaconda3\envs\EmotiVoice\python.exe",
        r"C:\Users\Think\miniconda3\envs\EmotiVoice\python.exe",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    found = _sh.which("python")
    if found:
        return found
    return candidates[0]


EMOTIVOICE_PYTHON = _resolve_python()
EMOTIVOICE_SPEAKER = os.getenv("EMOTIVOICE_SPEAKER", "8051")


@dataclass
class EmotiVoiceResult:
    provider: str
    model: str
    voice: str
    audio_format: str
    instructions: str
    audio_path: Optional[str]
    audio_base64: Optional[str]
    mime_type: str


def _helper_script() -> Path:
    return Path(__file__).resolve().parent / "emotivoice_cli.py"


def _required_paths() -> list[Path]:
    return [
        EMOTIVOICE_ROOT / "inference_am_vocoder_joint.py",
        EMOTIVOICE_ROOT / "outputs" / "prompt_tts_open_source_joint" / "ckpt" / "g_00140000",
        EMOTIVOICE_ROOT / "outputs" / "style_encoder" / "ckpt" / "checkpoint_163431",
        EMOTIVOICE_ROOT / "WangZeJun" / "simbert-base-chinese" / "config.json",
        EMOTIVOICE_ROOT / "WangZeJun" / "simbert-base-chinese" / "vocab.txt",
        EMOTIVOICE_ROOT / "WangZeJun" / "simbert-base-chinese" / "pytorch_model.bin",
        _helper_script(),
    ]


def _emotion_to_prompt(emotion: str) -> str:
    mapping = {
        "Anger": "语气紧一点，但仍然克制，有压住的火气。",
        "Disgust": "语气冷一些，带一点后撤和疏离感。",
        "Fear": "语气更轻，带一点发紧和迟疑，小心一点说。",
        "Happy": "语气明亮温暖，轻盈一点，像在靠近对方。",
        "Neutral": "语气自然平稳，安静、清晰、可对话。",
        "Sad": "语气轻柔、慢一点，带一点低落和安静的安抚感。",
        "Surprise": "语气更灵动，反应更快一点，像刚被点亮。",
    }
    return mapping.get(emotion, "语气自然、平稳、可对话。")


def is_available() -> bool:
    return availability_error() is None


def availability_error() -> Optional[str]:
    python_path = Path(EMOTIVOICE_PYTHON)
    if not python_path.exists():
        return (
            f"EmotiVoice Python 解释器不存在：{EMOTIVOICE_PYTHON}\n"
            "  → 请设置环境变量 EMOTIVOICE_PYTHON 指向正确的 python.exe 路径，"
            "或在 EmotiVoice conda 环境中运行本项目。\n"
            "  → 模型下载命令：\n"
            "      cd third_party/EmotiVoice\n"
            "      git lfs install\n"
            "      git lfs clone https://huggingface.co/WangZeJun/simbert-base-chinese WangZeJun/simbert-base-chinese\n"
            "      git clone https://www.modelscope.cn/syq163/outputs.git\n"
            "      # outputs 目录下的 prompt_tts_open_source_joint 和 style_encoder 即为模型权重"
        )
    missing = [str(path) for path in _required_paths() if not path.exists()]
    if missing:
        missing_names = "\n    ".join(missing)
        return (
            f"EmotiVoice 缺少必需文件，共 {len(missing)} 项：\n    {missing_names}\n"
            "  → 模型下载命令：\n"
            "      cd third_party/EmotiVoice\n"
            "      git lfs install\n"
            "      git lfs clone https://huggingface.co/WangZeJun/simbert-base-chinese WangZeJun/simbert-base-chinese\n"
            "      git clone https://www.modelscope.cn/syq163/outputs.git\n"
        )
    return None


def synthesize(
    *,
    text: str,
    emotion: str,
    request_id: str,
    output_dir: Path,
    include_audio_base64: bool,
    save_audio_to_file: bool,
    preferred_voice: Optional[str] = None,
) -> EmotiVoiceResult:
    error = availability_error()
    if error:
        raise RuntimeError(error)

    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{request_id}.wav"
    input_json = output_dir / f"{request_id}.emotivoice.json"
    payload = {
        "text": text,
        "prompt": _emotion_to_prompt(emotion),
        "speaker": preferred_voice or EMOTIVOICE_SPEAKER,
    }
    input_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    try:
        completed = subprocess.run(
            [
                EMOTIVOICE_PYTHON,
                str(_helper_script()),
                "--repo-root",
                str(EMOTIVOICE_ROOT),
                "--input-json",
                str(input_json),
                "--output",
                str(final_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(EMOTIVOICE_ROOT),
            check=False,
            encoding="utf-8",
            errors="ignore",
        )
    finally:
        try:
            input_json.unlink()
        except FileNotFoundError:
            pass

    if completed.returncode != 0:
        stderr_text = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"EmotiVoice 合成失败：{stderr_text}")
    if not final_path.exists():
        raise RuntimeError("EmotiVoice 没有生成音频文件")

    audio_bytes = final_path.read_bytes()
    audio_path = str(final_path.resolve()) if save_audio_to_file else None
    if not save_audio_to_file:
        final_path.unlink(missing_ok=True)

    return EmotiVoiceResult(
        provider="emotivoice",
        model="EmotiVoice",
        voice=payload["speaker"],
        audio_format="wav",
        instructions=f"EmotiVoice speaker={payload['speaker']}, prompt={payload['prompt']}",
        audio_path=audio_path,
        audio_base64=base64.b64encode(audio_bytes).decode("ascii") if include_audio_base64 else None,
        mime_type="audio/wav",
    )
