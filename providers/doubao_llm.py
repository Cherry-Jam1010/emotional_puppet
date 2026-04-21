"""
豆包（Doubao / 火山引擎）大模型接入模块

支持两种调用方式：
1. 中转 API（推荐）：使用 laozhang.ai 等兼容 OpenAI 接口格式的中转服务，
   配置简单，仅需一个 API Key，无需 AK/SK 签名。
2. 火山引擎直连：使用火山引擎官方 API，需要 Access Key + Secret Key，
   通过 Signature V4 签名认证。

使用方法（二选一）：
  方式一（中转 API）：
    豆包_API_KEY=your_relay_key
    豆包_API_BASE=https://api.laozhang.ai/v1   # 或其他中转地址
    豆包_MODEL=doubao-pro-32k                 # 根据中转平台支持的模型名填写

  方式二（火山引擎直连）：
    火山_AK=your_access_key
    火山_SK=your_secret_secret
    火山_REGION=cn-beijing
    火山_SERVICE=cvf_platform
    豆包_MODEL=doubao-pro-32k

环境变量说明：
  豆包_API_KEY      中转 API 的 Key（方式一）
  豆包_API_BASE     中转 API 的 base URL，末尾不要加斜杠（方式一）
  豆包_MODEL        使用的模型名称，如 doubao-pro-32k、doubao-lite 等
  豆包_API_KEY      中转 Key（方式一，同 OPENAI_API_KEY 用法）
  火山_AK           火山引擎 Access Key（方式二）
  火山_SK           火山引擎 Secret Key（方式二）
  火山_REGION        区域，默认 cn-beijing（方式二）
  火山_SERVICE      服务名，默认 cvf_platform（方式二）
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from volcengine.auth.SignerV4 import SignerV4

import requests

# ---------------------------------------------------------------------------
# 外部依赖提示（仅在直连模式时导入）
# ---------------------------------------------------------------------------

_VOLCE_ENGINE_AVAILABLE = False
try:
    from volcengine.auth.SignerV4 import SignerV4

    _VOLCE_ENGINE_AVAILABLE = True
except ImportError:
    SignerV4 = None  # type: ignore


# ---------------------------------------------------------------------------
# 配置读取
# ---------------------------------------------------------------------------

DOUBAO_API_KEY: Optional[str] = os.getenv("豆包_API_KEY") or os.getenv("DOUBAO_API_KEY")
DOUBAO_API_BASE: Optional[str] = os.getenv("豆包_API_BASE") or os.getenv("DOUBAO_API_BASE")
DOUBAO_MODEL: Optional[str] = os.getenv("豆包_MODEL") or os.getenv("DOUBAO_MODEL") or "doubao-pro-32k"

# 火山引擎直连参数
VOLC_AK: Optional[str] = os.getenv("火山_AK") or os.getenv("VOLC_AK")
VOLC_SK: Optional[str] = os.getenv("火山_SK") or os.getenv("VOLC_SK")
VOLC_REGION: str = os.getenv("火山_REGION") or os.getenv("VOLC_REGION") or "cn-beijing"
VOLC_SERVICE: str = os.getenv("火山_SERVICE") or os.getenv("VOLC_SERVICE") or "cvf_platform"

# 默认参数
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT = 30


# ---------------------------------------------------------------------------
# 对话消息类型
# ---------------------------------------------------------------------------

Message = Dict[str, str]


def _build_messages(
    system_prompt: str,
    user_prompt: str,
    history: Optional[List[Message]] = None,
) -> List[Dict[str, Any]]:
    """构建发送给豆包的 messages 列表。"""
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    return messages


# ---------------------------------------------------------------------------
# 中转 API 模式（OpenAI 兼容格式）
# ---------------------------------------------------------------------------

def _call_via_relay(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    通过兼容 OpenAI 接口格式的中转服务调用豆包。
    中转服务如：laozhang.ai、openai-proxy 等。
    """
    if not DOUBAO_API_KEY:
        raise RuntimeError("未设置豆包 API Key，请设置环境变量 豆包_API_KEY 或 DOUBAO_API_KEY")
    if not DOUBAO_API_BASE:
        raise RuntimeError("未设置豆包 API Base URL，请设置环境变量 豆包_API_BASE")

    url = f"{DOUBAO_API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _extract_text_from_relay_response(data: Dict[str, Any]) -> str:
    """从中转 API 响应中提取文本。"""
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"豆包返回格式异常，缺少 choices：{data}")

    first = choices[0]
    # OpenAI 中转格式
    content = first.get("message", {}).get("content")
    if content:
        return content.strip()

    # 某些中转平台用 delta 格式（流式场景返回的）
    content = first.get("delta", {}).get("content", "")
    if content:
        return content.strip()

    raise RuntimeError(f"无法从豆包响应中提取文本：{data}")


# ---------------------------------------------------------------------------
# 火山引擎直连模式（AK/SK 签名）
# ---------------------------------------------------------------------------

def _call_via_volcengine(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    通过火山引擎官方 API 直接调用豆包。
    需要安装 volcengine SDK：pip install volcengine
    """
    if not _VOLCE_ENGINE_AVAILABLE:
        raise RuntimeError(
            "火山引擎 SDK 未安装，请运行：pip install volcengine\n"
            "或者切换到中转 API 模式（推荐），配置 豆包_API_KEY 和 豆包_API_BASE"
        )
    if not VOLC_AK or not VOLC_SK:
        raise RuntimeError("未设置火山引擎 AK/SK，请设置环境变量 火山_AK 和 火山_SK")

    import datetime

    host = "ark.cn-beijing.volces.com"
    url = f"https://{host}/api/v3/chat/completions"

    # 构建请求体（兼容 OpenAI 格式）
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Host": host,
    }

    # 使用 Signature V4 签名
    signer = SignerV4(VOLC_AK, VOLC_SK, VOLC_REGION, VOLC_SERVICE)
    body_str = json.dumps(payload)

    signed_headers = signer.sign_request("POST", url, headers, body_str)
    signed_headers["Content-Type"] = "application/json"

    response = requests.post(url, headers=signed_headers, data=body_str, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _extract_text_from_volc_response(data: Dict[str, Any]) -> str:
    """从火山引擎 API 响应中提取文本。"""
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"火山引擎返回格式异常：{data}")

    content = choices[0].get("message", {}).get("content", "")
    if content:
        return content.strip()

    raise RuntimeError(f"无法从火山引擎响应中提取文本：{data}")


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def chat(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    history: Optional[List[Message]] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """
    调用豆包大模型生成文本。

    参数：
        system_prompt  系统提示词（设定角色、语气等）
        user_prompt    用户输入的文本
        model          模型名称，默认从环境变量 豆包_MODEL 读取
        temperature    温度参数，控制随机性，默认 0.7
        max_tokens     最大生成 token 数，默认 512
        history        历史对话消息列表，用于多轮对话
        timeout        请求超时秒数，默认 30

    返回：
        模型生成的文本字符串

    示例：
        text = chat(
            system_prompt="你是一个温暖有同理心的倾听者。",
            user_prompt="我今天心情很差。",
            temperature=0.8,
        )
    """
    model = model or DOUBAO_MODEL
    messages = _build_messages(system_prompt, user_prompt, history)

    # 根据是否配置了 AK/SK 决定使用哪种调用方式
    if VOLC_AK and VOLC_SK and not DOUBAO_API_KEY:
        data = _call_via_volcengine(messages, model, temperature, max_tokens, timeout)
        return _extract_text_from_volc_response(data)
    elif DOUBAO_API_KEY and DOUBAO_API_BASE:
        data = _call_via_relay(messages, model, temperature, max_tokens, timeout)
        return _extract_text_from_relay_response(data)
    elif DOUBAO_API_KEY:
        raise RuntimeError(
            "配置了 豆包_API_KEY 但没有配置 豆包_API_BASE，"
            "请同时设置环境变量 豆包_API_BASE（如 https://api.laozhang.ai/v1）"
        )
    else:
        raise RuntimeError(
            "未配置豆包 API，请选择以下方式之一：\n"
            "  方式一（推荐）：设置 豆包_API_KEY 和 豆包_API_BASE\n"
            "  方式二：设置 火山_AK 和 火山_SK\n"
            "详见本文件开头的使用说明。"
        )


def is_available() -> bool:
    """检查豆包 API 是否已配置（可用于前端显示状态）。"""
    return bool(
        (DOUBAO_API_KEY and DOUBAO_API_BASE)
        or (VOLC_AK and VOLC_SK)
    )


# ---------------------------------------------------------------------------
# 快速测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("豆包 API 接入测试")
    print("=" * 60)

    if is_available():
        print("[OK] 豆包 API 已配置")
        print(f"    模型：{DOUBAO_MODEL}")
        if DOUBAO_API_BASE:
            print(f"    API Base：{DOUBAO_API_BASE}")
        else:
            print(f"    模式：火山引擎直连（Region: {VOLC_REGION}, Service: {VOLC_SERVICE}）")

        print()
        print(">>> 发送测试请求...")
        try:
            result = chat(
                system_prompt="你是一个温暖有同理心的倾听者，回复简洁自然。",
                user_prompt="我今天心情很差，感觉什么事情都提不起劲。",
                temperature=0.8,
                max_tokens=200,
            )
            print(f"\n[模型回复]\n{result}")
        except Exception as e:
            print(f"\n[错误] {e}")
    else:
        print("[未配置] 请先设置以下环境变量之一：")
        print("  方式一（推荐）：export 豆包_API_KEY=xxx  豆包_API_BASE=https://api.laozhang.ai/v1")
        print("  方式二：export 火山_AK=xxx  火山_SK=xxx")
