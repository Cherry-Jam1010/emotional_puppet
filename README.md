# 情绪木偶 — Emotion Puppet

基于脑电（EEG）情绪识别驱动的 Live2D 数字生命系统。

给定一段 EEG 特征数据，系统会推断出 7 类情绪，然后驱动 Live2D 角色产生夸张的肢体与表情动画，同步生成自然语言回复和语音。

---

## 目录

1. [核心架构](#核心架构)
2. [关键文献](#关键文献)
3. [功能模块详解](#功能模块详解)
   - [EEG 情绪推理](#eeg-情绪推理)
   - [情感维度投影](#情感维度投影)
   - [Live2D 夸张动画系统](#live2d-夸张动画系统)
   - [LLM 文本生成](#llm-文本生成)
   - [TTS 语音合成](#tts-语音合成)
   - [前端交互界面](#前端交互界面)
4. [快速开始](#快速开始)
5. [API 参考](#api-参考)
6. [文件结构](#文件结构)
7. [配置与环境变量](#配置与环境变量)
8. [常见问题](#常见问题)

---

## 核心架构

```
EEG 特征 (.npy)
    │
    ▼
┌─────────────────────────┐
│  ConvFeatureClassifier   │  ← Conv1D 三层卷积分类器 (seed_vii_models.py)
│  输入: 62ch × 800 samples│  ← 4s 滑动窗口, 200Hz 采样率
│  输出: 7 类情绪概率       │
└──────────┬──────────────┘
           │
           ▼
    ┌──────────────┐
    │ VAD 投影     │  ← valence / arousal / dominance (Russell 1980)
    └──────┬───────┘
           │
     ┌─────┴─────┬──────────┬──────────────┐
     ▼           ▼          ▼              ▼
  Avatar      LLM Prompt  TTS Style    Frontend UI
  动画系统     文本生成     语音参数       面板展示
     │           │          │              │
     ▼           ▼          ▼              ▼
Live2D 模型   Doubao /    Edge TTS /    实时渲染
 Hiyori       OpenAI      EmotiVoice    + 音频播放
```

---

## 关键文献

### 情绪识别基础

| 论文 | 引用 | 说明 |
|------|------|------|
| SEED-IV 数据集 | [Zheng et al., IEEE T-AI 2022](https://ieeexplore.ieee.org/document/9839257) | 45 名受试者，4 类情绪（高兴、悲伤、恐惧、中性），15 通道 EEG，3 个 session |
| **SEED-VII 数据集** | **Zheng et al., Science Bulletin 2025** | **本项目使用的数据集。68 名受试者，7 类情绪（Anger/Disgust/Fear/Happy/Neutral/Sad/Surprise），62 通道 EEG，5 个 session，共约 23 万样本。强调伦理合规与真实世界泛化** |
| EEG 情绪分类综述 | [Liu et al., IEEE T-AI 2023](https://ieeexplore.ieee.org/document/10160120) | 系统综述了基于 EEG 的情绪识别方法 |

### 情感维度模型

| 论文 | 引用 | 说明 |
|------|------|------|
| Circumplex Model of Affect | Russell, *Journal of Personality and Social Psychology*, 1980 | 情绪的两维表征：效价（Valence）+ 唤醒度（Arousal）。本项目 VAD 参数的来源 |
| PAD Emotional State Model | Mehrabian, 1996 | 愉悦度-唤醒度-支配度三维模型，本项目用于 TTS 韵律控制 |

### Live2D / 数字人

| 论文/资源 | 说明 |
|-----------|------|
| Live2D Cubism 4 Manual | Live2D Cubism 4 官方文档，参数体系、动作文件格式（.motion3.json）、表情文件（.exp3.json） |
| pixi-live2d-display | PixiJS + Live2D 的 Web 渲染库，本项目前端使用（cubism4.js） |

### 语音合成

| 论文/资源 | 说明 |
|-----------|------|
| EmotiVoice | [网易伏羲开源 TTS 引擎](https://github.com/netease-youdao/EmotiVoice)，支持情感标签控制 |
| Edge TTS | 微软 Azure 语音服务，WebSocket 接口，中文高质量神经网络声音 |
| Piper | 开源本地神经网络 TTS，RNN-T 模型，可在离线环境运行 |
| VALLE | *Microsoft VALLE: Large-Scale Language Models for Speech Synthesis*, 2023 |

### LLM 驱动的人格生成

| 论文/资源 | 说明 |
|-----------|------|
| Prompting for Emotion | 本项目使用情绪感知提示词策略，将 VAD 参数嵌入 system prompt，引导大模型生成符合情绪状态的回复 |
| Doubao (豆包) | 字节跳动大模型，本项目通过 ARK API 调用 |

---

## 功能模块详解

### EEG 情绪推理

**入口**: `infer_seed_vii_emotion.py`

**模型架构** (`seed_vii_models.py → ConvFeatureClassifier`):

```
输入: (batch, 62, 800)   # 62 通道 EEG, 4 秒窗口, 200Hz
  │
  ├─ Conv1d(62→256, k=3) + BN + GELU + Dropout
  ├─ Conv1d(256→128, k=3) + BN + GELU + Dropout
  ├─ Conv1d(128→64, k=3) + BN + GELU + AdaptiveAvgPool
  └─ Linear(64→7) → softmax → 7 类概率
```

**推理流程**:
1. 从 `.npy` 文件加载 EEG 特征，shape = `(62, N)`，N 为样本数
2. 用滑动窗口切分（默认窗口 4s，步长 2s），每个窗口做 z-score 标准化
3. 逐窗口推理，取所有窗口概率的**均值**作为最终预测
4. 输出 7 类情绪概率 + 主导情绪标签

**7 类情绪标签**:
```
0: Anger    (愤怒)
1: Disgust  (厌恶)
2: Fear     (恐惧)
3: Happy    (快乐)
4: Neutral  (平静)
5: Sad      (悲伤)
6: Surprise (惊讶)
```

---

### 情感维度投影

**入口**: `orchestrator_api.py → _project_affect()`

每种情绪预定义了 VAD 基准值，通过概率加权求和投影到连续维度：

```
valence  = Σ P(e) × valence_base[e]    # 效价：积极/消极
arousal  = Σ P(e) × arousal_base[e]    # 唤醒度：兴奋/平静
intensity = (|valence| + arousal) / 2  # 情绪强度
stability = P(dominant) - P(runner_up)  # 分类置信度
```

**各情绪的 VAD 基准值**（`_project_affect` 中定义）:

| 情绪 | Valence | Arousal | 典型表现 |
|------|---------|---------|---------|
| Anger | -0.80 | 0.85 | 紧绷、力量感 |
| Disgust | -0.75 | 0.55 | 后撤、冷淡 |
| Fear | -0.90 | 0.95 | 发紧、警觉 |
| Happy | +0.95 | 0.70 | 敞开、跃动 |
| Neutral | 0.00 | 0.15 | 平稳、安静 |
| Sad | -0.85 | 0.25 | 柔软、下沉 |
| Surprise | +0.20 | 0.90 | 点亮、聚焦 |

---

### Live2D 夸张动画系统

**入口**: `frontend/live2d-manager.js`

这是本项目的核心视觉模块。包含以下子系统：

#### 夸张表情预设 (`EXAGGERATED_EXPRESSION_PRESETS`)

为每种情绪定义了夸张的参数目标，包含：
- `ParamBrowLY/RY`（眉毛上下）
- `ParamBrowLAngle/RAngle`（眉毛角度）
- `ParamEyeLOpen/ROpen`（眼睛睁大）
- `ParamEyeLSmile/RSmile`（眼微笑）
- `ParamMouthForm/OpenY`（嘴巴形状）
- `ParamCheek`（脸红）
- `ParamAngleX/Y/Z`（头部三轴旋转）

情绪越强烈，参数越极端（如惊讶时嘴巴全开、头部旋转 16°）。

#### 连续动画引擎

在每帧通过 `beforeModelUpdate` 回调驱动：

| 子系统 | 作用 |
|--------|------|
| **夸张表情插值** | 平滑过渡到目标夸张参数，高唤醒情绪用更快的 blendSpeed (0.28) |
| **呼吸动画** | 正弦波驱动头部前后倾斜，唤醒度越高频率越快 |
| **微妙摇摆** | 低频正弦波驱动头部左右/上下微动，模拟自然站姿 |
| **夸张脉冲** | 高唤醒情绪下，眉/眼/嘴参数叠加低频脉冲包络，产生颤动效果 |
| **微表情突发** | 情绪切换时触发短暂的眉毛/嘴巴闪烁，快速衰减 |
| **真实感眨眼** | 非均匀间隔眨眼，高唤醒时眨眼更频繁 |
| **循环动作序列** | 高唤醒情绪在多个 Live2D 动作片段间循环切换，产生丰富肢体运动 |

#### 动画映射表

| 情绪 | 动作序列 | 呼吸频率 | 呼吸幅度 |
|------|---------|---------|---------|
| Happy/Smile | buoyant_idle 循环 | 快 (0.45-1.0) | 大 |
| Angry/Disgust | tense_idle + TapBody | 中等 | 中 |
| Fear/Surprise | alert_idle + TapBody | 非常快 | 极大 |
| Sad | withdrawn_idle 循环 | 慢 | 小 |
| Neutral/Calm | calm_idle | 非常慢 | 非常小 |

#### 角色模型

使用 Live2D Cubism 4 的 **Hiyori** 模型（`live2d/hiyori/runtime/Hiyori.model3.json`），包含：
- 9 个 Idle 动作片段（m01-m10）
- 1 个 TapBody 动作片段
- 内置 EyeBlink 和 LipSync 参数组

---

### LLM 文本生成

**入口**: `orchestrator_api.py`

根据主导情绪和 VAD 参数，从预定义的 **情绪心境映射** 生成 system prompt，引导 LLM 以特定口吻回复：

| 情绪 | 心境 | 对话风格 |
|------|------|---------|
| Happy | "敞开、愉快、愿意靠近" | 温暖明亮、鼓励式 |
| Sad | "柔软、低落、心口有点沉" | 轻柔放慢、低声安抚 |
| Angry | "压着火气，但还在控制" | 紧绷短句、有力度但克制 |
| Fear | "发紧、警觉、小心翼翼" | 迟疑警觉、轻微发紧 |
| Disgust | "有点后撤，也带着怀疑" | 克制冷静、拉开距离感 |
| Surprise | "被点亮了一下，注意力迅速聚拢" | 灵动敏捷、反应感强 |
| Neutral | "平稳、安静、在观察" | 自然专注、可对话 |

支持的 LLM Provider：
- **豆包**（`doubao`）：通过 `ARK_API_KEY` 调用豆包大模型，适合中文自然对话
- **OpenAI 兼容接口**（`openai`）：通过 `OPENAI_API_KEY` 调用任意兼容 API
- **本地 Stub**（`stub`）：使用预定义的情绪模板文本，无需网络

---

### TTS 语音合成

根据情绪和 VAD 参数计算韵律参数，支持多种 Provider：

| Provider | 说明 | 韵律映射 |
|----------|------|---------|
| `edge_tts` | 微软 Edge TTS，在线高质量中文神经网络 | 速率、音高、音量随 VAD 连续调节 |
| `emotivoice` | 网易 EmotiVoice，开源本地情感 TTS | 情感标签直接传入 |
| `voice_soundboard` | 预录音频素材库 | 按情绪选择素材 |
| `pyttsx3` | Python 本地 TTS，跨平台离线 | 速率/音量缩放 |
| `sapi` | Windows SAPI，本地合成 | 速率/音量缩放 |
| `piper` | 本地神经网络 TTS，离线高质量 | 纯文本输入 |
| `openai` | OpenAI Audio API | instruction 驱动情感控制 |
| `stub` | 仅生成参数计划，不合成 | — |

韵律计算公式（`_build_tts_style`）:
```
speaking_rate = clamp(0.90 + arousal × 0.30 + max(valence,0) × 0.05, 0.75, 1.35)
pitch         = clamp(0.95 + valence × 0.12 + (arousal-0.5) × 0.10, 0.75, 1.25)
energy        = clamp(0.75 + arousal × 0.45, 0.55, 1.35)
pause_scale   = clamp(1.15 - arousal × 0.35 + max(-valence,0) × 0.15, 0.75, 1.35)
```

---

### 前端交互界面

**入口**: `frontend/index.html`, `frontend/app.js`

页面分为三个区域：
1. **Hero 区**：Live2D 角色展示 + 情绪指标卡片
2. **控制面板**：输入参数、运行联调
3. **结果面板**：连接状态、情绪分布、联动参数

核心交互流程：
1. 填写受试者编号、试次编号、输入文本
2. 选择文本模式和语音模式
3. 点击"开始联调"
4. 页面展示 EEG 情绪判断结果 → 驱动 Live2D 夸张动画 + 播放语音

---

## 快速开始

### 环境要求

- Python 3.10+
- Node.js（前端可选，用于本地静态文件服务）
- Windows（部分 TTS Provider 依赖 Windows SAPI）

### 1. 准备 EEG 数据

确保目录下存在：
```
SEED-VII_EEG_preprocessed/SEED-VII/EEG_preprocessed/   ← 原始 .mat 文件
emotion_label_and_stimuli_order.xlsx                   ← 标签文件
artifacts/seed_vii/feature_cache/de_LDS/manifest.csv   ← 特征缓存清单
```

如需重新构建特征缓存：
```bash
python seed_vii_data.py \
    --eeg-dir ./SEED-VII_EEG_preprocessed/SEED-VII/EEG_preprocessed \
    --label-file ./emotion_label_and_stimuli_order.xlsx \
    --output-dir ./artifacts/seed_vii
```

### 2. 安装依赖

```bash
pip install -r requirements-py311.txt
```

### 3. 配置环境变量（如需使用在线服务）

```bash
# 豆包大模型
set ARK_API_KEY=your_key_here
set ARK_MODEL=your_model_here

# OpenAI 兼容接口
set OPENAI_API_KEY=your_key_here
set OPENAI_BASE_URL=https://api.openai.com
```

### 4. 启动后端

```bash
cd f:\SEED_EEG_3D
python -m uvicorn orchestrator_api:app --host 127.0.0.1 --port 8000 --reload
```

### 5. 打开前端

在浏览器中访问：
```
http://127.0.0.1:8000
```

### 6. 运行一次联调

1. 点击"刷新连接状态"
2. 填入受试者编号（如 `1`）和试次编号（如 `1`）
3. 选择文本模式（`豆包自然回复`）和语音模式（`Edge 中文语音`）
4. 点击"开始联调"
5. 观察 Live2D 角色的夸张动画、情绪面板和语音播放

---

## API 参考

### `GET /health`

检查服务状态，返回模型加载情况和可用 Provider。

### `GET /providers`

返回各 Provider 的就绪状态。

### `POST /orchestrate`

核心端点，输入 EEG 特征，返回完整的情绪、动画、文本、语音数据。

**请求体**：
```json
{
  "subject_id": 1,
  "trial_id": 1,
  "stride": 4,
  "user_text": "你现在想对我说什么？",
  "llm_provider": "doubao",
  "tts_provider": "edge_tts"
}
```

**响应**（`OrchestrateResponse`）：
```json
{
  "meta": { "protocol_version", "request_id", "model_checkpoint", "window_count" },
  "source": { "feature_path", "subject_id", "trial_id" },
  "emotion": {
    "dominant_emotion": "Happy",
    "probabilities": [{"emotion": "Happy", "probability": 0.82}, ...],
    "valence": 0.71,
    "arousal": 0.54,
    "intensity": 0.62,
    "stability": 0.41
  },
  "avatar": {
    "expression": "smile",
    "animation_state": "buoyant_idle",
    "motion_scale": 0.73,
    "breathing_rate": 0.54
  },
  "llm": { "response_text": "我现在像被暖光轻轻托起来...", "response_style": "温暖明亮..." },
  "tts": { "audio_url": "/media/orchestrator/audio/abc123.mp3", "voice_style": "warm_bright" },
  "frontend": { "text": "我现在像被暖光轻轻托起来...", "should_play_audio": true }
}
```

---

## 文件结构

```
f:\SEED_EEG_3D\
├── orchestrator_api.py          # FastAPI 主入口，所有 Provider 调度
├── infer_seed_vii_emotion.py    # EEG 特征 → 情绪推理
├── seed_vii_models.py            # PyTorch 模型定义 ConvFeatureClassifier
├── seed_vii_data.py              # 数据预处理：MAT → 窗口缓存
│
├── frontend/
│   ├── index.html                # 单页应用入口
│   ├── app.js                    # 前端交互逻辑
│   ├── live2d-manager.js         # Live2D 夸张动画引擎
│   ├── styles.css                # 情绪感知 CSS 动画
│   ├── cubism4.js                # pixi-live2d-display runtime
│   └── live2dcubismcore.min.js   # Live2D Cubism Core SDK
│
├── live2d/
│   └── hiyori/runtime/           # Live2D Hiyori 模型文件
│       ├── Hiyori.model3.json   # 模型描述文件
│       ├── motions/              # 动作片段 (.motion3.json)
│       ├── expressions/          # 表情文件 (.exp3.json) — 预留
│       └── *.moc3               # 核心模型数据
│
├── providers/
│   ├── doubao_natural_llm.py     # 豆包 LLM Provider
│   ├── emotivoice_tts.py         # EmotiVoice TTS Provider
│   └── voice_soundboard_tts.py   # 音频素材库 Provider
│
└── third_party/
    └── EmotiVoice/               # 网易伏羲情感 TTS 引擎
```

---

## 配置与环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ARK_API_KEY` | — | 豆包 ARK API 密钥 |
| `ARK_MODEL` | — | 豆包模型名称 |
| `OPENAI_API_KEY` | — | OpenAI 兼容接口密钥 |
| `OPENAI_BASE_URL` | `https://api.openai.com` | API Base URL |
| `OPENAI_TEXT_MODEL` | `gpt-5` | 文本生成模型 |
| `OPENAI_TTS_MODEL` | `gpt-4o-mini-tts` | 语音合成模型 |
| `EDGE_TTS_ZH_VOICE` | `zh-CN-XiaoxiaoNeural` | 默认中文语音 |
| `EDGE_TTS_HAPPY_VOICE` | `zh-CN-XiaoxiaoNeural` | 快乐情绪专用语音 |
| `EDGE_TTS_ANGER_VOICE` | `zh-CN-YunxiNeural` | 愤怒情绪专用语音 |
| `PIPER_EXE` | `piper` | Piper 可执行文件路径 |
| `PIPER_MODEL` | — | Piper 模型文件路径 |

---

## 常见问题

**Q: Live2D 模型加载失败**  
检查 `frontend/live2dcubismcore.min.js` 是否为 Cubism 4 官方 Core。若模型为 Cubism 5 导出，需替换为 Cubism 5 Web Core。

**Q: 豆包未就绪**  
确认启动后端的终端中已设置 `ARK_API_KEY` 和 `ARK_MODEL`，然后点击"刷新连接状态"。

**Q: EmotiVoice 未就绪**  
确认 `third_party/EmotiVoice/outputs` 配置完整，`WangZeJun/simbert-base-chinese` 权重已下载。参考 `third_party/EmotiVoice/README_小白安装教程.md`。

**Q: EEG 特征文件找不到**  
检查 `artifacts/seed_vii/feature_cache/de_LDS/manifest.csv` 是否存在，可运行 `python seed_vii_data.py` 重建缓存。

**Q: 动画不够夸张**  
高唤醒情绪（Happy、Angry、Fear、Surprise）自动启用夸张模式。如需进一步调整，可修改 `live2d-manager.js` 中的 `EXAGGERATED_EXPRESSION_PRESETS` 参数值，以及 `pulseState.amplitude` 和 `breathState` 的目标值。
