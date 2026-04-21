# 数据与模型文件下载说明

以下文件因体积过大或涉及许可证，不纳入 Git 仓库。请按本说明下载并放置到对应路径后，系统即可正常运行。

---

## 1. SEED-VII EEG 原始数据（约 170 GB）

**下载链接**: [SEED-VII Dataset](https://www.researchgate.net/publication/351557985_SEED_VII_An_EEG_Dataset_for_Inferring_Alogogical_Behavioral_Information) (需注册下载)

**放置路径**:
```
f:\SEED_EEG_3D\
└── SEED-VII_EEG_preprocessed\
    └── SEED-VII\
        └── EEG_preprocessed\
            ├── 1.mat   ← 受试者 1 的 EEG 数据
            ├── 2.mat
            ├── ...
            └── 45.mat  ← 共 45 名受试者
```

> 注意：原始数据下载后，还需运行数据预处理脚本生成特征缓存：
> ```bash
> python seed_vii_data.py \
>     --eeg-dir ./SEED-VII_EEG_preprocessed/SEED-VII/EEG_preprocessed \
>     --label-file ./emotion_label_and_stimuli_order.xlsx \
>     --output-dir ./artifacts/seed_vii
> ```
> 预处理后生成 `artifacts/seed_vii/feature_cache/de_LDS/manifest.csv` 及各受试者的 `.npy` 特征文件（约 83 MB）。

---

## 2. Kokoro TTS 模型（约 100 MB，可选）

仅在使用 **Voice Soundboard** TTS Provider 时需要。

**下载链接**: [kokoro-onnx releases](https://github.com/remsky/kokoro-onnx/releases)

**放置路径**:
```
f:\SEED_EEG_3D\
└── models\
    ├── kokoro-v1.0.onnx   ← 主模型文件
    └── voices-v1.0.bin    ← 声音音色包
```

---

## 3. EmotiVoice 模型权重（需注册 + 申请，约 2 GB，可选）

仅在使用 **EmotiVoice** TTS Provider 时需要。

**申请步骤**:
1. 访问 [EmotiVoice 官网](https://垂音.com/) 或 GitHub 页面注册账号
2. 提交模型权重下载申请（需审核）
3. 获批后下载并解压到指定目录

**放置路径**:
```
f:\SEED_EEG_3D\
└── third_party\
    └── EmotiVoice\
        ├── outputs\
        │   └── .mv\            ← 模型权重文件
        └── WangZeJun\
            └── simbert-base-chinese\  ← SimBERT 语义模型
```

> **注意**: EmotiVoice 需同意其[用户协议](https://github.com/netease-youdao/EmotiVoice/blob/master/EmotiVoice_UserAgreement_最终用户协议.pdf)。

---

## 4. SEED-VII 标签文件

**必需文件**: `emotion_label_and_stimuli_order.xlsx`

此文件应随数据集一并下载，放置在项目根目录：

```
f:\SEED_EEG_3D\
└── emotion_label_and_stimuli_order.xlsx
```

---

## 快速验证

所有文件就位后，运行以下命令验证：

```bash
python -c "from orchestrator_api import app; print('OK')"
```

启动后端：
```bash
python -m uvicorn orchestrator_api:app --host 127.0.0.1 --port 8000
```

然后访问 http://127.0.0.1:8000 查看前端是否正常加载。
