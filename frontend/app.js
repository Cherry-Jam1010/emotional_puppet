const state = {
  apiBase: "http://127.0.0.1:8000",
  providers: null,
  lastResponse: null,
};

const demoTrials = [
  { subjectId: 1, trialId: 1, text: "你现在想对我说什么？" },
  { subjectId: 4, trialId: 15, text: "如果你有情绪，那会是什么感觉？" },
  { subjectId: 9, trialId: 32, text: "你会怎么安慰一个正在难过的人？" },
  { subjectId: 12, trialId: 51, text: "这一刻你的内心像什么颜色？" },
  { subjectId: 17, trialId: 73, text: "你更想靠近我，还是先安静一下？" },
];

const emotionNames = {
  Anger: "愤怒",
  Disgust: "厌恶",
  Fear: "恐惧",
  Happy: "快乐",
  Neutral: "平静",
  Sad: "悲伤",
  Surprise: "惊讶",
};

const llmProviderLabels = {
  stub: "本地示例回复",
  doubao: "豆包自然回复",
  openai: "兼容接口大模型",
  none: "只做情绪判断",
};

const ttsProviderLabels = {
  pyttsx3: "本地免费语音",
  edge_tts: "Edge 中文语音",
  emotivoice: "EmotiVoice 语音",
  voice_soundboard: "Voice Soundboard",
  stub: "只生成语音计划",
  piper: "本地 Piper 语音",
  openai: "接口语音",
  none: "不生成语音",
};

const animationLabels = {
  calm_idle: "平静待机",
  tense_idle: "紧张待机",
  bright_idle: "轻快待机",
  alert_idle: "警觉待机",
  withdrawn_idle: "低落收束",
  recoiling_idle: "排斥回避",
  startled_focus: "惊讶聚焦",
  reserved_idle: "克制观察",
  buoyant_idle: "愉快跃动",
  slow_idle: "缓慢呼吸",
  reactive_idle: "敏捷反应",
};

const ttsStyleLabels = {
  firm_intense: "克制而有力度",
  cool_distant: "冷静疏离",
  uneasy_breathing: "轻颤而谨慎",
  warm_bright: "温暖明亮",
  natural_conversational: "自然交谈",
  soft_gentle: "低声柔和",
  animated_reactive: "灵动敏捷",
};

const dom = {
  form: document.getElementById("orchestrateForm"),
  apiBaseInput: document.getElementById("apiBaseInput"),
  subjectIdInput: document.getElementById("subjectIdInput"),
  trialIdInput: document.getElementById("trialIdInput"),
  strideInput: document.getElementById("strideInput"),
  llmProviderSelect: document.getElementById("llmProviderSelect"),
  ttsProviderSelect: document.getElementById("ttsProviderSelect"),
  userTextInput: document.getElementById("userTextInput"),
  refreshProvidersBtn: document.getElementById("refreshProvidersBtn"),
  runButton: document.getElementById("runButton"),
  demoButton: document.getElementById("demoButton"),
  openGuideBtn: document.getElementById("openGuideBtn"),
  guideBtn: document.getElementById("guideBtn"),
  closeGuideBtn: document.getElementById("closeGuideBtn"),
  jumpControlBtn: document.getElementById("jumpControlBtn"),
  guideDialog: document.getElementById("guideDialog"),
  providerSummary: document.getElementById("providerSummary"),
  dominantEmotion: document.getElementById("dominantEmotion"),
  animationState: document.getElementById("animationState"),
  valenceValue: document.getElementById("valenceValue"),
  arousalValue: document.getElementById("arousalValue"),
  confidenceValue: document.getElementById("confidenceValue"),
  intensityValue: document.getElementById("intensityValue"),
  stabilityValue: document.getElementById("stabilityValue"),
  breathingValue: document.getElementById("breathingValue"),
  frontendText: document.getElementById("frontendText"),
  systemMoodText: document.getElementById("systemMoodText"),
  responseStyleText: document.getElementById("responseStyleText"),
  promptHintText: document.getElementById("promptHintText"),
  ttsStyleText: document.getElementById("ttsStyleText"),
  probabilityList: document.getElementById("probabilityList"),
  avatarStage: document.getElementById("avatarStage"),
  audioPlayer: document.getElementById("audioPlayer"),
  doubaoStatus: document.getElementById("doubaoStatus"),
  openaiStatus: document.getElementById("openaiStatus"),
  emotivoiceStatus: document.getElementById("emotivoiceStatus"),
  voiceSoundboardStatus: document.getElementById("voiceSoundboardStatus"),
  pyttsx3Status: document.getElementById("pyttsx3Status"),
  edgeTtsStatus: document.getElementById("edgeTtsStatus"),
  piperStatus: document.getElementById("piperStatus"),
};

function formatNumber(value) {
  return Number(value || 0).toFixed(2);
}

function apiUrl(path) {
  return `${state.apiBase.replace(/\/$/, "")}${path}`;
}

function translateEmotion(name) {
  return emotionNames[name] || name || "未知";
}

function translateAnimation(name) {
  return animationLabels[name] || name || "等待中";
}

function translateTtsStyle(name) {
  return ttsStyleLabels[name] || name || "待生成";
}

function updateProviderSummary() {
  const llmLabel = llmProviderLabels[dom.llmProviderSelect.value] || dom.llmProviderSelect.value;
  const ttsLabel = ttsProviderLabels[dom.ttsProviderSelect.value] || dom.ttsProviderSelect.value;
  dom.providerSummary.textContent = `文本模式：${llmLabel} / 语音模式：${ttsLabel}`;
}

function renderProbabilities(items = []) {
  dom.probabilityList.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "probability-item";
    row.innerHTML = `
      <strong>${translateEmotion(item.emotion)}</strong>
      <div class="probability-track">
        <div class="probability-fill" style="width:${Math.max(2, item.probability * 100)}%"></div>
      </div>
      <span>${Math.round(item.probability * 100)}%</span>
    `;
    dom.probabilityList.appendChild(row);
  });
}

function normalizeExpression(expression) {
  if (expression === "surprise") {
    return "surprised";
  }
  return expression || "neutral";
}

function resolveAudioSrc(tts) {
  if (!tts) {
    return "";
  }
  if (tts.audio_base64 && tts.mime_type) {
    return `data:${tts.mime_type};base64,${tts.audio_base64}`;
  }
  if (tts.audio_url) {
    return apiUrl(tts.audio_url);
  }
  return "";
}

function showMessage(text) {
  dom.frontendText.textContent = text;
}

function renderResponse(response) {
  state.lastResponse = response;

  dom.dominantEmotion.textContent = translateEmotion(response.emotion?.dominant_emotion);
  dom.animationState.textContent = translateAnimation(response.avatar?.animation_state);
  dom.valenceValue.textContent = formatNumber(response.emotion?.valence);
  dom.arousalValue.textContent = formatNumber(response.emotion?.arousal);
  dom.confidenceValue.textContent = formatNumber(response.emotion?.confidence);
  dom.intensityValue.textContent = formatNumber(response.emotion?.intensity);
  dom.stabilityValue.textContent = formatNumber(response.emotion?.stability);
  dom.breathingValue.textContent = formatNumber(response.avatar?.breathing_rate);
  dom.systemMoodText.textContent = response.llm?.system_mood || "暂无";
  dom.responseStyleText.textContent = response.llm?.response_style || "暂无";
  dom.promptHintText.textContent = response.llm?.prompt_hint || "暂无";
  dom.ttsStyleText.textContent = translateTtsStyle(response.tts?.voice_style);
  dom.avatarStage.dataset.expression = normalizeExpression(response.avatar?.expression);
  dom.avatarStage.dataset.motionScale = String(response.avatar?.motion_scale ?? 1.0);
  if (typeof window.applyLive2DExpression === "function") {
    window.applyLive2DExpression(response.avatar?.expression);
  }
  if (typeof window.applyLive2DAnimationState === "function") {
    window.applyLive2DAnimationState(response.avatar?.animation_state);
  }
  renderProbabilities(response.emotion?.probabilities || []);
  showMessage(response.frontend?.text || "这次没有生成可展示的文本。");

  const audioSrc = resolveAudioSrc(response.tts);
  if (audioSrc) {
    dom.audioPlayer.src = audioSrc;
    dom.audioPlayer.load();
    if (response.frontend?.should_play_audio) {
      dom.audioPlayer.play().catch(() => {});
    }
  } else {
    dom.audioPlayer.removeAttribute("src");
    dom.audioPlayer.load();
  }
}

function setProviderStatuses(data) {
  state.providers = data;
  dom.doubaoStatus.textContent = data.doubao_ready ? "已就绪" : "未就绪";
  dom.openaiStatus.textContent = data.openai_api_key_present ? "已配置" : "未配置";
  dom.emotivoiceStatus.textContent = data.emotivoice_ready ? "已就绪" : "未就绪";
  dom.voiceSoundboardStatus.textContent = data.voice_soundboard_ready ? "已就绪" : "未就绪";
  dom.pyttsx3Status.textContent = data.local_tts?.pyttsx3_available ? "可用" : "未安装";
  dom.edgeTtsStatus.textContent = data.local_tts?.edge_tts_available
    ? `可用：${data.local_tts.edge_tts_voice || "zh-CN-XiaoxiaoNeural"}`
    : "未安装";
  dom.piperStatus.textContent = data.local_tts?.piper_executable ? "可用" : "未配置";

  if (!data.emotivoice_ready && data.emotivoice_error) {
    showMessage(`EmotiVoice 未就绪：${data.emotivoice_error}`);
  }
}

async function fetchProviders() {
  state.apiBase = dom.apiBaseInput.value.trim();
  updateProviderSummary();

  const response = await fetch(apiUrl("/providers"));
  if (!response.ok) {
    throw new Error("无法读取连接状态");
  }

  const data = await response.json();
  setProviderStatuses(data);
  if (data.doubao_ready) {
    showMessage("后端已连接，豆包文本已就绪，可以直接开始联调。");
  } else {
    showMessage("后端已连接，可以开始联调；如果要用豆包，请先确认当前终端已经配置好豆包环境变量。");
  }
}

async function runOrchestrate(event) {
  event.preventDefault();
  state.apiBase = dom.apiBaseInput.value.trim();
  updateProviderSummary();

  const payload = {
    subject_id: Number(dom.subjectIdInput.value),
    trial_id: Number(dom.trialIdInput.value),
    stride: Number(dom.strideInput.value),
    user_text: dom.userTextInput.value.trim(),
    llm_provider: dom.llmProviderSelect.value,
    tts_provider: dom.ttsProviderSelect.value,
    save_audio_to_file: true,
    include_audio_base64: false,
  };

  dom.runButton.disabled = true;
  dom.runButton.textContent = "联调中...";
  showMessage("正在读取脑电特征并组织回复，请稍等。");

  try {
    const response = await fetch(apiUrl("/orchestrate"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json; charset=utf-8",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "请求失败");
    }

    renderResponse(data);
  } catch (error) {
    showMessage(`联调失败：${error.message}`);
  } finally {
    dom.runButton.disabled = false;
    dom.runButton.textContent = "开始联调";
  }
}

function applyDemoTrial() {
  const picked = demoTrials[Math.floor(Math.random() * demoTrials.length)];
  dom.subjectIdInput.value = picked.subjectId;
  dom.trialIdInput.value = picked.trialId;
  dom.userTextInput.value = picked.text;
  showMessage("已替换成一个示例输入，可以直接点击“开始联调”。");
}

function openGuide() {
  dom.guideDialog.showModal();
}

function closeGuide() {
  dom.guideDialog.close();
}

function maybeOpenGuideOnFirstVisit() {
  const seenKey = "emotion-puppet-guide-seen";
  if (!window.localStorage.getItem(seenKey)) {
    openGuide();
    window.localStorage.setItem(seenKey, "yes");
  }
}

function bindEvents() {
  dom.form.addEventListener("submit", runOrchestrate);
  dom.refreshProvidersBtn.addEventListener("click", () => {
    fetchProviders().catch((error) => {
      showMessage(`能力检测失败：${error.message}`);
    });
  });
  dom.demoButton.addEventListener("click", applyDemoTrial);
  dom.llmProviderSelect.addEventListener("change", updateProviderSummary);
  dom.ttsProviderSelect.addEventListener("change", updateProviderSummary);
  dom.openGuideBtn.addEventListener("click", openGuide);
  dom.guideBtn.addEventListener("click", openGuide);
  dom.closeGuideBtn.addEventListener("click", closeGuide);
  dom.jumpControlBtn.addEventListener("click", () => {
    document.getElementById("controlSection").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  dom.guideDialog.addEventListener("click", (event) => {
    const rect = dom.guideDialog.getBoundingClientRect();
    const clickedInDialog =
      rect.top <= event.clientY &&
      event.clientY <= rect.top + rect.height &&
      rect.left <= event.clientX &&
      event.clientX <= rect.left + rect.width;

    if (!clickedInDialog) {
      closeGuide();
    }
  });
}

async function init() {
  bindEvents();
  updateProviderSummary();
  renderProbabilities([
    { emotion: "Anger", probability: 0.32 },
    { emotion: "Disgust", probability: 0.27 },
    { emotion: "Surprise", probability: 0.15 },
    { emotion: "Neutral", probability: 0.09 },
  ]);
  maybeOpenGuideOnFirstVisit();

  try {
    await fetchProviders();
  } catch (error) {
    showMessage(`还没有连上后端：${error.message}`);
  }
}

init();
