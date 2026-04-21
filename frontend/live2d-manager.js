/**
 * Live2D Cubism integration for the Emotion Puppet page.
 *
 * Model file: /live2d/hiyori/runtime/Hiyori.model3.json
 * Runtime global: PIXI.live2d.Live2DModel
 */

(function () {
  const canvas = document.getElementById("live2dCanvas");
  if (!canvas) return;

  const MODEL_URL = window.LIVE2D_MODEL_URL || "/live2d/hiyori/runtime/Hiyori.model3.json";
  const WAIT_INTERVAL_MS = 200;
  const WAIT_MAX_ATTEMPTS = 60;

  const EXPRESSION_MAP = {
    neutral: "exp_01",
    calm: "exp_01",
    happy: "exp_04",
    smile: "exp_04",
    sad: "exp_03",
    angry: "exp_02",
    anger: "exp_02",
    disgust: "exp_02",
    fear: "exp_05",
    surprised: "exp_05",
    surprise: "exp_05",
  };

  const SYNTHETIC_EXPRESSION_PRESETS = {
    neutral: {},
    calm: {
      ParamBrowLY: 0.08,
      ParamBrowRY: 0.08,
      ParamBrowLAngle: -0.08,
      ParamBrowRAngle: 0.08,
      ParamMouthForm: 0.12,
      ParamMouthOpenY: 0.03,
      ParamEyeLSmile: 0.08,
      ParamEyeRSmile: 0.08,
    },
    happy: {
      ParamCheek: 0.65,
      ParamEyeLSmile: 0.75,
      ParamEyeRSmile: 0.75,
      ParamBrowLY: 0.2,
      ParamBrowRY: 0.2,
      ParamMouthForm: 0.9,
      ParamMouthOpenY: 0.28,
    },
    smile: {
      ParamCheek: 0.45,
      ParamEyeLSmile: 0.55,
      ParamEyeRSmile: 0.55,
      ParamMouthForm: 0.7,
      ParamMouthOpenY: 0.18,
    },
    sad: {
      ParamBrowLY: -0.55,
      ParamBrowRY: -0.55,
      ParamBrowLAngle: -0.35,
      ParamBrowRAngle: 0.35,
      ParamEyeLOpen: 0.72,
      ParamEyeROpen: 0.72,
      ParamMouthForm: -0.7,
      ParamMouthOpenY: 0.08,
    },
    angry: {
      ParamBrowLY: -0.35,
      ParamBrowRY: -0.35,
      ParamBrowLAngle: 0.75,
      ParamBrowRAngle: -0.75,
      ParamBrowLForm: -0.45,
      ParamBrowRForm: -0.45,
      ParamEyeLOpen: 0.82,
      ParamEyeROpen: 0.82,
      ParamMouthForm: -0.5,
      ParamMouthOpenY: 0.14,
    },
    anger: {
      ParamBrowLY: -0.35,
      ParamBrowRY: -0.35,
      ParamBrowLAngle: 0.75,
      ParamBrowRAngle: -0.75,
      ParamBrowLForm: -0.45,
      ParamBrowRForm: -0.45,
      ParamEyeLOpen: 0.82,
      ParamEyeROpen: 0.82,
      ParamMouthForm: -0.5,
      ParamMouthOpenY: 0.14,
    },
    disgust: {
      ParamBrowLY: -0.15,
      ParamBrowRY: -0.15,
      ParamBrowLAngle: 0.35,
      ParamBrowRAngle: -0.35,
      ParamMouthForm: -0.9,
      ParamMouthOpenY: 0.05,
      ParamAngleY: -4,
    },
    fear: {
      ParamBrowLY: 0.45,
      ParamBrowRY: 0.45,
      ParamEyeLOpen: 1,
      ParamEyeROpen: 1,
      ParamMouthForm: -0.18,
      ParamMouthOpenY: 0.42,
      ParamAngleY: 6,
    },
    surprised: {
      ParamBrowLY: 0.55,
      ParamBrowRY: 0.55,
      ParamEyeLOpen: 1,
      ParamEyeROpen: 1,
      ParamMouthForm: 0.05,
      ParamMouthOpenY: 0.62,
      ParamAngleY: 8,
    },
    surprise: {
      ParamBrowLY: 0.55,
      ParamBrowRY: 0.55,
      ParamEyeLOpen: 1,
      ParamEyeROpen: 1,
      ParamMouthForm: 0.05,
      ParamMouthOpenY: 0.62,
      ParamAngleY: 8,
    },
  };

  // Exaggerated expression presets — multiplied for dramatic effect
  const EXAGGERATED_EXPRESSION_PRESETS = {
    happy: {
      ParamCheek: 0.95,
      ParamEyeLSmile: 1.0,
      ParamEyeRSmile: 1.0,
      ParamBrowLY: 0.45,
      ParamBrowRY: 0.45,
      ParamMouthForm: 1.0,
      ParamMouthOpenY: 0.55,
      ParamAngleX: 12,
      ParamAngleY: 6,
    },
    smile: {
      ParamCheek: 0.75,
      ParamEyeLSmile: 0.9,
      ParamEyeRSmile: 0.9,
      ParamMouthForm: 0.95,
      ParamMouthOpenY: 0.35,
      ParamAngleX: 6,
      ParamAngleY: 3,
    },
    sad: {
      ParamBrowLY: -0.85,
      ParamBrowRY: -0.85,
      ParamBrowLAngle: -0.65,
      ParamBrowRAngle: 0.65,
      ParamEyeLOpen: 1.0,
      ParamEyeROpen: 1.0,
      ParamMouthForm: -1.0,
      ParamMouthOpenY: 0.05,
      ParamAngleX: -18,
      ParamAngleY: -4,
    },
    angry: {
      ParamBrowLY: -0.75,
      ParamBrowRY: -0.75,
      ParamBrowLAngle: 1.0,
      ParamBrowRAngle: -1.0,
      ParamBrowLForm: -0.85,
      ParamBrowRForm: -0.85,
      ParamEyeLOpen: 1.0,
      ParamEyeROpen: 1.0,
      ParamMouthForm: -0.85,
      ParamMouthOpenY: 0.22,
      ParamAngleX: 8,
      ParamAngleY: -8,
    },
    anger: {
      ParamBrowLY: -0.75,
      ParamBrowRY: -0.75,
      ParamBrowLAngle: 1.0,
      ParamBrowRAngle: -1.0,
      ParamBrowLForm: -0.85,
      ParamBrowRForm: -0.85,
      ParamEyeLOpen: 1.0,
      ParamEyeROpen: 1.0,
      ParamMouthForm: -0.85,
      ParamMouthOpenY: 0.22,
      ParamAngleX: 8,
      ParamAngleY: -8,
    },
    disgust: {
      ParamBrowLY: -0.45,
      ParamBrowRY: -0.45,
      ParamBrowLAngle: 0.65,
      ParamBrowRAngle: -0.65,
      ParamMouthForm: -1.0,
      ParamMouthOpenY: 0.12,
      ParamAngleY: -12,
      ParamAngleX: -5,
    },
    fear: {
      ParamBrowLY: 0.85,
      ParamBrowRY: 0.85,
      ParamEyeLOpen: 1.0,
      ParamEyeROpen: 1.0,
      ParamMouthForm: -0.45,
      ParamMouthOpenY: 0.72,
      ParamAngleY: 14,
      ParamAngleX: 6,
    },
    surprised: {
      ParamBrowLY: 1.0,
      ParamBrowRY: 1.0,
      ParamEyeLOpen: 1.0,
      ParamEyeROpen: 1.0,
      ParamMouthForm: 0.25,
      ParamMouthOpenY: 1.0,
      ParamAngleY: 16,
      ParamAngleX: 10,
    },
    surprise: {
      ParamBrowLY: 1.0,
      ParamBrowRY: 1.0,
      ParamEyeLOpen: 1.0,
      ParamEyeROpen: 1.0,
      ParamMouthForm: 0.25,
      ParamMouthOpenY: 1.0,
      ParamAngleY: 16,
      ParamAngleX: 10,
    },
    neutral: {},
    calm: {
      ParamBrowLY: 0.08,
      ParamBrowRY: 0.08,
      ParamBrowLAngle: -0.08,
      ParamBrowRAngle: 0.08,
      ParamMouthForm: 0.12,
      ParamMouthOpenY: 0.03,
      ParamEyeLSmile: 0.08,
      ParamEyeRSmile: 0.08,
    },
  };

  const SYNTHETIC_PARAMETER_IDS = [
    "ParamAngleX",
    "ParamAngleY",
    "ParamAngleZ",
    "ParamCheek",
    "ParamEyeLOpen",
    "ParamEyeROpen",
    "ParamEyeLSmile",
    "ParamEyeRSmile",
    "ParamBrowLY",
    "ParamBrowRY",
    "ParamBrowLAngle",
    "ParamBrowRAngle",
    "ParamBrowLForm",
    "ParamBrowRForm",
    "ParamMouthForm",
    "ParamMouthOpenY",
    "ParamBodyAngleX",
    "ParamBodyAngleY",
    "ParamBodyAngleZ",
  ];

  // Motion sequences for exaggerated (high-arousal) mode — cycles through Idle entries
  const EXAGGERATED_MOTION_MAP = {
    happy:     [{ group: "Idle", index: 2 }, { group: "Idle", index: 6 }],
    smile:     [{ group: "Idle", index: 2 }, { group: "Idle", index: 6 }],
    angry:     [{ group: "Idle", index: 1 }, { group: "Idle", index: 3 }],
    anger:     [{ group: "Idle", index: 1 }, { group: "Idle", index: 3 }],
    fear:      [{ group: "Idle", index: 3 }, { group: "Idle", index: 4 }],
    surprised: [{ group: "TapBody", index: 0 }, { group: "Idle", index: 8 }],
    surprise:  [{ group: "TapBody", index: 0 }, { group: "Idle", index: 8 }],
    disgust:   [{ group: "Idle", index: 4 }, { group: "Idle", index: 5 }],
    neutral:   [{ group: "Idle", index: 0 }, { group: "Idle", index: 7 }],
    calm:      [{ group: "Idle", index: 0 }, { group: "Idle", index: 7 }],
    sad:       [{ group: "Idle", index: 7 }, { group: "Idle", index: 4 }],
  };

  const ANIMATION_MOTION_MAP = {
    calm_idle: { group: "Idle", index: 0 },
    tense_idle: { group: "Idle", index: 1 },
    bright_idle: { group: "Idle", index: 2 },
    alert_idle: { group: "Idle", index: 3 },
    withdrawn_idle: { group: "Idle", index: 4 },
    reserved_idle: { group: "Idle", index: 5 },
    buoyant_idle: { group: "Idle", index: 6 },
    slow_idle: { group: "Idle", index: 7 },
    reactive_idle: { group: "Idle", index: 8 },
    recoiling_idle: {
      group: "TapBody",
      index: 0,
      fallback: { group: "Idle", index: 4, delayMs: 1600 },
    },
    startled_focus: {
      group: "TapBody",
      index: 0,
      fallback: { group: "Idle", index: 3, delayMs: 1600 },
    },
  };

  window.live2dReady = false;
  window.live2dFailed = false;
  window.live2dError = null;

  let live2dApp = null;
  let model = null;
  let currentExpressionName = null;
  let currentEmotionKey = "neutral";
  let syntheticExpressionState = {};
  let currentAnimationState = null;
  let currentAnimEmotionKey = null; // track which emotion the current animation is for
  let motionFallbackTimer = null;

  // Continuous animation state
  let lastMotionSwitchTime = 0;
  let currentMotionIndex = 0;
  let currentMotionSequence = null;
  let baseEmotionKey = "neutral";
  let isExaggeratedMode = false;
  let motionScale = 1.0;

  // Blink state
  let blinkState = {
    timer: 0,
    duration: 0,
    phase: "open",
    nextBlinkAt: 2000,
    baseOpenness: 1.0,
    blinkSpeed: 0.12,
  };

  // Micro-expression burst state
  let microBurstState = {
    timer: 0,
    targetParams: {},
    burstActive: false,
    decayFactor: 0,
  };

  // Breathing state
  let breathState = {
    t: 0,
    currentRate: 0.25,
    targetRate: 0.25,
    currentAmplitude: 0.08,
    targetAmplitude: 0.08,
  };

  // Subtle sway state
  let swayState = {
    t: 0,
    speed: 0.4,
    amplitudeX: 0,
    amplitudeY: 0,
  };

  // Exaggerated pulse state for extreme emotions
  let pulseState = {
    t: 0,
    frequency: 1.2,
    amplitude: 0,
    phase: 0,
  };

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function getLive2DModelCtor() {
    return window.PIXI && window.PIXI.live2d && window.PIXI.live2d.Live2DModel;
  }

  function hasNativeExpressions() {
    const expressionManager = model
      && model.internalModel
      && model.internalModel.motionManager
      && model.internalModel.motionManager.expressionManager;
    const definitions = expressionManager && expressionManager.definitions;
    return Array.isArray(definitions) && definitions.length > 0;
  }

  function normalizeEmotionKey(emotion) {
    const key = emotion ? emotion.toLowerCase() : "neutral";
    return SYNTHETIC_EXPRESSION_PRESETS[key] ? key : "neutral";
  }

  // Check if emotion has high arousal (more dramatic motion)
  function isHighArousalEmotion(key) {
    return ["happy", "smile", "angry", "anger", "fear", "surprised", "surprise", "disgust"].indexOf(key) !== -1;
  }

  // Blend two emotion presets with a factor
  function blendedTarget(emotionKey, secondaryKey, blend) {
    const base = EXAGGERATED_EXPRESSION_PRESETS[emotionKey] || {};
    const secondary = EXAGGERATED_EXPRESSION_PRESETS[secondaryKey] || {};
    const result = {};
    const allKeys = {};
    Object.keys(base).forEach(function (k) { allKeys[k] = true; });
    Object.keys(secondary).forEach(function (k) { allKeys[k] = true; });
    Object.keys(allKeys).forEach(function (id) {
      const a = id in base ? base[id] : 0;
      const b = id in secondary ? secondary[id] : 0;
      result[id] = lerp(a, b, blend);
    });
    return result;
  }

  // Schedule next blink using a realistic distribution
  function scheduleNextBlink() {
    const baseInterval = 2500 + Math.random() * 3000;
    const arousalBonus = (motionScale - 0.5) * 800;
    blinkState.nextBlinkAt = blinkState.timer + baseInterval - arousalBonus;
  }

  function updateBlink(dt) {
    blinkState.timer += dt;
    const b = blinkState;

    if (b.phase === "open" && b.timer >= b.nextBlinkAt) {
      b.phase = "closing";
      b.duration = 0;
    } else if (b.phase === "closing") {
      b.duration += dt;
      const progress = Math.min(b.duration / (b.blinkSpeed * 1000), 1.0);
      b.baseOpenness = 1.0 - progress;
      if (progress >= 1.0) {
        b.phase = "closed";
        b.duration = 0;
      }
    } else if (b.phase === "closed") {
      b.duration += dt;
      if (b.duration >= 80 + Math.random() * 60) {
        b.phase = "opening";
        b.duration = 0;
      }
    } else if (b.phase === "opening") {
      b.duration += dt;
      const progress = Math.min(b.duration / (b.blinkSpeed * 1000 * 1.3), 1.0);
      b.baseOpenness = progress;
      if (progress >= 1.0) {
        b.phase = "open";
        b.baseOpenness = 1.0;
        scheduleNextBlink();
      }
    }
  }

  // Trigger a micro-expression burst (quick eyebrow/mouth flicker)
  function triggerMicroBurst(emotionKey) {
    const bursts = {
      happy: { ParamEyeLSmile: 0.4, ParamEyeRSmile: 0.4, ParamMouthOpenY: 0.2 },
      sad: { ParamBrowLY: -0.2, ParamMouthForm: -0.3 },
      angry: { ParamBrowLAngle: 0.3, ParamBrowRAngle: -0.3, ParamMouthOpenY: 0.15 },
      fear: { ParamEyeLOpen: 0.5, ParamEyeROpen: 0.5, ParamMouthOpenY: 0.3 },
      surprised: { ParamMouthOpenY: 0.5, ParamBrowLY: 0.3 },
      disgust: { ParamMouthForm: -0.4, ParamAngleY: -3 },
    };
    const preset = bursts[emotionKey];
    if (!preset) return;
    microBurstState.targetParams = preset;
    microBurstState.burstActive = true;
    microBurstState.timer = 0;
    microBurstState.decayFactor = 0.92;
  }

  function updateMicroBurst(dt) {
    if (!microBurstState.burstActive) return;
    microBurstState.timer += dt;
    if (microBurstState.timer > 500) {
      microBurstState.burstActive = false;
    }
  }

  function getMicroBurstValue(paramId) {
    if (!microBurstState.burstActive) return 0;
    return (microBurstState.targetParams[paramId] || 0) * Math.pow(microBurstState.decayFactor, microBurstState.timer / 50);
  }

  function updateBreathing(dt) {
    breathState.t += dt;
    breathState.currentRate = lerp(breathState.currentRate, breathState.targetRate, 0.02);
    breathState.currentAmplitude = lerp(breathState.currentAmplitude, breathState.targetAmplitude, 0.02);
    const breathPhase = breathState.t * breathState.currentRate * 0.001 * Math.PI * 2;
    const swayPhase = swayState.t * swayState.speed * 0.001;
    swayState.t += dt;
    return {
      breath: Math.sin(breathPhase) * breathState.currentAmplitude,
      swayX: Math.sin(swayPhase * 0.7) * swayState.amplitudeX,
      swayY: Math.sin(swayPhase * 0.5) * swayState.amplitudeY,
    };
  }

  function updatePulse(dt) {
    pulseState.t += dt;
    const envelope = pulseState.amplitude * (0.6 + 0.4 * Math.sin(pulseState.t * 0.001 * pulseState.frequency * Math.PI * 2 + pulseState.phase));
    return envelope;
  }

  const BLINK_IDS = ["ParamEyeLOpen", "ParamEyeROpen"];

  function attachSyntheticExpressionDriver() {
    if (!model || !model.internalModel || typeof model.internalModel.on !== "function") {
      return;
    }

    let lastFrameTime = 0;
    scheduleNextBlink();

    model.internalModel.on("beforeModelUpdate", function () {
      const now = Date.now();
      const dt = lastFrameTime > 0 ? Math.min(now - lastFrameTime, 100) : 16;
      lastFrameTime = now;

      const coreModel = model.internalModel && model.internalModel.coreModel;
      if (!coreModel) {
        return;
      }

      // --- Update continuous animation sub-systems ---
      updateBlink(dt);
      updateMicroBurst(dt);

      // --- Determine base target from current emotion ---
      const rawTarget = blendedTarget(baseEmotionKey, "neutral", 0);
      const highArousal = isHighArousalEmotion(baseEmotionKey);
      const blendSpeed = highArousal ? 0.28 : 0.12;

      // --- Get breathing/sway ---
      const { breath, swayX, swayY } = updateBreathing(dt);

      // --- Get pulse for exaggerated mode ---
      const pulse = updatePulse(dt);

      // --- Set all parameters ---
      SYNTHETIC_PARAMETER_IDS.forEach(function (id) {
        let goal = Object.prototype.hasOwnProperty.call(rawTarget, id) ? rawTarget[id] : 0;

        // Breathing: affects head and body angles
        if (id === "ParamAngleX") {
          goal += breath * 4;
        }
        if (id === "ParamBodyAngleX") {
          goal += breath * 2.5;
        }
        if (id === "ParamAngleY") {
          goal += swayY;
        }
        if (id === "ParamAngleZ") {
          goal += swayX;
        }

        // Exaggerated pulse: amplify for extreme emotions
        if (isExaggeratedMode && (id === "ParamBrowLY" || id === "ParamBrowRY" || id === "ParamMouthOpenY" || id === "ParamEyeLSmile" || id === "ParamEyeRSmile")) {
          goal += pulse * 0.4;
        }

        // Blink: override eye openness with blink state
        if (BLINK_IDS.indexOf(id) !== -1) {
          const baseOpenness = Object.prototype.hasOwnProperty.call(rawTarget, id) ? rawTarget[id] : 1.0;
          goal = baseOpenness * blinkState.baseOpenness;
        }

        // Micro-expression burst
        goal += getMicroBurstValue(id);

        // Smooth interpolation with emotion-aware speed
        const previous = Object.prototype.hasOwnProperty.call(syntheticExpressionState, id)
          ? syntheticExpressionState[id]
          : 0;
        const next = previous + (goal - previous) * blendSpeed;
        syntheticExpressionState[id] = next;
        coreModel.setParameterValueById(id, next, 1);
      });
    });
  }

  function showFallbackAvatar() {
    const stage = canvas.parentElement;
    if (!stage) return;

    if (stage.querySelector(".live2d-fallback")) {
      return;
    }

    const fallback = document.createElement("div");
    fallback.className = "live2d-fallback";
    fallback.innerHTML = [
      '<svg viewBox="0 0 200 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%;">',
      "<defs>",
      '<radialGradient id="fg" cx="40%" cy="35%" r="60%">',
      '<stop offset="0%" stop-color="#c8a4e8"/>',
      '<stop offset="100%" stop-color="#7b5fa8"/>',
      "</radialGradient>",
      '<filter id="glow">',
      '<feGaussianBlur stdDeviation="3" result="blur"/>',
      '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>',
      "</filter>",
      "</defs>",
      '<ellipse cx="100" cy="70" rx="72" ry="65" fill="#9b7fc8" opacity="0.6" filter="url(#glow)"/>',
      '<ellipse cx="100" cy="90" rx="60" ry="65" fill="url(#fg)" stroke="#6b4f8a" stroke-width="1.5"/>',
      '<ellipse cx="75" cy="88" rx="14" ry="15" fill="#2a1a40"/>',
      '<ellipse cx="125" cy="88" rx="14" ry="15" fill="#2a1a40"/>',
      '<ellipse cx="77" cy="86" rx="6" ry="7" fill="#fff" opacity="0.8"/>',
      '<ellipse cx="127" cy="86" rx="6" ry="7" fill="#fff" opacity="0.8"/>',
      '<path d="M60,72 Q75,68 88,73" stroke="#5a3f7a" stroke-width="2" fill="none" stroke-linecap="round"/>',
      '<path d="M112,73 Q125,68 140,72" stroke="#5a3f7a" stroke-width="2" fill="none" stroke-linecap="round"/>',
      '<path d="M80,115 Q100,122 120,115" stroke="#5a3f7a" stroke-width="2" fill="none" stroke-linecap="round"/>',
      '<ellipse cx="100" cy="200" rx="55" ry="45" fill="#9b7fc8" opacity="0.5"/>',
      '<text x="100" y="200" text-anchor="middle" dominant-baseline="middle" fill="#c8a4e8" font-size="11" font-family="sans-serif">Hiyori Momose</text>',
      '<text x="100" y="214" text-anchor="middle" dominant-baseline="middle" fill="#9b7fc8" font-size="9" font-family="sans-serif">Live2D load failed</text>',
      "</svg>",
    ].join("");

    stage.appendChild(fallback);
    console.warn("Live2D: showing fallback avatar");
  }

  function fitCanvas() {
    const stage = canvas.parentElement;
    if (!stage) return;

    const width = stage.clientWidth || 540;
    const height = Math.round(width * 1.25);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";

    if (live2dApp) {
      live2dApp.renderer.resize(width, height);
      if (model) {
        model.x = live2dApp.screen.width / 2;
      }
    }
  }

  function startMotion(groupName) {
    if (!model) return;

    try {
      if (typeof model.motion === "function") {
        model.motion(groupName);
        return;
      }

      const motionManager = model.internalModel && model.internalModel.motionManager;
      if (motionManager && typeof motionManager.startMotion === "function") {
        motionManager.startMotion(groupName);
      }
    } catch (error) {
      console.debug("Live2D: motion unavailable", groupName, error && error.message);
    }
  }

  function startMappedMotion(groupName, index) {
    if (!model) return;

    try {
      if (typeof model.motion === "function") {
        model.motion(groupName, index, 3);
        return;
      }

      const motionManager = model.internalModel && model.internalModel.motionManager;
      if (motionManager && typeof motionManager.startMotion === "function") {
        motionManager.startMotion(groupName, index, 3);
      }
    } catch (error) {
      console.debug("Live2D: mapped motion unavailable", groupName, index, error && error.message);
    }
  }

  // Advance to the next motion in the exaggerated sequence
  function advanceExaggeratedMotion() {
    if (!model || !currentMotionSequence || currentMotionSequence.length === 0) return;
    const entry = currentMotionSequence[currentMotionIndex];
    startMappedMotion(entry.group, entry.index);

    const nextIndex = (currentMotionIndex + 1) % currentMotionSequence.length;
    const nextEntry = currentMotionSequence[nextIndex];
    const holdDuration = isHighArousalEmotion(baseEmotionKey)
      ? 600 + Math.random() * 600
      : 1200 + Math.random() * 1200;

    if (motionFallbackTimer) {
      window.clearTimeout(motionFallbackTimer);
    }
    motionFallbackTimer = window.setTimeout(function () {
      currentMotionIndex = nextIndex;
      advanceExaggeratedMotion();
    }, holdDuration);
  }

  function applyAnimationState(animationState) {
    if (!model || !window.live2dReady) return;

    const key = animationState || "calm_idle";
    // Re-apply if emotion changed even if animation_state label is the same
    const emotionChanged = currentAnimEmotionKey !== baseEmotionKey;
    if (!emotionChanged && currentAnimationState === key) {
      return;
    }
    currentAnimationState = key;
    currentAnimEmotionKey = baseEmotionKey;

    if (motionFallbackTimer) {
      window.clearTimeout(motionFallbackTimer);
      motionFallbackTimer = null;
    }

    // --- Configure exaggerated vs normal mode ---
    isExaggeratedMode = isHighArousalEmotion(baseEmotionKey);

    if (isExaggeratedMode) {
      // Cycle through the exaggerated motion sequence for this emotion
      currentMotionSequence = EXAGGERATED_MOTION_MAP[baseEmotionKey] || EXAGGERATED_MOTION_MAP.neutral;
      currentMotionIndex = 0;

      // Trigger micro-burst for dramatic emotion transitions
      triggerMicroBurst(baseEmotionKey);

      // Set exaggerated pulse parameters
      pulseState.amplitude = motionScale * 0.15;
      pulseState.frequency = 1.0 + motionScale * 0.8;
      pulseState.phase = Math.random() * Math.PI * 2;

      // Fast breathing for high arousal
      breathState.targetRate = 0.45 + motionScale * 0.55;
      breathState.targetAmplitude = 0.10 + motionScale * 0.18;
      swayState.amplitudeX = 2.5 + motionScale * 3.5;
      swayState.amplitudeY = 1.5 + motionScale * 2.5;
      swayState.speed = 0.5 + motionScale * 0.5;

      advanceExaggeratedMotion();
      console.info("Live2D: exaggerated animation applied", key, "motionScale=" + motionScale.toFixed(2));

    } else {
      // Calm/normal mode: gentle idle
      currentMotionSequence = null;
      pulseState.amplitude = 0;

      const target = ANIMATION_MOTION_MAP[key] || ANIMATION_MOTION_MAP.calm_idle;
      startMappedMotion(target.group, target.index);

      if (target.fallback) {
        motionFallbackTimer = window.setTimeout(function () {
          startMappedMotion(target.fallback.group, target.fallback.index);
        }, target.fallback.delayMs || 1500);
      }

      // Gentle breathing for calm emotions
      breathState.targetRate = 0.20;
      breathState.targetAmplitude = 0.06;
      swayState.amplitudeX = 0.5;
      swayState.amplitudeY = 0.3;
      swayState.speed = 0.35;

      console.info("Live2D: animation applied", key, target.group, target.index);
    }
  }

  function applyExpression(emotion) {
    if (!model || !window.live2dReady) return;

    const key = normalizeEmotionKey(emotion);
    currentEmotionKey = key;
    baseEmotionKey = key;

    const stage = canvas && canvas.parentElement;
    if (stage) {
      const scaleAttr = stage.dataset.motionScale;
      if (scaleAttr !== undefined) {
        motionScale = parseFloat(scaleAttr) || 1.0;
      }
    }

    const expressionName = EXPRESSION_MAP[key] || "exp_01";
    const syntheticOnly = !hasNativeExpressions();

    if (currentExpressionName === expressionName && !syntheticOnly) {
      return;
    }
    currentExpressionName = expressionName;

    try {
      if (syntheticOnly) {
        console.info("Live2D: synthetic expression applied", key, "scale=" + motionScale.toFixed(2));
        return;
      }

      if (typeof model.expression === "function") {
        model.expression(expressionName);
        console.info("Live2D: expression applied", expressionName, key);
        return;
      }

      const expressionManager = model.internalModel
        && model.internalModel.motionManager
        && model.internalModel.motionManager.expressionManager;

      if (expressionManager && typeof expressionManager.setExpression === "function") {
        expressionManager.setExpression(expressionName);
        console.info("Live2D: expression applied", expressionName, key);
      }
    } catch (error) {
      console.warn("Live2D: expression failed", expressionName, error && error.message);
    }
  }

  async function initLive2D() {
    fitCanvas();
    window.addEventListener("resize", fitCanvas);

    try {
      const Live2DModel = getLive2DModelCtor();
      if (!Live2DModel) {
        throw new Error("PIXI.live2d.Live2DModel is unavailable");
      }

      live2dApp = new PIXI.Application({
        view: canvas,
        width: canvas.clientWidth || 540,
        height: Math.round((canvas.clientWidth || 540) * 1.25),
        autoDensity: true,
        resolution: window.devicePixelRatio || 1,
        autoStart: true,
        backgroundAlpha: 0,
        clearBeforeRender: true,
        antialias: true,
      });

      model = await Live2DModel.from(MODEL_URL);
      live2dApp.stage.addChild(model);

      model.anchor.set(0.5, 0.5);
      model.x = live2dApp.screen.width / 2;
      model.y = live2dApp.screen.height * 0.58;

      const safeWidth = Math.max(model.width, 1);
      const scale = (live2dApp.screen.width / safeWidth) * 0.92;
      model.scale.set(scale);

      startMotion("Idle");
      attachSyntheticExpressionDriver();

      window.live2dReady = true;
      window.live2dFailed = false;
      window.live2dError = null;
      console.info("Live2D: model loaded");

      window.setTimeout(function () {
        applyExpression("neutral");
        applyAnimationState("calm_idle");
      }, 600);
    } catch (error) {
      window.live2dReady = false;
      window.live2dFailed = true;
      window.live2dError = error && error.message ? error.message : String(error);
      if (
        /Failed to CubismMoc\.create\(\)/.test(window.live2dError)
        || /moc3 ver:\[6\]/i.test(String(error))
      ) {
        window.live2dError =
          "Current Live2D Core is too old for this model. Replace frontend/live2dcubismcore.min.js with an official Cubism 5 Web Core, or re-export the model as Cubism 4.2 compatible.";
      }
      console.error("Live2D: initialization failed", window.live2dError, error);
      showFallbackAvatar();
    }
  }

  window.applyLive2DExpression = function (emotion) {
    if (window.live2dFailed || !window.live2dReady) {
      return;
    }

    applyExpression(emotion);
  };

  window.applyLive2DAnimationState = function (animationState) {
    if (window.live2dFailed || !window.live2dReady) {
      return;
    }

    applyAnimationState(animationState);
  };

  function waitForLibs(callback, attempts) {
    const count = attempts || 0;
    const ready = (
      typeof window.PIXI !== "undefined"
      && typeof window.Live2DCubismCore !== "undefined"
      && !!getLive2DModelCtor()
    );

    if (ready) {
      callback();
      return;
    }

    if (count > WAIT_MAX_ATTEMPTS) {
      window.live2dFailed = true;
      window.live2dError = "Live2D runtime scripts did not finish loading";
      console.error("Live2D: SDK scripts did not load in time");
      showFallbackAvatar();
      return;
    }

    window.setTimeout(function () {
      waitForLibs(callback, count + 1);
    }, WAIT_INTERVAL_MS);
  }

  waitForLibs(initLive2D, 0);
})();
