import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState
} from "react";
import { createRoot } from "react-dom/client";
import { Camera, FileVideo, Pause, Play, RotateCcw, ScanLine, Sparkles, Wand2, Waves } from "lucide-react";
import { DrawingUtils, FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import "./styles.css";

const wasmBase = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";
const labHistoryKey = "dmpa.lab.history.v1";
const mediaPipeSettingsKey = "dmpa.mediapipe.settings.v1";

const modelUrls = {
  lite: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
  full: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
  heavy: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
};

const defaultMediaPipeSettings = {
  modelVariant: "lite",
  delegate: "GPU",
  numPoses: 1,
  minPoseDetectionConfidence: 0.45,
  minPosePresenceConfidence: 0.45,
  minTrackingConfidence: 0.45,
  outputSegmentationMasks: false,
  scanFps: 5,
  landmarkSet: "core13",
  regions: {
    arms: true,
    torso: true,
    legs: true
  }
};

const landmarkNames = {
  0: "нос",
  11: "левое плечо",
  12: "правое плечо",
  13: "левый локоть",
  14: "правый локоть",
  15: "левое запястье",
  16: "правое запястье",
  23: "левое бедро",
  24: "правое бедро",
  25: "левое колено",
  26: "правое колено",
  27: "левая стопа",
  28: "правая стопа"
};

const coreLandmarkIds = Object.keys(landmarkNames).map(Number);

const poseLandmarkCatalog = [
  "нос",
  "левый глаз внутри",
  "левый глаз",
  "левый глаз снаружи",
  "правый глаз внутри",
  "правый глаз",
  "правый глаз снаружи",
  "левое ухо",
  "правое ухо",
  "левый угол рта",
  "правый угол рта",
  "левое плечо",
  "правое плечо",
  "левый локоть",
  "правый локоть",
  "левое запястье",
  "правое запястье",
  "левый мизинец",
  "правый мизинец",
  "левый указательный",
  "правый указательный",
  "левый большой палец",
  "правый большой палец",
  "левое бедро",
  "правое бедро",
  "левое колено",
  "правое колено",
  "левая лодыжка",
  "правая лодыжка",
  "левая пятка",
  "правая пятка",
  "носок левой стопы",
  "носок правой стопы"
];

const angleSpecs = [
  { id: "leftElbow", title: "Левый локоть", points: [11, 13, 15], weight: 1, region: "arms" },
  { id: "rightElbow", title: "Правый локоть", points: [12, 14, 16], weight: 1, region: "arms" },
  { id: "leftShoulder", title: "Левое плечо", points: [13, 11, 23], weight: 1.1, region: "arms" },
  { id: "rightShoulder", title: "Правое плечо", points: [14, 12, 24], weight: 1.1, region: "arms" },
  { id: "leftHip", title: "Левое бедро", points: [11, 23, 25], weight: 1.1, region: "torso" },
  { id: "rightHip", title: "Правое бедро", points: [12, 24, 26], weight: 1.1, region: "torso" },
  { id: "leftKnee", title: "Левое колено", points: [23, 25, 27], weight: 1.2, region: "legs" },
  { id: "rightKnee", title: "Правое колено", points: [24, 26, 28], weight: 1.2, region: "legs" },
  { id: "torsoTilt", title: "Наклон корпуса", points: [11, 23, 24], weight: 1.2, region: "torso" },
  { id: "shoulderLine", title: "Линия плеч", points: [23, 11, 12], weight: 0.8, region: "torso" }
];

function activeAngleSpecs(regions = defaultMediaPipeSettings.regions) {
  const selected = angleSpecs.filter((spec) => regions?.[spec.region]);
  return selected.length ? selected : angleSpecs;
}

function angle(a, b, c) {
  if (!a || !b || !c) return null;
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  if (!mag) return null;
  const value = Math.acos(Math.max(-1, Math.min(1, dot / mag))) * (180 / Math.PI);
  return Number(value.toFixed(1));
}

function poseAngles(landmarks, specs = angleSpecs) {
  if (!landmarks?.length) return {};
  return Object.fromEntries(
    specs.map((spec) => [spec.id, angle(landmarks[spec.points[0]], landmarks[spec.points[1]], landmarks[spec.points[2]])])
  );
}

function containRect(containerWidth, containerHeight, mediaWidth, mediaHeight) {
  if (!containerWidth || !containerHeight || !mediaWidth || !mediaHeight) {
    return { x: 0, y: 0, width: containerWidth, height: containerHeight };
  }
  const containerRatio = containerWidth / containerHeight;
  const mediaRatio = mediaWidth / mediaHeight;
  if (mediaRatio > containerRatio) {
    const width = containerWidth;
    const height = width / mediaRatio;
    return { x: 0, y: (containerHeight - height) / 2, width, height };
  }
  const height = containerHeight;
  const width = height * mediaRatio;
  return { x: (containerWidth - width) / 2, y: 0, width, height };
}

function projectLandmarksToCanvas(landmarks, rect, canvasWidth, canvasHeight) {
  return landmarks.map((landmark) => ({
    ...landmark,
    x: (rect.x + landmark.x * rect.width) / canvasWidth,
    y: (rect.y + landmark.y * rect.height) / canvasHeight
  }));
}

function comparePoseFrames(left, right, regions = defaultMediaPipeSettings.regions) {
  if (!left?.landmarks?.length || !right?.landmarks?.length) {
    return {
      ready: false,
      score: 0,
      verdict: "Сначала отсканируйте скелет в левом и правом видео.",
      rows: [],
      suggestions: []
    };
  }

  const specs = activeAngleSpecs(regions);
  const leftAngles = poseAngles(left.landmarks, specs);
  const rightAngles = poseAngles(right.landmarks, specs);
  let weightedScore = 0;
  let totalWeight = 0;

  const rows = specs
    .map((spec) => {
      const leftValue = leftAngles[spec.id];
      const rightValue = rightAngles[spec.id];
      if (leftValue === null || rightValue === null) return null;
      const diff = Math.abs(leftValue - rightValue);
      const score = Math.max(0, 100 - diff * 2.15);
      weightedScore += score * spec.weight;
      totalWeight += spec.weight;
      return {
        ...spec,
        leftValue,
        rightValue,
        diff: Number(diff.toFixed(1)),
        score: Math.round(score)
      };
    })
    .filter(Boolean);

  const centerDiff = normalizedCenterDifference(left.landmarks, right.landmarks);
  const centerPenalty = Math.min(22, centerDiff * 120);
  const score = Math.max(0, Math.min(100, Math.round(weightedScore / Math.max(1, totalWeight) - centerPenalty)));
  const worst = [...rows].sort((a, b) => b.diff - a.diff).slice(0, 4);
  const suggestions = worst.map((row) => {
    const side = row.leftValue > row.rightValue ? "у эталона угол больше" : "в правом видео угол больше";
    return `${row.title}: разница ${row.diff}°, ${side}.`;
  });

  return {
    ready: true,
    score,
    rows,
    suggestions,
    verdict: verdictForScore(score, suggestions)
  };
}

function compareScans(leftScan, rightScan, sync, regions = defaultMediaPipeSettings.regions) {
  if (!leftScan?.frames?.length || !rightScan?.frames?.length) return comparePoseFrames(null, null, regions);

  const offset = sync?.ready ? sync.offsetSeconds : 0;
  const usableFrames = leftScan.frames
    .filter((frame) => frame.landmarks?.length)
    .map((leftFrame) => {
      const rightFrame = nearestFrame(rightScan.frames, leftFrame.time + offset);
      if (!rightFrame?.landmarks?.length) return null;
      return {
        leftTime: leftFrame.time,
        rightTime: rightFrame.time,
        comparison: comparePoseFrames(leftFrame, rightFrame, regions)
      };
    })
    .filter(Boolean);

  if (!usableFrames.length) return comparePoseFrames(null, null, regions);

  const totals = new Map();
  let scoreSum = 0;
  let bestScore = -Infinity;
  let worstScore = Infinity;
  let worstMoment = null;
  for (const item of usableFrames) {
    const frameScore = item.comparison.score;
    scoreSum += frameScore;
    if (frameScore > bestScore) bestScore = frameScore;
    if (frameScore < worstScore) {
      worstScore = frameScore;
      worstMoment = item;
    }
    for (const row of item.comparison.rows) {
      const current = totals.get(row.id) || { ...row, diff: 0, leftValue: 0, rightValue: 0, count: 0 };
      current.diff += row.diff;
      current.leftValue += row.leftValue;
      current.rightValue += row.rightValue;
      current.count += 1;
      totals.set(row.id, current);
    }
  }

  const rows = Array.from(totals.values()).map((row) => ({
    ...row,
    diff: Number((row.diff / row.count).toFixed(1)),
    leftValue: Number((row.leftValue / row.count).toFixed(1)),
    rightValue: Number((row.rightValue / row.count).toFixed(1)),
    score: Math.max(0, Math.round(100 - (row.diff / row.count) * 2.15))
  }));

  const score = Math.round(scoreSum / usableFrames.length);
  const worst = [...rows].sort((a, b) => b.diff - a.diff).slice(0, 4);
  const suggestions = worst.map((row) => {
    const side = row.leftValue > row.rightValue ? "у эталона угол больше" : "в правом видео угол больше";
    return `${row.title}: средняя разница ${row.diff}°, ${side}.`;
  });

  return {
    ready: true,
    score,
    rows,
    suggestions,
    framesCompared: usableFrames.length,
    bestScore: Math.round(bestScore),
    worstScore: Math.round(worstScore),
    durationCompared: Number((usableFrames.at(-1).leftTime - usableFrames[0].leftTime).toFixed(1)),
    worstMoment: worstMoment
      ? {
          leftTime: worstMoment.leftTime,
          rightTime: worstMoment.rightTime,
          score: worstMoment.comparison.score
        }
      : null,
    verdict: verdictForScore(score, suggestions)
  };
}

function nearestFrame(frames, time) {
  if (!frames?.length || !Number.isFinite(time)) return null;
  let best = null;
  let bestDiff = Infinity;
  for (const frame of frames) {
    const diff = Math.abs(frame.time - time);
    if (diff < bestDiff) {
      best = frame;
      bestDiff = diff;
    }
  }
  return bestDiff <= 0.35 ? best : null;
}

function normalizedCenterDifference(leftLandmarks, rightLandmarks) {
  const ids = [11, 12, 23, 24, 25, 26];
  const leftCenter = averagePoint(ids.map((id) => leftLandmarks[id]).filter(Boolean));
  const rightCenter = averagePoint(ids.map((id) => rightLandmarks[id]).filter(Boolean));
  if (!leftCenter || !rightCenter) return 0;
  return Math.hypot(leftCenter.x - rightCenter.x, leftCenter.y - rightCenter.y);
}

function averagePoint(points) {
  if (!points.length) return null;
  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length
  };
}

function verdictForScore(score, suggestions) {
  if (score >= 86) return "Правое видео очень близко повторяет эталон. Отличия небольшие.";
  if (score >= 70) return `Правое видео хорошо попадает в эталон, но есть зоны для подгонки: ${suggestions.slice(0, 2).join(" ")}`;
  if (score >= 52) return `Среднее совпадение с эталоном. В первую очередь исправить: ${suggestions.slice(0, 3).join(" ")}`;
  return `Правое видео заметно выбивается из эталона. Начните с корпуса, коленей и плеч: ${suggestions.slice(0, 3).join(" ")}`;
}

function pendingFullRunComparison() {
  return {
    ready: false,
    score: 0,
    rows: [],
    suggestions: [],
    framesCompared: 0,
    verdict: "Скелеты сохранены. Запустите синхронный анализ всего видео, чтобы получить итоговую статистику, а не оценку одного кадра."
  };
}

function normalizeMediaPipeSettings(settings) {
  return {
    ...defaultMediaPipeSettings,
    ...(settings || {}),
    modelVariant: modelUrls[settings?.modelVariant] ? settings.modelVariant : defaultMediaPipeSettings.modelVariant,
    delegate: settings?.delegate === "CPU" ? "CPU" : "GPU",
    numPoses: Math.max(1, Math.min(4, Number(settings?.numPoses || defaultMediaPipeSettings.numPoses))),
    scanFps: Math.max(1, Math.min(15, Number(settings?.scanFps || defaultMediaPipeSettings.scanFps))),
    outputSegmentationMasks: Boolean(settings?.outputSegmentationMasks),
    landmarkSet: settings?.landmarkSet === "full33" ? "full33" : "core13",
    regions: {
      ...defaultMediaPipeSettings.regions,
      ...(settings?.regions || {})
    }
  };
}

function loadMediaPipeSettings() {
  try {
    return normalizeMediaPipeSettings(JSON.parse(localStorage.getItem(mediaPipeSettingsKey) || "null"));
  } catch {
    return defaultMediaPipeSettings;
  }
}

function detectorOptions(settings, runningMode) {
  return {
    baseOptions: {
      modelAssetPath: modelUrls[settings.modelVariant],
      delegate: settings.delegate
    },
    runningMode,
    numPoses: settings.numPoses,
    minPoseDetectionConfidence: settings.minPoseDetectionConfidence,
    minPosePresenceConfidence: settings.minPosePresenceConfidence,
    minTrackingConfidence: settings.minTrackingConfidence,
    outputSegmentationMasks: settings.outputSegmentationMasks
  };
}

function usePoseLandmarker(settings) {
  const [landmarker, setLandmarker] = useState(null);
  const [scanLandmarker, setScanLandmarker] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function boot() {
      setLandmarker(null);
      setScanLandmarker(null);
      setError("");
      try {
        const fileset = await FilesetResolver.forVisionTasks(wasmBase);
        const detector = await PoseLandmarker.createFromOptions(fileset, detectorOptions(settings, "VIDEO"));
        const scanDetector = await PoseLandmarker.createFromOptions(fileset, detectorOptions(settings, "IMAGE"));
        if (!cancelled) {
          setLandmarker(detector);
          setScanLandmarker(scanDetector);
        }
      } catch (err) {
        setError("Не удалось загрузить MediaPipe. Проверьте интернет или CDN-доступ.");
        console.error(err);
      }
    }
    boot();
    return () => {
      cancelled = true;
      landmarker?.close?.();
      scanLandmarker?.close?.();
    };
  }, [settings]);

  return { landmarker, scanLandmarker, error };
}

function waitForVideoEvent(video, eventName) {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error(`Timeout waiting for ${eventName}`));
    }, 5000);
    function cleanup() {
      window.clearTimeout(timer);
      video.removeEventListener(eventName, resolveOnce);
      video.removeEventListener("error", rejectOnce);
    }
    function resolveOnce() {
      cleanup();
      resolve();
    }
    function rejectOnce() {
      cleanup();
      reject(new Error("Video error"));
    }
    video.addEventListener(eventName, resolveOnce, { once: true });
    video.addEventListener("error", rejectOnce, { once: true });
  });
}

async function scanVideoPose(video, scanLandmarker, onProgress, range = null, settings = defaultMediaPipeSettings) {
  if (!video || !scanLandmarker || !Number.isFinite(video.duration)) {
    throw new Error("Видео еще не готово для сканирования.");
  }

  const wasPaused = video.paused;
  const previousTime = video.currentTime;
  video.pause();

  const duration = video.duration;
  const scanStart = Math.max(0, Math.min(range?.start ?? 0, duration));
  const scanEnd = Math.max(scanStart, Math.min(range?.end ?? duration, duration));
  const scanDuration = Math.max(0.01, scanEnd - scanStart);
  const step = 1 / Math.max(1, Number(settings.scanFps || defaultMediaPipeSettings.scanFps));
  const specs = activeAngleSpecs(settings.regions);
  const frames = [];

  for (let time = scanStart; time <= scanEnd; time += step) {
    const targetTime = Math.min(time, Math.max(0, duration - 0.02));
    if (Math.abs(video.currentTime - targetTime) > 0.01) {
      video.currentTime = targetTime;
      await waitForVideoEvent(video, "seeked");
    }
    const result = scanLandmarker.detect(video);
    const landmarks = result.landmarks?.[0] || [];
    frames.push({
      time: Number(video.currentTime.toFixed(3)),
      landmarks,
      angles: poseAngles(landmarks, specs),
      confidence: landmarks.length ? averageVisibility(landmarks) : 0
    });
    onProgress?.(Math.min(100, Math.round(((time - scanStart) / scanDuration) * 100)));
  }

  video.currentTime = previousTime;
  if (!wasPaused) await video.play();
  onProgress?.(100);

  const trackedFrames = frames.filter((frame) => frame.landmarks.length);
  return {
    duration,
    range: { start: scanStart, end: scanEnd },
    settings: {
      scanFps: settings.scanFps,
      landmarkSet: settings.landmarkSet,
      regions: settings.regions,
      activeAngles: specs.map((spec) => spec.id)
    },
    frames,
    trackedFrames: trackedFrames.length,
    averageConfidence: trackedFrames.length
      ? trackedFrames.reduce((sum, frame) => sum + frame.confidence, 0) / trackedFrames.length
      : 0,
    scannedAt: new Date().toISOString()
  };
}

async function analyzeAudioFile(file) {
  if (!file) throw new Error("Для аудио-синхронизации нужен видеофайл.");
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContextClass();
  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  const channel = audioBuffer.getChannelData(0);
  const sampleRate = audioBuffer.sampleRate;
  const waveform = buildRmsSeries(channel, 900);
  const hopSeconds = 0.05;
  const hopSize = Math.max(1, Math.floor(sampleRate * hopSeconds));
  const windowSize = Math.max(hopSize, Math.floor(sampleRate * 0.08));
  const envelope = [];

  for (let start = 0; start < channel.length - windowSize; start += hopSize) {
    envelope.push(rms(channel, start, start + windowSize));
  }

  normalizeInPlace(waveform);
  normalizeInPlace(envelope);
  await audioContext.close();

  return {
    duration: audioBuffer.duration,
    waveform,
    envelope,
    hopSeconds,
    peaks: detectPeaks(envelope, hopSeconds)
  };
}

function buildRmsSeries(channel, bins) {
  const values = [];
  for (let i = 0; i < bins; i += 1) {
    const start = Math.floor((i / bins) * channel.length);
    const end = Math.floor(((i + 1) / bins) * channel.length);
    values.push(rms(channel, start, end));
  }
  return values;
}

function rms(samples, start, end) {
  let sum = 0;
  let count = 0;
  for (let i = start; i < end && i < samples.length; i += 1) {
    sum += samples[i] * samples[i];
    count += 1;
  }
  return count ? Math.sqrt(sum / count) : 0;
}

function normalizeInPlace(values) {
  const max = Math.max(...values, 0.000001);
  for (let i = 0; i < values.length; i += 1) values[i] = values[i] / max;
}

function detectPeaks(envelope, hopSeconds) {
  const peaks = [];
  for (let i = 2; i < envelope.length - 2; i += 1) {
    const value = envelope[i];
    if (value > 0.55 && value > envelope[i - 1] && value > envelope[i + 1]) {
      peaks.push({ time: Number((i * hopSeconds).toFixed(2)), value });
    }
  }
  return peaks.slice(0, 80);
}

function estimateAudioSync(leftAudio, rightAudio) {
  if (!leftAudio?.envelope?.length || !rightAudio?.envelope?.length) {
    return { ready: false, offsetSeconds: 0, confidence: 0, message: "Сначала загрузите два видеофайла с аудио." };
  }

  const hop = leftAudio.hopSeconds || 0.05;
  const maxLag = Math.min(Math.round(20 / hop), Math.floor(Math.min(leftAudio.envelope.length, rightAudio.envelope.length) * 0.45));
  const left = zNormalize(leftAudio.envelope);
  const right = zNormalize(rightAudio.envelope);
  let bestLag = 0;
  let bestScore = -Infinity;

  for (let lag = -maxLag; lag <= maxLag; lag += 1) {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < left.length; i += 1) {
      const j = i + lag;
      if (j < 0 || j >= right.length) continue;
      sum += left[i] * right[j];
      count += 1;
    }
    if (count < 20) continue;
    const score = sum / count;
    if (score > bestScore) {
      bestScore = score;
      bestLag = lag;
    }
  }

  const offsetSeconds = Number((bestLag * hop).toFixed(2));
  const confidence = Math.max(0, Math.min(100, Math.round(((bestScore + 1) / 2) * 100)));
  return {
    ready: true,
    offsetSeconds,
    confidence,
    message:
      offsetSeconds >= 0
        ? `Правое видео читается на ${offsetSeconds.toFixed(2)} сек. вперед относительно эталона.`
        : `Правое видео читается на ${Math.abs(offsetSeconds).toFixed(2)} сек. назад относительно эталона.`
  };
}

function zNormalize(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const std = Math.sqrt(variance) || 1;
  return values.map((value) => (value - mean) / std);
}

function formatTime(seconds) {
  const safeSeconds = Math.max(0, seconds || 0);
  const minutes = Math.floor(safeSeconds / 60);
  const rest = Math.floor(safeSeconds % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${rest}`;
}

const VideoPane = forwardRef(function VideoPane(
  {
    title,
    roleLabel,
    side,
    landmarker,
    scanLandmarker,
    nextTimestamp,
    onPose,
    onFile,
    onScanComplete,
    scan,
    active,
    comment,
    onCommentChange,
    analysisRange,
    onAnalysisRangeChange,
    mediaPipeSettings,
    showAnalysisRange = false
  },
  ref
) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const [sourceName, setSourceName] = useState("Источник не выбран");
  const [isPlaying, setIsPlaying] = useState(false);
  const [mode, setMode] = useState("empty");
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState("");
  const [scanProgress, setScanProgress] = useState(0);
  const [isScanning, setIsScanning] = useState(false);
  const [showCameraPrompt, setShowCameraPrompt] = useState(false);
  const isScanningRef = useRef(false);
  const specs = useMemo(() => activeAngleSpecs(mediaPipeSettings?.regions), [mediaPipeSettings?.regions]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  useImperativeHandle(ref, () => ({
    video: videoRef.current,
    async playAt(time, withSound = false) {
      const video = videoRef.current;
      if (!video) return false;
      video.currentTime = Math.max(0, Math.min(time, video.duration || time));
      video.muted = !withSound;
      video.loop = false;
      try {
        await video.play();
        return true;
      } catch (err) {
        if (withSound) {
          video.muted = true;
          await video.play();
          return true;
        }
        throw err;
      }
    },
    pause() {
      videoRef.current?.pause();
    },
    setLoop(loop) {
      if (videoRef.current) videoRef.current.loop = loop;
    }
  }));

  const analyzeFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (isScanningRef.current) {
      rafRef.current = requestAnimationFrame(analyzeFrame);
      return;
    }
    if (!video || !canvas || !landmarker || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(analyzeFrame);
      return;
    }

    const canvasRect = canvas.getBoundingClientRect();
    const width = Math.round(canvasRect.width || canvas.clientWidth || video.videoWidth);
    const height = Math.round(canvasRect.height || canvas.clientHeight || video.videoHeight);
    if (!width || !height) {
      rafRef.current = requestAnimationFrame(analyzeFrame);
      return;
    }
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    if (lastVideoTimeRef.current !== video.currentTime) {
      const result = landmarker.detectForVideo(video, nextTimestamp());
      const landmarks = result.landmarks?.[0] || [];
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const isInsideAnalysisRange =
        !showAnalysisRange ||
        !analysisRange ||
        (video.currentTime >= analysisRange.start && video.currentTime <= analysisRange.end);

      if (landmarks.length && isInsideAnalysisRange) {
        const rect = containRect(canvas.width, canvas.height, video.videoWidth, video.videoHeight);
        const projectedLandmarks = projectLandmarksToCanvas(landmarks, rect, canvas.width, canvas.height);
        const visualScale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight);
        const drawingUtils = new DrawingUtils(ctx);
        drawingUtils.drawConnectors(projectedLandmarks, PoseLandmarker.POSE_CONNECTIONS, {
          color: side === "left" ? "#28d7a4" : "#55a4ff",
          lineWidth: Math.max(1.5, 4 * visualScale)
        });
        drawingUtils.drawLandmarks(projectedLandmarks, {
          color: side === "left" ? "#28d7a4" : "#55a4ff",
          fillColor: side === "left" ? "#28d7a4" : "#55a4ff",
          radius: Math.max(1.2, 2.4 * visualScale)
        });
      }

      onPose({
        landmarks: isInsideAnalysisRange ? landmarks : [],
        angles: isInsideAnalysisRange ? poseAngles(landmarks, specs) : {},
        timestamp: video.currentTime,
        confidence: landmarks.length && isInsideAnalysisRange ? averageVisibility(landmarks) : 0
      });
      lastVideoTimeRef.current = video.currentTime;
    }

    rafRef.current = requestAnimationFrame(analyzeFrame);
  }, [analysisRange, landmarker, onPose, showAnalysisRange, side, specs]);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(analyzeFrame);
    return () => cancelAnimationFrame(rafRef.current);
  }, [analyzeFrame]);

  useEffect(() => stopCamera, [stopCamera]);

  function useCamera() {
    setError("");
    setShowCameraPrompt(true);
  }

  async function requestCameraAccess() {
    setError("");
    try {
      stopCamera();
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      video.loop = false;
      video.muted = true;
      await video.play();
      setIsPlaying(true);
      setMode("camera");
      setSourceName("Камера ноутбука");
      setShowCameraPrompt(false);
      onFile(null);
    } catch (err) {
      const isDenied = err?.name === "NotAllowedError" || err?.name === "PermissionDeniedError";
      setError(
        isDenied
          ? "Доступ к камере заблокирован. Нажмите значок замка рядом с адресом сайта и разрешите камеру, затем попробуйте снова."
          : "Камера недоступна. Проверьте, что камера подключена и не занята другим приложением."
      );
      console.error(err);
    }
  }

  function loadFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    stopCamera();
    const video = videoRef.current;
    video.srcObject = null;
    video.src = URL.createObjectURL(file);
    video.loop = true;
    video.muted = true;
    video.play();
    setIsPlaying(true);
    setMode("file");
    setSourceName(file.name);
    setError("");
    setScanProgress(0);
    onFile(file);
    onScanComplete(null);
  }

  function handleLoadedMetadata() {
    const video = videoRef.current;
    const nextDuration = video?.duration || 0;
    setDuration(nextDuration);
    if (showAnalysisRange && nextDuration && onAnalysisRangeChange) {
      onAnalysisRangeChange({ start: 0, end: Number(nextDuration.toFixed(2)) });
    }
  }

  function togglePlay() {
    const video = videoRef.current;
    if (!video?.src && !video?.srcObject) return;
    if (video.paused) {
      video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  }

  function reset() {
    const video = videoRef.current;
    if (video && mode === "file") video.currentTime = 0;
  }

  function seekVideo(event) {
    const video = videoRef.current;
    if (!video) return;
    const nextTime = Number(event.target.value);
    video.currentTime = nextTime;
    setCurrentTime(nextTime);
  }

  async function scanSkeleton() {
    const video = videoRef.current;
    if (!video || mode === "empty") {
      setError("Сначала загрузите видео.");
      return;
    }
    if (!scanLandmarker) {
      setError("MediaPipe еще загружается.");
      return;
    }
    setError("");
    setIsScanning(true);
    isScanningRef.current = true;
    setScanProgress(1);
    try {
      const result = await scanVideoPose(
        video,
        scanLandmarker,
        setScanProgress,
        showAnalysisRange ? analysisRange : null,
        mediaPipeSettings
      );
      onScanComplete(result);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setError(`Не получилось отсканировать скелет: ${detail}`);
      console.error(err);
    } finally {
      isScanningRef.current = false;
      setIsScanning(false);
    }
  }

  return (
    <section className={`video-pane ${active || scan?.trackedFrames ? "active" : ""}`}>
      <header className="pane-header">
        <div>
          <p className="eyebrow">{title}</p>
          <h2>{sourceName}</h2>
          <small className="role-label">{roleLabel}</small>
        </div>
        <span className={`status ${scan?.trackedFrames ? "status-good" : ""}`}>
          {scan?.trackedFrames ? `скан сохранен: ${scan.trackedFrames} кадров` : active ? "скелет найден" : "ожидание скана"}
        </span>
      </header>

      <div className="source-toolbar">
        <button type="button" className="camera-primary" onClick={useCamera}>
          <Camera size={18} />
          Включить камеру
        </button>
        <span>{mode === "camera" ? "Камера активна" : "Можно загрузить видео или сразу включить камеру устройства"}</span>
      </div>

      {showCameraPrompt && (
        <div className="camera-permission">
          <div>
            <strong>Разрешить доступ к камере?</strong>
            <span>После нажатия браузер откроет системное окно разрешения камеры для этого сайта.</span>
          </div>
          <div className="permission-actions">
            <button type="button" onClick={() => setShowCameraPrompt(false)}>
              Отмена
            </button>
            <button type="button" className="camera-primary" onClick={requestCameraAccess}>
              Разрешить камеру
            </button>
          </div>
        </div>
      )}

      <div className="stage">
        <video
          ref={videoRef}
          playsInline
          muted
          onLoadedMetadata={handleLoadedMetadata}
          onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
        />
        <canvas ref={canvasRef} />
        {mode === "empty" && (
          <div className="empty-state">
            <ScanLine size={34} />
            <strong>Загрузите видео или включите камеру</strong>
          </div>
        )}
      </div>

      {duration > 0 && (
        <div className="playback-timeline">
          <div className="time-row">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
          <input type="range" min="0" max={duration} step="0.05" value={currentTime} onChange={seekVideo} />
        </div>
      )}

      {showAnalysisRange && duration > 0 && analysisRange && (
        <div className="analysis-range">
          <div className="range-header">
            <strong>Диапазон анализа скелета</strong>
            <span>
              {formatTime(analysisRange.start)} - {formatTime(analysisRange.end)}
            </span>
          </div>
          <div className="range-track">
            <i
              style={{
                left: `${(analysisRange.start / duration) * 100}%`,
                width: `${Math.max(0, ((analysisRange.end - analysisRange.start) / duration) * 100)}%`
              }}
            />
          </div>
          <label>
            Начало
            <input
              type="range"
              min="0"
              max={duration}
              step="0.1"
              value={analysisRange.start}
              onChange={(event) =>
                onAnalysisRangeChange({
                  start: Math.min(Number(event.target.value), analysisRange.end - 0.1),
                  end: analysisRange.end
                })
              }
            />
          </label>
          <label>
            Конец
            <input
              type="range"
              min="0"
              max={duration}
              step="0.1"
              value={analysisRange.end}
              onChange={(event) =>
                onAnalysisRangeChange({
                  start: analysisRange.start,
                  end: Math.max(Number(event.target.value), analysisRange.start + 0.1)
                })
              }
            />
          </label>
          <div className="range-actions">
            <button
              type="button"
              onClick={() =>
                onAnalysisRangeChange({
                  start: Math.min(videoRef.current?.currentTime || 0, analysisRange.end - 0.1),
                  end: analysisRange.end
                })
              }
            >
              Начало = текущий кадр
            </button>
            <button
              type="button"
              onClick={() =>
                onAnalysisRangeChange({
                  start: analysisRange.start,
                  end: Math.max(videoRef.current?.currentTime || duration, analysisRange.start + 0.1)
                })
              }
            >
              Конец = текущий кадр
            </button>
          </div>
        </div>
      )}

      <div className="controls">
        <label className="button">
          <FileVideo size={18} />
          Видео
          <input type="file" accept="video/*" onChange={loadFile} />
        </label>
        <button type="button" onClick={useCamera}>
          <Camera size={18} />
          Камера
        </button>
        <button type="button" onClick={togglePlay}>
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          {isPlaying ? "Пауза" : "Старт"}
        </button>
        <button type="button" onClick={reset}>
          <RotateCcw size={18} />
          Сброс
        </button>
        <button type="button" className="scan-button" onClick={scanSkeleton} disabled={isScanning}>
          <Wand2 size={18} />
          {isScanning ? `${scanProgress}%` : "Сканировать скелет"}
        </button>
      </div>

      <div className="scan-meta">
        <span>Данные позы: {scan?.frames?.length ? `${scan.frames.length} точек таймлайна` : "не сохранены"}</span>
        <span>Точность: {scan?.averageConfidence ? `${Math.round(scan.averageConfidence * 100)}%` : "-"}</span>
      </div>
      <label className="video-comment">
        Комментарий для датасета
        <textarea
          value={comment}
          onChange={(event) => onCommentChange(event.target.value)}
          placeholder={
            side === "left"
              ? "Например: танец 1, эталон, оригинальная хореография"
              : "Например: танец 1 дубль 2, тот же танец / танец 2, другой стиль"
          }
        />
      </label>
      {error && <p className="pane-error">{error}</p>}
    </section>
  );
});

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function loadLabHistory() {
  try {
    return JSON.parse(localStorage.getItem(labHistoryKey) || "[]");
  } catch {
    return [];
  }
}

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function MediaPipeSettingsPanel({ settings, onChange, isReady }) {
  const specs = activeAngleSpecs(settings.regions);
  const update = (patch) => onChange(normalizeMediaPipeSettings({ ...settings, ...patch }));
  const updateNumber = (key, value) => update({ [key]: Number(value) });
  const updateRegion = (key, checked) => update({ regions: { ...settings.regions, [key]: checked } });

  return (
    <section className="settings-panel">
      <div className="settings-title">
        <div>
          <p className="eyebrow">MediaPipe Settings</p>
          <h2>Параметры трекинга и датасета</h2>
        </div>
        <span className={`status ${isReady ? "status-good" : ""}`}>{isReady ? "модель активна" : "модель загружается"}</span>
      </div>

      <div className="settings-grid">
        <label>
          Модель Pose Landmarker
          <select value={settings.modelVariant} onChange={(event) => update({ modelVariant: event.target.value })}>
            <option value="lite">Lite - быстрее</option>
            <option value="full">Full - баланс</option>
            <option value="heavy">Heavy - точнее, медленнее</option>
          </select>
        </label>
        <label>
          Вычисление
          <select value={settings.delegate} onChange={(event) => update({ delegate: event.target.value })}>
            <option value="GPU">GPU</option>
            <option value="CPU">CPU</option>
          </select>
        </label>
        <label>
          Максимум поз
          <input
            type="number"
            min="1"
            max="4"
            value={settings.numPoses}
            onChange={(event) => updateNumber("numPoses", event.target.value)}
          />
        </label>
        <label>
          FPS сканирования
          <select value={settings.scanFps} onChange={(event) => updateNumber("scanFps", event.target.value)}>
            <option value="2">2 кадра/сек</option>
            <option value="3">3 кадра/сек</option>
            <option value="5">5 кадров/сек</option>
            <option value="10">10 кадров/сек</option>
            <option value="15">15 кадров/сек</option>
          </select>
        </label>
        <label>
          Точки в датасете
          <select value={settings.landmarkSet} onChange={(event) => update({ landmarkSet: event.target.value })}>
            <option value="core13">13 ключевых точек</option>
            <option value="full33">Все 33 точки MediaPipe</option>
          </select>
        </label>
        <label className="checkbox-setting">
          <input
            type="checkbox"
            checked={settings.outputSegmentationMasks}
            onChange={(event) => update({ outputSegmentationMasks: event.target.checked })}
          />
          Маска сегментации тела
        </label>
        <label>
          Detection confidence
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={settings.minPoseDetectionConfidence}
            onChange={(event) => updateNumber("minPoseDetectionConfidence", event.target.value)}
          />
          <small>{settings.minPoseDetectionConfidence.toFixed(2)}</small>
        </label>
        <label>
          Presence confidence
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={settings.minPosePresenceConfidence}
            onChange={(event) => updateNumber("minPosePresenceConfidence", event.target.value)}
          />
          <small>{settings.minPosePresenceConfidence.toFixed(2)}</small>
        </label>
        <label>
          Tracking confidence
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={settings.minTrackingConfidence}
            onChange={(event) => updateNumber("minTrackingConfidence", event.target.value)}
          />
          <small>{settings.minTrackingConfidence.toFixed(2)}</small>
        </label>
      </div>

      <div className="region-settings">
        <strong>Области сравнения</strong>
        <label>
          <input type="checkbox" checked={settings.regions.arms} onChange={(event) => updateRegion("arms", event.target.checked)} />
          Руки и плечи
        </label>
        <label>
          <input type="checkbox" checked={settings.regions.torso} onChange={(event) => updateRegion("torso", event.target.checked)} />
          Корпус и бедра
        </label>
        <label>
          <input type="checkbox" checked={settings.regions.legs} onChange={(event) => updateRegion("legs", event.target.checked)} />
          Ноги и колени
        </label>
        <span>
          MediaPipe возвращает 33 точки. В сравнении сейчас активно {specs.length} углов, в датасет сохраняется{" "}
          {settings.landmarkSet === "full33" ? "33 точки" : "13 ключевых точек"}.
        </span>
      </div>
    </section>
  );
}

function serializeLandmarks(landmarks, landmarkSet = "core13") {
  const ids = landmarkSet === "full33" ? poseLandmarkCatalog.map((_, index) => index) : coreLandmarkIds;
  return ids
    .map((id) => {
      const point = landmarks?.[id];
      if (!point) return null;
      return {
        id,
        name: poseLandmarkCatalog[id] || landmarkNames[id] || `landmark ${id}`,
        x: Number(point.x.toFixed(5)),
        y: Number(point.y.toFixed(5)),
        z: Number((point.z || 0).toFixed(5)),
        visibility: Number((point.visibility || 0).toFixed(3))
      };
    })
    .filter(Boolean);
}

function serializeScanSkeleton(scan, settings = defaultMediaPipeSettings) {
  if (!scan?.frames?.length) return null;
  return {
    duration: scan.duration,
    range: scan.range || null,
    trackedFrames: scan.trackedFrames,
    averageConfidence: Number((scan.averageConfidence || 0).toFixed(4)),
    landmarkSet: settings.landmarkSet,
    frames: scan.frames
      .filter((frame) => frame.landmarks?.length)
      .map((frame) => ({
        time: frame.time,
        confidence: Number((frame.confidence || 0).toFixed(4)),
        angles: frame.angles,
        landmarks: serializeLandmarks(frame.landmarks, settings.landmarkSet)
      }))
  };
}

function serializeSkeletonPairs(leftScan, rightScan, sync, settings = defaultMediaPipeSettings) {
  if (!leftScan?.frames?.length || !rightScan?.frames?.length) return [];
  const offset = sync?.ready ? sync.offsetSeconds : 0;
  return leftScan.frames
    .filter((frame) => frame.landmarks?.length)
    .map((leftFrame) => {
      const rightFrame = nearestFrame(rightScan.frames, leftFrame.time + offset);
      if (!rightFrame?.landmarks?.length) return null;
      return {
        leftTime: leftFrame.time,
        rightTime: rightFrame.time,
        leftLandmarks: serializeLandmarks(leftFrame.landmarks, settings.landmarkSet),
        rightLandmarks: serializeLandmarks(rightFrame.landmarks, settings.landmarkSet)
      };
    })
    .filter(Boolean);
}

function LabHistoryPanel({ history, expectedScore, onExpectedScoreChange, onSave, onExport, onClear, canSave }) {
  return (
    <section className="lab-panel">
      <div className="lab-header">
        <div>
          <p className="eyebrow">Algorithm Lab</p>
          <h2>История обучающих примеров</h2>
        </div>
        <span>{history.length} записей</span>
      </div>

      <div className="lab-controls">
        <label>
          Ожидаемая схожесть, %
          <input
            type="number"
            min="0"
            max="100"
            value={expectedScore}
            onChange={(event) => onExpectedScoreChange(event.target.value)}
            placeholder="например 0 или 100"
          />
        </label>
        <button type="button" onClick={onSave} disabled={!canSave}>
          Сохранить пример
        </button>
        <button type="button" onClick={onExport} disabled={!history.length}>
          Экспорт JSON
        </button>
        <button type="button" onClick={onClear} disabled={!history.length}>
          Очистить историю
        </button>
      </div>

      <div className="history-list">
        {history.length ? (
          history.slice(0, 8).map((item) => (
            <article key={item.id} className="history-item">
              <strong>
                {item.score}% схожесть{item.expectedScore !== null ? ` / ожидалось ${item.expectedScore}%` : ""}
              </strong>
              <span>{new Date(item.createdAt).toLocaleString()}</span>
              <p>
                Эталон: {item.leftComment || "без комментария"} | Правое: {item.rightComment || "без комментария"}
              </p>
              <p>
                Скелеты: {item.skeletons?.left?.frames?.length || 0} кадров эталона,{" "}
                {item.skeletons?.right?.frames?.length || 0} кадров правого,{" "}
                {item.skeletons?.synchronizedPairs?.length || 0} синхронных пар.
              </p>
            </article>
          ))
        ) : (
          <p className="empty-history">После полного анализа сохраняйте примеры, чтобы собрать датасет для настройки алгоритма.</p>
        )}
      </div>
    </section>
  );
}

function WaveformTimeline({ leftAudio, rightAudio, sync }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#f6f9fd";
    ctx.fillRect(0, 0, rect.width, rect.height);
    drawGrid(ctx, rect.width, rect.height);
    drawWave(ctx, leftAudio?.waveform, rect.width, rect.height, "#df4a5f", 0);
    drawWave(ctx, rightAudio?.waveform, rect.width, rect.height, "#407ee8", sync?.ready ? sync.offsetSeconds : 0);
    if (sync?.ready) {
      const x = rect.width / 2;
      ctx.strokeStyle = "#17202a";
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 6]);
      ctx.beginPath();
      ctx.moveTo(x, 8);
      ctx.lineTo(x, rect.height - 8);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [leftAudio, rightAudio, sync]);

  return (
    <section className="sync-panel">
      <div className="sync-header">
        <div>
          <p className="eyebrow">Audio Sync</p>
          <h2>Таймлайн аудиодорожек</h2>
        </div>
        <div className="legend">
          <span>
            <i className="left-color" /> эталон
          </span>
          <span>
            <i className="right-color" /> правое видео
          </span>
        </div>
      </div>
      <canvas ref={canvasRef} className="waveform-canvas" />
      <p className="sync-note">
        {sync?.ready
          ? `${sync.message} Уверенность аудио-сопоставления: ${sync.confidence}%.`
          : "После загрузки видео аудиоволны появятся здесь. Синхронизация ищет совпадающие всплески и ритмический рисунок музыки."}
      </p>
    </section>
  );
}

function drawGrid(ctx, width, height) {
  ctx.strokeStyle = "#dfe8f2";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i += 1) {
    const x = (width / 8) * i;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  ctx.strokeStyle = "#cfdbea";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
}

function drawWave(ctx, waveform, width, height, color, offsetSeconds) {
  if (!waveform?.length) return;
  const center = height / 2;
  const scale = height * 0.36;
  const offsetPx = offsetSeconds * 14;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  waveform.forEach((value, index) => {
    const x = (index / (waveform.length - 1)) * width - offsetPx;
    const y = center - value * scale;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  for (let index = waveform.length - 1; index >= 0; index -= 1) {
    const value = waveform[index];
    const x = (index / (waveform.length - 1)) * width - offsetPx;
    const y = center + value * scale;
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.globalAlpha = 0.22;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.stroke();
}

function App() {
  const [mediaPipeSettings, setMediaPipeSettings] = useState(() => loadMediaPipeSettings());
  const { landmarker, scanLandmarker, error } = usePoseLandmarker(mediaPipeSettings);
  const leftVideoRef = useRef(null);
  const rightVideoRef = useRef(null);
  const analysisRafRef = useRef(0);
  const mediaPipeTimestampRef = useRef(0);
  const [leftPose, setLeftPose] = useState(null);
  const [rightPose, setRightPose] = useState(null);
  const [leftFile, setLeftFile] = useState(null);
  const [rightFile, setRightFile] = useState(null);
  const [leftComment, setLeftComment] = useState("");
  const [rightComment, setRightComment] = useState("");
  const [leftScan, setLeftScan] = useState(null);
  const [rightScan, setRightScan] = useState(null);
  const [leftAudio, setLeftAudio] = useState(null);
  const [rightAudio, setRightAudio] = useState(null);
  const [leftAnalysisRange, setLeftAnalysisRange] = useState({ start: 0, end: 0 });
  const [sync, setSync] = useState({ ready: false, offsetSeconds: 0, confidence: 0 });
  const [audioStatus, setAudioStatus] = useState("");
  const [runState, setRunState] = useState({ status: "idle", progress: 0, result: null, message: "" });
  const [expectedScore, setExpectedScore] = useState("");
  const [labHistory, setLabHistory] = useState(() => loadLabHistory());
  const activeSpecs = useMemo(() => activeAngleSpecs(mediaPipeSettings.regions), [mediaPipeSettings.regions]);
  const liveComparison = useMemo(
    () => comparePoseFrames(leftPose, rightPose, mediaPipeSettings.regions),
    [leftPose, rightPose, mediaPipeSettings.regions]
  );
  const scanComparison = useMemo(
    () => compareScans(leftScan, rightScan, sync, mediaPipeSettings.regions),
    [leftScan, rightScan, sync, mediaPipeSettings.regions]
  );
  const comparison = runState.result || (leftScan?.frames?.length && rightScan?.frames?.length ? pendingFullRunComparison() : liveComparison);
  const confidence = Math.round(
    scanComparison.ready
      ? ((leftScan?.averageConfidence || 0) + (rightScan?.averageConfidence || 0)) * 50
      : ((leftPose?.confidence || 0) + (rightPose?.confidence || 0)) * 50
  );

  useEffect(() => () => cancelAnimationFrame(analysisRafRef.current), []);

  useEffect(() => {
    localStorage.setItem(labHistoryKey, JSON.stringify(labHistory));
  }, [labHistory]);

  useEffect(() => {
    localStorage.setItem(mediaPipeSettingsKey, JSON.stringify(mediaPipeSettings));
  }, [mediaPipeSettings]);

  const nextMediaPipeTimestamp = useCallback(() => {
    const now = performance.now();
    const next = Math.max(now, mediaPipeTimestampRef.current + 1);
    mediaPipeTimestampRef.current = next;
    return next;
  }, []);

  async function handleAudio(side, file) {
    setSync({ ready: false, offsetSeconds: 0, confidence: 0 });
    setRunState({ status: "idle", progress: 0, result: null, message: "" });
    if (!file) {
      if (side === "left") setLeftAudio(null);
      else setRightAudio(null);
      return;
    }
    setAudioStatus(`Сканирую аудио: ${file.name}`);
    try {
      const audio = await analyzeAudioFile(file);
      if (side === "left") setLeftAudio(audio);
      else setRightAudio(audio);
      setAudioStatus("Аудиодорожка отсканирована.");
    } catch (err) {
      setAudioStatus("Не удалось прочитать аудио из этого файла. Попробуйте MP4/MOV с обычной аудиодорожкой.");
      console.error(err);
    }
  }

  function synchronizeAudio() {
    const result = estimateAudioSync(leftAudio, rightAudio);
    setSync(result);
    setRunState({
      status: "ready",
      progress: 0,
      result: null,
      message: result.ready ? "Синхронизация рассчитана. Запустите полный прогон видео." : result.message
    });
    if (result.ready && leftScan?.frames?.length && rightScan?.frames?.length) {
      window.setTimeout(() => playSynchronizedAnalysis(result), 0);
    }
  }

  async function playSynchronizedAnalysis(syncOverride = sync) {
    if (!leftScan?.frames?.length || !rightScan?.frames?.length) {
      setRunState({
        status: "error",
        progress: 0,
        result: null,
        message: "Сначала отсканируйте скелет в обоих видео."
      });
      return;
    }

    const activeSync = syncOverride?.ready ? syncOverride : sync;
    const offset = activeSync.ready ? activeSync.offsetSeconds : 0;
    let leftStart = leftScan.frames?.[0]?.time ?? Math.max(0, -offset);
    let rightStart = leftStart + offset;
    if (rightStart < 0) {
      leftStart = Math.min(leftScan.frames?.at(-1)?.time ?? leftStart, leftStart - rightStart);
      rightStart = 0;
    }
    const leftVideo = leftVideoRef.current?.video;
    const rightVideo = rightVideoRef.current?.video;
    const playableSeconds = Math.min(
      Math.max(0, (leftScan.frames?.at(-1)?.time ?? leftVideo?.duration ?? leftScan.duration ?? 0) - leftStart),
      Math.max(0, (rightVideo?.duration || rightScan.duration || 0) - rightStart)
    );

    if (!playableSeconds) {
      setRunState({
        status: "error",
        progress: 0,
        result: null,
        message: "Не удалось определить общий синхронный фрагмент для воспроизведения."
      });
      return;
    }

    cancelAnimationFrame(analysisRafRef.current);
    setRunState({
      status: "running",
      progress: 0,
      result: null,
      message: "Идет полный синхронный прогон. После окончания появится итоговая статистика всего видео."
    });
    leftVideoRef.current?.setLoop(false);
    rightVideoRef.current?.setLoop(false);
    try {
      await Promise.all([leftVideoRef.current?.playAt(leftStart, false), rightVideoRef.current?.playAt(rightStart, false)]);
    } catch (err) {
      const result = compareScans(leftScan, rightScan, activeSync, mediaPipeSettings.regions);
      setRunState({
        status: "done",
        progress: 100,
        result,
        message:
          "Мобильный браузер заблокировал воспроизведение, поэтому статистика рассчитана сразу по сохраненным сканам без видеопрогона."
      });
      console.error(err);
      return;
    }

    const tick = () => {
      const currentLeft = leftVideoRef.current?.video?.currentTime ?? leftStart;
      const elapsed = Math.max(0, currentLeft - leftStart);
      const progress = Math.max(0, Math.min(100, Math.round((elapsed / playableSeconds) * 100)));

      if (elapsed >= playableSeconds - 0.05) {
        leftVideoRef.current?.pause();
        rightVideoRef.current?.pause();
        const result = compareScans(leftScan, rightScan, activeSync, mediaPipeSettings.regions);
        setRunState({
          status: "done",
          progress: 100,
          result,
          message: `Прогон завершен: проанализировано ${result.framesCompared || 0} синхронизированных кадров.`
        });
        return;
      }

      setRunState((previous) => ({ ...previous, progress }));
      analysisRafRef.current = requestAnimationFrame(tick);
    };

    analysisRafRef.current = requestAnimationFrame(tick);
  }

  function saveLabExample() {
    if (!runState.result?.ready) return;
    const nextItem = {
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      leftComment,
      rightComment,
      leftFileName: leftFile?.name || "",
      rightFileName: rightFile?.name || "",
      expectedScore: expectedScore === "" ? null : Number(expectedScore),
      score: runState.result.score,
      mediaPipeSettings: {
        ...mediaPipeSettings,
        activeAngles: activeSpecs.map((spec) => ({
          id: spec.id,
          title: spec.title,
          region: spec.region,
          points: spec.points
        }))
      },
      sync,
      leftAnalysisRange,
      metrics: {
        framesCompared: runState.result.framesCompared || 0,
        bestScore: runState.result.bestScore ?? null,
        worstScore: runState.result.worstScore ?? null,
        durationCompared: runState.result.durationCompared ?? null,
        worstMoment: runState.result.worstMoment ?? null
      },
      angleRows: runState.result.rows,
      skeletons: {
        left: serializeScanSkeleton(leftScan, mediaPipeSettings),
        right: serializeScanSkeleton(rightScan, mediaPipeSettings),
        synchronizedPairs: serializeSkeletonPairs(leftScan, rightScan, sync, mediaPipeSettings)
      },
      suggestions: runState.result.suggestions,
      verdict: runState.result.verdict
    };
    setLabHistory((items) => [nextItem, ...items]);
  }

  function clearLabHistory() {
    setLabHistory([]);
  }

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <p className="eyebrow">MediaPipe Pose Analyzer</p>
          <h1>Сравнение скелетов в двух видео</h1>
          <p className="subtitle">Левое видео - эталон. Правое видео оценивается и подгоняется относительно эталона.</p>
        </div>
        <div className="model-state">
          <Sparkles size={18} />
          {landmarker && scanLandmarker ? "MediaPipe готов" : "Загрузка модели..."}
        </div>
      </header>

      <MediaPipeSettingsPanel
        settings={mediaPipeSettings}
        onChange={(settings) => {
          setMediaPipeSettings(settings);
          setLeftScan(null);
          setRightScan(null);
          setRunState({
            status: "idle",
            progress: 0,
            result: null,
            message: "Настройки MediaPipe изменены. Пересканируйте оба видео для новой версии датасета."
          });
        }}
        isReady={Boolean(landmarker && scanLandmarker)}
      />

      {error && <div className="global-error">{error}</div>}

      <section className="video-grid">
        <VideoPane
          ref={leftVideoRef}
          title="Видео A / эталон"
          roleLabel="Сканируем и сохраняем позу как базовый образец"
          side="left"
          landmarker={landmarker}
          scanLandmarker={scanLandmarker}
          nextTimestamp={nextMediaPipeTimestamp}
          onPose={setLeftPose}
          onFile={(file) => {
            setLeftFile(file);
            setLeftScan(null);
            setLeftAnalysisRange({ start: 0, end: 0 });
            handleAudio("left", file);
          }}
          onScanComplete={(scan) => {
            setLeftScan(scan);
            setRunState({ status: "idle", progress: 0, result: null, message: "" });
          }}
          scan={leftScan}
          active={Boolean(leftPose?.landmarks?.length)}
          comment={leftComment}
          onCommentChange={setLeftComment}
          showAnalysisRange
          analysisRange={leftAnalysisRange}
          mediaPipeSettings={mediaPipeSettings}
          onAnalysisRangeChange={(range) => {
            setLeftAnalysisRange({
              start: Number(Math.max(0, range.start).toFixed(2)),
              end: Number(Math.max(0, range.end).toFixed(2))
            });
            setLeftScan(null);
            setRunState({ status: "idle", progress: 0, result: null, message: "" });
          }}
        />
        <VideoPane
          ref={rightVideoRef}
          title="Видео B / подгонка"
          roleLabel="Это видео сравнивается с эталоном по музыке и позе"
          side="right"
          landmarker={landmarker}
          scanLandmarker={scanLandmarker}
          nextTimestamp={nextMediaPipeTimestamp}
          onPose={setRightPose}
          onFile={(file) => {
            setRightFile(file);
            setRightScan(null);
            handleAudio("right", file);
          }}
          onScanComplete={(scan) => {
            setRightScan(scan);
            setRunState({ status: "idle", progress: 0, result: null, message: "" });
          }}
          scan={rightScan}
          active={Boolean(rightPose?.landmarks?.length)}
          comment={rightComment}
          onCommentChange={setRightComment}
          mediaPipeSettings={mediaPipeSettings}
        />
      </section>

      <WaveformTimeline leftAudio={leftAudio} rightAudio={rightAudio} sync={sync} />

      <div className="sync-actions">
        <button type="button" onClick={synchronizeAudio} disabled={!leftAudio || !rightAudio}>
          <Waves size={18} />
          Синхронизировать по аудио
        </button>
        <button
          type="button"
          onClick={() => playSynchronizedAnalysis()}
          disabled={!leftScan?.frames?.length || !rightScan?.frames?.length || runState.status === "running"}
        >
          <Play size={18} />
          {runState.status === "running" ? `Анализ ${runState.progress}%` : "Повторить анализ всего видео"}
        </button>
        <span>{audioStatus || "Аудио используется для выравнивания одинаковых музыкальных фрагментов."}</span>
      </div>

      <div className={`run-status ${runState.status}`}>
        <div>
          <strong>
            {runState.status === "done"
              ? "Итоговый прогон завершен"
              : runState.status === "running"
                ? "Идет синхронное воспроизведение"
                : runState.status === "ready"
                  ? "Готово к полному прогону"
                  : "Полный прогон еще не запускался"}
          </strong>
          <span>{runState.message || "После синхронизации запустите видео один раз, чтобы получить статистику всего танца."}</span>
        </div>
        <div className="run-progress" aria-label="Прогресс анализа">
          <i style={{ width: `${runState.progress}%` }} />
        </div>
      </div>

      <section className="analysis-panel">
        <div className="score-ring" style={{ "--score": comparison.score }}>
          <div>
            <span>{comparison.score}%</span>
            <small>схожесть</small>
          </div>
        </div>

        <div className="analysis-body">
          <div className="metrics">
            <MetricCard label="Уверенность трекинга" value={`${confidence}%`} />
            <MetricCard label="Кадров сравнено" value={comparison.framesCompared || comparison.rows.length} />
            <MetricCard label="Смещение аудио" value={sync.ready ? `${sync.offsetSeconds.toFixed(2)} сек` : "нет"} />
            <MetricCard label="Лучший момент" value={comparison.bestScore != null ? `${comparison.bestScore}%` : "-"} />
            <MetricCard label="Худший момент" value={comparison.worstScore != null ? `${comparison.worstScore}%` : "-"} />
            <MetricCard label="Длительность анализа" value={comparison.durationCompared ? `${comparison.durationCompared} сек` : "-"} />
          </div>

          <div className="verdict">
            <h2>Анализ правого видео относительно эталона</h2>
            <p>{comparison.verdict}</p>
            {comparison.worstMoment && (
              <p className="worst-moment">
                Самый слабый момент: эталон {comparison.worstMoment.leftTime.toFixed(1)} сек, правое видео{" "}
                {comparison.worstMoment.rightTime.toFixed(1)} сек, схожесть {comparison.worstMoment.score}%.
              </p>
            )}
          </div>

          <div className="angle-table">
            <div className="table-head">
              <span>Сегмент</span>
              <span>Эталон</span>
              <span>Правое</span>
              <span>Разница</span>
            </div>
            {(comparison.rows.length ? comparison.rows : activeSpecs).map((row) => (
              <div className="table-row" key={row.id}>
                <span>{row.title}</span>
                <span>{row.leftValue ?? "-"}°</span>
                <span>{row.rightValue ?? "-"}°</span>
                <span className={row.diff > 18 ? "bad" : row.diff > 9 ? "warn" : "good"}>{row.diff ?? "-"}°</span>
              </div>
            ))}
          </div>

          <div className="suggestions">
            <h3>Мнение системы</h3>
            {comparison.suggestions.length ? (
              comparison.suggestions.map((item) => <p key={item}>{item}</p>)
            ) : (
              <p>После сканирования двух скелетов и аудио-синхронизации здесь появится анализ подгонки правого видео под эталон.</p>
            )}
          </div>
        </div>
      </section>

      <LabHistoryPanel
        history={labHistory}
        expectedScore={expectedScore}
        onExpectedScoreChange={setExpectedScore}
        onSave={saveLabExample}
        onExport={() => downloadJson("dmpa-lab-history.json", labHistory)}
        onClear={clearLabHistory}
        canSave={Boolean(runState.result?.ready)}
      />
    </main>
  );
}

function averageVisibility(landmarks) {
  const tracked = Object.keys(landmarkNames).map(Number);
  const visible = tracked.map((id) => landmarks[id]?.visibility ?? 0).filter((value) => value > 0);
  if (!visible.length) return 0;
  return visible.reduce((sum, value) => sum + value, 0) / visible.length;
}

createRoot(document.getElementById("root")).render(<App />);
