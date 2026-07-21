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
import { compareSkeletons_2026_07_06 } from "./skeletonComparison20260706.mjs";
import { compareSkeletons_2026_07_12 } from "./skeletonComparison20260712.mjs";
import { compareSkeletons_2026_07_13, filterSkeletonFrames_2026_07_13 } from "./skeletonComparison20260713.mjs";
import "./styles.css";

const wasmBase = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";
const labHistoryKey = "dmpa.lab.history.v1";
const mediaPipeSettingsKey = "dmpa.mediapipe.settings.v1";
const captureEngineKey = "dmpa.capture.engine.v1";
const hybridMethodSettingsKey = "dmpa.hybrid.methods.v1";
const maxStoredLabItems = 20;
const maxStoredSkeletonFrames = 80;
const maxStoredAngleRows = 60;
const appVersion = {
  name: "DMPA Lab",
  version: "0.7.0",
  versionLabel: "v0.7.0",
  build: "sequential-gate-lab-2026-07-21"
};

const captureEngines = {
  mediapipe: {
    id: "mediapipe",
    title: "MediaPipe",
    shortTitle: "MediaPipe",
    description: "Быстрое браузерное сканирование 2D/псевдо-3D скелета прямо из видео или камеры."
  },
  motioncap: {
    id: "motioncap",
    title: "MotionCap / FreeMoCap",
    shortTitle: "MotionCap",
    description: "Импорт исследовательского 3D-скелета FreeMoCap: body_3d_xyz.csv или *_by_frame.csv."
  }
};

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
    legs: true,
    hands: false,
    face: false
  }
};

const mediaPipeHelp = {
  modelVariant:
    "Выбор версии MediaPipe Pose. Lite быстрее и легче для телефона, Full дает баланс, Heavy точнее, но сильнее нагружает устройство.",
  delegate: "Где считать модель: GPU обычно быстрее, CPU полезен, если видеокарта или браузер работают нестабильно.",
  numPoses: "Сколько людей MediaPipe пытается найти в кадре. Для сравнения одного танцора обычно оставляем 1.",
  scanFps: "Сколько кадров в секунду сохранять в скелет. Чем выше FPS, тем точнее движение, но дольше сканирование и тяжелее файл.",
  landmarkSet: "Сколько точек сохранять в датасет: 13 ключевых точек легче, 33 точки дают больше деталей MediaPipe Pose.",
  outputSegmentationMasks: "Маска тела отделяет человека от фона. Для сравнения скелетов обычно не обязательна и может замедлять сканирование.",
  minPoseDetectionConfidence: "Минимальная уверенность первичного обнаружения человека в кадре. Выше значение строже, но может терять сложные позы.",
  minPosePresenceConfidence: "Минимальная уверенность, что поза действительно присутствует в текущем кадре. Влияет на пропуск сомнительных кадров.",
  minTrackingConfidence: "Минимальная уверенность сопровождения уже найденной позы между кадрами. Выше значение уменьшает шум, но может чаще терять тело.",
  regions: "Какие зоны тела учитывать в сравнении и сохранять как настройки эксперимента.",
  hands:
    "MediaPipe Hands нужен для детального анализа кистей и пальцев. Сейчас это отдельный флаг лаборатории; базовые запястья и пальцы Pose уже есть в 33 точках.",
  face:
    "MediaPipe Face Landmarker нужен для детального анализа лица. Сейчас это отдельный флаг лаборатории; базовые точки лица Pose доступны только грубо."
};

const comparisonModels = {
  angles: {
    id: "angles",
    version: "1.3.0",
    versionLabel: "v1.3.0",
    algorithmBuild: "angles-torso-fit-2026-07-12",
    title: "Углы",
    shortTitle: "1. Углы",
    description:
      "Базовая модель: сначала точно совмещает корпус правого скелета с эталоном, затем сравнивает углы локтей, плеч, бедер, коленей и корпуса на синхронных кадрах."
  },
  overlay: {
    id: "overlay",
    version: "1.1.0",
    versionLabel: "v1.1.0",
    algorithmBuild: "overlay-torso-fit-2026-07-12",
    title: "Наложение",
    shortTitle: "2. Наложение",
    description:
      "Скелеты нормализуются по центру корпуса и масштабу тела, затем накладываются друг на друга. Оценка строится по средней дистанции выбранных точек."
  },
  poses: {
    id: "poses",
    version: "1.0.0",
    versionLabel: "v1.0.0",
    algorithmBuild: "audio-impulse-poses-2026-07-05",
    title: "Позы",
    shortTitle: "3. Позы",
    description:
      "Модель ищет сильные импульсы в музыке, делает ключевые снимки позы на этих моментах и сравнивает каждую позу гибридно: углы плюс наложение."
  },
  "2026-07-06": {
    id: "2026-07-06",
    version: "1.2.0",
    versionLabel: "v1.2.0",
    algorithmBuild: "body-normalized-soft-sync-2026-07-06",
    name: "06.07.2026",
    title: "06.07.2026",
    shortTitle: "4. 06.07.2026",
    description:
      "Сравнение скелетов с нормализацией комплекции и мягкой синхронизацией под музыку."
  },
  "2026-07-12": {
    id: "2026-07-12",
    version: "1.4.0",
    versionLabel: "v1.4.0",
    algorithmBuild: "elastic-dtw-tracking-filter-2026-07-12",
    name: "12.07.2026",
    title: "12.07.2026",
    shortTitle: "5. 12.07.2026",
    description:
      "Эластичное сравнение танца: нормализует тело и сравнивает последовательность поз через DTW, траектории и амплитуду движения."
  },
  "2026-07-13": {
    id: "2026-07-13",
    version: "4.1.0",
    versionLabel: "v4.1.0",
    algorithmBuild: "choreography-repeat-evidence-2026-07-13",
    name: "13.07.2026",
    title: "13.07.2026",
    shortTitle: "6. 13.07.2026",
    description:
      "Хореографическое сравнение: корпус как якорь, ключевые позы, мягкая задержка рук и робастная оценка двигательной фразы."
  },
  "all-auto": {
    id: "all-auto",
    version: "1.0.0",
    versionLabel: "v1.0.0",
    algorithmBuild: "all-model-auto-run-2026-07-13",
    name: "Все модели + автосканирование",
    title: "Все модели + автосканирование",
    shortTitle: "7. Все модели",
    description:
      "Лабораторный режим: одной кнопкой сканирует оба видео, затем прогоняет все доступные модели и сохраняет результаты в историю."
  },
  "openai-expert": {
    id: "openai-expert",
    version: "0.7.0",
    versionLabel: "v0.7.0",
    algorithmBuild: "openai-evidence-gate-2026-07-14",
    name: "OpenAI эксперт",
    title: "OpenAI эксперт",
    shortTitle: "8. OpenAI эксперт",
    description:
      "Серверная AI-модель: требует доказательство выполнения хореографической фразы, отделяет качество скана от качества танца и строго ограничивает оценку при отсутствии движения."
  },
  "joint-areas": {
    id: "joint-areas",
    version: "0.1.0",
    versionLabel: "v0.1.0",
    algorithmBuild: "joint-area-hit-bone-normalized-2026-07-19",
    name: "Области",
    title: "Области",
    shortTitle: "9. Области",
    description:
      "Сравнивает попадание активных суставов правого видео в области вокруг суставов эталона после нормализации корпуса и костей."
  },
  "trajectory-drawing": {
    id: "trajectory-drawing",
    version: "0.1.0",
    versionLabel: "v0.1.0",
    algorithmBuild: "active-joint-trajectory-drawing-2026-07-19",
    name: "Рисунок",
    title: "Рисунок",
    shortTitle: "10. Рисунок",
    description:
      "Сравнивает рисунки траекторий активных точек по коротким фразам: куда и как двигались кисти, локти, колени и стопы."
  },
  "zone-grid": {
    id: "zone-grid",
    version: "0.1.0",
    versionLabel: "v0.1.0",
    algorithmBuild: "centered-bone-normalized-zone-grid-2026-07-19",
    name: "Зоны",
    title: "Зоны",
    shortTitle: "11. Зоны",
    description:
      "Скелеты центрируются отдельно, длины костей приводятся к эталону, а точки сравниваются по попаданию в одинаковые квадраты сетки."
  },
  activity: {
    id: "activity",
    version: "0.1.0",
    versionLabel: "v0.1.0",
    algorithmBuild: "skeleton-activity-level-2026-07-19",
    name: "Активность",
    title: "Активность",
    shortTitle: "12. Активность",
    description:
      "Оценивает уровень движения скелета: сколько и с какой амплитудой двигаются руки, ноги и корпус, а затем сравнивает активность правого видео с эталоном."
  }
};

const defaultHybridMethodSettings = {
  zones: true,
  drawing: true
};

const zoneDrawingJointSpecs = [
  { id: 15, key: "leftWrist", title: "Левая кисть", group: "arms", radius: 0.28 },
  { id: 16, key: "rightWrist", title: "Правая кисть", group: "arms", radius: 0.28 },
  { id: 13, key: "leftElbow", title: "Левый локоть", group: "arms", radius: 0.22 },
  { id: 14, key: "rightElbow", title: "Правый локоть", group: "arms", radius: 0.22 },
  { id: 25, key: "leftKnee", title: "Левое колено", group: "legs", radius: 0.24 },
  { id: 26, key: "rightKnee", title: "Правое колено", group: "legs", radius: 0.24 },
  { id: 27, key: "leftAnkle", title: "Левая стопа", group: "legs", radius: 0.28 },
  { id: 28, key: "rightAnkle", title: "Правая стопа", group: "legs", radius: 0.28 }
];

const zoneGridConfig = {
  columns: 16,
  rows: 14,
  xMin: -2.2,
  xMax: 2.2,
  yMin: -2.6,
  yMax: 2.4
};

const runnableComparisonModelIds = Object.keys(comparisonModels).filter((id) => id !== "all-auto");

const comparisonModelKey = "dmpa.comparison.model.v1";
const maxScanFrames = 3600;
const mobileMaxScanFrames = 1200;
const maxOverlayPreviewFrames = 240;

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

const poseConnections = [
  [11, 12],
  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [17, 19],
  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [18, 20],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [27, 29],
  [27, 31],
  [29, 31],
  [24, 26],
  [26, 28],
  [28, 30],
  [28, 32],
  [30, 32]
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

function nearestScanFrame(frames = [], time = 0) {
  if (!frames.length) return null;
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if ((frames[mid]?.time || 0) < time) low = mid + 1;
    else high = mid;
  }
  const next = frames[low];
  const previous = frames[Math.max(0, low - 1)];
  if (!previous) return next;
  if (!next) return previous;
  return Math.abs((previous.time || 0) - time) <= Math.abs((next.time || 0) - time) ? previous : next;
}

function drawVideoSkeleton(ctx, landmarks, canvas, video, side) {
  if (!landmarks?.length || !canvas?.width || !canvas?.height || !video?.videoWidth || !video?.videoHeight) return;
  const rect = containRect(canvas.width, canvas.height, video.videoWidth, video.videoHeight);
  const projectedLandmarks = projectLandmarksToCanvas(landmarks, rect, canvas.width, canvas.height);
  const visualScale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight);
  const color = side === "left" ? "#28d7a4" : "#55a4ff";
  const drawingUtils = new DrawingUtils(ctx);
  drawingUtils.drawConnectors(projectedLandmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color,
    lineWidth: Math.max(1.5, 4 * visualScale)
  });
  drawingUtils.drawLandmarks(projectedLandmarks, {
    color,
    fillColor: color,
    radius: Math.max(1.2, 2.4 * visualScale)
  });
}

function comparePoseFrames(left, right, regions = defaultMediaPipeSettings.regions, options = {}) {
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
  const fitted = torsoFittedLandmarks(left.landmarks, right.landmarks, options.leftAspect, options.rightAspect);
  const leftAngles = poseAngles(fitted.left, specs);
  const rightAngles = poseAngles(fitted.right, specs);
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

  const score = Math.max(0, Math.min(100, Math.round(weightedScore / Math.max(1, totalWeight))));
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
    diagnostics: {
      torsoLocked: fitted.ready
    },
    verdict: verdictForScore(score, suggestions)
  };
}

function torsoFittedLandmarks(leftLandmarks, rightLandmarks, leftAspect = 1, rightAspect = 1) {
  const left = normalizeSkeleton(leftLandmarks, leftAspect);
  const right = normalizeSkeleton(rightLandmarks, rightAspect);
  if (!left?.length || !right?.length) return { left: leftLandmarks, right: rightLandmarks, ready: false };
  return {
    left,
    right: fitNormalizedSkeletonToReference(left, right),
    ready: true
  };
}

function compareScans(leftScan, rightScan, sync, regions = defaultMediaPipeSettings.regions) {
  if (!leftScan?.frames?.length || !rightScan?.frames?.length) return comparePoseFrames(null, null, regions);

  const offset = sync?.ready ? sync.offsetSeconds : 0;
  const anglePairs = synchronizedAngleFramePairs(leftScan, rightScan, offset);
  const usableFrames = anglePairs.pairs.map((pair) => ({
    ...pair,
    comparison: comparePoseFrames(pair.leftFrame, pair.rightFrame, regions, {
      leftAspect: leftScan?.video?.aspect,
      rightAspect: rightScan?.video?.aspect
    })
  }));

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
    diagnostics: {
      trackingOutliersSkipped: anglePairs.skipped,
      referenceOutliersSkipped: anglePairs.leftSkipped,
      userOutliersSkipped: anglePairs.rightSkipped
    },
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

function fittedPairLandmarks(pair, leftScan, rightScan) {
  const left = normalizeSkeleton(pair.leftFrame.landmarks, leftScan?.video?.aspect);
  const right = normalizeSkeleton(pair.rightFrame.landmarks, rightScan?.video?.aspect);
  if (!left?.length || !right?.length) return null;
  const torsoFittedRight = fitNormalizedSkeletonToReference(left, right);
  return {
    left,
    right: matchBoneLengthsToReference(left, torsoFittedRight),
    torsoFittedRight
  };
}

const boneLengthMatchChains = [
  [11, 13, 15],
  [12, 14, 16],
  [23, 25, 27],
  [24, 26, 28]
];

function matchBoneLengthsToReference(referenceLandmarks, userLandmarks) {
  if (!referenceLandmarks?.length || !userLandmarks?.length) return userLandmarks;
  const fitted = userLandmarks.map((point) => (point ? { ...point } : point));

  for (const [rootId, middleId, endId] of boneLengthMatchChains) {
    fitBoneSegment(referenceLandmarks, fitted, rootId, middleId);
    fitBoneSegment(referenceLandmarks, fitted, middleId, endId);
  }

  return fitted;
}

function fitBoneSegment(referenceLandmarks, targetLandmarks, rootId, childId) {
  const referenceRoot = referenceLandmarks?.[rootId];
  const referenceChild = referenceLandmarks?.[childId];
  const targetRoot = targetLandmarks?.[rootId];
  const targetChild = targetLandmarks?.[childId];
  const referenceLength = pointDistance(referenceRoot, referenceChild);
  if (!referenceRoot || !referenceChild || !targetRoot || !targetChild || !Number.isFinite(referenceLength) || referenceLength <= 0) return;

  const dx = targetChild.x - targetRoot.x;
  const dy = targetChild.y - targetRoot.y;
  const dz = (targetChild.z || 0) - (targetRoot.z || 0);
  const targetLength = Math.hypot(dx, dy, dz);
  if (!Number.isFinite(targetLength) || targetLength <= 0.000001) {
    targetLandmarks[childId] = { ...targetChild, x: referenceChild.x, y: referenceChild.y, z: referenceChild.z || 0 };
    return;
  }

  const scale = referenceLength / targetLength;
  targetLandmarks[childId] = {
    ...targetChild,
    x: targetRoot.x + dx * scale,
    y: targetRoot.y + dy * scale,
    z: (targetRoot.z || 0) + dz * scale
  };
}

function compareZonesDrawingScans(leftScan, rightScan, sync, hybridMethods = defaultHybridMethodSettings) {
  if (!leftScan?.frames?.length || !rightScan?.frames?.length) return comparePoseFrames(null, null);

  const methods = normalizeHybridMethodSettings(hybridMethods);
  const offset = sync?.ready ? sync.offsetSeconds : 0;
  const anglePairs = synchronizedAngleFramePairs(leftScan, rightScan, offset);
  const pairs = anglePairs.pairs
    .map((pair) => ({ ...pair, fitted: fittedPairLandmarks(pair, leftScan, rightScan) }))
    .filter((pair) => pair.fitted?.left?.length && pair.fitted?.right?.length);

  if (!pairs.length) return comparePoseFrames(null, null);

  const methodResults = [];
  if (methods.zones) methodResults.push(compareJointZonesPairs(pairs));
  if (methods.drawing) methodResults.push(compareTrajectoryDrawingPairs(pairs));

  const scoreWeight = methodResults.reduce((sum, item) => sum + item.weight, 0);
  const score = clampPercent(methodResults.reduce((sum, item) => sum + item.score * item.weight, 0) / Math.max(1, scoreWeight));
  const arms = clampPercent(averageNumbers(methodResults.map((item) => item.bodyParts?.arms).filter(Number.isFinite)));
  const legs = clampPercent(averageNumbers(methodResults.map((item) => item.bodyParts?.legs).filter(Number.isFinite)));
  const rows = methodResults.flatMap((item) => item.rows);
  const suggestions = methodResults.flatMap((item) => item.suggestions).slice(0, 5);
  const frameScores = pairs.map((pair) => compareZoneFrameScore(pair).score).filter(Number.isFinite);
  const worstFrame = pairs
    .map((pair) => ({ pair, score: compareZoneFrameScore(pair).score }))
    .filter((item) => Number.isFinite(item.score))
    .sort((a, b) => a.score - b.score)[0];

  return {
    ready: true,
    method: "Области + Рисунок",
    score,
    finalScore: score,
    rows,
    suggestions,
    framesCompared: pairs.length,
    bestScore: clampPercent(Math.max(...frameScores)),
    worstScore: clampPercent(Math.min(...frameScores)),
    durationCompared: Number((pairs.at(-1).leftTime - pairs[0].leftTime).toFixed(1)),
    worstMoment: worstFrame
      ? {
          leftTime: worstFrame.pair.leftTime,
          rightTime: worstFrame.pair.rightTime,
          score: clampPercent(worstFrame.score)
        }
      : null,
    bodyParts: {
      arms,
      legs,
      torso: 100,
      head: 0,
      rhythm: sync?.ready ? clampPercent((sync.confidence || 0) * 100) : 75
    },
    diagnostics: {
      hybridMethods: methods,
      boneLengthNormalization: "reference-limb-bones",
      zoneDrawingJoints: zoneDrawingJointSpecs.length,
      trackingOutliersSkipped: anglePairs.skipped,
      referenceOutliersSkipped: anglePairs.leftSkipped,
      userOutliersSkipped: anglePairs.rightSkipped
    },
    verdict: verdictForScore(score, suggestions)
  };
}

function synchronizedFittedPairs(leftScan, rightScan, sync) {
  const offset = sync?.ready ? sync.offsetSeconds : 0;
  const anglePairs = synchronizedAngleFramePairs(leftScan, rightScan, offset);
  const pairs = anglePairs.pairs
    .map((pair) => ({ ...pair, fitted: fittedPairLandmarks(pair, leftScan, rightScan) }))
    .filter((pair) => pair.fitted?.left?.length && pair.fitted?.right?.length);
  return { ...anglePairs, pairs };
}

function compareJointAreasScans(leftScan, rightScan, sync) {
  if (!leftScan?.frames?.length || !rightScan?.frames?.length) return comparePoseFrames(null, null);
  const anglePairs = synchronizedFittedPairs(leftScan, rightScan, sync);
  const pairs = anglePairs.pairs;
  if (!pairs.length) return comparePoseFrames(null, null);

  const areas = compareJointZonesPairs(pairs, { prefix: "joint-areas", totalTitle: "Области: все активные суставы" });
  const frameScores = pairs.map((pair) => compareZoneFrameScore(pair).score).filter(Number.isFinite);
  const worstFrame = pairs
    .map((pair) => ({ pair, score: compareZoneFrameScore(pair).score }))
    .filter((item) => Number.isFinite(item.score))
    .sort((a, b) => a.score - b.score)[0];
  const score = clampPercent(areas.score);

  return {
    ready: true,
    method: "Области",
    score,
    finalScore: score,
    rows: areas.rows,
    suggestions: areas.suggestions,
    framesCompared: pairs.length,
    bestScore: clampPercent(Math.max(...frameScores)),
    worstScore: clampPercent(Math.min(...frameScores)),
    durationCompared: Number((pairs.at(-1).leftTime - pairs[0].leftTime).toFixed(1)),
    worstMoment: worstFrame
      ? {
          leftTime: worstFrame.pair.leftTime,
          rightTime: worstFrame.pair.rightTime,
          score: clampPercent(worstFrame.score)
        }
      : null,
    bodyParts: {
      arms: clampPercent(areas.bodyParts?.arms),
      legs: clampPercent(areas.bodyParts?.legs),
      torso: 100,
      head: 0,
      rhythm: sync?.ready ? clampPercent((sync.confidence || 0) * 100) : 75
    },
    diagnostics: {
      areaHitRule: "inside-area-is-100-outside-has-falloff",
      boneLengthNormalization: "reference-limb-bones",
      trackingOutliersSkipped: anglePairs.skipped,
      referenceOutliersSkipped: anglePairs.leftSkipped,
      userOutliersSkipped: anglePairs.rightSkipped
    },
    verdict: verdictForScore(score, areas.suggestions)
  };
}

function compareTrajectoryDrawingScans(leftScan, rightScan, sync) {
  if (!leftScan?.frames?.length || !rightScan?.frames?.length) return comparePoseFrames(null, null);
  const anglePairs = synchronizedFittedPairs(leftScan, rightScan, sync);
  const pairs = anglePairs.pairs;
  if (!pairs.length) return comparePoseFrames(null, null);

  const drawing = compareTrajectoryDrawingPairs(pairs, { prefix: "trajectory-drawing" });
  const score = clampPercent(drawing.score);
  return {
    ready: true,
    method: "Рисунок",
    score,
    finalScore: score,
    rows: drawing.rows,
    suggestions: drawing.suggestions,
    framesCompared: pairs.length,
    bestScore: null,
    worstScore: null,
    durationCompared: Number((pairs.at(-1).leftTime - pairs[0].leftTime).toFixed(1)),
    worstMoment: null,
    bodyParts: {
      arms: clampPercent(drawing.bodyParts?.arms),
      legs: clampPercent(drawing.bodyParts?.legs),
      torso: 100,
      head: 0,
      rhythm: sync?.ready ? clampPercent((sync.confidence || 0) * 100) : 75
    },
    diagnostics: {
      trajectorySegmentSeconds: 2,
      boneLengthNormalization: "reference-limb-bones",
      trackingOutliersSkipped: anglePairs.skipped,
      referenceOutliersSkipped: anglePairs.leftSkipped,
      userOutliersSkipped: anglePairs.rightSkipped
    },
    verdict: verdictForScore(score, drawing.suggestions)
  };
}

function gridPairLandmarks(pair, leftScan, rightScan) {
  const left = normalizeSkeleton(pair.leftFrame.landmarks, leftScan?.video?.aspect);
  const right = normalizeSkeleton(pair.rightFrame.landmarks, rightScan?.video?.aspect);
  if (!left?.length || !right?.length) return null;
  return {
    left,
    right: matchBoneLengthsToReference(left, right)
  };
}

function compareZoneGridScans(leftScan, rightScan, sync) {
  if (!leftScan?.frames?.length || !rightScan?.frames?.length) return comparePoseFrames(null, null);

  const offset = sync?.ready ? sync.offsetSeconds : 0;
  const anglePairs = synchronizedAngleFramePairs(leftScan, rightScan, offset);
  const pairs = anglePairs.pairs
    .map((pair) => ({ ...pair, fitted: gridPairLandmarks(pair, leftScan, rightScan) }))
    .filter((pair) => pair.fitted?.left?.length && pair.fitted?.right?.length);

  if (!pairs.length) return comparePoseFrames(null, null);

  const scores = [];
  const groupScores = { arms: [], legs: [] };
  const jointAverages = new Map();
  const frameScores = [];
  let worstFrame = null;

  for (const pair of pairs) {
    const frameScoresForJoints = [];
    for (const spec of zoneDrawingJointSpecs) {
      const leftCell = pointToZoneCell(pair.fitted.left?.[spec.id]);
      const rightCell = pointToZoneCell(pair.fitted.right?.[spec.id]);
      if (!leftCell || !rightCell) continue;
      const score = zoneCellScore(leftCell, rightCell);
      scores.push(score);
      frameScoresForJoints.push(score);
      groupScores[spec.group]?.push(score);
      const current = jointAverages.get(spec.id) || { spec, scores: [] };
      current.scores.push(score);
      jointAverages.set(spec.id, current);
    }
    const frameScore = averageNumbers(frameScoresForJoints);
    if (Number.isFinite(frameScore)) {
      frameScores.push(frameScore);
      if (!worstFrame || frameScore < worstFrame.score) worstFrame = { pair, score: frameScore };
    }
  }

  const jointRows = Array.from(jointAverages.values())
    .map(({ spec, scores: itemScores }) => rowPercent(`zone-grid-${spec.key}`, `Сетка: ${spec.title}`, averageNumbers(itemScores)))
    .sort((a, b) => a.score - b.score);
  const worst = jointRows.slice(0, 4);
  const score = clampPercent(averageNumbers(scores));
  const suggestions = worst.map((row) => `${row.title.replace("Сетка: ", "")}: чаще попадает в другой сектор сетки.`);

  return {
    ready: true,
    method: "Зоны",
    score,
    finalScore: score,
    rows: [
      rowPercent("zone-grid-total", "Зоны: все активные точки", score),
      rowPercent("zone-grid-arms", "Зоны рук", averageNumbers(groupScores.arms)),
      rowPercent("zone-grid-legs", "Зоны ног", averageNumbers(groupScores.legs)),
      ...worst
    ],
    suggestions,
    framesCompared: pairs.length,
    bestScore: clampPercent(Math.max(...frameScores)),
    worstScore: clampPercent(Math.min(...frameScores)),
    durationCompared: Number((pairs.at(-1).leftTime - pairs[0].leftTime).toFixed(1)),
    worstMoment: worstFrame
      ? {
          leftTime: worstFrame.pair.leftTime,
          rightTime: worstFrame.pair.rightTime,
          score: clampPercent(worstFrame.score)
        }
      : null,
    bodyParts: {
      arms: clampPercent(averageNumbers(groupScores.arms)),
      legs: clampPercent(averageNumbers(groupScores.legs)),
      torso: 100,
      head: 0,
      rhythm: sync?.ready ? clampPercent((sync.confidence || 0) * 100) : 75
    },
    diagnostics: {
      zoneGrid: zoneGridConfig,
      comparedJoints: zoneDrawingJointSpecs.map((spec) => spec.key),
      boneLengthNormalization: "reference-limb-bones",
      skeletonAlignment: "separate-centered-grids",
      trackingOutliersSkipped: anglePairs.skipped,
      referenceOutliersSkipped: anglePairs.leftSkipped,
      userOutliersSkipped: anglePairs.rightSkipped
    },
    verdict: verdictForScore(score, suggestions)
  };
}

function pointToZoneCell(point) {
  if (!point) return null;
  const { columns, rows, xMin, xMax, yMin, yMax } = zoneGridConfig;
  const xRatio = (point.x - xMin) / (xMax - xMin);
  const yRatio = (point.y - yMin) / (yMax - yMin);
  if (!Number.isFinite(xRatio) || !Number.isFinite(yRatio)) return null;
  return {
    col: Math.max(0, Math.min(columns - 1, Math.floor(xRatio * columns))),
    row: Math.max(0, Math.min(rows - 1, Math.floor(yRatio * rows)))
  };
}

function zoneCellScore(leftCell, rightCell) {
  const distance = Math.hypot(leftCell.col - rightCell.col, leftCell.row - rightCell.row);
  if (distance <= 0) return 100;
  if (distance <= 1) return 86;
  return clampPercent(100 - distance * 18);
}

function compareJointZonesPairs(pairs, options = {}) {
  const prefix = options.prefix || "zones-drawing";
  const totalTitle = options.totalTitle || "Области суставов";
  const scores = [];
  const groupScores = { arms: [], legs: [] };
  const jointAverages = new Map();

  for (const pair of pairs) {
    for (const spec of zoneDrawingJointSpecs) {
      const leftPoint = pair.fitted.left?.[spec.id];
      const rightPoint = pair.fitted.right?.[spec.id];
      const distance = pointDistance(leftPoint, rightPoint);
      if (!Number.isFinite(distance)) continue;
      const score = areaHitScore(distance, spec.radius);
      scores.push(score);
      groupScores[spec.group]?.push(score);
      const current = jointAverages.get(spec.id) || { spec, scores: [] };
      current.scores.push(score);
      jointAverages.set(spec.id, current);
    }
  }

  const jointRows = Array.from(jointAverages.values())
    .map(({ spec, scores: itemScores }) => rowPercent(`${prefix}-zone-${spec.key}`, `Область: ${spec.title}`, averageNumbers(itemScores)))
    .sort((a, b) => a.score - b.score);
  const worst = jointRows.slice(0, 3);

  return {
    id: "zones",
    weight: 0.45,
    score: clampPercent(averageNumbers(scores)),
    bodyParts: {
      arms: averageNumbers(groupScores.arms),
      legs: averageNumbers(groupScores.legs)
    },
    rows: [
      rowPercent(`${prefix}-zones`, totalTitle, averageNumbers(scores)),
      rowPercent(`${prefix}-zones-arms`, "Области рук", averageNumbers(groupScores.arms)),
      rowPercent(`${prefix}-zones-legs`, "Области ног", averageNumbers(groupScores.legs)),
      ...worst
    ],
    suggestions: worst.map((row) => `${row.title.replace("Область: ", "")}: чаще всего выходит из своей зоны.`)
  };
}

function areaHitScore(distance, radius) {
  if (distance <= radius) return 100;
  return clampPercent(100 - ((distance - radius) / Math.max(radius, 0.001)) * 70);
}

function compareTrajectoryDrawingPairs(pairs, options = {}) {
  const prefix = options.prefix || "zones-drawing";
  const segmentSeconds = 2;
  const startTime = pairs[0]?.leftTime || 0;
  const segmentMap = new Map();
  for (const pair of pairs) {
    const segmentId = Math.floor(((pair.leftTime || 0) - startTime) / segmentSeconds);
    if (!segmentMap.has(segmentId)) segmentMap.set(segmentId, []);
    segmentMap.get(segmentId).push(pair);
  }

  const scores = [];
  const groupScores = { arms: [], legs: [] };
  const rows = [];

  for (const spec of zoneDrawingJointSpecs) {
    const jointScores = [];
    for (const segmentPairs of segmentMap.values()) {
      const score = compareTrajectorySegment(segmentPairs, spec.id);
      if (!Number.isFinite(score)) continue;
      jointScores.push(score);
      scores.push(score);
      groupScores[spec.group]?.push(score);
    }
    if (jointScores.length) {
      rows.push(rowPercent(`${prefix}-trajectory-${spec.key}`, `Рисунок: ${spec.title}`, averageNumbers(jointScores)));
    }
  }

  const worst = rows.sort((a, b) => a.score - b.score).slice(0, 3);
  return {
    id: "drawing",
    weight: 0.55,
    score: clampPercent(averageNumbers(scores)),
    bodyParts: {
      arms: averageNumbers(groupScores.arms),
      legs: averageNumbers(groupScores.legs)
    },
    rows: [
      rowPercent(`${prefix}-trajectory`, "Рисунок движения", averageNumbers(scores)),
      rowPercent(`${prefix}-trajectory-arms`, "Рисунок рук", averageNumbers(groupScores.arms)),
      rowPercent(`${prefix}-trajectory-legs`, "Рисунок ног", averageNumbers(groupScores.legs)),
      ...worst
    ],
    suggestions: worst.map((row) => `${row.title.replace("Рисунок: ", "")}: траектория отличается от эталона.`)
  };
}

function compareZoneFrameScore(pair) {
  const scores = zoneDrawingJointSpecs
    .map((spec) => {
      const distance = pointDistance(pair.fitted.left?.[spec.id], pair.fitted.right?.[spec.id]);
      return Number.isFinite(distance) ? areaHitScore(distance, spec.radius) : null;
    })
    .filter(Number.isFinite);
  return { score: averageNumbers(scores) };
}

function compareTrajectorySegment(segmentPairs, jointId) {
  const leftPath = segmentPairs.map((pair) => pair.fitted.left?.[jointId]).filter(Boolean);
  const rightPath = segmentPairs.map((pair) => pair.fitted.right?.[jointId]).filter(Boolean);
  if (leftPath.length < 3 || rightPath.length < 3) return null;

  const count = Math.min(leftPath.length, rightPath.length);
  const leftSample = resamplePath(leftPath, count);
  const rightSample = resamplePath(rightPath, count);
  const leftLength = pathLength(leftSample);
  const rightLength = pathLength(rightSample);
  const movement = Math.max(leftLength, rightLength);
  if (movement < 0.08) return null;

  const leftNormalized = normalizeTrajectory(leftSample);
  const rightNormalized = normalizeTrajectory(rightSample);
  const shapeDistance = averageNumbers(leftNormalized.map((point, index) => pointDistance(point, rightNormalized[index])));
  const shapeScore = clampPercent(100 - shapeDistance * 80);
  const lengthScore = clampPercent(100 - Math.abs(leftLength - rightLength) / Math.max(0.001, Math.max(leftLength, rightLength)) * 100);
  const directionScore = trajectoryDirectionScore(leftSample, rightSample);
  return clampPercent(shapeScore * 0.58 + directionScore * 0.24 + lengthScore * 0.18);
}

function resamplePath(points, count) {
  if (points.length === count) return points;
  return Array.from({ length: count }, (_, index) => {
    const sourceIndex = (index / Math.max(1, count - 1)) * (points.length - 1);
    const low = Math.floor(sourceIndex);
    const high = Math.min(points.length - 1, Math.ceil(sourceIndex));
    const t = sourceIndex - low;
    const a = points[low];
    const b = points[high];
    return {
      x: a.x + (b.x - a.x) * t,
      y: a.y + (b.y - a.y) * t,
      z: (a.z || 0) + ((b.z || 0) - (a.z || 0)) * t
    };
  });
}

function pathLength(points) {
  let length = 0;
  for (let index = 1; index < points.length; index += 1) {
    length += pointDistance(points[index - 1], points[index]) || 0;
  }
  return length;
}

function normalizeTrajectory(points) {
  const origin = points[0];
  const scale = Math.max(0.001, pathLength(points));
  return points.map((point) => ({
    x: (point.x - origin.x) / scale,
    y: (point.y - origin.y) / scale,
    z: ((point.z || 0) - (origin.z || 0)) / scale
  }));
}

const activityJointSpecs = [
  { id: 11, key: "leftShoulder", title: "Левое плечо", group: "torso", weight: 0.7 },
  { id: 12, key: "rightShoulder", title: "Правое плечо", group: "torso", weight: 0.7 },
  { id: 13, key: "leftElbow", title: "Левый локоть", group: "arms", weight: 1 },
  { id: 14, key: "rightElbow", title: "Правый локоть", group: "arms", weight: 1 },
  { id: 15, key: "leftWrist", title: "Левая кисть", group: "arms", weight: 1.25 },
  { id: 16, key: "rightWrist", title: "Правая кисть", group: "arms", weight: 1.25 },
  { id: 23, key: "leftHip", title: "Левое бедро", group: "torso", weight: 0.7 },
  { id: 24, key: "rightHip", title: "Правое бедро", group: "torso", weight: 0.7 },
  { id: 25, key: "leftKnee", title: "Левое колено", group: "legs", weight: 1 },
  { id: 26, key: "rightKnee", title: "Правое колено", group: "legs", weight: 1 },
  { id: 27, key: "leftAnkle", title: "Левая стопа", group: "legs", weight: 1.15 },
  { id: 28, key: "rightAnkle", title: "Правая стопа", group: "legs", weight: 1.15 }
];

function compareActivityScans(leftScan, rightScan, sync) {
  if (!leftScan?.frames?.length || !rightScan?.frames?.length) return comparePoseFrames(null, null);

  const offset = sync?.ready ? sync.offsetSeconds : 0;
  const anglePairs = synchronizedAngleFramePairs(leftScan, rightScan, offset);
  if (!anglePairs.pairs.length) return comparePoseFrames(null, null);

  const leftProfile = buildActivityProfile(leftScan);
  const rightProfile = buildActivityProfile({
    ...rightScan,
    frames: rightScan.frames.map((frame) => ({ ...frame, time: Number((frame.time - offset).toFixed(3)) }))
  });
  const frameActivity = buildPairedActivityFrames(anglePairs.pairs, leftScan, rightScan);
  const activityMatch = activitySimilarityScore(leftProfile.total.activity, rightProfile.total.activity);
  const amplitudeMatch = activitySimilarityScore(leftProfile.total.amplitudePercent, rightProfile.total.amplitudePercent);
  const phraseMatch = clampPercent(averageNumbers(frameActivity.map((item) => item.score)));
  const score = clampPercent(activityMatch * 0.52 + phraseMatch * 0.32 + amplitudeMatch * 0.16);
  const groupMatches = ["arms", "legs", "torso"].map((group) => ({
    group,
    title: activityGroupTitle(group),
    score: activitySimilarityScore(leftProfile.groups[group]?.activity || 0, rightProfile.groups[group]?.activity || 0)
  }));
  const weakestGroup = [...groupMatches].sort((a, b) => a.score - b.score)[0];
  const worstFrame = [...frameActivity].sort((a, b) => a.score - b.score)[0] || null;
  const frameScores = frameActivity.map((item) => item.score);

  return {
    ready: true,
    method: "Активность",
    score,
    finalScore: score,
    rows: [
      rowPercent("activity-match-total", "Активность: совпадение уровня", score),
      rowPercent("activity-reference-level", "Активность эталона", leftProfile.total.activity),
      rowPercent("activity-user-level", "Активность правого видео", rightProfile.total.activity),
      rowPercent("activity-arms", "Активность рук совпадает", groupMatches.find((item) => item.group === "arms")?.score),
      rowPercent("activity-legs", "Активность ног совпадает", groupMatches.find((item) => item.group === "legs")?.score),
      rowPercent("activity-torso", "Активность корпуса совпадает", groupMatches.find((item) => item.group === "torso")?.score),
      rowPercent("activity-phrase", "Активность по фразе", phraseMatch)
    ],
    suggestions: activitySuggestions(leftProfile, rightProfile, weakestGroup),
    framesCompared: anglePairs.pairs.length,
    bestScore: frameScores.length ? clampPercent(Math.max(...frameScores)) : null,
    worstScore: frameScores.length ? clampPercent(Math.min(...frameScores)) : null,
    durationCompared: Number((anglePairs.pairs.at(-1).leftTime - anglePairs.pairs[0].leftTime).toFixed(1)),
    worstMoment: worstFrame
      ? {
          leftTime: worstFrame.leftTime,
          rightTime: worstFrame.rightTime,
          score: worstFrame.score
        }
      : null,
    bodyParts: {
      arms: groupMatches.find((item) => item.group === "arms")?.score || 0,
      legs: groupMatches.find((item) => item.group === "legs")?.score || 0,
      torso: groupMatches.find((item) => item.group === "torso")?.score || 0,
      head: 0,
      rhythm: sync?.ready ? clampPercent((sync.confidence || 0) * 100) : 75
    },
    diagnostics: {
      activityReference: leftProfile,
      activityUser: rightProfile,
      activityMatch,
      amplitudeMatch,
      phraseMatch,
      trackingOutliersSkipped: anglePairs.skipped,
      referenceOutliersSkipped: anglePairs.leftSkipped,
      userOutliersSkipped: anglePairs.rightSkipped
    },
    verdict: activityVerdict(score, leftProfile.total.activity, rightProfile.total.activity, weakestGroup)
  };
}

function buildActivityProfile(scan) {
  const filtered = filterAngleScanFrames(scan);
  const normalizedFrames = filtered.frames
    .map((frame) => ({
      frame,
      landmarks: normalizeSkeleton(frame.landmarks, scan?.video?.aspect)
    }))
    .filter((item) => item.landmarks?.length);
  const byJoint = new Map();
  const groups = {
    arms: emptyActivityBucket(),
    legs: emptyActivityBucket(),
    torso: emptyActivityBucket()
  };

  for (const spec of activityJointSpecs) byJoint.set(spec.id, { spec, speeds: [], points: [] });

  for (const item of normalizedFrames) {
    for (const spec of activityJointSpecs) {
      const point = item.landmarks?.[spec.id];
      if (point) byJoint.get(spec.id).points.push(point);
    }
  }

  for (let index = 1; index < normalizedFrames.length; index += 1) {
    const previous = normalizedFrames[index - 1];
    const current = normalizedFrames[index];
    const dt = Math.max(0.001, (current.frame.time || 0) - (previous.frame.time || 0));
    if (dt > 1.15) continue;
    for (const spec of activityJointSpecs) {
      const distance = pointDistance(previous.landmarks?.[spec.id], current.landmarks?.[spec.id]);
      if (!Number.isFinite(distance) || distance > 1.45) continue;
      byJoint.get(spec.id).speeds.push(distance / dt);
    }
  }

  const jointProfiles = Array.from(byJoint.values()).map(({ spec, speeds, points }) => {
    const speed = trimmedAverage(speeds, 0.12);
    const amplitude = activityPathAmplitude(points);
    const speedPercent = activityRawToPercent(speed);
    const amplitudePercent = activityRawToPercent(amplitude);
    const activity = clampPercent(speedPercent * 0.72 + amplitudePercent * 0.28);
    const profile = {
      id: spec.id,
      key: spec.key,
      title: spec.title,
      group: spec.group,
      speed: Number(speed.toFixed(4)),
      amplitude: Number(amplitude.toFixed(4)),
      speedPercent,
      amplitudePercent,
      activity,
      weight: spec.weight
    };
    const bucket = groups[spec.group];
    bucket.weightedActivity += activity * spec.weight;
    bucket.weightedAmplitudePercent += amplitudePercent * spec.weight;
    bucket.weight += spec.weight;
    return profile;
  });

  for (const bucket of Object.values(groups)) {
    bucket.activity = clampPercent(bucket.weightedActivity / Math.max(0.001, bucket.weight));
    bucket.amplitudePercent = clampPercent(bucket.weightedAmplitudePercent / Math.max(0.001, bucket.weight));
  }

  const totalWeight = jointProfiles.reduce((sum, item) => sum + item.weight, 0);
  const totalActivity = jointProfiles.reduce((sum, item) => sum + item.activity * item.weight, 0) / Math.max(0.001, totalWeight);
  const totalAmplitude = jointProfiles.reduce((sum, item) => sum + item.amplitudePercent * item.weight, 0) / Math.max(0.001, totalWeight);

  return {
    frames: normalizedFrames.length,
    skipped: filtered.skipped,
    duration: Number(((normalizedFrames.at(-1)?.frame.time || 0) - (normalizedFrames[0]?.frame.time || 0)).toFixed(2)),
    total: {
      activity: clampPercent(totalActivity),
      amplitudePercent: clampPercent(totalAmplitude)
    },
    groups: {
      arms: compactActivityBucket(groups.arms),
      legs: compactActivityBucket(groups.legs),
      torso: compactActivityBucket(groups.torso)
    },
    joints: jointProfiles
  };
}

function buildPairedActivityFrames(pairs, leftScan, rightScan) {
  const result = [];
  let previous = null;
  for (const pair of pairs) {
    const fitted = fittedPairLandmarks(pair, leftScan, rightScan);
    if (!fitted?.left?.length || !fitted?.right?.length) continue;
    if (previous) {
      const leftDt = Math.max(0.001, pair.leftTime - previous.leftTime);
      const rightDt = Math.max(0.001, pair.rightTime - previous.rightTime);
      const leftMovement = averageJointMovement(previous.fitted.left, fitted.left, leftDt);
      const rightMovement = averageJointMovement(previous.fitted.right, fitted.right, rightDt);
      result.push({
        leftTime: pair.leftTime,
        rightTime: pair.rightTime,
        leftMovement,
        rightMovement,
        score: activitySimilarityScore(activityRawToPercent(leftMovement), activityRawToPercent(rightMovement))
      });
    }
    previous = { ...pair, fitted };
  }
  return result;
}

function averageJointMovement(previous, current, dt) {
  const movements = activityJointSpecs
    .map((spec) => {
      const distance = pointDistance(previous?.[spec.id], current?.[spec.id]);
      return Number.isFinite(distance) && distance <= 1.45 ? (distance / dt) * spec.weight : null;
    })
    .filter(Number.isFinite);
  return trimmedAverage(movements, 0.12);
}

function activityRawToPercent(value) {
  if (!Number.isFinite(value) || value <= 0.004) return 0;
  return clampPercent(100 * (1 - Math.exp(-value / 0.62)));
}

function activitySimilarityScore(referenceActivity, userActivity) {
  const reference = clampPercent(referenceActivity);
  const user = clampPercent(userActivity);
  if (reference < 8 && user < 8) return 100;
  if (reference >= 18 && user < 8) return clampPercent(12 + user * 1.4);
  const ratioDiff = Math.abs(reference - user) / Math.max(reference, 18);
  return clampPercent(100 - ratioDiff * 105);
}

function activityPathAmplitude(points) {
  if (!points?.length) return 0;
  const xs = points.map((point) => point.x).filter(Number.isFinite);
  const ys = points.map((point) => point.y).filter(Number.isFinite);
  const zs = points.map((point) => point.z || 0).filter(Number.isFinite);
  return Math.hypot(rangeNumbers(xs), rangeNumbers(ys), rangeNumbers(zs));
}

function trimmedAverage(values, trimRatio = 0.1) {
  const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!clean.length) return 0;
  const trim = Math.floor(clean.length * trimRatio);
  const trimmed = clean.slice(trim, Math.max(trim + 1, clean.length - trim));
  return averageNumbers(trimmed);
}

function emptyActivityBucket() {
  return {
    weightedActivity: 0,
    weightedAmplitudePercent: 0,
    weight: 0,
    activity: 0,
    amplitudePercent: 0
  };
}

function compactActivityBucket(bucket) {
  return {
    activity: clampPercent(bucket.activity),
    amplitudePercent: clampPercent(bucket.amplitudePercent)
  };
}

function activityGroupTitle(group) {
  if (group === "arms") return "руки";
  if (group === "legs") return "ноги";
  return "корпус";
}

function activitySuggestions(reference, user, weakestGroup) {
  const suggestions = [];
  if (reference.total.activity >= 25 && user.total.activity < 12) {
    suggestions.push("Правое видео почти не набирает движение относительно активного эталона.");
  } else if (user.total.activity < reference.total.activity - 18) {
    suggestions.push("В правом видео активности меньше: движение выглядит более сдержанным, чем в эталоне.");
  } else if (user.total.activity > reference.total.activity + 22) {
    suggestions.push("В правом видео активности больше: движение может быть резче или суетливее эталона.");
  } else {
    suggestions.push("Общий уровень движения близок к эталону.");
  }
  if (weakestGroup) suggestions.push(`Сильнее всего отличается активность группы: ${weakestGroup.title}.`);
  return suggestions;
}

function activityVerdict(score, referenceActivity, userActivity, weakestGroup) {
  if (referenceActivity >= 25 && userActivity < 12) {
    return "Эталон активно двигается, а в правом видео движение почти отсутствует. Эта модель уверенно видит стояние или очень слабую активность.";
  }
  if (score >= 86) return "Уровень активности правого видео хорошо совпадает с эталоном: движение по энергии и амплитуде близкое.";
  if (score >= 68) return `Активность в целом похожа, но ${weakestGroup?.title || "одна из зон"} двигается не так энергично, как в эталоне.`;
  if (score >= 45) return `Активность совпадает частично: правое видео заметно отличается по уровню движения, особенно в зоне ${weakestGroup?.title || "рук или ног"}.`;
  return "Активность правого видео сильно не похожа на эталон. Это хороший сигнал для отсеивания стоящего человека или слишком слабого выполнения.";
}

function trajectoryDirectionScore(leftPath, rightPath) {
  const leftStart = leftPath[0];
  const leftEnd = leftPath.at(-1);
  const rightStart = rightPath[0];
  const rightEnd = rightPath.at(-1);
  const leftVector = { x: leftEnd.x - leftStart.x, y: leftEnd.y - leftStart.y };
  const rightVector = { x: rightEnd.x - rightStart.x, y: rightEnd.y - rightStart.y };
  const leftLength = Math.hypot(leftVector.x, leftVector.y);
  const rightLength = Math.hypot(rightVector.x, rightVector.y);
  if (leftLength < 0.001 || rightLength < 0.001) return 50;
  const cosine = (leftVector.x * rightVector.x + leftVector.y * rightVector.y) / (leftLength * rightLength);
  return clampPercent(((Math.max(-1, Math.min(1, cosine)) + 1) / 2) * 100);
}

function compareByModel(model, leftScan, rightScan, sync, regions, leftAudio, hybridMethods = defaultHybridMethodSettings) {
  if (model === "openai-expert") return pendingOpenAiComparison();
  if (model === "overlay") return compareOverlayScans(leftScan, rightScan, sync, regions);
  if (model === "poses") return compareImpulsePoseScans(leftScan, rightScan, sync, regions, leftAudio);
  if (model === "2026-07-06") return compareScans20260706(leftScan, rightScan, sync);
  if (model === "2026-07-12") return compareScans20260712(leftScan, rightScan, sync);
  if (model === "2026-07-13") return compareScans20260713(leftScan, rightScan, sync);
  if (model === "zones-drawing") return compareZonesDrawingScans(leftScan, rightScan, sync, hybridMethods);
  if (model === "joint-areas") return compareJointAreasScans(leftScan, rightScan, sync);
  if (model === "trajectory-drawing") return compareTrajectoryDrawingScans(leftScan, rightScan, sync);
  if (model === "zone-grid") return compareZoneGridScans(leftScan, rightScan, sync);
  if (model === "activity") return compareActivityScans(leftScan, rightScan, sync);
  return compareScans(leftScan, rightScan, sync, regions);
}

async function compareByModelAsync(model, leftScan, rightScan, sync, regions, leftAudio, mediaPipeSettings, hybridMethods = defaultHybridMethodSettings) {
  if (model === "openai-expert") return compareScansOpenAiExpert(leftScan, rightScan, sync, regions, leftAudio, mediaPipeSettings);
  return compareByModel(model, leftScan, rightScan, sync, regions, leftAudio, hybridMethods);
}

function pendingOpenAiComparison() {
  return {
    ready: false,
    method: "OpenAI эксперт",
    score: 0,
    finalScore: 0,
    rows: [],
    suggestions: [],
    framesCompared: 0,
    diagnostics: {
      weakPoints: ["Запустите полный анализ: OpenAI эксперт работает через серверный API и не считается в live-режиме."],
      confidence: 0
    },
    verdict: "OpenAI эксперт готов к расчету после полного анализа сохраненных скелетов."
  };
}

async function compareScansOpenAiExpert(leftScan, rightScan, sync, regions, leftAudio, mediaPipeSettings) {
  const local = compareScans20260713(leftScan, rightScan, sync);
  const payload = buildOpenAiComparisonPayload({ leftScan, rightScan, sync, regions, leftAudio, mediaPipeSettings, local });
  let response = null;
  let data = null;
  try {
    response = await fetch("/api/openai-compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    data = await response.json().catch(() => null);
  } catch (err) {
    data = { message: err instanceof Error ? err.message : String(err) };
  }
  if (!response?.ok || !data?.ready) {
    const detail = data?.message || data?.error || `HTTP ${response?.status || 0}`;
    return {
      ...local,
      method: "OpenAI эксперт",
      score: local.score,
      finalScore: local.score,
      diagnostics: {
        ...(local.diagnostics || {}),
        openAiReady: false,
        openAiError: detail
      },
      rows: [
        ...(local.rows || []),
        rowPercent("openai-expert-status", "OpenAI эксперт: статус", 0)
      ],
      suggestions: [`OpenAI эксперт не рассчитался: ${detail}`, ...(local.suggestions || [])].slice(0, 8),
      verdict: `OpenAI эксперт недоступен, показана локальная оценка ${local.score}%. Причина: ${detail}`
    };
  }
  const aiScore = clampPercent(data.finalDisplayedScore ?? data.score ?? data.finalScore ?? local.score);
  const rows = [
    rowPercent("openai-expert-score", "OpenAI эксперт: итоговая оценка", aiScore),
    rowPercent("openai-expert-choreography", "OpenAI эксперт: хореография", data.choreographyScore ?? aiScore),
    rowPercent("openai-expert-tracking", "OpenAI эксперт: надежность скана", data.trackingQualityScore ?? local.diagnostics?.trackingQualityGate?.ceiling ?? 100),
    ...(local.rows || []).slice(0, 8)
  ];
  return {
    ...local,
    ready: true,
    method: "OpenAI эксперт",
    score: aiScore,
    finalScore: aiScore,
    poseScore: clampPercent(data.choreographyScore ?? local.poseScore),
    motionScore: clampPercent(data.choreographyScore ?? local.motionScore),
    timingScore: clampPercent(data.rhythmScore ?? local.timingScore),
    rows,
    suggestions: Array.isArray(data.suggestions) && data.suggestions.length ? data.suggestions.slice(0, 8) : local.suggestions,
    diagnostics: {
      ...(local.diagnostics || {}),
      openAiReady: true,
      openAiModel: data.openAiModel || "server-default",
      openAiConfidence: clampPercent(data.confidence ?? 70),
      openAiTrackingQualityScore: clampPercent(data.trackingQualityScore ?? 100),
      openAiReasoning: data.reasoning || "",
      openAiEvidenceGateApplied: Boolean(data.evidenceGateApplied),
      openAiEvidenceGateReason: data.evidenceGateReason || ""
    },
    verdict: data.verdict || "OpenAI эксперт видит, что ученик в целом повторяет эталон, но отдельные моменты стоит разобрать внимательнее.",
    bestScore: clampPercent(data.bestScore ?? local.bestScore),
    worstScore: clampPercent(data.worstScore ?? local.worstScore)
  };
}

function buildOpenAiComparisonPayload({ leftScan, rightScan, sync, regions, leftAudio, mediaPipeSettings, local }) {
  return {
    appVersion,
    model: comparisonModels["openai-expert"],
    task:
      "Оцени, насколько правое видео ученика повторяет левое эталонное видео педагога. Не путай качество трекинга MediaPipe с качеством танца.",
    scoringPolicy: {
      primaryQuestion:
        "Насколько ученик визуально и хореографически повторяет эталонную фразу, как это оценил бы внимательный педагог.",
      phraseFirst:
        "Сначала смотри на всю композицию и крупные фрагменты движения: начало, развитие, акценты и финал. Не делай главный вывод по одному кадру или одной позе.",
      choreographyEvents:
        "Сравнивай танец как цепочку понятных действий: ноги раскрылись или закрылись, руки поднялись или опустились, руки сделали круг, корпус повернулся, вес перенесся, прыжок или шаг случился, акцент был пойман. Если действие совпало по смыслу и месту в музыке, это важнее мелкой разницы в точке кисти или локтя.",
      evidenceRule:
        "Высокая оценка возможна только если есть доказательство выполнения хореографической фразы. Стабильный корпус, хороший ритм, надежный trackingQualityScore или похожая средняя поза сами по себе не доказывают, что ученик станцевал связку.",
      scoreSeparation: {
        choreographyScore:
          "Оценивает только то, насколько ученик выполнил ту же хореографию, последовательность действий, акценты и направления.",
        trackingQualityScore:
          "Оценивает только надежность скана MediaPipe. Это не бонус к хореографии.",
        finalDisplayedScore:
          "Клиентская оценка ученика после evidence gate. trackingQualityScore никогда не должен повышать choreographyScore."
      },
      evidenceGate: [
        "Если амплитуда движения ученика почти отсутствует относительно эталона, finalDisplayedScore не выше 15.",
        "Если trajectoryScore < 45 и keyPoseScore < 70, finalDisplayedScore не выше 40.",
        "Если trajectoryScore < 45, frameHitScore < 72 и anglePatternScore < 70, finalDisplayedScore не выше 45.",
        "Если ученик стоит или делает только минимальные движения, не называй это упрощенной версией той же связки. Пиши: хореографическая фраза в основном не выполнена.",
        "Если движение есть, но последовательность, направления корпуса, рук и акценты отличаются, не оценивай выше 40-50, даже если ритм частично похож.",
        "Если ученик действительно повторяет ту же фразу, но с человеческими отличиями, можно давать 80-95.",
        "Если ученик хорошо повторяет эталон, но скан частично плохой, choreographyScore может быть высоким, но добавь предупреждение о trackingQuality."
      ],
      negativeCases: [
        "Человек стоит: 0-15.",
        "Другая хореография: 0-40.",
        "Движение частично похоже, но фраза распадается: 40-60.",
        "Та же фраза, но слабая амплитуда, руки или акценты: 60-80.",
        "Хорошее повторение: 80-95.",
        "Почти идеальное повторение другим человеком: 90-100."
      ],
      segmentLogic: [
        "Оцени, повторяется ли общий рисунок танца от начала до конца.",
        "Сначала найди совпадающие хореографические события: ноги в стороны, руки вверх, круг руками, поворот, шаг, присед, акцент.",
        "Считай хорошим повтором, когда ученик сделал то же действие в той же части музыки, даже если рука немного не дотянулась, кадр смазался или пик движения пойман чуть раньше.",
        "Смотри на устойчивые различия, которые держатся несколько моментов подряд.",
        "Одиночную спорную позу учитывай только как маленькое замечание, если до и после движение похоже.",
        "Если рука или нога приходит в нужную точку чуть раньше или чуть позже, но траектория и музыкальный акцент совпадают, это не грубая ошибка.",
        "Если быстрое движение руки смазалось и один скан поймал пик, а другой соседнюю фазу того же движения, оцени это как нормальный допуск видеосъемки и трекинга.",
        "Если ученик танцует ту же хореографию, но немного иначе по камере, росту или амплитуде, оценка должна оставаться высокой.",
        "Если ошибка видна в конкретном месте фразы, можно назвать примерное время, но не превращать комментарий в таблицу."
      ],
      ignoreAsStudentErrors: [
        "микродрожание точек MediaPipe",
        "одиночные или короткие серии сломанных кадров",
        "улетевшие кисти, ноги или голова, если соседние кадры показывают нормальную позу",
        "смаз быстрых рук или ног, когда движение по соседним кадрам идет в правильную сторону",
        "небольшое опережение или отставание конечности внутри той же хореографической фразы",
        "резкие невозможные скачки суставов за доли секунды",
        "мелкие отличия камеры, роста, комплекции, масштаба и положения в комнате"
      ],
      keepAsTrackingDiagnostics: [
        "процент плохих или отброшенных кадров",
        "потеря тела MediaPipe",
        "перепутанные конечности",
        "низкая уверенность скана"
      ],
      compareAsHumanWould: [
        "общая хореографическая фраза",
        "цельность композиции",
        "повторяемость крупных фрагментов",
        "совпадение хореографических действий",
        "последовательность действий: что за чем произошло",
        "ритм и музыкальное попадание",
        "корпус как главный якорь",
        "амплитуда и направление движения",
        "ключевые позы",
        "руки и ноги с мягким допуском на 1-2 кадра",
        "быстрые движения как траекторию, а не только как замороженную позу в одном кадре"
      ],
      warning:
        "Если локальная метрика резко снижена из-за trajectory/микрошумов, смаза быстрых рук, небольшого фазового сдвига или мелкой разницы кисти/локтя, но цепочка действий совпала с эталоном, итоговую оценку нужно делать ближе к человеческой визуальной оценке. Но если доказательства выполнения фразы нет, высокий trackingQualityScore, стабильный корпус или похожая средняя поза не должны поднимать оценку."
    },
    commentStyle: {
      goal:
        "Текстовые поля verdict, reasoning и suggestions должны звучать как комментарий педагога-хореографа ребенку или начинающему ученику.",
      use:
        "Пиши коротко, просто и конкретно: что получилось, что поправить, где рука, локоть, корпус, нога или акцент уходят от эталона.",
      avoid:
        "Не используй в текстовых комментариях проценты, сухую статистику, названия внутренних метрик, HTTP/API детали, технические слова про расчеты и сложные метафоры.",
      allowed:
        "Можно указать примерное время ошибки, если это помогает ученику найти место в видео. Время должно быть педагогической подсказкой, а не статистикой.",
      numericFields:
        "Числовые оценки возвращай только в отдельных JSON-полях choreographyScore, trackingQualityScore, finalDisplayedScore, confidence, bestScore и worstScore."
    },
    localBaseline: {
      method: local.method,
      score: local.score,
      rows: (local.rows || []).slice(0, 14),
      bodyParts: local.bodyParts || null,
      diagnostics: local.diagnostics || {},
      suggestions: local.suggestions || [],
      verdict: local.verdict || ""
    },
    sync: sync || null,
    mediaPipeSettings: mediaPipeSettings || null,
    regions: regions || null,
    audio: {
      hasReferenceAudio: Boolean(leftAudio),
      referenceDuration: leftAudio?.duration || leftScan?.duration || 0,
      userDuration: rightScan?.duration || 0
    },
    scans: {
      reference: compactScanForAi(leftScan),
      user: compactScanForAi(rightScan)
    },
    expectedOutput:
      "Верни JSON: choreographyScore, trackingQualityScore, finalDisplayedScore, evidenceGateApplied, evidenceGateReason, confidence, bestScore, worstScore, verdict, reasoning, suggestions. В score-полях оставь числа. verdict, reasoning и suggestions напиши простым языком хореографа для ученика: честно скажи, выполнена ли хореографическая фраза. Не начисляй похожесть за стабильное стояние. Не используй проценты, статистику и внутренние названия метрик. Можно назвать примерную секунду, если это помогает найти ошибку."
  };
}

function compactScanForAi(scan) {
  const frames = scan?.frames || [];
  const sampled = sampleEvenly(frames, 36).map((frame) => ({
    time: Number((frame.time ?? 0).toFixed(2)),
    confidence: Number((frame.confidence ?? 0).toFixed(2)),
    angles: compactAngles(frame.angles),
    landmarks: compactAiLandmarks(frame.landmarks)
  }));
  return {
    duration: Number((scan?.duration || frames.at(-1)?.time || 0).toFixed(2)),
    frameCount: frames.length,
    range: scan?.range || null,
    video: scan?.video || null,
    frames: sampled
  };
}

function compactAngles(angles = {}) {
  return Object.fromEntries(
    Object.entries(angles)
      .filter(([, value]) => Number.isFinite(value))
      .map(([key, value]) => [key, Number(value.toFixed(1))])
  );
}

function compactAiLandmarks(landmarks = []) {
  const ids = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  return Object.fromEntries(
    ids
      .map((id) => {
        const point = landmarks?.[id];
        if (!point) return null;
        return [
          id,
          {
            x: Number(point.x.toFixed(4)),
            y: Number(point.y.toFixed(4)),
            z: Number((point.z || 0).toFixed(4)),
            visibility: Number((point.visibility ?? 0).toFixed(2))
          }
        ];
      })
      .filter(Boolean)
  );
}

function rowPercent(id, title, value) {
  const score = clampPercent(value);
  return { id, title, leftValue: score, rightValue: 100, diff: 100 - score, unit: "%", score };
}

function clampPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(100, Math.round(number)));
}

function compareScans20260706(leftScan, rightScan, sync) {
  const offsetMs = sync?.ready ? sync.offsetSeconds * 1000 : 0;
  const userScan = {
    ...(rightScan || {}),
    frames: (rightScan?.frames || []).map((frame) => ({
      ...frame,
      timestamp: Math.round(frame.time * 1000 - offsetMs)
    }))
  };
  return compareSkeletons_2026_07_06(leftScan, userScan, {
    syncWindowMs: 300,
    useBodyNormalization: true,
    compareAngles: true,
    compareBoneDirections: true,
    compareVelocity: true
  });
}

function compareScans20260712(leftScan, rightScan, sync) {
  const offsetMs = sync?.ready ? sync.offsetSeconds * 1000 : 0;
  const userScan = {
    ...(rightScan || {}),
    frames: (rightScan?.frames || []).map((frame) => ({
      ...frame,
      timestamp: Math.round(frame.time * 1000 - offsetMs)
    }))
  };
  return compareSkeletons_2026_07_12(leftScan, userScan);
}

function compareScans20260713(leftScan, rightScan, sync) {
  const offsetMs = sync?.ready ? sync.offsetSeconds * 1000 : 0;
  const userScan = {
    ...(rightScan || {}),
    frames: (rightScan?.frames || []).map((frame) => ({
      ...frame,
      timestamp: Math.round(frame.time * 1000 - offsetMs)
    }))
  };
  return compareSkeletons_2026_07_13(leftScan, userScan);
}

function compareOverlayFrames(left, right, regions = defaultMediaPipeSettings.regions) {
  if (!left?.landmarks?.length || !right?.landmarks?.length) {
    return {
      ready: false,
      score: 0,
      verdict: "Сначала отсканируйте скелет в левом и правом видео.",
      rows: [],
      suggestions: []
    };
  }

  const ids = overlayLandmarkIds(regions);
  const normalizedLeft = normalizeSkeleton(left.landmarks);
  const normalizedRight = normalizeSkeleton(right.landmarks);
  if (!normalizedLeft || !normalizedRight) return comparePoseFrames(null, null, regions);

  let sum = 0;
  let count = 0;
  const regionTotals = new Map();
  for (const id of ids) {
    const leftPoint = normalizedLeft[id];
    const rightPoint = normalizedRight[id];
    if (!leftPoint || !rightPoint) continue;
    const distance = Math.hypot(leftPoint.x - rightPoint.x, leftPoint.y - rightPoint.y);
    sum += distance;
    count += 1;
    const region = regionForLandmark(id);
    const current = regionTotals.get(region) || { distance: 0, count: 0 };
    current.distance += distance;
    current.count += 1;
    regionTotals.set(region, current);
  }

  if (!count) return comparePoseFrames(null, null, regions);

  const averageDistance = sum / count;
  const score = Math.max(0, Math.min(100, Math.round(100 - averageDistance * 145)));
  const rows = Array.from(regionTotals.entries()).map(([region, value]) => {
    const diff = Number((value.distance / value.count).toFixed(3));
    return {
      id: `overlay-${region}`,
      title: regionTitle(region),
      leftValue: "-",
      rightValue: "-",
      diff,
      unit: "",
      score: Math.max(0, Math.round(100 - diff * 145))
    };
  });
  const worst = [...rows].sort((a, b) => b.diff - a.diff).slice(0, 3);
  const suggestions = worst.map((row) => `${row.title}: средняя дистанция наложения ${row.diff}.`);

  return {
    ready: true,
    score,
    rows,
    overlayDistance: Number(averageDistance.toFixed(4)),
    suggestions,
    verdict: overlayVerdict(score, suggestions)
  };
}

function compareOverlayScans(leftScan, rightScan, sync, regions = defaultMediaPipeSettings.regions) {
  if (!leftScan?.frames?.length || !rightScan?.frames?.length) return compareOverlayFrames(null, null, regions);

  const offset = sync?.ready ? sync.offsetSeconds : 0;
  const usableFrames = synchronizedFramePairs(leftScan, rightScan, offset)
    .map((pair) => ({ ...pair, comparison: compareOverlayFrames(pair.leftFrame, pair.rightFrame, regions) }))
    .filter((pair) => pair.comparison.ready);

  if (!usableFrames.length) return compareOverlayFrames(null, null, regions);
  return aggregateComparisons(usableFrames, "overlay");
}

function compareImpulsePoseScans(leftScan, rightScan, sync, regions = defaultMediaPipeSettings.regions, leftAudio = null) {
  if (!leftScan?.frames?.length || !rightScan?.frames?.length) return comparePoseFrames(null, null, regions);

  const offset = sync?.ready ? sync.offsetSeconds : 0;
  const impulseTimes = impulseTimesForAnalysis(leftAudio, leftScan);
  const usableFrames = impulseTimes
    .map((time) => {
      const leftFrame = nearestFrame(leftScan.frames, time);
      const rightFrame = nearestFrame(rightScan.frames, time + offset);
      if (!leftFrame?.landmarks?.length || !rightFrame?.landmarks?.length) return null;
      const angles = comparePoseFrames(leftFrame, rightFrame, regions);
      const overlay = compareOverlayFrames(leftFrame, rightFrame, regions);
      const score = Math.round(angles.score * 0.58 + overlay.score * 0.42);
      return {
        leftTime: leftFrame.time,
        rightTime: rightFrame.time,
        impulseTime: time,
        leftFrame,
        rightFrame,
        comparison: {
          ready: true,
          score,
          rows: [
            {
              id: `pose-${time}-angles`,
              title: `Импульс ${formatTime(time)}: углы`,
              leftValue: angles.score,
              rightValue: overlay.score,
              diff: Math.abs(angles.score - overlay.score),
              unit: "%",
              score
            }
          ],
          suggestions: [...angles.suggestions.slice(0, 1), ...overlay.suggestions.slice(0, 1)]
        }
      };
    })
    .filter(Boolean);

  if (!usableFrames.length) return comparePoseFrames(null, null, regions);
  const result = aggregateComparisons(usableFrames, "poses");
  return {
    ...result,
    poseMoments: usableFrames.slice(0, 12).map((item) => ({
      leftTime: item.leftTime,
      rightTime: item.rightTime,
      impulseTime: item.impulseTime,
      score: item.comparison.score,
      leftLandmarks: item.leftFrame.landmarks,
      rightLandmarks: item.rightFrame.landmarks
    })),
    verdict: poseVerdict(result.score, result.suggestions)
  };
}

function aggregateComparisons(usableFrames, model) {
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
      current.diff += Number(row.diff) || 0;
      current.leftValue += Number(row.leftValue) || 0;
      current.rightValue += Number(row.rightValue) || 0;
      current.count += 1;
      totals.set(row.id, current);
    }
  }

  const rows = Array.from(totals.values()).map((row) => ({
    ...row,
    diff: Number((row.diff / row.count).toFixed(row.unit === "" ? 3 : 1)),
    leftValue: Number.isFinite(row.leftValue) ? Number((row.leftValue / row.count).toFixed(1)) : "-",
    rightValue: Number.isFinite(row.rightValue) ? Number((row.rightValue / row.count).toFixed(1)) : "-",
    score: Math.max(0, Math.round(100 - (row.diff / row.count) * (row.unit === "" ? 145 : 2.15)))
  }));

  const score = Math.round(scoreSum / usableFrames.length);
  const worst = [...rows].sort((a, b) => (b.diff || 0) - (a.diff || 0)).slice(0, 4);
  const suggestions =
    model === "overlay"
      ? worst.map((row) => `${row.title}: средняя дистанция наложения ${row.diff}.`)
      : worst.map((row) => `${row.title}: ключевой момент проседает до ${row.score}%.`);

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
    verdict: model === "overlay" ? overlayVerdict(score, suggestions) : poseVerdict(score, suggestions)
  };
}

function synchronizedFramePairs(leftScan, rightScan, offset) {
  return leftScan.frames
    .filter((frame) => frame.landmarks?.length)
    .map((leftFrame) => {
      const rightFrame = nearestFrame(rightScan.frames, leftFrame.time + offset);
      if (!rightFrame?.landmarks?.length) return null;
      return { leftFrame, rightFrame, leftTime: leftFrame.time, rightTime: rightFrame.time };
    })
    .filter(Boolean);
}

function overlayLandmarkIds(regions = defaultMediaPipeSettings.regions) {
  const ids = new Set();
  activeAngleSpecs(regions).forEach((spec) => spec.points.forEach((id) => ids.add(id)));
  return Array.from(ids);
}

function normalizeSkeleton(landmarks, aspect = 1) {
  const safeAspect = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const corrected = (landmarks || []).map((point) =>
    point
      ? {
          ...point,
          x: point.x * safeAspect
        }
      : point
  );
  const center = averagePoint([corrected[11], corrected[12], corrected[23], corrected[24]].filter(Boolean));
  if (!center) return null;
  const scalePoints = [corrected[11], corrected[12], corrected[23], corrected[24], corrected[25], corrected[26]].filter(Boolean);
  const scale =
    scalePoints.reduce((sum, point) => sum + Math.hypot(point.x - center.x, point.y - center.y), 0) / Math.max(1, scalePoints.length) ||
    0.1;
  return corrected.map((point) =>
    point
      ? {
          ...point,
          x: (point.x - center.x) / scale,
          y: (point.y - center.y) / scale
        }
      : point
  );
}

function fitNormalizedSkeletonToReference(referenceLandmarks, userLandmarks) {
  const fitIds = [11, 12, 23, 24];
  const pairs = fitIds.map((id) => [referenceLandmarks?.[id], userLandmarks?.[id]]).filter(([reference, user]) => reference && user);
  if (pairs.length < 2) return userLandmarks;
  const referenceCenter = averagePoint(pairs.map(([reference]) => reference));
  const userCenter = averagePoint(pairs.map(([, user]) => user));
  let dotSum = 0;
  let crossSum = 0;
  let referenceEnergy = 0;
  let userEnergy = 0;
  for (const [reference, user] of pairs) {
    const rx = reference.x - referenceCenter.x;
    const ry = reference.y - referenceCenter.y;
    const ux = user.x - userCenter.x;
    const uy = user.y - userCenter.y;
    dotSum += ux * rx + uy * ry;
    crossSum += ux * ry - uy * rx;
    referenceEnergy += rx * rx + ry * ry;
    userEnergy += ux * ux + uy * uy;
  }
  const rotation = Math.atan2(crossSum, dotSum);
  const scale = Math.sqrt(referenceEnergy / Math.max(userEnergy, 0.000001));
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  const fitted = userLandmarks.map((point) => {
    if (!point) return point;
    const x = point.x - userCenter.x;
    const y = point.y - userCenter.y;
    return {
      ...point,
      x: referenceCenter.x + (x * cos - y * sin) * scale,
      y: referenceCenter.y + (x * sin + y * cos) * scale,
      z: (point.z || 0) * scale
    };
  });
  for (const id of fitIds) {
    if (referenceLandmarks?.[id] && fitted?.[id]) fitted[id] = { ...fitted[id], ...referenceLandmarks[id] };
  }
  return fitted;
}

function fitNormalizedSkeletonForPreview(referenceLandmarks, userLandmarks) {
  const fitIds = [11, 12, 23, 24];
  const pairs = fitIds.map((id) => [referenceLandmarks?.[id], userLandmarks?.[id]]).filter(([reference, user]) => reference && user);
  if (pairs.length < 2) return userLandmarks;
  const referenceCenter = averagePoint(pairs.map(([reference]) => reference));
  const userCenter = averagePoint(pairs.map(([, user]) => user));
  let dotSum = 0;
  let crossSum = 0;
  let referenceEnergy = 0;
  let userEnergy = 0;
  for (const [reference, user] of pairs) {
    const rx = reference.x - referenceCenter.x;
    const ry = reference.y - referenceCenter.y;
    const ux = user.x - userCenter.x;
    const uy = user.y - userCenter.y;
    dotSum += ux * rx + uy * ry;
    crossSum += ux * ry - uy * rx;
    referenceEnergy += rx * rx + ry * ry;
    userEnergy += ux * ux + uy * uy;
  }
  const rotation = Math.atan2(crossSum, dotSum);
  const rawScale = Math.sqrt(referenceEnergy / Math.max(userEnergy, 0.000001));
  const scale = Math.max(0.72, Math.min(1.38, Number.isFinite(rawScale) ? rawScale : 1));
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  return userLandmarks.map((point) => {
    if (!point) return point;
    const x = point.x - userCenter.x;
    const y = point.y - userCenter.y;
    return {
      ...point,
      x: referenceCenter.x + (x * cos - y * sin) * scale,
      y: referenceCenter.y + (x * sin + y * cos) * scale,
      z: (point.z || 0) * scale
    };
  });
}

function filterAngleScanFrames(scan) {
  const sourceFrames = scan?.frames?.filter((frame) => frame.landmarks?.length) || [];
  if (sourceFrames.length < 6) return { frames: sourceFrames, skipped: 0 };
  const aspect = scan?.video?.aspect || 1;
  const measured = sourceFrames
    .map((frame) => {
      const normalized = normalizeSkeleton(frame.landmarks, aspect);
      return normalized?.length
        ? {
            frame,
            normalized,
            spread: normalizedSkeletonSpread(normalized),
            maxBone: maxSkeletonBoneLength(normalized),
            boneLengths: skeletonBoneLengths(normalized),
            visibility: averageNumbers(coreLandmarkIds.map((id) => frame.landmarks[id]?.visibility).filter(Number.isFinite))
          }
        : null;
    })
    .filter(Boolean);
  if (measured.length < 6) return { frames: sourceFrames, skipped: 0 };

  const medianSpread = medianNumbers(measured.map((item) => item.spread));
  const medianMaxBone = medianNumbers(measured.map((item) => item.maxBone));
  const medianBoneLengths = Object.fromEntries(
    stableBoneSpecs.map((bone) => [bone.id, medianNumbers(measured.map((item) => item.boneLengths[bone.id]))])
  );
  const jumps = measured.slice(1).map((item, index) => skeletonJump(measured[index].normalized, item.normalized));
  const medianAverageJump = medianNumbers(jumps.map((jump) => jump.average));
  const medianMaxJump = medianNumbers(jumps.map((jump) => jump.max));
  const averageJumpLimit = Math.max(0.75, medianAverageJump * 5.5);
  const maxJumpLimit = Math.max(1.55, medianMaxJump * 6);
  const localJumpLimit = Math.max(0.62, medianAverageJump * 4.25);

  const kept = measured.filter((item, index) => {
    const spreadRatio = item.spread / Math.max(medianSpread, 0.000001);
    const boneRatio = item.maxBone / Math.max(medianMaxBone, 0.000001);
    const previousJump = index > 0 ? skeletonJump(measured[index - 1].normalized, item.normalized) : { average: 0, max: 0 };
    const nextJump = index < measured.length - 1 ? skeletonJump(item.normalized, measured[index + 1].normalized) : { average: 0, max: 0 };
    const impossibleAverageJump = previousJump.average > averageJumpLimit && nextJump.average > averageJumpLimit;
    const impossibleJointJump = previousJump.max > maxJumpLimit && nextJump.max > maxJumpLimit;
    const shortTrackingRun = localTrackingRunIsBroken(measured, index, localJumpLimit, maxJumpLimit);
    const anatomyBreaks = skeletonAnatomyBreaks(item, medianBoneLengths);
    return (
      item.visibility >= 0.22 &&
      spreadRatio >= 0.35 &&
      spreadRatio <= 2.45 &&
      boneRatio <= 2.65 &&
      anatomyBreaks <= 2 &&
      !impossibleAverageJump &&
      !impossibleJointJump &&
      !shortTrackingRun
    );
  });

  if (kept.length < Math.max(4, measured.length * 0.35)) return { frames: sourceFrames, skipped: 0 };
  return {
    frames: kept.map((item) => item.frame),
    skipped: sourceFrames.length - kept.length
  };
}

const stableBoneSpecs = [
  { id: "leftUpperArm", points: [11, 13] },
  { id: "leftForearm", points: [13, 15] },
  { id: "rightUpperArm", points: [12, 14] },
  { id: "rightForearm", points: [14, 16] },
  { id: "leftThigh", points: [23, 25] },
  { id: "leftShin", points: [25, 27] },
  { id: "rightThigh", points: [24, 26] },
  { id: "rightShin", points: [26, 28] },
  { id: "leftTorso", points: [11, 23] },
  { id: "rightTorso", points: [12, 24] },
  { id: "shoulders", points: [11, 12] },
  { id: "hips", points: [23, 24] }
];

function skeletonBoneLengths(landmarks) {
  return Object.fromEntries(
    stableBoneSpecs.map((bone) => [bone.id, pointDistance(landmarks?.[bone.points[0]], landmarks?.[bone.points[1]])])
  );
}

function skeletonAnatomyBreaks(item, medianBoneLengths) {
  const lengths = item.boneLengths || {};
  let breaks = 0;
  for (const bone of stableBoneSpecs) {
    const length = lengths[bone.id];
    const medianLength = medianBoneLengths[bone.id];
    if (!Number.isFinite(length) || !Number.isFinite(medianLength) || medianLength <= 0.000001) continue;
    const ratio = length / medianLength;
    if (ratio > 2.25 || ratio < 0.18) breaks += bone.id.includes("Torso") || bone.id === "shoulders" || bone.id === "hips" ? 2 : 1;
  }
  breaks += limbChainBreaks(item.normalized);
  return breaks;
}

function limbChainBreaks(landmarks) {
  const chains = [
    [11, 13, 15],
    [12, 14, 16],
    [23, 25, 27],
    [24, 26, 28]
  ];
  return chains.reduce((count, [root, middle, end]) => {
    const rootToMiddle = pointDistance(landmarks?.[root], landmarks?.[middle]);
    const middleToEnd = pointDistance(landmarks?.[middle], landmarks?.[end]);
    const rootToEnd = pointDistance(landmarks?.[root], landmarks?.[end]);
    if (![rootToMiddle, middleToEnd, rootToEnd].every(Number.isFinite)) return count;
    const chainLength = rootToMiddle + middleToEnd;
    if (chainLength <= 0.000001) return count + 1;
    const foldedRatio = rootToEnd / chainLength;
    const segmentRatio = Math.max(rootToMiddle, middleToEnd) / Math.max(0.000001, Math.min(rootToMiddle, middleToEnd));
    return count + (foldedRatio < 0.045 || segmentRatio > 6.5 ? 1 : 0);
  }, 0);
}

function localTrackingRunIsBroken(measured, index, averageLimit, maxLimit) {
  const previous = measured[index - 1];
  const next = measured[index + 1];
  if (!previous || !next) return false;
  const previousToCurrent = skeletonJump(previous.normalized, measured[index].normalized);
  const currentToNext = skeletonJump(measured[index].normalized, next.normalized);
  if (previousToCurrent.average > averageLimit && currentToNext.average > averageLimit) return true;
  if (previousToCurrent.max > maxLimit && currentToNext.max > maxLimit) return true;

  const before = measured[index - 2];
  const after = measured[index + 2];
  if (!before || !after) return false;
  const beforeJump = skeletonJump(before.normalized, measured[index].normalized);
  const afterJump = skeletonJump(measured[index].normalized, after.normalized);
  const bridgeJump = skeletonJump(before.normalized, after.normalized);
  const shortBadIsland =
    beforeJump.average > averageLimit &&
    afterJump.average > averageLimit &&
    bridgeJump.average < averageLimit * 0.9;
  const shortBadJointIsland = beforeJump.max > maxLimit && afterJump.max > maxLimit && bridgeJump.max < maxLimit * 0.95;
  return shortBadIsland || shortBadJointIsland;
}

function normalizedSkeletonSpread(landmarks) {
  const points = coreLandmarkIds.map((id) => landmarks[id]).filter(Boolean);
  if (!points.length) return 0;
  return Math.hypot(rangeNumbers(points.map((point) => point.x)), rangeNumbers(points.map((point) => point.y)));
}

function maxSkeletonBoneLength(landmarks) {
  const lengths = poseConnections
    .map(([a, b]) => (landmarks[a] && landmarks[b] ? pointDistance(landmarks[a], landmarks[b]) : null))
    .filter(Number.isFinite);
  return lengths.length ? Math.max(...lengths) : 0;
}

function skeletonJump(previous, current) {
  const distances = coreLandmarkIds.map((id) => pointDistance(previous[id], current[id])).filter(Number.isFinite);
  return {
    average: averageNumbers(distances),
    max: distances.length ? Math.max(...distances) : 0
  };
}

function pointDistance(a, b) {
  if (!a || !b) return null;
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function averageNumbers(values) {
  const clean = values.filter(Number.isFinite);
  return clean.length ? clean.reduce((sum, value) => sum + value, 0) / clean.length : 0;
}

function medianNumbers(values) {
  const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
  return clean.length ? clean[Math.floor(clean.length / 2)] : 0;
}

function rangeNumbers(values) {
  const clean = values.filter(Number.isFinite);
  return clean.length ? Math.max(...clean) - Math.min(...clean) : 0;
}

function synchronizedAngleFramePairs(leftScan, rightScan, offset) {
  const leftFiltered = filterAngleScanFrames(leftScan);
  const rightFiltered = filterAngleScanFrames(rightScan);
  const pairs = leftFiltered.frames
    .map((leftFrame) => {
      const rightFrame = nearestFrame(rightFiltered.frames, leftFrame.time + offset);
      if (!rightFrame?.landmarks?.length) return null;
      return { leftFrame, rightFrame, leftTime: leftFrame.time, rightTime: rightFrame.time };
    })
    .filter(Boolean);
  return {
    pairs,
    skipped: leftFiltered.skipped + rightFiltered.skipped,
    leftSkipped: leftFiltered.skipped,
    rightSkipped: rightFiltered.skipped
  };
}

function regionForLandmark(id) {
  if ([13, 14, 15, 16, 11, 12].includes(id)) return "arms";
  if ([25, 26, 27, 28, 29, 30, 31, 32].includes(id)) return "legs";
  return "torso";
}

function regionTitle(region) {
  if (region === "arms") return "Наложение рук и плеч";
  if (region === "legs") return "Наложение ног";
  return "Наложение корпуса";
}

function impulseTimesForAnalysis(leftAudio, leftScan) {
  const start = leftScan?.range?.start ?? 0;
  const end = leftScan?.range?.end ?? leftScan?.duration ?? 0;
  const peaks = (leftAudio?.peaks || [])
    .filter((peak) => peak.time >= start && peak.time <= end)
    .sort((a, b) => b.value - a.value);
  const selected = [];
  for (const peak of peaks) {
    if (selected.every((time) => Math.abs(time - peak.time) > 0.45)) selected.push(peak.time);
    if (selected.length >= 12) break;
  }
  if (selected.length) return selected.sort((a, b) => a - b);
  const frames = leftScan?.frames?.filter((frame) => frame.landmarks?.length) || [];
  const stride = Math.max(1, Math.floor(frames.length / 8));
  return frames.filter((_, index) => index % stride === 0).slice(0, 8).map((frame) => frame.time);
}

function overlayVerdict(score, suggestions) {
  if (score >= 86) return "Скелеты хорошо ложатся друг на друга после нормализации корпуса и масштаба.";
  if (score >= 70) return `Наложение в целом стабильное, но заметны зоны расхождения: ${suggestions.slice(0, 2).join(" ")}`;
  if (score >= 52) return `Скелеты совпадают частично. Главные смещения: ${suggestions.slice(0, 3).join(" ")}`;
  return `Наложение показывает сильное расхождение пластики и формы движения: ${suggestions.slice(0, 3).join(" ")}`;
}

function poseVerdict(score, suggestions) {
  if (score >= 86) return "Ключевые позы на музыкальных импульсах совпадают с эталоном очень близко.";
  if (score >= 70) return `Ключевые позы в музыке в целом похожи, но есть акцентные расхождения: ${suggestions.slice(0, 2).join(" ")}`;
  if (score >= 52) return `На музыкальных импульсах правое видео часто приходит в другую форму: ${suggestions.slice(0, 3).join(" ")}`;
  return `Ключевые позы почти не совпадают с эталоном: ${suggestions.slice(0, 3).join(" ")}`;
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
  return bestDiff <= 1.05 ? best : null;
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

function loadCaptureEngine() {
  try {
    const saved = localStorage.getItem(captureEngineKey);
    return captureEngines[saved] ? saved : "mediapipe";
  } catch {
    return "mediapipe";
  }
}

function normalizeHybridMethodSettings(settings) {
  const next = {
    ...defaultHybridMethodSettings,
    ...(settings || {})
  };
  const normalized = {
    zones: Boolean(next.zones),
    drawing: Boolean(next.drawing)
  };
  if (!normalized.zones && !normalized.drawing) return { ...normalized, zones: true };
  return normalized;
}

function loadHybridMethodSettings() {
  try {
    return normalizeHybridMethodSettings(JSON.parse(localStorage.getItem(hybridMethodSettingsKey) || "null"));
  } catch {
    return defaultHybridMethodSettings;
  }
}

function parseCsvText(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (char === '"' && quoted && next === '"') {
      cell += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell.trim());
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(cell.trim());
      if (row.some(Boolean)) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  row.push(cell.trim());
  if (row.some(Boolean)) rows.push(row);
  return rows;
}

function parseMotionCapCsv(text, fileName = "freemocap.csv") {
  const rows = parseCsvText(text);
  if (rows.length < 2) throw new Error("CSV пустой или без строк данных.");
  const headers = rows[0].map((header) => header.trim());
  const lowerHeaders = headers.map((header) => header.toLowerCase());
  const isTidy = ["frame", "keypoint", "x", "y", "z"].every((name) => lowerHeaders.includes(name));
  const frames = isTidy ? parseTidyMotionCapRows(headers, rows.slice(1)) : parseWideMotionCapRows(headers, rows.slice(1));
  if (!frames.length) throw new Error("Не удалось найти 3D-точки FreeMoCap в CSV.");
  const keypoints = new Set();
  let validValues = 0;
  for (const frame of frames) {
    for (const [name, point] of Object.entries(frame.points)) {
      keypoints.add(name);
      if ([point.x, point.y, point.z].every(Number.isFinite)) validValues += 1;
    }
  }
  const duration = frames.at(-1)?.time || frames.length / 30;
  return {
    ready: true,
    source: "freemocap",
    fileName,
    format: isTidy ? "tidy by_frame.csv" : "wide body_3d_xyz.csv",
    frames,
    duration,
    frameCount: frames.length,
    keypointCount: keypoints.size,
    validValues,
    keypoints: Array.from(keypoints).sort()
  };
}

function parseTidyMotionCapRows(headers, rows) {
  const index = Object.fromEntries(headers.map((header, idx) => [header.toLowerCase(), idx]));
  const byFrame = new Map();
  for (const row of rows) {
    const frameNumber = Number(row[index.frame]);
    const keypoint = String(row[index.keypoint] || "").trim();
    if (!Number.isFinite(frameNumber) || !keypoint) continue;
    const frame = byFrame.get(frameNumber) || {
      frame: frameNumber,
      time: Number.isFinite(Number(row[index.timestamp])) ? Number(row[index.timestamp]) : frameNumber / 30,
      points: {},
      reprojectionErrors: {}
    };
    frame.points[keypoint] = {
      x: parseMotionCapNumber(row[index.x]),
      y: parseMotionCapNumber(row[index.y]),
      z: parseMotionCapNumber(row[index.z])
    };
    if (index.reprojection_error !== undefined) {
      frame.reprojectionErrors[keypoint] = parseMotionCapNumber(row[index.reprojection_error]);
    }
    byFrame.set(frameNumber, frame);
  }
  return Array.from(byFrame.values()).sort((a, b) => a.frame - b.frame);
}

function parseWideMotionCapRows(headers, rows) {
  const groups = new Map();
  headers.forEach((header, index) => {
    const match = header.match(/^(.+)_([xyz])$/i);
    if (!match) return;
    const pointName = match[1];
    const axis = match[2].toLowerCase();
    groups.set(pointName, { ...(groups.get(pointName) || {}), [axis]: index });
  });
  const timestampIndex = headers.findIndex((header) => header.toLowerCase() === "timestamp");
  return rows
    .map((row, frameIndex) => {
      const points = {};
      for (const [name, axes] of groups.entries()) {
        if (axes.x === undefined || axes.y === undefined || axes.z === undefined) continue;
        points[name] = {
          x: parseMotionCapNumber(row[axes.x]),
          y: parseMotionCapNumber(row[axes.y]),
          z: parseMotionCapNumber(row[axes.z])
        };
      }
      return {
        frame: frameIndex,
        time: timestampIndex >= 0 && Number.isFinite(Number(row[timestampIndex])) ? Number(row[timestampIndex]) : frameIndex / 30,
        points
      };
    })
    .filter((frame) => Object.keys(frame.points).length);
}

function parseMotionCapNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function compareMotionCapScans(reference, user) {
  if (!reference?.frames?.length || !user?.frames?.length) {
    return {
      ready: false,
      score: 0,
      sharedKeypoints: 0,
      framesCompared: 0,
      verdict: "Загрузите FreeMoCap CSV для эталона и ученика."
    };
  }
  const shared = reference.keypoints.filter((name) => user.keypoints.includes(name));
  const useful = shared.filter((name) => !name.startsWith("com_"));
  const sharedKeypoints = useful.length ? useful : shared;
  const framesCompared = Math.min(reference.frames.length, user.frames.length);
  let sum = 0;
  let count = 0;
  for (let index = 0; index < framesCompared; index += 1) {
    const left = normalizeMotionCapFrame(reference.frames[index], sharedKeypoints);
    const right = normalizeMotionCapFrame(user.frames[index], sharedKeypoints);
    if (!left || !right) continue;
    for (const name of sharedKeypoints) {
      const a = left[name];
      const b = right[name];
      if (!a || !b) continue;
      const distance = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
      sum += Math.max(0, 1 - distance / 1.4);
      count += 1;
    }
  }
  const score = count ? clampPercent((sum / count) * 100) : 0;
  return {
    ready: count > 0,
    score,
    sharedKeypoints: sharedKeypoints.length,
    framesCompared,
    verdict:
      score >= 85
        ? "MotionCap 3D-скелеты очень близки по общей форме движения."
        : score >= 65
          ? "MotionCap видит похожую фразу, но есть заметные расхождения в 3D-траектории."
          : "MotionCap не видит устойчивого совпадения 3D-скелетов."
  };
}

function normalizeMotionCapFrame(frame, keypoints) {
  const points = frame?.points || {};
  const leftHip = points.left_hip;
  const rightHip = points.right_hip;
  const leftShoulder = points.left_shoulder;
  const rightShoulder = points.right_shoulder;
  const pelvis = midpoint3d(leftHip, rightHip) || points.hip || points.pelvis || points.com_full_body;
  const neck = midpoint3d(leftShoulder, rightShoulder) || points.neck;
  if (!pelvis) return null;
  const scale = Math.max(0.001, distance3d(pelvis, neck) || averageMotionCapSpan(points, keypoints) || 1);
  return Object.fromEntries(
    keypoints
      .map((name) => {
        const point = points[name];
        if (!point || ![point.x, point.y, point.z].every(Number.isFinite)) return null;
        return [
          name,
          {
            x: (point.x - pelvis.x) / scale,
            y: (point.y - pelvis.y) / scale,
            z: (point.z - pelvis.z) / scale
          }
        ];
      })
      .filter(Boolean)
  );
}

function midpoint3d(a, b) {
  if (!a || !b || ![a.x, a.y, a.z, b.x, b.y, b.z].every(Number.isFinite)) return null;
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

function distance3d(a, b) {
  if (!a || !b || ![a.x, a.y, a.z, b.x, b.y, b.z].every(Number.isFinite)) return 0;
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function averageMotionCapSpan(points, keypoints) {
  const valid = keypoints.map((name) => points[name]).filter((point) => point && [point.x, point.y, point.z].every(Number.isFinite));
  if (valid.length < 2) return 1;
  const center = valid.reduce((sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y, z: sum.z + point.z }), { x: 0, y: 0, z: 0 });
  center.x /= valid.length;
  center.y /= valid.length;
  center.z /= valid.length;
  return valid.reduce((sum, point) => sum + distance3d(point, center), 0) / valid.length || 1;
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

function isMemoryConstrainedDevice() {
  const ua = navigator.userAgent || "";
  const isIOS = /iPad|iPhone|iPod/.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  return isIOS || (navigator.deviceMemory && navigator.deviceMemory <= 4);
}

function compactLandmarks(landmarks) {
  return (landmarks || []).map((point) => ({
    x: Number(point.x.toFixed(5)),
    y: Number(point.y.toFixed(5)),
    z: Number((point.z || 0).toFixed(5)),
    visibility: Number((point.visibility || 0).toFixed(3))
  }));
}

function yieldToBrowser() {
  return new Promise((resolve) => window.setTimeout(resolve, 0));
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
  const mobileSafe = isMemoryConstrainedDevice();
  const frameBudget = mobileSafe ? mobileMaxScanFrames : maxScanFrames;
  const requestedFps = Math.max(1, Number(settings.scanFps || defaultMediaPipeSettings.scanFps));
  const requestedStep = 1 / requestedFps;
  const step = Math.max(requestedStep, scanDuration / frameBudget);
  const specs = activeAngleSpecs(settings.regions);
  const frames = [];
  let scannedFrames = 0;

  for (let time = scanStart; time <= scanEnd; time += step) {
    scannedFrames += 1;
    const targetTime = Math.min(time, Math.max(0, duration - 0.02));
    if (Math.abs(video.currentTime - targetTime) > 0.01) {
      video.currentTime = targetTime;
      await waitForVideoEvent(video, "seeked");
    }
    const result = scanLandmarker.detect(video);
    const rawLandmarks = result.landmarks?.[0] || [];
    if (rawLandmarks.length) {
      const landmarks = compactLandmarks(rawLandmarks);
      frames.push({
        time: Number(video.currentTime.toFixed(3)),
        landmarks,
        angles: poseAngles(landmarks, specs),
        confidence: averageVisibility(landmarks)
      });
    }
    onProgress?.(Math.min(100, Math.round(((time - scanStart) / scanDuration) * 100)));
    if (scannedFrames % 10 === 0) await yieldToBrowser();
  }

  video.currentTime = previousTime;
  if (!wasPaused) {
    try {
      await video.play();
    } catch {
      video.pause();
    }
  }
  onProgress?.(100);

  return {
    duration,
    video: {
      width: video.videoWidth || 0,
      height: video.videoHeight || 0,
      aspect: video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : 1
    },
    range: { start: scanStart, end: scanEnd },
    settings: {
      modelVariant: settings.modelVariant,
      delegate: settings.delegate,
      scanFps: settings.scanFps,
      effectiveScanFps: Number((1 / step).toFixed(2)),
      requestedFrames: Math.ceil(scanDuration * requestedFps),
      frameBudget,
      scannedFrames,
      mobileSafe,
      landmarkSet: settings.landmarkSet,
      regions: settings.regions,
      activeAngles: specs.map((spec) => spec.id)
    },
    frames,
    trackedFrames: frames.length,
    averageConfidence: frames.length
      ? frames.reduce((sum, frame) => sum + frame.confidence, 0) / frames.length
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
  const channel = mixAudioChannels(audioBuffer);
  const sampleRate = audioBuffer.sampleRate;
  const waveform = buildRmsSeries(channel, 900);
  const hopSeconds = 0.025;
  const hopSize = Math.max(1, Math.floor(sampleRate * hopSeconds));
  const windowSize = Math.max(hopSize, Math.floor(sampleRate * 0.12));
  const envelope = [];
  const flux = [];
  let previousEnergy = 0;

  for (let start = 0; start < channel.length - windowSize; start += hopSize) {
    const energy = rms(channel, start, start + windowSize);
    envelope.push(energy);
    flux.push(Math.max(0, energy - previousEnergy));
    previousEnergy = energy;
  }

  normalizeInPlace(waveform);
  normalizeInPlace(envelope);
  normalizeInPlace(flux);
  const smoothEnvelope = normalizeSeries(smoothSeries(envelope, 13));
  const contrast = normalizeSeries(envelope.map((value, index) => Math.max(0, value - (smoothEnvelope[index] || 0) * 0.72)));
  const clippedFlux = normalizeSeries(clipOutliers(flux, 0.88));
  const syncFeatures = buildSyncFeatures(envelope, clippedFlux, smoothEnvelope, contrast);
  await audioContext.close();

  return {
    duration: audioBuffer.duration,
    waveform,
    envelope,
    flux: clippedFlux,
    smoothEnvelope,
    contrast,
    syncFeatures,
    hopSeconds,
    peaks: detectPeaks(syncFeatures, hopSeconds)
  };
}

function mixAudioChannels(audioBuffer) {
  const length = audioBuffer.length;
  const channels = Math.min(2, audioBuffer.numberOfChannels || 1);
  if (channels === 1) return audioBuffer.getChannelData(0);
  const mixed = new Float32Array(length);
  for (let channelIndex = 0; channelIndex < channels; channelIndex += 1) {
    const channel = audioBuffer.getChannelData(channelIndex);
    for (let i = 0; i < length; i += 1) mixed[i] += channel[i] / channels;
  }
  return mixed;
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

function normalizeSeries(values) {
  const copy = [...values];
  normalizeInPlace(copy);
  return copy;
}

function smoothSeries(values, radius = 5) {
  return values.map((_, index) => {
    let sum = 0;
    let count = 0;
    for (let offset = -radius; offset <= radius; offset += 1) {
      const value = values[index + offset];
      if (!Number.isFinite(value)) continue;
      sum += value;
      count += 1;
    }
    return count ? sum / count : 0;
  });
}

function clipOutliers(values, percentile = 0.9) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  const limit = sorted.length ? sorted[Math.floor((sorted.length - 1) * percentile)] || 0.000001 : 0.000001;
  return values.map((value) => Math.min(value || 0, limit));
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
  const maxLag = Math.min(Math.round(35 / hop), Math.floor(Math.min(leftAudio.envelope.length, rightAudio.envelope.length) * 0.6));
  const left = zNormalize(leftAudio.syncFeatures || leftAudio.envelope);
  const right = zNormalize(rightAudio.syncFeatures || rightAudio.envelope);
  const leftMusic = zNormalize(leftAudio.smoothEnvelope || leftAudio.envelope);
  const rightMusic = zNormalize(rightAudio.smoothEnvelope || rightAudio.envelope);
  const leftFlux = zNormalize(leftAudio.flux || leftAudio.envelope);
  const rightFlux = zNormalize(rightAudio.flux || rightAudio.envelope);
  let bestLag = 0;
  let bestScore = -Infinity;
  let secondBestScore = -Infinity;

  for (let lag = -maxLag; lag <= maxLag; lag += 1) {
    let featureSum = 0;
    let musicSum = 0;
    let fluxSum = 0;
    let count = 0;
    for (let i = 0; i < left.length; i += 1) {
      const j = i + lag;
      if (j < 0 || j >= right.length) continue;
      featureSum += left[i] * right[j];
      musicSum += (leftMusic[i] || 0) * (rightMusic[j] || 0);
      fluxSum += (leftFlux[i] || 0) * (rightFlux[j] || 0);
      count += 1;
    }
    if (count < 20) continue;
    const overlapPenalty = Math.min(1, count / Math.min(left.length, right.length)) ** 1.35;
    const score = ((musicSum / count) * 0.62 + (featureSum / count) * 0.28 + (fluxSum / count) * 0.1) * overlapPenalty;
    if (score > bestScore) {
      secondBestScore = bestScore;
      bestScore = score;
      bestLag = lag;
    } else if (score > secondBestScore) {
      secondBestScore = score;
    }
  }

  const offsetSeconds = Number((bestLag * hop).toFixed(2));
  const peakSeparation = Math.max(0, bestScore - secondBestScore);
  const confidence = Math.max(0, Math.min(100, Math.round(((bestScore + 1) / 2) * 78 + Math.min(22, peakSeparation * 120))));
  return {
    ready: true,
    offsetSeconds,
    confidence,
    automaticOffsetSeconds: offsetSeconds,
    message:
      offsetSeconds >= 0
        ? `Правое видео читается на ${offsetSeconds.toFixed(2)} сек. вперед относительно эталона.`
        : `Правое видео читается на ${Math.abs(offsetSeconds).toFixed(2)} сек. назад относительно эталона.`
  };
}

function buildSyncFeatures(envelope, flux, smoothEnvelope = envelope, contrast = envelope) {
  return envelope.map((value, index) => {
    const previous = envelope[index - 1] ?? value;
    const next = envelope[index + 1] ?? value;
    const localContrast = Math.max(0, value - (previous + next) / 2);
    return (smoothEnvelope[index] || value) * 0.56 + value * 0.22 + (contrast[index] || localContrast) * 0.14 + (flux[index] || 0) * 0.08;
  });
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

function formatSeconds(seconds, digits = 1) {
  return Number.isFinite(seconds) ? `${seconds.toFixed(digits)} сек` : "-";
}

function effectiveSync(sync, manualSync) {
  const baseOffset = sync?.ready ? Number(sync.offsetSeconds || 0) : 0;
  const manualOffset = Number(manualSync?.offsetSeconds || 0);
  const offsetSeconds = Number((baseOffset + manualOffset).toFixed(2));
  return {
    ...(sync || {}),
    ready: Boolean(sync?.ready || manualOffset || manualSync?.useManualStarts),
    offsetSeconds,
    automaticOffsetSeconds: sync?.automaticOffsetSeconds ?? baseOffset,
    manualOffsetSeconds: manualOffset,
    message: `Итоговое смещение правого видео: ${offsetSeconds.toFixed(2)} сек.`
  };
}

function clampVideoTime(value, duration = 0) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Number(Math.max(0, Math.min(numeric, duration || numeric)).toFixed(2));
}

function formatMetricValue(value, unit) {
  if (value === null || value === undefined || value === "-") return "-";
  if (unit === "") return value;
  if (unit === "%") return `${value}%`;
  return `${value}°`;
}

function activeRegionLabels(regions = defaultMediaPipeSettings.regions) {
  const labels = [];
  if (regions?.arms) labels.push("руки");
  if (regions?.torso) labels.push("корпус");
  if (regions?.legs) labels.push("ноги");
  return labels.length ? labels : ["все области"];
}

function formatBytes(bytes) {
  if (!bytes) return "-";
  const units = ["Б", "КБ", "МБ", "ГБ"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex ? 1 : 0)} ${units[unitIndex]}`;
}

function fileProfile(file, scan, audio) {
  if (!file) {
    return {
      source: "camera-or-empty",
      name: "без файла",
      size: 0,
      sizeLabel: "-",
      type: "",
      lastModified: null,
      lastModifiedIso: null,
      duration: scan?.duration ?? audio?.duration ?? null,
      durationLabel: scan?.duration || audio?.duration ? formatTime(scan?.duration ?? audio?.duration) : "-"
    };
  }
  const duration = scan?.duration ?? audio?.duration ?? null;
  return {
    source: "file",
    name: file.name,
    size: file.size,
    sizeLabel: formatBytes(file.size),
    type: file.type || "unknown",
    lastModified: file.lastModified || null,
    lastModifiedIso: file.lastModified ? new Date(file.lastModified).toISOString() : null,
    duration,
    durationLabel: duration ? formatTime(duration) : "-",
    scanRange: scan?.range || null,
    scanFrames: scan?.frames?.length || 0,
    trackedFrames: scan?.trackedFrames || 0,
    audioDuration: audio?.duration || null
  };
}

function sameFileCandidate(left, right) {
  if (!left || !right || left.source !== "file" || right.source !== "file") return false;
  const durationDiff = Math.abs((left.duration || 0) - (right.duration || 0));
  return left.name === right.name && left.size === right.size && durationDiff < 0.05;
}

function modelComponentMetadata(modelId, result, hybridMethods = defaultHybridMethodSettings) {
  if (modelId === "joint-areas") return { mode: "single", components: ["areas"], label: "Только области суставов" };
  if (modelId === "trajectory-drawing") return { mode: "single", components: ["drawing"], label: "Только рисунок траекторий" };
  if (modelId === "zone-grid") return { mode: "single", components: ["zone-grid"], label: "Сравнение по одинаковым квадратам сетки" };
  if (modelId === "activity") return { mode: "single", components: ["activity"], label: "Сравнение уровня активности скелета" };
  if (modelId === "zones-drawing") {
    const methods = result?.diagnostics?.hybridMethods || normalizeHybridMethodSettings(hybridMethods);
    return {
      mode: "combined",
      components: Object.entries(methods)
        .filter(([, enabled]) => enabled)
        .map(([key]) => key),
      label: "Старая комбинированная модель области плюс рисунок",
      settings: methods
    };
  }
  if (modelId === "all-auto") return { mode: "batch", components: runnableComparisonModelIds, label: "Автопрогон всех моделей" };
  return { mode: "single", components: [modelId], label: comparisonModels[modelId]?.title || modelId };
}

function buildModelRunMetadata({
  app,
  modelId,
  modelDetails,
  result,
  saveMode,
  runGroupId,
  sync,
  mediaPipeSettings,
  activeSpecs,
  hybridMethods
}) {
  return {
    schemaVersion: 1,
    runGroupId: runGroupId || null,
    savedAtAppVersion: app.version,
    savedAtAppBuild: app.build,
    saveMode,
    model: {
      id: modelId,
      name: modelDetails.name,
      title: modelDetails.title,
      version: modelDetails.version,
      versionLabel: modelDetails.versionLabel,
      algorithmBuild: modelDetails.algorithmBuild,
      componentMetadata: modelComponentMetadata(modelId, result, hybridMethods)
    },
    result: {
      method: result?.method || modelDetails.title,
      score: result?.score ?? null,
      finalScore: result?.finalScore ?? result?.score ?? null,
      bestScore: result?.bestScore ?? null,
      worstScore: result?.worstScore ?? null,
      framesCompared: result?.framesCompared || 0,
      durationCompared: result?.durationCompared ?? null,
      worstMoment: result?.worstMoment ?? null,
      bodyParts: result?.bodyParts || null,
      diagnostics: result?.diagnostics || {}
    },
    inputSettings: {
      sync,
      mediaPipe: {
        modelVariant: mediaPipeSettings.modelVariant,
        delegate: mediaPipeSettings.delegate,
        numPoses: mediaPipeSettings.numPoses,
        scanFps: mediaPipeSettings.scanFps,
        landmarkSet: mediaPipeSettings.landmarkSet,
        regions: mediaPipeSettings.regions,
        activeAngles: activeSpecs.map((spec) => ({
          id: spec.id,
          title: spec.title,
          region: spec.region,
          points: spec.points
        }))
      }
    }
  };
}

function inferAutonomousCase(fileName = "") {
  const normalized = fileName.toLowerCase();
  if (/стоит|сто[яи]т|stand|still|static|stop/.test(normalized)) {
    return { type: "STANDING_STILL", label: "Стоит / почти не выполняет", target: "низко" };
  }
  if (/не\s*похож|непохож|different|wrong|друг/.test(normalized)) {
    return { type: "UNRELATED_DANCE", label: "Другая / непохожая фраза", target: "низко" };
  }
  if (/очень\s*похож|похож|good|repeat|match|учен/.test(normalized)) {
    return { type: "GOOD_REPEAT", label: "Хорошее повторение", target: "высоко" };
  }
  if (/эталон|reference|ref|teacher/.test(normalized)) {
    return { type: "SELF_CHECK_OR_REFERENCE", label: "Техническая проверка", target: "100 только для диагностики" };
  }
  return { type: "UNKNOWN", label: "Неизвестный тип", target: "проверить вручную" };
}

function autonomousEnsembleScore(modelScores = {}) {
  const weights = {
    "joint-areas": 0.28,
    "zone-grid": 0.16,
    "trajectory-drawing": 0.18,
    activity: 0.12,
    "2026-07-13": 0.2,
    "2026-07-12": 0.08,
    "openai-expert": 0.1
  };
  const entries = Object.entries(modelScores).filter(([, score]) => Number.isFinite(score));
  if (!entries.length) return null;
  const weighted = entries.reduce(
    (acc, [modelId, score]) => {
      const weight = weights[modelId] ?? 0.06;
      acc.sum += score * weight;
      acc.weight += weight;
      return acc;
    },
    { sum: 0, weight: 0 }
  );
  let score = weighted.weight ? weighted.sum / weighted.weight : averageNumbers(entries.map(([, value]) => value));
  const jointAreas = modelScores["joint-areas"];
  const zoneGrid = modelScores["zone-grid"];
  const trajectory = modelScores["trajectory-drawing"];
  const activity = modelScores.activity;
  if (Number.isFinite(jointAreas) && jointAreas < 35 && Number.isFinite(zoneGrid) && zoneGrid < 65) score = Math.min(score, 35);
  if (Number.isFinite(jointAreas) && jointAreas < 45 && Number.isFinite(trajectory) && trajectory < 45) score = Math.min(score, 40);
  if (Number.isFinite(activity) && activity < 25 && Number.isFinite(trajectory) && trajectory < 55) score = Math.min(score, 28);
  if (Number.isFinite(jointAreas) && jointAreas >= 82 && Number.isFinite(zoneGrid) && zoneGrid >= 84) score = Math.max(score, 84);
  return clampPercent(score);
}

function autonomousRecommendation(caseType, ensembleScore, modelScores = {}) {
  if (!Number.isFinite(ensembleScore)) return "Недостаточно результатов.";
  if (caseType === "STANDING_STILL" && ensembleScore > 25) return "Стояние все еще завышено: нужна motion-gate проверка амплитуды и событий движения.";
  if (caseType === "UNRELATED_DANCE" && ensembleScore > 40) return "Непохожий танец завышен: усилить вес траектории и ключевых переходов.";
  if (caseType === "GOOD_REPEAT" && ensembleScore < 80) return "Хорошее повторение занижено: проверить tolerance зон, синхронизацию и качество скана.";
  const spread = Math.max(...Object.values(modelScores).filter(Number.isFinite)) - Math.min(...Object.values(modelScores).filter(Number.isFinite));
  if (spread > 45) return "Модели сильно расходятся: этот пример полезен для настройки ансамбля.";
  return "Ансамбль ведет себя ожидаемо для этого имени файла.";
}

const sequentialGateThresholds = {
  activity: 58,
  overlay: 64,
  poses: 62,
  angles: 62,
  "joint-areas": 62,
  "trajectory-drawing": 55,
  "zone-grid": 60,
  "2026-07-06": 62,
  "2026-07-12": 62,
  "2026-07-13": 62,
  "openai-expert": 62
};

function sequentialGateDecision(modelId, result) {
  const threshold = sequentialGateThresholds[modelId] ?? 62;
  if (!result?.ready) {
    return {
      passed: false,
      threshold,
      reason: "Модель не смогла рассчитать результат."
    };
  }

  const score = clampPercent(result.finalScore ?? result.score);
  if (modelId === "activity") {
    const referenceActivity = result.diagnostics?.activityReference?.total?.activity ?? 0;
    const userActivity = result.diagnostics?.activityUser?.total?.activity ?? 0;
    if (referenceActivity >= 25 && userActivity < 12) {
      return {
        passed: false,
        threshold,
        reason: `Стоп-гейт: эталон активный (${referenceActivity}%), а правое видео почти стоит (${userActivity}%).`
      };
    }
    if (referenceActivity >= 35 && userActivity < referenceActivity * 0.45) {
      return {
        passed: false,
        threshold,
        reason: `Стоп-гейт: активности правого видео слишком мало относительно эталона (${userActivity}% против ${referenceActivity}%).`
      };
    }
  }

  return {
    passed: score >= threshold,
    threshold,
    reason:
      score >= threshold
        ? `Гейт пройден: ${score}% при пороге ${threshold}%.`
        : `Стоп-гейт: ${score}% ниже порога ${threshold}%.`
  };
}

function sequentialFinalScore(steps = []) {
  const passedScores = steps.filter((step) => step.passed && Number.isFinite(step.score)).map((step) => step.score);
  if (!passedScores.length) return 0;
  return clampPercent(averageNumbers(passedScores));
}

function shuffledItems(items = []) {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function autonomousCaseTarget(caseType) {
  if (caseType === "GOOD_REPEAT") return 90;
  if (caseType === "SELF_CHECK_OR_REFERENCE") return 100;
  if (caseType === "STANDING_STILL") return 8;
  if (caseType === "UNRELATED_DANCE") return 22;
  return 55;
}

function modelSubsetScore(modelScores = {}, modelIds = []) {
  const subset = Object.fromEntries(modelIds.map((modelId) => [modelId, modelScores[modelId]]));
  return autonomousEnsembleScore(subset);
}

function buildAutonomousTrials(results = [], selectedModels = []) {
  if (!results.length || !selectedModels.length) return [];
  const unique = new Map();
  const addTrial = (models) => {
    const clean = models.filter((modelId) => selectedModels.includes(modelId));
    if (!clean.length) return;
    const key = [...clean].sort().join("+");
    if (!unique.has(key)) unique.set(key, clean);
  };

  selectedModels.forEach((modelId) => addTrial([modelId]));
  addTrial(selectedModels);
  for (let i = 0; i < Math.min(18, Math.max(6, selectedModels.length * 4)); i += 1) {
    const shuffled = shuffledItems(selectedModels);
    const size = 1 + Math.floor(Math.random() * selectedModels.length);
    addTrial(shuffled.slice(0, size));
  }

  return Array.from(unique.values())
    .map((models) => scoreAutonomousTrial(results, models))
    .sort((a, b) => b.qualityScore - a.qualityScore)
    .slice(0, 12);
}

function scoreAutonomousTrial(results = [], models = []) {
  const evaluated = results.map((item) => {
    const score = modelSubsetScore(item.modelScores, models);
    const target = autonomousCaseTarget(item.caseType);
    const error = Number.isFinite(score) ? Math.abs(score - target) : 100;
    return {
      fileName: item.fileName,
      caseType: item.caseType,
      caseLabel: item.caseLabel,
      score,
      target,
      error
    };
  });
  const averageError = averageNumbers(evaluated.map((item) => item.error));
  const goodScores = evaluated.filter((item) => item.caseType === "GOOD_REPEAT").map((item) => item.score).filter(Number.isFinite);
  const lowScores = evaluated
    .filter((item) => item.caseType === "STANDING_STILL" || item.caseType === "UNRELATED_DANCE")
    .map((item) => item.score)
    .filter(Number.isFinite);
  const selfScores = evaluated.filter((item) => item.caseType === "SELF_CHECK_OR_REFERENCE").map((item) => item.score).filter(Number.isFinite);
  const separation = goodScores.length && lowScores.length ? averageNumbers(goodScores) - averageNumbers(lowScores) : 0;
  const selfPenalty = selfScores.length ? Math.abs(100 - averageNumbers(selfScores)) * 0.35 : 0;
  const qualityScore = clampPercent(100 - averageError + Math.max(0, separation) * 0.18 - selfPenalty);
  return {
    id: crypto.randomUUID(),
    models,
    modelLabel: models.map((modelId) => comparisonModels[modelId]?.title || modelId).join(" + "),
    qualityScore,
    averageError: Number(averageError.toFixed(1)),
    separation: Number(separation.toFixed(1)),
    evaluated,
    summary: {
      videosCompared: evaluated.length,
      goodAverage: goodScores.length ? clampPercent(averageNumbers(goodScores)) : null,
      lowAverage: lowScores.length ? clampPercent(averageNumbers(lowScores)) : null,
      selfAverage: selfScores.length ? clampPercent(averageNumbers(selfScores)) : null
    }
  };
}

async function loadFileIntoVideo(video, file) {
  if (!video || !file) throw new Error("Не выбран видеофайл.");
  const url = URL.createObjectURL(file);
  video.pause();
  video.removeAttribute("src");
  video.srcObject = null;
  video.src = url;
  video.muted = true;
  video.preload = "auto";
  try {
    await waitForVideoEvent(video, "loadedmetadata");
    return url;
  } catch (err) {
    URL.revokeObjectURL(url);
    throw err;
  }
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
    analysisRange,
    onAnalysisRangeChange,
    mediaPipeSettings,
    liveDetectionPaused = false,
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
  const [isMuted, setIsMuted] = useState(true);
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
    scanSkeleton,
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
    if (!video || !canvas || video.readyState < 2) {
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
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const isInsideAnalysisRange =
        !showAnalysisRange ||
        !analysisRange ||
        (video.currentTime >= analysisRange.start && video.currentTime <= analysisRange.end);

      if (scan?.frames?.length) {
        const isInsideScanRange =
          !scan.range || (video.currentTime >= scan.range.start && video.currentTime <= scan.range.end);
        const savedFrame = isInsideScanRange ? nearestScanFrame(scan.frames, video.currentTime) : null;
        if (savedFrame?.landmarks?.length) drawVideoSkeleton(ctx, savedFrame.landmarks, canvas, video, side);
        onPose({
          landmarks: savedFrame?.landmarks || [],
          angles: savedFrame?.angles || {},
          timestamp: video.currentTime,
          confidence: savedFrame?.confidence || 0,
          source: "scan"
        });
        lastVideoTimeRef.current = video.currentTime;
        rafRef.current = requestAnimationFrame(analyzeFrame);
        return;
      }

      if (liveDetectionPaused || !landmarker) {
        onPose({ landmarks: [], angles: {}, timestamp: video.currentTime, confidence: 0, source: "paused" });
        lastVideoTimeRef.current = video.currentTime;
        rafRef.current = requestAnimationFrame(analyzeFrame);
        return;
      }

      const result = landmarker.detectForVideo(video, nextTimestamp());
      const landmarks = result.landmarks?.[0] || [];
      if (landmarks.length && isInsideAnalysisRange) drawVideoSkeleton(ctx, landmarks, canvas, video, side);

      onPose({
        landmarks: isInsideAnalysisRange ? landmarks : [],
        angles: isInsideAnalysisRange ? poseAngles(landmarks, specs) : {},
        timestamp: video.currentTime,
        confidence: landmarks.length && isInsideAnalysisRange ? averageVisibility(landmarks) : 0,
        source: "live"
      });
      lastVideoTimeRef.current = video.currentTime;
    }

    rafRef.current = requestAnimationFrame(analyzeFrame);
  }, [analysisRange, landmarker, liveDetectionPaused, nextTimestamp, onPose, scan, showAnalysisRange, side, specs]);

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
      setIsMuted(true);
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
    video.muted = isMuted;
    video.pause();
    video.currentTime = 0;
    setCurrentTime(0);
    setIsPlaying(false);
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

  function toggleSound() {
    const video = videoRef.current;
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    if (video) video.muted = nextMuted;
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
      return result;
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setError(`Не получилось отсканировать скелет: ${detail}`);
      console.error(err);
      throw err;
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
          muted={isMuted}
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
          <div className="dual-range" style={{ "--start": `${(analysisRange.start / duration) * 100}%`, "--end": `${(analysisRange.end / duration) * 100}%` }}>
            <div className="range-track">
              <i />
            </div>
            <input
              aria-label="Начало диапазона анализа"
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
            <input
              aria-label="Конец диапазона анализа"
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
        <button type="button" onClick={toggleSound}>
          {isMuted ? "Включить звук" : "Выключить звук"}
        </button>
        <button type="button" onClick={reset}>
          <RotateCcw size={18} />
          Сброс
        </button>
        <button type="button" className="scan-button" onClick={scanSkeleton} disabled={isScanning || !scanLandmarker}>
          <Wand2 size={18} />
          {isScanning ? `${scanProgress}%` : scanLandmarker ? "Сканировать скелет" : "MediaPipe загружается"}
        </button>
      </div>

      <div className="scan-meta">
        <span>Данные позы: {scan?.frames?.length ? `${scan.frames.length} точек таймлайна` : "не сохранены"}</span>
        <span>Точность: {scan?.averageConfidence ? `${Math.round(scan.averageConfidence * 100)}%` : "-"}</span>
        <span>
          Скан: {scan?.settings ? `${scan.settings.modelVariant} / ${scan.settings.effectiveScanFps} fps / ${scan.settings.landmarkSet}` : "-"}
        </span>
      </div>
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

function TrackingEnginePanel({ engine, onChange }) {
  return (
    <section className="settings-panel engine-panel">
      <div className="settings-title">
        <div>
          <p className="eyebrow">Движок сканирования</p>
          <h2>Выбор источника скелета</h2>
        </div>
        <span className="status status-good">{captureEngines[engine].title}</span>
      </div>
      <div className="engine-switch">
        {Object.values(captureEngines).map((item) => (
          <button
            type="button"
            key={item.id}
            className={engine === item.id ? "model-card active" : "model-card"}
            onClick={() => onChange(item.id)}
          >
            <strong>{item.shortTitle}</strong>
            <span>{item.description}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function MotionCapLab({ leftScan, rightScan, status, onImport, comparison }) {
  return (
    <>
      <section className="settings-panel motioncap-panel">
        <div className="settings-title">
          <div>
            <p className="eyebrow">MotionCap Lab</p>
            <h2>FreeMoCap 3D-скелет</h2>
          </div>
          <span className="status status-good">CSV импорт</span>
        </div>
        <p className="motioncap-intro">
          FreeMoCap запускается как отдельная Python/desktop-система: записывает одну или несколько камер, синхронизирует видео,
          строит 2D-точки через SkellyTracker, триангулирует 3D через Anipose и сохраняет скелет в `output_data`.
          В DMPA сейчас можно загрузить экспорт `mediapipe_body_3d_xyz.csv` или `*_by_frame.csv` и сравнить 3D-скелеты.
        </p>
        <div className="motioncap-grid">
          <MotionCapImportCard title="Эталон FreeMoCap" scan={leftScan} side="left" onImport={onImport} />
          <MotionCapImportCard title="Ученик FreeMoCap" scan={rightScan} side="right" onImport={onImport} />
        </div>
        {status && <p className="motioncap-status">{status}</p>}
      </section>

      <section className="analysis-panel motioncap-analysis">
        <div className="score-ring" style={{ "--score": comparison.score }}>
          <div>
            <span>{comparison.score}%</span>
            <small>3D схожесть</small>
          </div>
        </div>
        <div className="analysis-body">
          <div className="metrics">
            <MetricCard label="Статус MotionCap" value={comparison.ready ? "готов" : "ожидание"} />
            <MetricCard label="Общих 3D-точек" value={comparison.sharedKeypoints || "-"} />
            <MetricCard label="Кадров сравнено" value={comparison.framesCompared || "-"} />
            <MetricCard label="Формат эталона" value={leftScan?.format || "-"} />
            <MetricCard label="Формат ученика" value={rightScan?.format || "-"} />
          </div>
          <div className="verdict">
            <h2>Анализ FreeMoCap относительно эталона</h2>
            <p>{comparison.verdict}</p>
          </div>
          <div className="suggestions">
            <h3>Как использовать дальше</h3>
            <p>Сначала прогоняем видео через FreeMoCap, затем импортируем CSV сюда и сравниваем уже не плоский MediaPipe-скелет, а 3D-траектории суставов.</p>
            <p>Следующий шаг - связать этот 3D-скелет с нашими моделями “фраза / события / evidence gate”, чтобы оценка танца опиралась на объемное движение.</p>
          </div>
        </div>
      </section>
    </>
  );
}

function MotionCapImportCard({ title, scan, side, onImport }) {
  return (
    <div className="motioncap-card">
      <div>
        <p className="eyebrow">{side === "left" ? "Левое / эталон" : "Правое / ученик"}</p>
        <h3>{title}</h3>
      </div>
      <label className="button camera-primary">
        <FileVideo size={18} />
        Загрузить CSV
        <input type="file" accept=".csv,text/csv" onChange={(event) => onImport(side, event.target.files?.[0] || null)} />
      </label>
      <div className="motioncap-card-stats">
        <MetricCard label="Файл" value={scan?.fileName || "-"} />
        <MetricCard label="Кадров" value={scan?.frameCount || "-"} />
        <MetricCard label="3D-точек" value={scan?.keypointCount || "-"} />
        <MetricCard label="Длительность" value={scan?.duration ? formatSeconds(scan.duration, 1) : "-"} />
      </div>
    </div>
  );
}

function loadLabHistory() {
  try {
    return compactLabHistory(JSON.parse(localStorage.getItem(labHistoryKey) || "[]"));
  } catch {
    return [];
  }
}

function compactLabHistory(items, options = {}) {
  const maxItems = options.maxItems ?? maxStoredLabItems;
  const includeSkeletons = options.includeSkeletons ?? true;
  return (Array.isArray(items) ? items : []).slice(0, maxItems).map((item) => compactLabHistoryItem(item, includeSkeletons));
}

function compactLabHistoryItem(item, includeSkeletons = true) {
  return {
    ...item,
    angleRows: sampleEvenly(item?.angleRows || [], maxStoredAngleRows),
    suggestions: (item?.suggestions || []).slice(0, 8),
    skeletons: includeSkeletons ? compactSkeletonBundle(item?.skeletons) : null
  };
}

function compactSkeletonBundle(skeletons) {
  if (!skeletons) return null;
  return {
    left: compactStoredSkeleton(skeletons.left),
    right: compactStoredSkeleton(skeletons.right),
    synchronizedPairs: sampleEvenly(skeletons.synchronizedPairs || [], maxStoredSkeletonFrames)
  };
}

function compactStoredSkeleton(skeleton) {
  if (!skeleton) return null;
  const frames = sampleEvenly(skeleton.frames || [], maxStoredSkeletonFrames);
  return {
    ...skeleton,
    storedFrames: frames.length,
    frames
  };
}

function saveLabHistoryToStorage(history) {
  const compact = compactLabHistory(history);
  try {
    localStorage.setItem(labHistoryKey, JSON.stringify(compact));
    return compact;
  } catch (err) {
    const tiny = compactLabHistory(history, { maxItems: 8, includeSkeletons: false });
    try {
      localStorage.setItem(labHistoryKey, JSON.stringify(tiny));
      return tiny;
    } catch {
      console.warn("Не удалось сохранить историю лаборатории: quota exceeded.", err);
      localStorage.removeItem(labHistoryKey);
      return [];
    }
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
  const FieldTitle = ({ children, help }) => (
    <span className="setting-label">
      {children}
      <span className="help-dot" tabIndex="0" aria-label={help} data-help={help}>
        ?
      </span>
    </span>
  );

  return (
    <section className="settings-panel">
      <div className="settings-title">
        <div>
          <p className="eyebrow">Настройки MediaPipe</p>
          <h2>Параметры трекинга и датасета</h2>
        </div>
        <span className={`status ${isReady ? "status-good" : ""}`}>{isReady ? "модель активна" : "модель загружается"}</span>
      </div>

      <div className="settings-grid">
        <label>
          <FieldTitle help={mediaPipeHelp.modelVariant}>Модель распознавания позы</FieldTitle>
          <select value={settings.modelVariant} onChange={(event) => update({ modelVariant: event.target.value })}>
            <option value="lite">Легкая - быстрее</option>
            <option value="full">Полная - баланс</option>
            <option value="heavy">Тяжелая - точнее, медленнее</option>
          </select>
        </label>
        <label>
          <FieldTitle help={mediaPipeHelp.delegate}>Устройство вычисления</FieldTitle>
          <select value={settings.delegate} onChange={(event) => update({ delegate: event.target.value })}>
            <option value="GPU">Видеокарта</option>
            <option value="CPU">Процессор</option>
          </select>
        </label>
        <label>
          <FieldTitle help={mediaPipeHelp.numPoses}>Количество людей в кадре</FieldTitle>
          <input
            type="number"
            min="1"
            max="4"
            value={settings.numPoses}
            onChange={(event) => updateNumber("numPoses", event.target.value)}
          />
        </label>
        <label>
          <FieldTitle help={mediaPipeHelp.scanFps}>Частота сканирования</FieldTitle>
          <select value={settings.scanFps} onChange={(event) => updateNumber("scanFps", event.target.value)}>
            <option value="2">2 кадра/сек</option>
            <option value="3">3 кадра/сек</option>
            <option value="5">5 кадров/сек</option>
            <option value="10">10 кадров/сек</option>
            <option value="15">15 кадров/сек</option>
          </select>
        </label>
        <label>
          <FieldTitle help={mediaPipeHelp.landmarkSet}>Точки скелета в датасете</FieldTitle>
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
          <FieldTitle help={mediaPipeHelp.outputSegmentationMasks}>Маска сегментации тела</FieldTitle>
        </label>
        <label>
          <FieldTitle help={mediaPipeHelp.minPoseDetectionConfidence}>Уверенность обнаружения</FieldTitle>
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
          <FieldTitle help={mediaPipeHelp.minPosePresenceConfidence}>Уверенность присутствия позы</FieldTitle>
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
          <FieldTitle help={mediaPipeHelp.minTrackingConfidence}>Уверенность сопровождения</FieldTitle>
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
        <strong>
          Области сравнения
          <span className="help-dot" tabIndex="0" aria-label={mediaPipeHelp.regions} data-help={mediaPipeHelp.regions}>
            ?
          </span>
        </strong>
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
        <label>
          <input type="checkbox" checked={settings.regions.hands} onChange={(event) => updateRegion("hands", event.target.checked)} />
          Кисти MediaPipe Hands
          <span className="help-dot" tabIndex="0" aria-label={mediaPipeHelp.hands} data-help={mediaPipeHelp.hands}>
            ?
          </span>
        </label>
        <label>
          <input type="checkbox" checked={settings.regions.face} onChange={(event) => updateRegion("face", event.target.checked)} />
          Лицо MediaPipe Face
          <span className="help-dot" tabIndex="0" aria-label={mediaPipeHelp.face} data-help={mediaPipeHelp.face}>
            ?
          </span>
        </label>
        <span className="region-summary">
          MediaPipe возвращает 33 точки. В сравнении сейчас активно {specs.length} углов, в датасет сохраняется{" "}
          {settings.landmarkSet === "full33" ? "33 точки" : "13 ключевых точек"}.
        </span>
      </div>
    </section>
  );
}

function ComparisonModelPanel({ model, onChange }) {
  return (
    <section className="comparison-model-panel">
      <div className="settings-title">
        <div>
          <p className="eyebrow">Comparison Model</p>
          <h2>Модель сравнения скелетов</h2>
        </div>
        <span className="status status-good">{comparisonModels[model].title}</span>
      </div>
      <div className="model-tabs">
        {Object.entries(comparisonModels).map(([key, item]) => (
          <div
            className={model === key ? "model-tab-card selected" : "model-tab-card"}
            key={key}
            role="button"
            tabIndex="0"
            onClick={() => onChange(key)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") onChange(key);
            }}
          >
            <strong>
              {item.shortTitle} {item.versionLabel ? `· ${item.versionLabel}` : ""}
            </strong>
            <span>{item.description}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function AutonomousEnsembleLab({
  referenceFile,
  candidateFiles,
  selectedModels,
  runState,
  onReferenceChange,
  onCandidatesChange,
  onToggleModel,
  onRun,
  onExport
}) {
  const selectableModels = runnableComparisonModelIds.filter((id) => id !== "openai-expert");
  return (
    <section className="autonomous-lab">
      <div className="lab-header">
        <div>
          <p className="eyebrow">Autonomous Ensemble Lab</p>
          <h2>Общий автономный анализ пачки видео</h2>
        </div>
        <span>{runState.results.length} результатов</span>
      </div>

      <div className="autonomous-grid">
        <label className="file-drop">
          <FileVideo size={18} />
          <span>
            <b>Эталон педагога</b>
            {referenceFile ? referenceFile.name : "Выберите один файл"}
          </span>
          <input type="file" accept="video/*" onChange={(event) => onReferenceChange(event.target.files?.[0] || null)} />
        </label>
        <label className="file-drop">
          <FileVideo size={18} />
          <span>
            <b>Видео учеников</b>
            {candidateFiles.length ? `${candidateFiles.length} файлов` : "Выберите несколько файлов"}
          </span>
          <input type="file" accept="video/*" multiple onChange={(event) => onCandidatesChange(Array.from(event.target.files || []))} />
        </label>
      </div>

      <div className="ensemble-models">
        {selectableModels.map((modelId) => {
          const modelItem = comparisonModels[modelId];
          return (
            <label key={modelId} className={selectedModels.includes(modelId) ? "selected" : ""}>
              <input type="checkbox" checked={selectedModels.includes(modelId)} onChange={() => onToggleModel(modelId)} />
              <span>
                <b>{modelItem.title}</b>
                {modelItem.versionLabel}
              </span>
            </label>
          );
        })}
      </div>

      <div className="sync-actions autonomous-actions">
        <button
          type="button"
          onClick={onRun}
          disabled={!referenceFile || !candidateFiles.length || !selectedModels.length || runState.status === "running"}
        >
          <Play size={18} />
          {runState.status === "running" ? `Автоанализ ${runState.progress}%` : "Сравнить пачку"}
        </button>
        <button type="button" onClick={onExport} disabled={!runState.results.length}>
          Экспорт ансамбля JSON
        </button>
        <span>{runState.message || "Имена файлов используются как подсказка: похожее, стоит, не похоже."}</span>
      </div>

      <div className={`run-status ${runState.status}`}>
        <div>
          <strong>{runState.status === "running" ? "Автономная лаборатория считает" : "Автономная лаборатория готова"}</strong>
          <span>{runState.message || "Выберите эталон, видео учеников и модели."}</span>
        </div>
        <div className="run-progress" aria-label="Прогресс автономной лаборатории">
          <i style={{ width: `${runState.progress}%` }} />
        </div>
      </div>

      {runState.results.length > 0 && (
        <div className="ensemble-table" style={{ "--model-count": selectedModels.length }}>
          <div className="ensemble-row ensemble-head">
            <span>Видео</span>
            <span>Тип по имени</span>
            <span>Ансамбль</span>
            {selectedModels.map((modelId) => (
              <span key={modelId}>{comparisonModels[modelId]?.title || modelId}</span>
            ))}
            <span>Вывод</span>
          </div>
          {runState.results.map((item) => (
            <div className="ensemble-row" key={item.id}>
              <span>{item.fileName}</span>
              <span>{item.caseLabel}</span>
              <strong>{item.ensembleScore == null ? "-" : `${item.ensembleScore}%`}</strong>
              {selectedModels.map((modelId) => (
                <span key={modelId}>{item.modelScores[modelId] == null ? "-" : `${item.modelScores[modelId]}%`}</span>
              ))}
              <span>{item.recommendation}</span>
            </div>
          ))}
        </div>
      )}

      {runState.bestTrials?.length > 0 && (
        <div className="ensemble-summary">
          <div className="best-model-card">
            <p className="eyebrow">Best Found</p>
            <h3>{runState.bestTrials[0].modelLabel}</h3>
            <div className="best-model-stats">
              <span>Качество поиска: <b>{runState.bestTrials[0].qualityScore}%</b></span>
              <span>Средняя ошибка: <b>{runState.bestTrials[0].averageError} п.п.</b></span>
              <span>Разделение good/low: <b>{runState.bestTrials[0].separation} п.п.</b></span>
              <span>Видео: <b>{runState.bestTrials[0].summary.videosCompared}</b></span>
            </div>
          </div>
          <div className="trial-list">
            <div className="trial-list-head">
              <strong>История перебора моделей</strong>
              <span>Порядок запуска: {(runState.executionOrder || selectedModels).map((id) => comparisonModels[id]?.title || id).join(" → ")}</span>
            </div>
            {runState.bestTrials.map((trial, index) => (
              <div className="trial-item" key={trial.id}>
                <b>{index + 1}. {trial.modelLabel}</b>
                <span>
                  качество {trial.qualityScore}% · ошибка {trial.averageError} · good{" "}
                  {trial.summary.goodAverage == null ? "-" : `${trial.summary.goodAverage}%`} · low{" "}
                  {trial.summary.lowAverage == null ? "-" : `${trial.summary.lowAverage}%`} · self{" "}
                  {trial.summary.selfAverage == null ? "-" : `${trial.summary.selfAverage}%`}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function SequentialGateLab({
  referenceFile,
  candidateFiles,
  modelSequence,
  runState,
  onReferenceChange,
  onCandidatesChange,
  onAddModel,
  onRemoveModelAt,
  onClearSequence,
  onRun,
  onExport
}) {
  const selectableModels = runnableComparisonModelIds.filter((id) => id !== "openai-expert");
  return (
    <section className="autonomous-lab sequential-lab">
      <div className="lab-header">
        <div>
          <p className="eyebrow">Sequential Gate Lab</p>
          <h2>Последовательный автономный анализ</h2>
        </div>
        <span>{runState.results.length} результатов</span>
      </div>

      <div className="autonomous-grid">
        <label className="file-drop">
          <FileVideo size={18} />
          <span>
            <b>Эталон педагога</b>
            {referenceFile ? referenceFile.name : "Выберите один файл"}
          </span>
          <input type="file" accept="video/*" onChange={(event) => onReferenceChange(event.target.files?.[0] || null)} />
        </label>
        <label className="file-drop">
          <FileVideo size={18} />
          <span>
            <b>Видео учеников</b>
            {candidateFiles.length ? `${candidateFiles.length} файлов` : "Выберите несколько файлов"}
          </span>
          <input type="file" accept="video/*" multiple onChange={(event) => onCandidatesChange(Array.from(event.target.files || []))} />
        </label>
      </div>

      <div className="sequence-builder">
        <div>
          <p className="eyebrow">Добавить гейт</p>
          <div className="sequence-models">
            {selectableModels.map((modelId) => {
              const modelItem = comparisonModels[modelId];
              return (
                <button type="button" key={modelId} onClick={() => onAddModel(modelId)}>
                  <b>{modelItem.title}</b>
                  <span>{modelItem.versionLabel}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="sequence-list">
          <div className="trial-list-head">
            <strong>Порядок прохождения</strong>
            <button type="button" onClick={onClearSequence} disabled={!modelSequence.length}>
              Очистить
            </button>
          </div>
          {modelSequence.length ? (
            modelSequence.map((modelId, index) => (
              <div className="sequence-step" key={`${modelId}-${index}`}>
                <span>
                  <b>{index + 1}. {comparisonModels[modelId]?.title || modelId}</b>
                  порог {sequentialGateThresholds[modelId] ?? 62}%
                </span>
                <button type="button" onClick={() => onRemoveModelAt(index)}>
                  Убрать
                </button>
              </div>
            ))
          ) : (
            <p className="sync-note">Нажимайте модели слева: первая станет первым пропускным гейтом, вторая проверит только прошедшие видео.</p>
          )}
        </div>
      </div>

      <div className="sync-actions autonomous-actions">
        <button
          type="button"
          onClick={onRun}
          disabled={!referenceFile || !candidateFiles.length || !modelSequence.length || runState.status === "running"}
        >
          <Play size={18} />
          {runState.status === "running" ? `Конвейер ${runState.progress}%` : "Запустить последовательность"}
        </button>
        <button type="button" onClick={onExport} disabled={!runState.results.length}>
          Экспорт конвейера JSON
        </button>
        <span>{runState.message || "Каждая модель решает, пропускать ли видео к следующему гейту."}</span>
      </div>

      <div className={`run-status ${runState.status}`}>
        <div>
          <strong>{runState.status === "running" ? "Последовательный анализ считает" : "Последовательный анализ готов"}</strong>
          <span>{runState.message || "Соберите порядок моделей, затем запустите пакет учеников."}</span>
        </div>
        <div className="run-progress" aria-label="Прогресс последовательной лаборатории">
          <i style={{ width: `${runState.progress}%` }} />
        </div>
      </div>

      {runState.results.length > 0 && (
        <div className="sequential-results">
          {runState.results.map((item) => (
            <div className={`sequential-result-card ${item.passed ? "passed" : "stopped"}`} key={item.id}>
              <div>
                <h3>{item.fileName}</h3>
                <span>{item.passed ? "прошло всю последовательность" : `остановлено на шаге ${item.stoppedAtStep || 1}`}</span>
              </div>
              <strong>{item.finalScore}%</strong>
              <p>{item.stopReason || "Все гейты пройдены, видео можно отправлять в более глубокий анализ."}</p>
              <div className="sequential-steps">
                {item.steps.map((step) => (
                  <span key={`${item.id}-${step.index}`} className={step.passed ? "passed" : "stopped"}>
                    {step.index}. {comparisonModels[step.modelId]?.title || step.modelId}: {step.score == null ? "-" : `${step.score}%`}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
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
  const frames = sampleEvenly(
    scan.frames.filter((frame) => frame.landmarks?.length),
    maxStoredSkeletonFrames
  );
  return {
    duration: scan.duration,
    range: scan.range || null,
    trackedFrames: scan.trackedFrames,
    storedFrames: frames.length,
    averageConfidence: Number((scan.averageConfidence || 0).toFixed(4)),
    landmarkSet: settings.landmarkSet,
    frames: frames.map((frame) => ({
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
  const leftFrames = sampleEvenly(
    leftScan.frames.filter((frame) => frame.landmarks?.length),
    maxStoredSkeletonFrames
  );
  return leftFrames
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

function sampleEvenly(items, maxItems) {
  if (!Array.isArray(items) || items.length <= maxItems) return items || [];
  if (maxItems <= 1) return items.slice(0, 1);
  const lastIndex = items.length - 1;
  return Array.from({ length: maxItems }, (_, index) => items[Math.round((index / (maxItems - 1)) * lastIndex)]);
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
          Сохранить пример вручную
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
                {item.score}% схожесть{item.expectedScore != null ? ` / ожидалось ${item.expectedScore}%` : ""}
              </strong>
              <span>{new Date(item.createdAt).toLocaleString()}</span>
              <div className="history-tags">
                <b>{item.appVersionLabel || item.appDetails?.versionLabel || "app без версии"}</b>
                <b>{comparisonModels[item.comparisonModel]?.title || "Углы"}</b>
                <b>{item.comparisonModelVersionLabel || item.comparisonModelDetails?.versionLabel || comparisonModels[item.comparisonModel]?.versionLabel || "без версии"}</b>
                <b>{item.mediaPipeSettings?.modelVariant || "lite"}</b>
                <b>{item.mediaPipeSettings?.landmarkSet === "full33" ? "33 точки" : "13 точек"}</b>
                <b>{item.metrics?.framesCompared || 0} кадров</b>
                <b>{item.saveMode === "auto" ? "авто" : "ручное"}</b>
                <b>{item.videos?.sameFileCandidate ? "похоже один файл" : "файлы различаются"}</b>
              </div>
              <div className="file-facts">
                <p>
                  <b>Эталон:</b> {item.videos?.left?.name || item.leftFileName || "без файла"} |{" "}
                  {item.videos?.left?.sizeLabel || "-"} | {item.videos?.left?.durationLabel || "-"} |{" "}
                  {item.videos?.left?.type || "-"}
                </p>
                <p>
                  <b>Правое:</b> {item.videos?.right?.name || item.rightFileName || "без файла"} |{" "}
                  {item.videos?.right?.sizeLabel || "-"} | {item.videos?.right?.durationLabel || "-"} |{" "}
                  {item.videos?.right?.type || "-"}
                </p>
              </div>
              <p>
                Метрики: лучший {item.metrics?.bestScore ?? "-"}%, худший {item.metrics?.worstScore ?? "-"}%, длительность{" "}
                {item.metrics?.durationCompared ?? "-"} сек, аудио-смещение {item.sync?.ready ? `${item.sync.offsetSeconds} сек` : "нет"}.
              </p>
              {item.expectedScore != null && (
                <p>
                  Ошибка модели относительно ожидания: {Math.abs(item.score - item.expectedScore)} п.п.
                </p>
              )}
              <p>
                Области: {activeRegionLabels(item.mediaPipeSettings?.regions).join(", ")}. FPS скана:{" "}
                {item.mediaPipeSettings?.scanFps || "-"}.
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

function WaveformTimeline({ leftAudio, rightAudio, sync, manualSync, onManualSyncChange, soundMode, onSoundModeChange, leftDuration = 0, rightDuration = 0 }) {
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
    drawWave(ctx, leftAudio?.waveform, rect.width, rect.height, "#df4a5f", 0, leftAudio?.duration);
    drawWave(ctx, rightAudio?.waveform, rect.width, rect.height, "#407ee8", sync?.ready ? sync.offsetSeconds : 0, rightAudio?.duration);
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

  const updateManual = (patch) => onManualSyncChange({ ...manualSync, ...patch });

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
      <div className="manual-sync-controls">
        <label>
          Ручная поправка дорожки, сек
          <input
            type="number"
            step="0.05"
            value={manualSync.offsetSeconds}
            onChange={(event) => updateManual({ offsetSeconds: Number(event.target.value) || 0 })}
          />
        </label>
        <label>
          <input
            type="checkbox"
            checked={manualSync.useManualStarts}
            onChange={(event) => updateManual({ useManualStarts: event.target.checked })}
          />
          Ручные старты видео
        </label>
        <label>
          Старт эталона, сек
          <input
            type="number"
            min="0"
            max={leftDuration || undefined}
            step="0.05"
            value={manualSync.leftStart}
            onChange={(event) => updateManual({ leftStart: clampVideoTime(event.target.value, leftDuration) })}
          />
        </label>
        <label>
          Старт правого, сек
          <input
            type="number"
            min="0"
            max={rightDuration || undefined}
            step="0.05"
            value={manualSync.rightStart}
            onChange={(event) => updateManual({ rightStart: clampVideoTime(event.target.value, rightDuration) })}
          />
        </label>
        <label>
          Звук при прогоне
          <select value={soundMode} onChange={(event) => onSoundModeChange(event.target.value)}>
            <option value="left">Эталон</option>
            <option value="right">Правое видео</option>
            <option value="both">Оба видео</option>
            <option value="muted">Без звука</option>
          </select>
        </label>
      </div>
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

function drawWave(ctx, waveform, width, height, color, offsetSeconds, duration = 0) {
  if (!waveform?.length) return;
  const center = height / 2;
  const scale = height * 0.36;
  const offsetPx = duration ? (offsetSeconds / duration) * width : offsetSeconds * 14;
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

function SkeletonOverlayViewer({ leftScan, rightScan, sync, regions, enabled }) {
  const canvasRef = useRef(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const pairs = useMemo(() => {
    if (!enabled || !leftScan?.frames?.length || !rightScan?.frames?.length) return [];
    const allPairs = synchronizedFramePairs(leftScan, rightScan, sync?.ready ? sync.offsetSeconds : 0);
    const stride = Math.max(1, Math.ceil(allPairs.length / maxOverlayPreviewFrames));
    return allPairs.filter((_, index) => index % stride === 0);
  }, [enabled, leftScan, rightScan, sync]);
  const pair = pairs[currentIndex] || null;

  useEffect(() => {
    setCurrentIndex(0);
    setIsPlaying(false);
  }, [pairs.length]);

  useEffect(() => {
    if (!enabled || !isPlaying || pairs.length < 2) return undefined;
    const timer = window.setInterval(() => {
      setCurrentIndex((index) => (index + 1) % pairs.length);
    }, 140);
    return () => window.clearInterval(timer);
  }, [enabled, isPlaying, pairs.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#101827";
    ctx.fillRect(0, 0, rect.width, rect.height);
    drawSkeletonGrid(ctx, rect.width, rect.height);

    if (pair) {
      const left = normalizeSkeleton(pair.leftFrame.landmarks, leftScan?.video?.aspect);
      const right = fitNormalizedSkeletonForPreview(left, normalizeSkeleton(pair.rightFrame.landmarks, rightScan?.video?.aspect));
      drawNormalizedSkeleton(ctx, left, rect.width, rect.height, "#28d7a4", regions);
      drawNormalizedSkeleton(ctx, right, rect.width, rect.height, "#55a4ff", regions);
      ctx.fillStyle = "#d8e8fa";
      ctx.font = "700 13px Inter, sans-serif";
      ctx.fillText(`Эталон ${formatSeconds(pair.leftTime)} / правое ${formatSeconds(pair.rightTime)}`, 14, 24);
    } else {
      ctx.fillStyle = "#d8e8fa";
      ctx.font = "700 15px Inter, sans-serif";
      ctx.fillText("Сначала отсканируйте оба скелета", 14, 28);
    }
  }, [leftScan?.video?.aspect, pair, regions, rightScan?.video?.aspect]);

  const frameScore = pair ? compareOverlayFrames(pair.leftFrame, pair.rightFrame, regions).score : null;

  return (
    <section className={`skeleton-lab ${enabled ? "" : "hidden"}`}>
      <div className="sync-header">
        <div>
          <p className="eyebrow">Overlay Lab</p>
          <h2>Наложение скелет на скелет</h2>
        </div>
        <div className="legend">
          <span>
            <i className="skeleton-left" /> эталон
          </span>
          <span>
            <i className="skeleton-right" /> правое видео
          </span>
        </div>
      </div>
      <canvas ref={canvasRef} className="skeleton-canvas" />
      <div className="overlay-controls">
        <button type="button" onClick={() => setIsPlaying((value) => !value)} disabled={!pairs.length}>
          {isPlaying ? <Pause size={17} /> : <Play size={17} />}
          {isPlaying ? "Пауза" : "Play"}
        </button>
        <button type="button" onClick={() => setCurrentIndex((index) => Math.max(0, index - 1))} disabled={!pairs.length}>
          Назад
        </button>
        <input
          type="range"
          min="0"
          max={Math.max(0, pairs.length - 1)}
          step="1"
          value={Math.min(currentIndex, Math.max(0, pairs.length - 1))}
          onChange={(event) => {
            setIsPlaying(false);
            setCurrentIndex(Number(event.target.value));
          }}
          disabled={!pairs.length}
        />
        <button
          type="button"
          onClick={() => setCurrentIndex((index) => Math.min(Math.max(0, pairs.length - 1), index + 1))}
          disabled={!pairs.length}
        >
          Вперед
        </button>
      </div>
      <div className="overlay-meta">
        <span>
          Кадр {pairs.length ? currentIndex + 1 : 0}/{pairs.length}
        </span>
        <span>Эталон: {pair ? formatSeconds(pair.leftTime, 2) : "-"}</span>
        <span>Правое: {pair ? formatSeconds(pair.rightTime, 2) : "-"}</span>
        <span>Схожесть кадра: {frameScore != null ? `${frameScore}%` : "-"}</span>
      </div>
      <p className="sync-note">
        Здесь оба скелета центрируются по корпусу и приводятся к одному масштабу. Чем меньше расстояние между точками, тем выше оценка
        наложения.
      </p>
    </section>
  );
}

function AngleComparisonViewer({ leftScan, rightScan, sync, comparison, regions, enabled }) {
  const canvasRef = useRef(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const pairs = useMemo(() => {
    if (!enabled || !leftScan?.frames?.length || !rightScan?.frames?.length) return [];
    const allPairs = synchronizedAngleFramePairs(leftScan, rightScan, sync?.ready ? sync.offsetSeconds : 0).pairs;
    const stride = Math.max(1, Math.ceil(allPairs.length / maxOverlayPreviewFrames));
    return allPairs.filter((_, index) => index % stride === 0);
  }, [enabled, leftScan, rightScan, sync]);
  const pair = pairs[currentIndex] || null;

  useEffect(() => {
    setCurrentIndex(0);
    setIsPlaying(false);
  }, [pairs.length]);

  useEffect(() => {
    if (!enabled || !isPlaying || pairs.length < 2) return undefined;
    const timer = window.setInterval(() => {
      setCurrentIndex((index) => (index + 1) % pairs.length);
    }, 180);
    return () => window.clearInterval(timer);
  }, [enabled, isPlaying, pairs.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#101827";
    ctx.fillRect(0, 0, rect.width, rect.height);
    drawSkeletonGrid(ctx, rect.width, rect.height);

    if (pair) {
      const left = normalizeSkeleton(pair.leftFrame.landmarks, leftScan?.video?.aspect);
      const right = fitNormalizedSkeletonForPreview(left, normalizeSkeleton(pair.rightFrame.landmarks, rightScan?.video?.aspect));
      drawSkeletonDifferences(ctx, left, right, rect.width, rect.height, regions);
      drawNormalizedSkeleton(ctx, left, rect.width, rect.height, "#28d7a4", regions);
      drawNormalizedSkeleton(ctx, right, rect.width, rect.height, "#55a4ff", regions);
      drawAngleHotspots(ctx, left, right, comparison?.rows || [], rect.width, rect.height, regions);
      ctx.fillStyle = "#d8e8fa";
      ctx.font = "700 13px Inter, sans-serif";
      ctx.fillText(`Углы: эталон ${formatSeconds(pair.leftTime, 2)} / правое ${formatSeconds(pair.rightTime, 2)}`, 14, 24);
    } else {
      ctx.fillStyle = "#d8e8fa";
      ctx.font = "700 15px Inter, sans-serif";
      ctx.fillText("Запустите полный анализ модели «Углы», чтобы увидеть сравнение.", 14, 28);
    }
  }, [comparison?.rows, leftScan?.video?.aspect, pair, regions, rightScan?.video?.aspect]);

  const frameScore = pair
    ? comparePoseFrames(pair.leftFrame, pair.rightFrame, regions, {
        leftAspect: leftScan?.video?.aspect,
        rightAspect: rightScan?.video?.aspect
      }).score
    : null;
  const angleRows = (comparison?.rows || []).slice(0, 6);

  return (
    <section className={`skeleton-lab angle-lab ${enabled ? "" : "hidden"}`}>
      <div className="sync-header">
        <div>
          <p className="eyebrow">Angle Lab</p>
          <h2>Визуализация модели «Углы»</h2>
        </div>
        <div className="legend">
          <span>
            <i className="skeleton-left" /> эталон
          </span>
          <span>
            <i className="skeleton-right" /> правое видео
          </span>
          <span>
            <i className="difference-color" /> угловое расхождение
          </span>
        </div>
      </div>
      <canvas ref={canvasRef} className="skeleton-canvas" />
      <div className="overlay-controls">
        <button type="button" onClick={() => setIsPlaying((value) => !value)} disabled={!pairs.length}>
          {isPlaying ? <Pause size={17} /> : <Play size={17} />}
          {isPlaying ? "Пауза" : "Play"}
        </button>
        <button type="button" onClick={() => setCurrentIndex((index) => Math.max(0, index - 1))} disabled={!pairs.length}>
          Назад
        </button>
        <input
          type="range"
          min="0"
          max={Math.max(0, pairs.length - 1)}
          step="1"
          value={Math.min(currentIndex, Math.max(0, pairs.length - 1))}
          onChange={(event) => {
            setIsPlaying(false);
            setCurrentIndex(Number(event.target.value));
          }}
          disabled={!pairs.length}
        />
        <button
          type="button"
          onClick={() => setCurrentIndex((index) => Math.min(Math.max(0, pairs.length - 1), index + 1))}
          disabled={!pairs.length}
        >
          Вперед
        </button>
      </div>
      <div className="overlay-meta">
        <span>
          Кадр {pairs.length ? currentIndex + 1 : 0}/{pairs.length}
        </span>
        <span>Эталон: {pair ? formatSeconds(pair.leftTime, 2) : "-"}</span>
        <span>Правое: {pair ? formatSeconds(pair.rightTime, 2) : "-"}</span>
        <span>Углы кадра: {frameScore != null ? `${frameScore}%` : "-"}</span>
        <span>Итог модели: {comparison?.ready ? `${comparison.score}%` : "-"}</span>
      </div>
      {angleRows.length > 0 && (
        <div className="elastic-metrics">
          {angleRows.map((row) => (
            <span key={row.id}>
              {row.title}: <b>{row.diff}°</b>
            </span>
          ))}
          {comparison?.diagnostics?.trackingOutliersSkipped > 0 && (
            <span>
              Срывы трекинга: <b>{comparison.diagnostics.trackingOutliersSkipped}</b>
            </span>
          )}
        </div>
      )}
      <p className="sync-note">
        Здесь правый скелет сначала точно подгоняется корпусом к эталону. Красные маркеры показывают суставы, где средняя разница углов
        самая заметная.
      </p>
    </section>
  );
}

function ElasticDanceViewer({ leftScan, rightScan, sync, comparison, regions, enabled, modelId = "2026-07-12" }) {
  const canvasRef = useRef(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const pairs = useMemo(() => {
    if (!enabled || !leftScan?.frames?.length || !rightScan?.frames?.length) return [];
    const cleanedLeftScan =
      modelId === "2026-07-13" ? { ...leftScan, frames: filterSkeletonFrames_2026_07_13(leftScan) } : leftScan;
    const cleanedRightScan =
      modelId === "2026-07-13" ? { ...rightScan, frames: filterSkeletonFrames_2026_07_13(rightScan) } : rightScan;
    const allPairs = synchronizedFramePairs(cleanedLeftScan, cleanedRightScan, sync?.ready ? sync.offsetSeconds : 0);
    const stride = Math.max(1, Math.ceil(allPairs.length / maxOverlayPreviewFrames));
    return allPairs.filter((_, index) => index % stride === 0);
  }, [enabled, leftScan, modelId, rightScan, sync]);
  const pair = pairs[currentIndex] || null;

  useEffect(() => {
    setCurrentIndex(0);
    setIsPlaying(false);
  }, [pairs.length]);

  useEffect(() => {
    if (!enabled || !isPlaying || pairs.length < 2) return undefined;
    const timer = window.setInterval(() => {
      setCurrentIndex((index) => (index + 1) % pairs.length);
    }, 180);
    return () => window.clearInterval(timer);
  }, [enabled, isPlaying, pairs.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#101827";
    ctx.fillRect(0, 0, rect.width, rect.height);
    drawSkeletonGrid(ctx, rect.width, rect.height);

    if (pair) {
      const left = normalizeSkeleton(pair.leftFrame.landmarks, leftScan?.video?.aspect);
      const right = fitNormalizedSkeletonForPreview(left, normalizeSkeleton(pair.rightFrame.landmarks, rightScan?.video?.aspect));
      drawSkeletonDifferences(ctx, left, right, rect.width, rect.height, regions);
      drawNormalizedSkeleton(ctx, left, rect.width, rect.height, "#28d7a4", regions);
      drawNormalizedSkeleton(ctx, right, rect.width, rect.height, "#55a4ff", regions);
      ctx.fillStyle = "#d8e8fa";
      ctx.font = "700 13px Inter, sans-serif";
      ctx.fillText(`${comparisonModels[modelId]?.title || "Elastic"}: эталон ${formatSeconds(pair.leftTime, 2)} / правое ${formatSeconds(pair.rightTime, 2)}`, 14, 24);
    } else {
      ctx.fillStyle = "#d8e8fa";
      ctx.font = "700 15px Inter, sans-serif";
      ctx.fillText(`Отсканируйте оба видео и запустите анализ модели ${comparisonModels[modelId]?.title || "Elastic"}`, 14, 28);
    }
  }, [leftScan?.video?.aspect, modelId, pair, regions, rightScan?.video?.aspect]);

  const frameScore = pair ? compareOverlayFrames(pair.leftFrame, pair.rightFrame, regions).score : null;
  const modelRows = (comparison?.rows || []).filter((row) => String(row.id || "").includes(modelId)).slice(0, 5);

  return (
    <section className={`skeleton-lab elastic-lab ${enabled ? "" : "hidden"}`}>
      <div className="sync-header">
        <div>
          <p className="eyebrow">Elastic Dance Lab</p>
          <h2>Визуализация модели {comparisonModels[modelId]?.title || "Elastic"}</h2>
        </div>
        <div className="legend">
          <span>
            <i className="skeleton-left" /> эталон
          </span>
          <span>
            <i className="skeleton-right" /> правое видео
          </span>
          <span>
            <i className="difference-color" /> расхождение
          </span>
        </div>
      </div>
      <canvas ref={canvasRef} className="skeleton-canvas elastic-canvas" />
      <div className="overlay-controls">
        <button type="button" onClick={() => setIsPlaying((value) => !value)} disabled={!pairs.length}>
          {isPlaying ? <Pause size={17} /> : <Play size={17} />}
          {isPlaying ? "Пауза" : "Play"}
        </button>
        <button type="button" onClick={() => setCurrentIndex((index) => Math.max(0, index - 1))} disabled={!pairs.length}>
          Назад
        </button>
        <input
          type="range"
          min="0"
          max={Math.max(0, pairs.length - 1)}
          step="1"
          value={Math.min(currentIndex, Math.max(0, pairs.length - 1))}
          onChange={(event) => {
            setIsPlaying(false);
            setCurrentIndex(Number(event.target.value));
          }}
          disabled={!pairs.length}
        />
        <button
          type="button"
          onClick={() => setCurrentIndex((index) => Math.min(Math.max(0, pairs.length - 1), index + 1))}
          disabled={!pairs.length}
        >
          Вперед
        </button>
      </div>
      <div className="overlay-meta">
        <span>
          Кадр {pairs.length ? currentIndex + 1 : 0}/{pairs.length}
        </span>
        <span>Эталон: {pair ? formatSeconds(pair.leftTime, 2) : "-"}</span>
        <span>Правое: {pair ? formatSeconds(pair.rightTime, 2) : "-"}</span>
        <span>Наложение кадра: {frameScore != null ? `${frameScore}%` : "-"}</span>
        <span>Итог модели: {comparison?.ready ? `${comparison.score}%` : "-"}</span>
      </div>
      {modelRows.length > 0 && (
        <div className="elastic-metrics">
          {modelRows.map((row) => (
            <span key={row.id}>
              {row.title.replace(`${comparisonModels[modelId]?.title || modelId}: `, "")}: <b>{row.score}%</b>
            </span>
          ))}
        </div>
      )}
      <p className="sync-note">
        Это окно показывает сохраненные скелеты после скана, а не живой MediaPipe-детект. Красные связи подсвечивают расстояние между
        одинаковыми точками эталона и правого видео на выбранном кадре.
      </p>
    </section>
  );
}

function AreasDrawingViewer({ leftScan, rightScan, sync, comparison, enabled, mode = "areas" }) {
  const canvasRef = useRef(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const methods = {
    zones: mode === "areas",
    drawing: mode === "drawing"
  };
  const title = mode === "drawing" ? "Рисунок" : "Области";
  const pairs = useMemo(() => {
    if (!enabled || !leftScan?.frames?.length || !rightScan?.frames?.length) return [];
    const allPairs = synchronizedAngleFramePairs(leftScan, rightScan, sync?.ready ? sync.offsetSeconds : 0).pairs
      .map((pair) => ({ ...pair, fitted: fittedPairLandmarks(pair, leftScan, rightScan) }))
      .filter((pair) => pair.fitted?.left?.length && pair.fitted?.right?.length);
    const stride = Math.max(1, Math.ceil(allPairs.length / maxOverlayPreviewFrames));
    return allPairs.filter((_, index) => index % stride === 0);
  }, [enabled, leftScan, rightScan, sync]);
  const pair = pairs[currentIndex] || null;
  const trajectoryPairs = useMemo(() => {
    if (!pair) return [];
    const start = pair.leftTime - 1;
    const end = pair.leftTime + 1;
    return pairs.filter((item) => item.leftTime >= start && item.leftTime <= end);
  }, [pair, pairs]);

  useEffect(() => {
    setCurrentIndex(0);
    setIsPlaying(false);
  }, [pairs.length]);

  useEffect(() => {
    if (!enabled || !isPlaying || pairs.length < 2) return undefined;
    const timer = window.setInterval(() => {
      setCurrentIndex((index) => (index + 1) % pairs.length);
    }, 180);
    return () => window.clearInterval(timer);
  }, [enabled, isPlaying, pairs.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#101827";
    ctx.fillRect(0, 0, rect.width, rect.height);
    drawSkeletonGrid(ctx, rect.width, rect.height);

    if (pair) {
      drawZonesDrawingScene(ctx, pair, trajectoryPairs, rect.width, rect.height, methods);
      ctx.fillStyle = "#d8e8fa";
      ctx.font = "700 13px Inter, sans-serif";
      ctx.fillText(`${title}: эталон ${formatSeconds(pair.leftTime, 2)} / правое ${formatSeconds(pair.rightTime, 2)}`, 14, 24);
    } else {
      ctx.fillStyle = "#d8e8fa";
      ctx.font = "700 15px Inter, sans-serif";
      ctx.fillText(`Запустите полный анализ модели «${title}», чтобы увидеть визуализацию.`, 14, 28);
    }
  }, [methods.drawing, methods.zones, pair, title, trajectoryPairs]);

  const frameScore = pair ? compareZoneFrameScore(pair).score : null;
  const rowPrefix = mode === "drawing" ? "trajectory-drawing" : "joint-areas";
  const modelRows = (comparison?.rows || []).filter((row) => String(row.id || "").startsWith(rowPrefix)).slice(0, 6);

  return (
    <section className={`skeleton-lab zones-drawing-lab ${enabled ? "" : "hidden"}`}>
      <div className="sync-header">
        <div>
          <p className="eyebrow">{mode === "drawing" ? "Drawing Lab" : "Area Lab"}</p>
          <h2>{`Визуализация модели «${title}»`}</h2>
        </div>
        <div className="legend">
          <span>
            <i className="skeleton-left" /> эталон
          </span>
          <span>
            <i className="skeleton-right" /> правое видео
          </span>
          {mode === "areas" && (
            <span>
              <i className="zone-color" /> области
            </span>
          )}
          {mode === "drawing" && (
            <span>
              <i className="trajectory-color" /> рисунки
            </span>
          )}
        </div>
      </div>
      <canvas ref={canvasRef} className="skeleton-canvas zones-drawing-canvas" />
      <div className="overlay-controls">
        <button type="button" onClick={() => setIsPlaying((value) => !value)} disabled={!pairs.length}>
          {isPlaying ? <Pause size={17} /> : <Play size={17} />}
          {isPlaying ? "Пауза" : "Play"}
        </button>
        <button type="button" onClick={() => setCurrentIndex((index) => Math.max(0, index - 1))} disabled={!pairs.length}>
          Назад
        </button>
        <input
          type="range"
          min="0"
          max={Math.max(0, pairs.length - 1)}
          step="1"
          value={Math.min(currentIndex, Math.max(0, pairs.length - 1))}
          onChange={(event) => {
            setIsPlaying(false);
            setCurrentIndex(Number(event.target.value));
          }}
          disabled={!pairs.length}
        />
        <button
          type="button"
          onClick={() => setCurrentIndex((index) => Math.min(Math.max(0, pairs.length - 1), index + 1))}
          disabled={!pairs.length}
        >
          Вперед
        </button>
      </div>
      <div className="overlay-meta">
        <span>
          Кадр {pairs.length ? currentIndex + 1 : 0}/{pairs.length}
        </span>
        <span>Эталон: {pair ? formatSeconds(pair.leftTime, 2) : "-"}</span>
        <span>Правое: {pair ? formatSeconds(pair.rightTime, 2) : "-"}</span>
        {mode === "areas" && <span>Попадание в области: {frameScore != null ? `${clampPercent(frameScore)}%` : "-"}</span>}
        <span>Итог модели: {comparison?.ready ? `${comparison.score}%` : "-"}</span>
      </div>
      {modelRows.length > 0 && (
        <div className="elastic-metrics">
          {modelRows.map((row) => (
            <span key={row.id}>
              {row.title}: <b>{row.score}%</b>
            </span>
          ))}
        </div>
      )}
      <p className="sync-note">
        {mode === "areas"
          ? "Две панели показывают попадание суставов в области отдельно: слева эталон, справа правое видео. Зеленая область означает, что точка находится внутри допустимой зоны."
          : "Две панели показывают рисунок движения отдельно: слева траектории эталона, справа траектории правого видео за короткий фрагмент вокруг выбранного кадра."}
      </p>
    </section>
  );
}

function ZoneGridViewer({ leftScan, rightScan, sync, comparison, enabled }) {
  const canvasRef = useRef(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const pairs = useMemo(() => {
    if (!enabled || !leftScan?.frames?.length || !rightScan?.frames?.length) return [];
    const allPairs = synchronizedAngleFramePairs(leftScan, rightScan, sync?.ready ? sync.offsetSeconds : 0).pairs
      .map((pair) => ({ ...pair, fitted: gridPairLandmarks(pair, leftScan, rightScan) }))
      .filter((pair) => pair.fitted?.left?.length && pair.fitted?.right?.length);
    const stride = Math.max(1, Math.ceil(allPairs.length / maxOverlayPreviewFrames));
    return allPairs.filter((_, index) => index % stride === 0);
  }, [enabled, leftScan, rightScan, sync]);
  const pair = pairs[currentIndex] || null;

  useEffect(() => {
    setCurrentIndex(0);
    setIsPlaying(false);
  }, [pairs.length]);

  useEffect(() => {
    if (!enabled || !isPlaying || pairs.length < 2) return undefined;
    const timer = window.setInterval(() => {
      setCurrentIndex((index) => (index + 1) % pairs.length);
    }, 180);
    return () => window.clearInterval(timer);
  }, [enabled, isPlaying, pairs.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#101827";
    ctx.fillRect(0, 0, rect.width, rect.height);

    if (pair) {
      drawZoneGridScene(ctx, pair, rect.width, rect.height);
      ctx.fillStyle = "#d8e8fa";
      ctx.font = "700 13px Inter, sans-serif";
      ctx.fillText(`Зоны: эталон ${formatSeconds(pair.leftTime, 2)} / правое ${formatSeconds(pair.rightTime, 2)}`, 14, 24);
    } else {
      ctx.fillStyle = "#d8e8fa";
      ctx.font = "700 15px Inter, sans-serif";
      ctx.fillText("Запустите полный анализ модели «Зоны», чтобы увидеть две сетки сравнения.", 14, 28);
    }
  }, [pair]);

  const frameScore = pair ? compareZoneGridFrameScore(pair) : null;
  const modelRows = (comparison?.rows || []).filter((row) => String(row.id || "").startsWith("zone-grid")).slice(0, 6);

  return (
    <section className={`skeleton-lab zone-grid-lab ${enabled ? "" : "hidden"}`}>
      <div className="sync-header">
        <div>
          <p className="eyebrow">Zone Grid Lab</p>
          <h2>Визуализация модели «Зоны»</h2>
        </div>
        <div className="legend">
          <span>
            <i className="skeleton-left" /> эталон
          </span>
          <span>
            <i className="skeleton-right" /> правое видео
          </span>
          <span>
            <i className="zone-match-color" /> тот же сектор
          </span>
          <span>
            <i className="zone-miss-color" /> другой сектор
          </span>
        </div>
      </div>
      <canvas ref={canvasRef} className="skeleton-canvas zone-grid-canvas" />
      <div className="overlay-controls">
        <button type="button" onClick={() => setIsPlaying((value) => !value)} disabled={!pairs.length}>
          {isPlaying ? <Pause size={17} /> : <Play size={17} />}
          {isPlaying ? "Пауза" : "Play"}
        </button>
        <button type="button" onClick={() => setCurrentIndex((index) => Math.max(0, index - 1))} disabled={!pairs.length}>
          Назад
        </button>
        <input
          type="range"
          min="0"
          max={Math.max(0, pairs.length - 1)}
          step="1"
          value={Math.min(currentIndex, Math.max(0, pairs.length - 1))}
          onChange={(event) => {
            setIsPlaying(false);
            setCurrentIndex(Number(event.target.value));
          }}
          disabled={!pairs.length}
        />
        <button
          type="button"
          onClick={() => setCurrentIndex((index) => Math.min(Math.max(0, pairs.length - 1), index + 1))}
          disabled={!pairs.length}
        >
          Вперед
        </button>
      </div>
      <div className="overlay-meta">
        <span>
          Кадр {pairs.length ? currentIndex + 1 : 0}/{pairs.length}
        </span>
        <span>Эталон: {pair ? formatSeconds(pair.leftTime, 2) : "-"}</span>
        <span>Правое: {pair ? formatSeconds(pair.rightTime, 2) : "-"}</span>
        <span>Попадание кадра: {frameScore != null ? `${clampPercent(frameScore)}%` : "-"}</span>
        <span>Сетка: {zoneGridConfig.columns * zoneGridConfig.rows} квадратов</span>
        <span>Итог модели: {comparison?.ready ? `${comparison.score}%` : "-"}</span>
      </div>
      {modelRows.length > 0 && (
        <div className="elastic-metrics">
          {modelRows.map((row) => (
            <span key={row.id}>
              {row.title}: <b>{row.score}%</b>
            </span>
          ))}
        </div>
      )}
      <p className="sync-note">
        Здесь скелеты не накладываются друг на друга: эталон и правое видео стоят в двух одинаковых сетках. Оценка строится по тому,
        насколько одинаковые суставы попадают в такие же или соседние квадраты.
      </p>
    </section>
  );
}

function compareZoneGridFrameScore(pair) {
  const scores = zoneDrawingJointSpecs
    .map((spec) => {
      const leftCell = pointToZoneCell(pair.fitted.left?.[spec.id]);
      const rightCell = pointToZoneCell(pair.fitted.right?.[spec.id]);
      return leftCell && rightCell ? zoneCellScore(leftCell, rightCell) : null;
    })
    .filter(Number.isFinite);
  return averageNumbers(scores);
}

function drawZoneGridScene(ctx, pair, width, height) {
  const gap = 18;
  const top = 44;
  const panelWidth = (width - gap * 3) / 2;
  const panelHeight = height - top - 18;
  const leftRect = { x: gap, y: top, width: panelWidth, height: panelHeight };
  const rightRect = { x: gap * 2 + panelWidth, y: top, width: panelWidth, height: panelHeight };

  drawZoneGridPanel(ctx, leftRect, "Эталон", pair.fitted.left, pair.fitted.right, "left");
  drawZoneGridPanel(ctx, rightRect, "Правое видео", pair.fitted.right, pair.fitted.left, "right");
}

function drawZoneGridPanel(ctx, rect, title, landmarks, oppositeLandmarks, side) {
  ctx.save();
  ctx.strokeStyle = "rgba(216, 232, 250, 0.18)";
  ctx.lineWidth = 1;
  ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
  drawZoneGridLines(ctx, rect);
  highlightZoneGridCells(ctx, rect, landmarks, oppositeLandmarks);
  drawSkeletonInZoneGrid(ctx, landmarks, rect, side === "left" ? "#28d7a4" : "#55a4ff");
  ctx.fillStyle = "#d8e8fa";
  ctx.font = "800 13px Inter, sans-serif";
  ctx.fillText(title, rect.x + 12, rect.y + 22);
  ctx.restore();
}

function drawZoneGridLines(ctx, rect) {
  const { columns, rows } = zoneGridConfig;
  ctx.strokeStyle = "rgba(216, 232, 250, 0.12)";
  ctx.lineWidth = 1;
  for (let col = 1; col < columns; col += 1) {
    const x = rect.x + (rect.width / columns) * col;
    ctx.beginPath();
    ctx.moveTo(x, rect.y);
    ctx.lineTo(x, rect.y + rect.height);
    ctx.stroke();
  }
  for (let row = 1; row < rows; row += 1) {
    const y = rect.y + (rect.height / rows) * row;
    ctx.beginPath();
    ctx.moveTo(rect.x, y);
    ctx.lineTo(rect.x + rect.width, y);
    ctx.stroke();
  }
}

function highlightZoneGridCells(ctx, rect, landmarks, oppositeLandmarks) {
  const { columns, rows } = zoneGridConfig;
  for (const spec of zoneDrawingJointSpecs) {
    const cell = pointToZoneCell(landmarks?.[spec.id]);
    const oppositeCell = pointToZoneCell(oppositeLandmarks?.[spec.id]);
    if (!cell || !oppositeCell) continue;
    const sameCell = cell.col === oppositeCell.col && cell.row === oppositeCell.row;
    ctx.fillStyle = sameCell ? "rgba(40, 215, 164, 0.16)" : "rgba(255, 103, 103, 0.15)";
    ctx.fillRect(rect.x + (rect.width / columns) * cell.col, rect.y + (rect.height / rows) * cell.row, rect.width / columns, rect.height / rows);
  }
}

function drawSkeletonInZoneGrid(ctx, landmarks, rect, color) {
  if (!landmarks?.length) return;
  const ids = new Set(zoneDrawingJointSpecs.map((spec) => spec.id).concat([11, 12, 23, 24]));
  const project = (point) => pointToZoneGridCanvas(point, rect);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  for (const [a, b] of poseConnections) {
    if (!ids.has(a) || !ids.has(b) || !landmarks[a] || !landmarks[b]) continue;
    const pa = project(landmarks[a]);
    const pb = project(landmarks[b]);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
  for (const id of ids) {
    if (!landmarks[id]) continue;
    const point = project(landmarks[id]);
    ctx.beginPath();
    ctx.arc(point.x, point.y, zoneDrawingJointSpecs.some((spec) => spec.id === id) ? 4 : 2.6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function pointToZoneGridCanvas(point, rect) {
  const { xMin, xMax, yMin, yMax } = zoneGridConfig;
  return {
    x: rect.x + ((point.x - xMin) / (xMax - xMin)) * rect.width,
    y: rect.y + ((point.y - yMin) / (yMax - yMin)) * rect.height
  };
}

function drawZonesDrawingScene(ctx, pair, trajectoryPairs, width, height, methods) {
  const gap = 18;
  const top = 44;
  const panelWidth = (width - gap * 3) / 2;
  const panelHeight = height - top - 18;
  const leftRect = { x: gap, y: top, width: panelWidth, height: panelHeight };
  const rightRect = { x: gap * 2 + panelWidth, y: top, width: panelWidth, height: panelHeight };

  drawZonesDrawingPanel(ctx, leftRect, "Эталон", pair.fitted.left, pair.fitted.right, trajectoryPairs, methods, "left");
  drawZonesDrawingPanel(ctx, rightRect, "Правое видео", pair.fitted.right, pair.fitted.left, trajectoryPairs, methods, "right");
}

function drawZonesDrawingPanel(ctx, rect, title, landmarks, oppositeLandmarks, trajectoryPairs, methods, side) {
  ctx.save();
  ctx.strokeStyle = "rgba(216, 232, 250, 0.18)";
  ctx.lineWidth = 1;
  ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
  drawPanelSkeletonGrid(ctx, rect);
  const project = panelSkeletonProjector(rect);
  if (methods.zones) drawJointZones(ctx, landmarks, oppositeLandmarks, project);
  if (methods.drawing) drawJointTrajectories(ctx, trajectoryPairs, project, side);
  drawSkeletonInPanel(ctx, landmarks, rect, side === "left" ? "#28d7a4" : "#55a4ff");
  ctx.fillStyle = "#d8e8fa";
  ctx.font = "800 13px Inter, sans-serif";
  ctx.fillText(title, rect.x + 12, rect.y + 22);
  ctx.restore();
}

function drawPanelSkeletonGrid(ctx, rect) {
  ctx.strokeStyle = "rgba(216, 232, 250, 0.12)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const x = rect.x + (rect.width / 4) * i;
    const y = rect.y + (rect.height / 4) * i;
    ctx.beginPath();
    ctx.moveTo(x, rect.y);
    ctx.lineTo(x, rect.y + rect.height);
    ctx.moveTo(rect.x, y);
    ctx.lineTo(rect.x + rect.width, y);
    ctx.stroke();
  }
}

function panelSkeletonProjector(rect) {
  const centerX = rect.x + rect.width / 2;
  const centerY = rect.y + rect.height * 0.54;
  const scale = Math.min(rect.width, rect.height) * 0.26;
  return {
    scale,
    point: (point) => ({ x: centerX + point.x * scale, y: centerY + point.y * scale })
  };
}

function drawJointZones(ctx, landmarks, oppositeLandmarks, project) {
  for (const spec of zoneDrawingJointSpecs) {
    const point = landmarks?.[spec.id];
    if (!point) continue;
    const center = project.point(point);
    const oppositePoint = oppositeLandmarks?.[spec.id] ? project.point(oppositeLandmarks[spec.id]) : null;
    const radius = spec.radius * project.scale;
    const hit = oppositePoint ? Math.hypot(center.x - oppositePoint.x, center.y - oppositePoint.y) <= radius : false;

    ctx.beginPath();
    ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = hit ? "rgba(40, 215, 164, 0.12)" : "rgba(255, 103, 103, 0.14)";
    ctx.strokeStyle = hit ? "rgba(40, 215, 164, 0.75)" : "rgba(255, 103, 103, 0.85)";
    ctx.lineWidth = 1.5;
    ctx.fill();
    ctx.stroke();
  }
}

function drawJointTrajectories(ctx, pairs, project, side = "left") {
  for (const spec of zoneDrawingJointSpecs) {
    const path = pairs.map((pair) => pair.fitted[side]?.[spec.id]).filter(Boolean);
    drawTrajectoryPath(ctx, path, project, side === "left" ? "rgba(40, 215, 164, 0.72)" : "rgba(85, 164, 255, 0.72)");
  }
}

function drawSkeletonInPanel(ctx, landmarks, rect, color) {
  if (!landmarks?.length) return;
  const ids = new Set(zoneDrawingJointSpecs.map((spec) => spec.id).concat([11, 12, 23, 24]));
  const project = panelSkeletonProjector(rect).point;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  for (const [a, b] of poseConnections) {
    if (!ids.has(a) || !ids.has(b) || !landmarks[a] || !landmarks[b]) continue;
    const pa = project(landmarks[a]);
    const pb = project(landmarks[b]);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
  for (const id of ids) {
    if (!landmarks[id]) continue;
    const point = project(landmarks[id]);
    ctx.beginPath();
    ctx.arc(point.x, point.y, zoneDrawingJointSpecs.some((spec) => spec.id === id) ? 4 : 2.6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawTrajectoryPath(ctx, points, project, color) {
  if (points.length < 2) return;
  ctx.beginPath();
  points.forEach((point, index) => {
    const projected = project.point(point);
    if (index === 0) ctx.moveTo(projected.x, projected.y);
    else ctx.lineTo(projected.x, projected.y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke();

  const first = project.point(points[0]);
  const last = project.point(points.at(-1));
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(first.x, first.y, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(last.x, last.y, 5, 0, Math.PI * 2);
  ctx.fill();
}

function KeyPoseViewer({ comparison, enabled, regions }) {
  const moments = comparison?.poseMoments || [];
  return (
    <section className={`skeleton-lab ${enabled ? "" : "hidden"}`}>
      <div className="sync-header">
        <div>
          <p className="eyebrow">Pose Impulses</p>
          <h2>Ключевые позы на импульсах музыки</h2>
        </div>
        <span className="status">{moments.length ? `${moments.length} поз` : "ожидание анализа"}</span>
      </div>
      {moments.length ? (
        <div className="pose-grid">
          {moments.map((moment) => (
            <PoseCard key={`${moment.leftTime}-${moment.rightTime}`} moment={moment} regions={regions} />
          ))}
        </div>
      ) : (
        <p className="empty-history">После полного анализа модель покажет позы на сильных музыкальных импульсах.</p>
      )}
    </section>
  );
}

function PoseCard({ moment, regions }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#101827";
    ctx.fillRect(0, 0, rect.width, rect.height);
    drawSkeletonGrid(ctx, rect.width, rect.height);
    drawNormalizedSkeleton(ctx, normalizeSkeleton(moment.leftLandmarks), rect.width, rect.height, "#28d7a4", regions);
    drawNormalizedSkeleton(ctx, normalizeSkeleton(moment.rightLandmarks), rect.width, rect.height, "#55a4ff", regions);
  }, [moment, regions]);

  return (
    <article className="pose-card">
      <canvas ref={canvasRef} />
      <strong>{moment.score}%</strong>
      <span>Импульс {formatTime(moment.impulseTime)}</span>
    </article>
  );
}

function drawSkeletonGrid(ctx, width, height) {
  ctx.strokeStyle = "rgba(216, 232, 250, 0.14)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const x = (width / 4) * i;
    const y = (height / 4) * i;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawNormalizedSkeleton(ctx, landmarks, width, height, color, regions) {
  if (!landmarks?.length) return;
  const ids = new Set(overlayLandmarkIds(regions));
  const centerX = width / 2;
  const centerY = height * 0.52;
  const scale = Math.min(width, height) * 0.22;
  const project = (point) => ({ x: centerX + point.x * scale, y: centerY + point.y * scale });
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;

  for (const [a, b] of poseConnections) {
    if (!ids.has(a) || !ids.has(b) || !landmarks[a] || !landmarks[b]) continue;
    const pa = project(landmarks[a]);
    const pb = project(landmarks[b]);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
  for (const id of ids) {
    if (!landmarks[id]) continue;
    const point = project(landmarks[id]);
    ctx.beginPath();
    ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawAngleHotspots(ctx, leftLandmarks, rightLandmarks, rows, width, height, regions) {
  if (!leftLandmarks?.length || !rightLandmarks?.length || !rows?.length) return;
  const specs = new Map(activeAngleSpecs(regions).map((spec) => [spec.id, spec]));
  const centerX = width / 2;
  const centerY = height * 0.52;
  const scale = Math.min(width, height) * 0.22;
  const project = (point) => ({ x: centerX + point.x * scale, y: centerY + point.y * scale });
  const hotRows = [...rows].sort((a, b) => (b.diff || 0) - (a.diff || 0)).slice(0, 4);
  ctx.save();
  ctx.strokeStyle = "rgba(255, 107, 107, 0.9)";
  ctx.fillStyle = "rgba(255, 107, 107, 0.22)";
  ctx.lineWidth = 2;
  for (const row of hotRows) {
    const spec = specs.get(row.id);
    if (!spec) continue;
    const jointId = spec.points[1];
    const left = leftLandmarks[jointId];
    const right = rightLandmarks[jointId];
    if (!left || !right) continue;
    for (const point of [project(left), project(right)]) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawSkeletonDifferences(ctx, leftLandmarks, rightLandmarks, width, height, regions) {
  if (!leftLandmarks?.length || !rightLandmarks?.length) return;
  const ids = overlayLandmarkIds(regions);
  const centerX = width / 2;
  const centerY = height * 0.52;
  const scale = Math.min(width, height) * 0.22;
  const project = (point) => ({ x: centerX + point.x * scale, y: centerY + point.y * scale });
  ctx.save();
  ctx.strokeStyle = "rgba(255, 107, 107, 0.72)";
  ctx.lineWidth = 1.4;
  ctx.setLineDash([4, 5]);
  for (const id of ids) {
    if (!leftLandmarks[id] || !rightLandmarks[id]) continue;
    const left = project(leftLandmarks[id]);
    const right = project(rightLandmarks[id]);
    const gap = Math.hypot(left.x - right.x, left.y - right.y);
    if (gap < 4) continue;
    ctx.beginPath();
    ctx.moveTo(left.x, left.y);
    ctx.lineTo(right.x, right.y);
    ctx.stroke();
  }
  ctx.restore();
}

function App() {
  const [captureEngine, setCaptureEngine] = useState(() => loadCaptureEngine());
  const [motionCapLeftScan, setMotionCapLeftScan] = useState(null);
  const [motionCapRightScan, setMotionCapRightScan] = useState(null);
  const [motionCapStatus, setMotionCapStatus] = useState("");
  const [mediaPipeSettings, setMediaPipeSettings] = useState(() => loadMediaPipeSettings());
  const [comparisonModel, setComparisonModel] = useState(() => {
    const savedModel = localStorage.getItem(comparisonModelKey);
    return comparisonModels[savedModel] ? savedModel : "angles";
  });
  const [hybridMethods, setHybridMethods] = useState(() => loadHybridMethodSettings());
  const { landmarker, scanLandmarker, error } = usePoseLandmarker(mediaPipeSettings);
  const leftVideoRef = useRef(null);
  const rightVideoRef = useRef(null);
  const autonomousVideoRef = useRef(null);
  const analysisRafRef = useRef(0);
  const mediaPipeTimestampRef = useRef(0);
  const [leftPose, setLeftPose] = useState(null);
  const [rightPose, setRightPose] = useState(null);
  const [leftFile, setLeftFile] = useState(null);
  const [rightFile, setRightFile] = useState(null);
  const [leftScan, setLeftScan] = useState(null);
  const [rightScan, setRightScan] = useState(null);
  const [leftAudio, setLeftAudio] = useState(null);
  const [rightAudio, setRightAudio] = useState(null);
  const [leftAnalysisRange, setLeftAnalysisRange] = useState({ start: 0, end: 0 });
  const [sync, setSync] = useState({ ready: false, offsetSeconds: 0, confidence: 0 });
  const [manualSync, setManualSync] = useState({ offsetSeconds: 0, leftStart: 0, rightStart: 0, useManualStarts: false });
  const [soundMode, setSoundMode] = useState("left");
  const [audioStatus, setAudioStatus] = useState("");
  const [runState, setRunState] = useState({ status: "idle", progress: 0, result: null, message: "" });
  const [expectedScore, setExpectedScore] = useState("");
  const [labHistory, setLabHistory] = useState(() => loadLabHistory());
  const [autonomousReferenceFile, setAutonomousReferenceFile] = useState(null);
  const [autonomousCandidateFiles, setAutonomousCandidateFiles] = useState([]);
  const [autonomousSelectedModels, setAutonomousSelectedModels] = useState(["joint-areas", "zone-grid", "trajectory-drawing", "activity", "2026-07-13"]);
  const [autonomousRunState, setAutonomousRunState] = useState({
    status: "idle",
    progress: 0,
    message: "",
    results: [],
    bestTrials: [],
    executionOrder: []
  });
  const [sequentialReferenceFile, setSequentialReferenceFile] = useState(null);
  const [sequentialCandidateFiles, setSequentialCandidateFiles] = useState([]);
  const [sequentialModelSequence, setSequentialModelSequence] = useState(["activity", "joint-areas"]);
  const [sequentialRunState, setSequentialRunState] = useState({
    status: "idle",
    progress: 0,
    message: "",
    results: [],
    executionOrder: []
  });
  const activeSpecs = useMemo(() => activeAngleSpecs(mediaPipeSettings.regions), [mediaPipeSettings.regions]);
  const activeSync = useMemo(() => effectiveSync(sync, manualSync), [sync, manualSync]);
  const motionCapComparison = useMemo(
    () => compareMotionCapScans(motionCapLeftScan, motionCapRightScan),
    [motionCapLeftScan, motionCapRightScan]
  );
  const liveComparison = useMemo(() => {
    if (comparisonModel === "openai-expert") return pendingOpenAiComparison();
    if (comparisonModel === "overlay") return compareOverlayFrames(leftPose, rightPose, mediaPipeSettings.regions);
    if (comparisonModel === "2026-07-06") {
      return compareSkeletons_2026_07_06(
        leftPose ? { frames: [{ ...leftPose, timestamp: Math.round((leftPose.time || 0) * 1000) }] } : null,
        rightPose ? { frames: [{ ...rightPose, timestamp: Math.round((rightPose.time || 0) * 1000) }] } : null
      );
    }
    if (comparisonModel === "2026-07-12") {
      return compareSkeletons_2026_07_12(
        leftPose ? { frames: [{ ...leftPose, timestamp: Math.round((leftPose.time || 0) * 1000) }] } : null,
        rightPose ? { frames: [{ ...rightPose, timestamp: Math.round((rightPose.time || 0) * 1000) }] } : null
      );
    }
    if (comparisonModel === "2026-07-13") {
      return compareSkeletons_2026_07_13(
        leftPose ? { frames: [{ ...leftPose, timestamp: Math.round((leftPose.time || 0) * 1000) }] } : null,
        rightPose ? { frames: [{ ...rightPose, timestamp: Math.round((rightPose.time || 0) * 1000) }] } : null
      );
    }
    if (comparisonModel === "zones-drawing") return compareZonesDrawingScans(leftScan, rightScan, activeSync, hybridMethods);
    if (comparisonModel === "joint-areas") return compareJointAreasScans(leftScan, rightScan, activeSync);
    if (comparisonModel === "trajectory-drawing") return compareTrajectoryDrawingScans(leftScan, rightScan, activeSync);
    if (comparisonModel === "zone-grid") return compareZoneGridScans(leftScan, rightScan, activeSync);
    if (comparisonModel === "activity") return compareActivityScans(leftScan, rightScan, activeSync);
    return comparePoseFrames(leftPose, rightPose, mediaPipeSettings.regions);
  }, [activeSync, comparisonModel, hybridMethods, leftPose, leftScan, mediaPipeSettings.regions, rightPose, rightScan]);
  const comparison = runState.result || (leftScan?.frames?.length && rightScan?.frames?.length ? pendingFullRunComparison() : liveComparison);
  const confidence = Math.round(
    leftScan?.frames?.length || rightScan?.frames?.length
      ? ((leftScan?.averageConfidence || 0) + (rightScan?.averageConfidence || 0)) * 50
      : ((leftPose?.confidence || 0) + (rightPose?.confidence || 0)) * 50
  );

  useEffect(() => () => cancelAnimationFrame(analysisRafRef.current), []);

  useEffect(() => {
    saveLabHistoryToStorage(labHistory);
  }, [labHistory]);

  useEffect(() => {
    try {
      localStorage.setItem(mediaPipeSettingsKey, JSON.stringify(mediaPipeSettings));
    } catch (err) {
      console.warn("Не удалось сохранить настройки MediaPipe.", err);
    }
  }, [mediaPipeSettings]);

  useEffect(() => {
    try {
      localStorage.setItem(captureEngineKey, captureEngine);
    } catch (err) {
      console.warn("Не удалось сохранить движок сканирования.", err);
    }
  }, [captureEngine]);

  const importMotionCapCsv = useCallback((side, file) => {
    if (!file) return;
    setMotionCapStatus(`Читаю ${file.name}...`);
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const scan = parseMotionCapCsv(String(reader.result || ""), file.name);
        if (side === "left") setMotionCapLeftScan(scan);
        if (side === "right") setMotionCapRightScan(scan);
        setMotionCapStatus(`${file.name}: импортировано ${scan.frameCount} кадров и ${scan.keypointCount} 3D-точек.`);
      } catch (err) {
        setMotionCapStatus(`Не удалось импортировать ${file.name}: ${err instanceof Error ? err.message : String(err)}`);
      }
    };
    reader.onerror = () => {
      setMotionCapStatus(`Не удалось прочитать ${file.name}.`);
    };
    reader.readAsText(file);
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(comparisonModelKey, comparisonModel);
    } catch (err) {
      console.warn("Не удалось сохранить выбранную модель сравнения.", err);
    }
  }, [comparisonModel]);

  useEffect(() => {
    try {
      localStorage.setItem(hybridMethodSettingsKey, JSON.stringify(hybridMethods));
    } catch (err) {
      console.warn("Не удалось сохранить методы гибридной модели.", err);
    }
  }, [hybridMethods]);

  const nextMediaPipeTimestamp = useCallback(() => {
    const now = performance.now();
    const next = Math.max(now, mediaPipeTimestampRef.current + 1);
    mediaPipeTimestampRef.current = next;
    return next;
  }, []);

  async function handleAudio(side, file) {
    setSync({ ready: false, offsetSeconds: 0, confidence: 0 });
    setManualSync({ offsetSeconds: 0, leftStart: 0, rightStart: 0, useManualStarts: false });
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
    setManualSync((previous) => ({ ...previous, offsetSeconds: 0 }));
    setRunState({
      status: "ready",
      progress: 0,
      result: null,
      message: result.ready ? "Синхронизация рассчитана. При необходимости подвиньте дорожку вручную и запустите полный прогон." : result.message
    });
  }

  async function playSynchronizedAnalysis(syncOverride = activeSync) {
    if (!leftScan?.frames?.length || !rightScan?.frames?.length) {
      setRunState({
        status: "error",
        progress: 0,
        result: null,
        message: "Сначала отсканируйте скелет в обоих видео."
      });
      return;
    }

    const runSync = syncOverride?.ready ? syncOverride : activeSync;
    const offset = runSync.ready ? runSync.offsetSeconds : 0;
    let leftStart = manualSync.useManualStarts ? manualSync.leftStart : (leftScan.frames?.[0]?.time ?? Math.max(0, -offset));
    let rightStart = manualSync.useManualStarts ? manualSync.rightStart : leftStart + offset;
    if (rightStart < 0) {
      leftStart = Math.min(leftScan.frames?.at(-1)?.time ?? leftStart, leftStart - rightStart);
      rightStart = 0;
    }
    const playbackSync = {
      ...runSync,
      ready: true,
      offsetSeconds: Number((rightStart - leftStart).toFixed(2)),
      message: `Итоговое смещение правого видео: ${(rightStart - leftStart).toFixed(2)} сек.`
    };
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
      await Promise.all([
        leftVideoRef.current?.playAt(leftStart, soundMode === "left" || soundMode === "both"),
        rightVideoRef.current?.playAt(rightStart, soundMode === "right" || soundMode === "both")
      ]);
    } catch (err) {
      const result = await compareByModelAsync(
        comparisonModel,
        leftScan,
        rightScan,
        playbackSync,
        mediaPipeSettings.regions,
        leftAudio,
        mediaPipeSettings,
        hybridMethods
      );
      saveLabExample(result, playbackSync, "auto");
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
        setRunState((previous) => ({
          ...previous,
          progress: 96,
          message: comparisonModel === "openai-expert" ? "Видео прогнано. OpenAI эксперт анализирует метрики и скелеты." : previous.message
        }));
        compareByModelAsync(
          comparisonModel,
          leftScan,
          rightScan,
          playbackSync,
          mediaPipeSettings.regions,
          leftAudio,
          mediaPipeSettings,
          hybridMethods
        )
          .then((result) => {
            saveLabExample(result, playbackSync, "auto");
            setRunState({
              status: "done",
              progress: 100,
              result,
              message: `Прогон завершен: проанализировано ${result.framesCompared || 0} синхронизированных кадров.`
            });
          })
          .catch((err) => {
            const detail = err instanceof Error ? err.message : String(err);
            setRunState({
              status: "error",
              progress: 0,
              result: null,
              message: `OpenAI эксперт не смог завершить анализ: ${detail}`
            });
          });
        return;
      }

      setRunState((previous) => ({ ...previous, progress }));
      analysisRafRef.current = requestAnimationFrame(tick);
    };

    analysisRafRef.current = requestAnimationFrame(tick);
  }

  async function runAllModelsAuto() {
    if (runState.status === "running") return;
    cancelAnimationFrame(analysisRafRef.current);
    setRunState({
      status: "running",
      progress: 3,
      result: null,
      message: "Автолаборатория: сканирую левое эталонное видео."
    });

    try {
      const freshLeftScan = leftScan?.frames?.length ? leftScan : await leftVideoRef.current?.scanSkeleton();
      if (!freshLeftScan?.frames?.length) throw new Error("Не удалось получить скан левого видео.");
      setLeftScan(freshLeftScan);
      setRunState((previous) => ({
        ...previous,
        progress: 35,
        message: "Автолаборатория: левый скелет готов, сканирую правое видео."
      }));

      const freshRightScan = rightScan?.frames?.length ? rightScan : await rightVideoRef.current?.scanSkeleton();
      if (!freshRightScan?.frames?.length) throw new Error("Не удалось получить скан правого видео.");
      setRightScan(freshRightScan);

      let activeRunSync = activeSync;
      if (leftAudio && rightAudio) {
        const automaticSync = estimateAudioSync(leftAudio, rightAudio);
        setSync(automaticSync);
        activeRunSync = effectiveSync(automaticSync, manualSync);
        setAudioStatus(automaticSync.ready ? "Автолаборатория рассчитала синхронизацию по аудио." : automaticSync.message);
      }

      setRunState((previous) => ({
        ...previous,
        progress: 72,
        message: "Автолаборатория: прогоняю все модели сравнения."
      }));

      const autoRunGroupId = crypto.randomUUID();
      const results = [];
      for (const [index, modelId] of runnableComparisonModelIds.entries()) {
        setRunState((previous) => ({
          ...previous,
          progress: Math.min(96, 72 + Math.round((index / Math.max(1, runnableComparisonModelIds.length)) * 24)),
          message:
            modelId === "openai-expert"
              ? "Автолаборатория: OpenAI эксперт анализирует сжатые метрики."
              : `Автолаборатория: рассчитываю модель "${comparisonModels[modelId]?.title || modelId}".`
        }));
        results.push({
          modelId,
          result: await compareByModelAsync(
            modelId,
            freshLeftScan,
            freshRightScan,
            activeRunSync,
            mediaPipeSettings.regions,
            leftAudio,
            mediaPipeSettings,
            hybridMethods
          )
        });
      }

      for (const { modelId, result } of results) {
        saveLabExample(result, activeRunSync, "auto", modelId, freshLeftScan, freshRightScan, autoRunGroupId);
      }

      const primary = results.find((item) => item.modelId === "2026-07-06")?.result || results[0]?.result || null;
      setRunState({
        status: "done",
        progress: 100,
        result: primary,
        message: `Автолаборатория завершена: ${results.length} моделей рассчитаны и сохранены в историю.`
      });
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setRunState({
        status: "error",
        progress: 0,
        result: null,
        message: `Автолаборатория остановилась: ${detail}`
      });
      console.error(err);
    }
  }

  function toggleAutonomousModel(modelId) {
    setAutonomousSelectedModels((items) =>
      items.includes(modelId) ? items.filter((item) => item !== modelId) : [...items, modelId]
    );
  }

  function addSequentialModel(modelId) {
    setSequentialModelSequence((items) => [...items, modelId]);
  }

  function removeSequentialModelAt(indexToRemove) {
    setSequentialModelSequence((items) => items.filter((_, index) => index !== indexToRemove));
  }

  async function scanAutonomousFile(file, progressLabel, onProgress) {
    const video = autonomousVideoRef.current;
    if (!video) throw new Error("Скрытый видео-сканер не готов.");
    const url = await loadFileIntoVideo(video, file);
    try {
      const scan = await scanVideoPose(
        video,
        scanLandmarker,
        (progress) => onProgress?.(`${progressLabel}: скан ${progress}%`),
        null,
        mediaPipeSettings
      );
      let audio = null;
      try {
        audio = await analyzeAudioFile(file);
      } catch {
        audio = null;
      }
      return { scan, audio };
    } finally {
      video.pause();
      video.removeAttribute("src");
      video.load();
      URL.revokeObjectURL(url);
    }
  }

  async function runAutonomousEnsembleLab() {
    if (autonomousRunState.status === "running") return;
    if (!autonomousReferenceFile || !autonomousCandidateFiles.length || !autonomousSelectedModels.length) {
      setAutonomousRunState((previous) => ({
        ...previous,
        status: "error",
        message: "Выберите эталон, хотя бы одно видео ученика и одну модель."
      }));
      return;
    }

    const executionOrder = shuffledItems(autonomousSelectedModels);
    const totalWork = 1 + autonomousCandidateFiles.length * (1 + executionOrder.length);
    let completedWork = 0;
    const updateProgress = (message) => {
      setAutonomousRunState((previous) => ({
        ...previous,
        status: "running",
        progress: Math.min(99, Math.round((completedWork / Math.max(1, totalWork)) * 100)),
        message
      }));
    };

    setAutonomousRunState({
      status: "running",
      progress: 0,
      message: "Сканирую эталон педагога.",
      results: [],
      bestTrials: [],
      executionOrder
    });
    try {
      const reference = await scanAutonomousFile(autonomousReferenceFile, "Эталон", (message) => updateProgress(message));
      completedWork += 1;
      const results = [];

      for (const candidateFile of autonomousCandidateFiles) {
        updateProgress(`Сканирую ${candidateFile.name}.`);
        const candidate = await scanAutonomousFile(candidateFile, candidateFile.name, (message) => updateProgress(message));
        completedWork += 1;

        const runSync =
          reference.audio && candidate.audio
            ? estimateAudioSync(reference.audio, candidate.audio)
            : { ready: false, offsetSeconds: 0, confidence: 0, message: "Аудио-синхронизация недоступна." };
        const modelScores = {};
        const modelDetails = {};

        for (const modelId of executionOrder) {
          updateProgress(`${candidateFile.name}: модель "${comparisonModels[modelId]?.title || modelId}".`);
          try {
            const result = await compareByModelAsync(
              modelId,
              reference.scan,
              candidate.scan,
              runSync,
              mediaPipeSettings.regions,
              reference.audio,
              mediaPipeSettings,
              hybridMethods
            );
            modelScores[modelId] = result?.ready ? result.score : null;
            modelDetails[modelId] = {
              ready: Boolean(result?.ready),
              score: result?.score ?? null,
              bestScore: result?.bestScore ?? null,
              worstScore: result?.worstScore ?? null,
              framesCompared: result?.framesCompared ?? 0,
              verdict: result?.verdict || "",
              suggestions: result?.suggestions || []
            };
          } catch (err) {
            modelScores[modelId] = null;
            modelDetails[modelId] = {
              ready: false,
              error: err instanceof Error ? err.message : String(err)
            };
          }
          completedWork += 1;
        }

        const caseInfo = inferAutonomousCase(candidateFile.name);
        const ensembleScore = autonomousEnsembleScore(modelScores);
        const item = {
          id: crypto.randomUUID(),
          createdAt: new Date().toISOString(),
          referenceFileName: autonomousReferenceFile.name,
          fileName: candidateFile.name,
          caseType: caseInfo.type,
          caseLabel: caseInfo.label,
          target: caseInfo.target,
          ensembleScore,
          recommendation: autonomousRecommendation(caseInfo.type, ensembleScore, modelScores),
          modelScores,
          modelDetails,
          sync: runSync,
          mediaPipeSettings,
          appVersion,
          selectedModels: executionOrder
        };
        results.push(item);
        const bestTrials = buildAutonomousTrials(results, autonomousSelectedModels);
        setAutonomousRunState({
          status: "running",
          progress: Math.min(99, Math.round((completedWork / Math.max(1, totalWork)) * 100)),
          message: `${candidateFile.name}: готово.`,
          results: [...results],
          bestTrials,
          executionOrder
        });
        await yieldToBrowser();
      }

      const bestTrials = buildAutonomousTrials(results, autonomousSelectedModels);
      setAutonomousRunState({
        status: "done",
        progress: 100,
        message: `Автономный анализ завершен: ${results.length} видео, ${executionOrder.length} моделей. Лучший вариант: ${bestTrials[0]?.modelLabel || "не найден"}.`,
        results,
        bestTrials,
        executionOrder
      });
    } catch (err) {
      setAutonomousRunState((previous) => ({
        ...previous,
        status: "error",
        progress: 0,
        message: `Автономная лаборатория остановилась: ${err instanceof Error ? err.message : String(err)}`
      }));
    }
  }

  async function runSequentialGateLab() {
    if (sequentialRunState.status === "running") return;
    if (!sequentialReferenceFile || !sequentialCandidateFiles.length || !sequentialModelSequence.length) {
      setSequentialRunState((previous) => ({
        ...previous,
        status: "error",
        message: "Выберите эталон, видео учеников и хотя бы один гейт."
      }));
      return;
    }

    const executionOrder = [...sequentialModelSequence];
    const totalWork = 1 + sequentialCandidateFiles.length * (1 + executionOrder.length);
    let completedWork = 0;
    const updateProgress = (message) => {
      setSequentialRunState((previous) => ({
        ...previous,
        status: "running",
        progress: Math.min(99, Math.round((completedWork / Math.max(1, totalWork)) * 100)),
        message
      }));
    };

    setSequentialRunState({
      status: "running",
      progress: 0,
      message: "Последовательный анализ: сканирую эталон педагога.",
      results: [],
      executionOrder
    });

    try {
      const reference = await scanAutonomousFile(sequentialReferenceFile, "Эталон", (message) => updateProgress(message));
      completedWork += 1;
      const results = [];

      for (const candidateFile of sequentialCandidateFiles) {
        updateProgress(`Сканирую ${candidateFile.name}.`);
        const candidate = await scanAutonomousFile(candidateFile, candidateFile.name, (message) => updateProgress(message));
        completedWork += 1;

        const runSync =
          reference.audio && candidate.audio
            ? estimateAudioSync(reference.audio, candidate.audio)
            : { ready: false, offsetSeconds: 0, confidence: 0, message: "Аудио-синхронизация недоступна." };
        const steps = [];
        let passed = true;
        let stopReason = "";
        let stoppedAtStep = null;

        for (const [index, modelId] of executionOrder.entries()) {
          updateProgress(`${candidateFile.name}: гейт ${index + 1} "${comparisonModels[modelId]?.title || modelId}".`);
          try {
            const result = await compareByModelAsync(
              modelId,
              reference.scan,
              candidate.scan,
              runSync,
              mediaPipeSettings.regions,
              reference.audio,
              mediaPipeSettings,
              hybridMethods
            );
            const decision = sequentialGateDecision(modelId, result);
            steps.push({
              index: index + 1,
              modelId,
              modelTitle: comparisonModels[modelId]?.title || modelId,
              modelVersion: comparisonModels[modelId]?.versionLabel || "",
              score: result?.ready ? result.score : null,
              finalScore: result?.finalScore ?? result?.score ?? null,
              threshold: decision.threshold,
              passed: decision.passed,
              reason: decision.reason,
              diagnostics: result?.diagnostics || {},
              bodyParts: result?.bodyParts || null,
              framesCompared: result?.framesCompared || 0,
              verdict: result?.verdict || ""
            });
            completedWork += 1;
            if (!decision.passed) {
              passed = false;
              stopReason = decision.reason;
              stoppedAtStep = index + 1;
              completedWork += executionOrder.length - index - 1;
              break;
            }
          } catch (err) {
            passed = false;
            stopReason = `Стоп-гейт: модель не рассчиталась (${err instanceof Error ? err.message : String(err)}).`;
            stoppedAtStep = index + 1;
            steps.push({
              index: index + 1,
              modelId,
              modelTitle: comparisonModels[modelId]?.title || modelId,
              modelVersion: comparisonModels[modelId]?.versionLabel || "",
              score: null,
              finalScore: null,
              threshold: sequentialGateThresholds[modelId] ?? 62,
              passed: false,
              reason: stopReason,
              error: err instanceof Error ? err.message : String(err)
            });
            completedWork += executionOrder.length - index;
            break;
          }
          await yieldToBrowser();
        }

        const item = {
          id: crypto.randomUUID(),
          createdAt: new Date().toISOString(),
          referenceFileName: sequentialReferenceFile.name,
          fileName: candidateFile.name,
          passed,
          stoppedAtStep,
          stopReason,
          finalScore: sequentialFinalScore(steps),
          steps,
          sync: runSync,
          mediaPipeSettings,
          appVersion,
          executionOrder
        };
        results.push(item);
        setSequentialRunState({
          status: "running",
          progress: Math.min(99, Math.round((completedWork / Math.max(1, totalWork)) * 100)),
          message: passed ? `${candidateFile.name}: прошел всю последовательность.` : `${candidateFile.name}: ${stopReason}`,
          results: [...results],
          executionOrder
        });
        await yieldToBrowser();
      }

      setSequentialRunState({
        status: "done",
        progress: 100,
        message: `Последовательный анализ завершен: ${results.filter((item) => item.passed).length}/${results.length} видео прошли все гейты.`,
        results,
        executionOrder
      });
    } catch (err) {
      setSequentialRunState((previous) => ({
        ...previous,
        status: "error",
        progress: 0,
        message: `Последовательный анализ остановился: ${err instanceof Error ? err.message : String(err)}`
      }));
    }
  }

  function saveLabExample(
    resultOverride = null,
    syncOverride = sync,
    saveMode = "manual",
    modelOverride = comparisonModel,
    leftScanOverride = leftScan,
    rightScanOverride = rightScan,
    runGroupId = null
  ) {
    const result = resultOverride || runState.result;
    if (!result?.ready) return;
    const savedModel = modelOverride === "all-auto" ? "angles" : modelOverride;
    const savedModelDetails = comparisonModels[savedModel] || comparisonModels.angles;
    const activeLeftScan = leftScanOverride || leftScan;
    const activeRightScan = rightScanOverride || rightScan;
    const leftVideoProfile = fileProfile(leftFile, activeLeftScan, leftAudio);
    const rightVideoProfile = fileProfile(rightFile, activeRightScan, rightAudio);
    const nextItem = {
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      appVersion: appVersion.version,
      appVersionLabel: appVersion.versionLabel,
      appBuild: appVersion.build,
      appDetails: appVersion,
      leftFileName: leftFile?.name || "",
      rightFileName: rightFile?.name || "",
      saveMode,
      runGroupId,
      expectedScore: expectedScore === "" ? null : Number(expectedScore),
      score: result.score,
      videos: {
        left: leftVideoProfile,
        right: rightVideoProfile,
        sameFileCandidate: sameFileCandidate(leftVideoProfile, rightVideoProfile)
      },
      comparisonModel: savedModel,
      comparisonModelVersion: savedModelDetails.version,
      comparisonModelVersionLabel: savedModelDetails.versionLabel,
      comparisonAlgorithmBuild: savedModelDetails.algorithmBuild,
      comparisonModelDetails: savedModelDetails,
      hybridMethods: savedModel === "zones-drawing" ? normalizeHybridMethodSettings(hybridMethods) : null,
      modelRunMetadata: buildModelRunMetadata({
        app: appVersion,
        modelId: savedModel,
        modelDetails: savedModelDetails,
        result,
        saveMode,
        runGroupId,
        sync: syncOverride,
        mediaPipeSettings,
        activeSpecs,
        hybridMethods
      }),
      mediaPipeSettings: {
        ...mediaPipeSettings,
        activeAngles: activeSpecs.map((spec) => ({
          id: spec.id,
          title: spec.title,
          region: spec.region,
          points: spec.points
        }))
      },
      sync: syncOverride,
      leftAnalysisRange,
      metrics: {
        framesCompared: result.framesCompared || 0,
        bestScore: result.bestScore ?? null,
        worstScore: result.worstScore ?? null,
        durationCompared: result.durationCompared ?? null,
        worstMoment: result.worstMoment ?? null,
        openAiReady: result.diagnostics?.openAiReady ?? null,
        openAiModel: result.diagnostics?.openAiModel ?? null,
        openAiConfidence: result.diagnostics?.openAiConfidence ?? null,
        openAiTrackingQualityScore: result.diagnostics?.openAiTrackingQualityScore ?? null,
        openAiEvidenceGateApplied: result.diagnostics?.openAiEvidenceGateApplied ?? null,
        openAiEvidenceGateReason: result.diagnostics?.openAiEvidenceGateReason ?? null,
        openAiError: result.diagnostics?.openAiError ?? null,
        openAiReasoning: result.diagnostics?.openAiReasoning ?? null,
        openAiVerdict: result.diagnostics?.openAiReady ? result.verdict : null,
        openAiSuggestions: result.diagnostics?.openAiReady ? result.suggestions || [] : []
      },
      angleRows: sampleEvenly(result.rows || [], maxStoredAngleRows),
      skeletons: {
        left: serializeScanSkeleton(activeLeftScan, mediaPipeSettings),
        right: serializeScanSkeleton(activeRightScan, mediaPipeSettings),
        synchronizedPairs: serializeSkeletonPairs(activeLeftScan, activeRightScan, syncOverride, mediaPipeSettings)
      },
      suggestions: result.suggestions,
      verdict: result.verdict
    };
    setLabHistory((items) => compactLabHistory([nextItem, ...items]));
  }

  function clearLabHistory() {
    setLabHistory([]);
  }

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <p className="eyebrow">{captureEngines[captureEngine].title}</p>
          <div className="title-row">
            <h1>Сравнение скелетов в двух видео</h1>
            <span className="app-version-badge">{appVersion.name} {appVersion.versionLabel}</span>
          </div>
          <p className="subtitle">Левое видео - эталон. Правое видео оценивается и подгоняется относительно эталона.</p>
        </div>
        <div className="model-state">
          <Sparkles size={18} />
          {captureEngine === "motioncap"
            ? "MotionCap импорт CSV"
            : landmarker && scanLandmarker
              ? "MediaPipe готов"
              : "Загрузка модели..."}
        </div>
      </header>

      <TrackingEnginePanel
        engine={captureEngine}
        onChange={(engine) => {
          setCaptureEngine(engine);
          setRunState({ status: "idle", progress: 0, result: null, message: "" });
        }}
      />

      {captureEngine === "motioncap" ? (
        <MotionCapLab
          leftScan={motionCapLeftScan}
          rightScan={motionCapRightScan}
          status={motionCapStatus}
          onImport={importMotionCapCsv}
          comparison={motionCapComparison}
        />
      ) : (
        <>
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

      <ComparisonModelPanel
        model={comparisonModel}
        onChange={(model) => {
          setComparisonModel(model);
          setRunState({
            status: "idle",
            progress: 0,
            result: null,
            message: `Выбрана модель "${comparisonModels[model].title}". Запустите полный анализ для расчета по этой логике.`
          });
        }}
      />

      <AutonomousEnsembleLab
        referenceFile={autonomousReferenceFile}
        candidateFiles={autonomousCandidateFiles}
        selectedModels={autonomousSelectedModels}
        runState={autonomousRunState}
        onReferenceChange={(file) => {
          setAutonomousReferenceFile(file);
          setAutonomousRunState({ status: "idle", progress: 0, message: "", results: [], bestTrials: [], executionOrder: [] });
        }}
        onCandidatesChange={(files) => {
          setAutonomousCandidateFiles(files);
          setAutonomousRunState({ status: "idle", progress: 0, message: "", results: [], bestTrials: [], executionOrder: [] });
        }}
        onToggleModel={toggleAutonomousModel}
        onRun={runAutonomousEnsembleLab}
        onExport={() =>
          downloadJson("dmpa-autonomous-ensemble.json", {
            appVersion,
            referenceFileName: autonomousReferenceFile?.name || "",
            candidateFileNames: autonomousCandidateFiles.map((file) => file.name),
            selectedModels: autonomousSelectedModels,
            executionOrder: autonomousRunState.executionOrder,
            bestTrials: autonomousRunState.bestTrials,
            results: autonomousRunState.results
          })
        }
      />

      <SequentialGateLab
        referenceFile={sequentialReferenceFile}
        candidateFiles={sequentialCandidateFiles}
        modelSequence={sequentialModelSequence}
        runState={sequentialRunState}
        onReferenceChange={(file) => {
          setSequentialReferenceFile(file);
          setSequentialRunState({ status: "idle", progress: 0, message: "", results: [], executionOrder: [] });
        }}
        onCandidatesChange={(files) => {
          setSequentialCandidateFiles(files);
          setSequentialRunState({ status: "idle", progress: 0, message: "", results: [], executionOrder: [] });
        }}
        onAddModel={addSequentialModel}
        onRemoveModelAt={removeSequentialModelAt}
        onClearSequence={() => setSequentialModelSequence([])}
        onRun={runSequentialGateLab}
        onExport={() =>
          downloadJson("dmpa-sequential-gate-lab.json", {
            appVersion,
            referenceFileName: sequentialReferenceFile?.name || "",
            candidateFileNames: sequentialCandidateFiles.map((file) => file.name),
            modelSequence: sequentialModelSequence,
            gateThresholds: sequentialGateThresholds,
            executionOrder: sequentialRunState.executionOrder,
            results: sequentialRunState.results
          })
        }
      />
      <video ref={autonomousVideoRef} className="hidden-batch-video" muted playsInline />

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
          liveDetectionPaused={runState.status === "running"}
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
          liveDetectionPaused={runState.status === "running"}
          mediaPipeSettings={mediaPipeSettings}
        />
      </section>

      <WaveformTimeline
        leftAudio={leftAudio}
        rightAudio={rightAudio}
        sync={activeSync}
        manualSync={manualSync}
        onManualSyncChange={setManualSync}
        soundMode={soundMode}
        onSoundModeChange={setSoundMode}
        leftDuration={leftScan?.duration || leftAudio?.duration || 0}
        rightDuration={rightScan?.duration || rightAudio?.duration || 0}
      />

      <div className="sync-actions">
        <button type="button" onClick={synchronizeAudio} disabled={!leftAudio || !rightAudio}>
          <Waves size={18} />
          Синхронизировать по аудио
        </button>
        {comparisonModel === "all-auto" ? (
          <button type="button" onClick={runAllModelsAuto} disabled={!scanLandmarker || runState.status === "running"}>
            <Play size={18} />
            {runState.status === "running" ? `Автолаборатория ${runState.progress}%` : "Начать"}
          </button>
        ) : (
          <button
            type="button"
            onClick={() => playSynchronizedAnalysis()}
            disabled={!leftScan?.frames?.length || !rightScan?.frames?.length || runState.status === "running"}
          >
            <Play size={18} />
            {runState.status === "running" ? `Анализ ${runState.progress}%` : "Повторить анализ всего видео"}
          </button>
        )}
        <span>{audioStatus || "Аудио используется для выравнивания одинаковых музыкальных фрагментов."}</span>
      </div>

      <div className={`run-status ${runState.status}`}>
        <div>
          <strong>
            {runState.status === "done"
              ? "Итоговый прогон завершен"
              : runState.status === "running"
                ? comparisonModel === "all-auto"
                  ? "Идет автолаборатория"
                  : "Идет синхронное воспроизведение"
                : runState.status === "ready"
                  ? "Готово к полному прогону"
                  : "Полный прогон еще не запускался"}
          </strong>
          <span>
            {runState.message ||
              (comparisonModel === "all-auto"
                ? "Загрузите оба видео и нажмите «Начать»: приложение само отсканирует скелеты и сохранит результаты всех моделей."
                : "После синхронизации запустите видео один раз, чтобы получить статистику всего танца.")}
          </span>
        </div>
        <div className="run-progress" aria-label="Прогресс анализа">
          <i style={{ width: `${runState.progress}%` }} />
        </div>
      </div>

      <SkeletonOverlayViewer
        leftScan={leftScan}
        rightScan={rightScan}
        sync={activeSync}
        regions={mediaPipeSettings.regions}
        enabled={comparisonModel === "overlay"}
      />

      <KeyPoseViewer comparison={comparison} enabled={comparisonModel === "poses"} regions={mediaPipeSettings.regions} />

      <AngleComparisonViewer
        leftScan={leftScan}
        rightScan={rightScan}
        sync={activeSync}
        comparison={comparison}
        regions={mediaPipeSettings.regions}
        enabled={comparisonModel === "angles" && Boolean(runState.result?.ready)}
      />

      <ElasticDanceViewer
        leftScan={leftScan}
        rightScan={rightScan}
        sync={activeSync}
        comparison={comparison}
        regions={mediaPipeSettings.regions}
        enabled={(comparisonModel === "2026-07-12" || comparisonModel === "2026-07-13") && Boolean(runState.result?.ready)}
        modelId={comparisonModel === "2026-07-13" ? "2026-07-13" : "2026-07-12"}
      />

      <AreasDrawingViewer
        leftScan={leftScan}
        rightScan={rightScan}
        sync={activeSync}
        comparison={comparison}
        enabled={(comparisonModel === "joint-areas" || comparisonModel === "trajectory-drawing") && Boolean(runState.result?.ready)}
        mode={comparisonModel === "trajectory-drawing" ? "drawing" : "areas"}
      />

      <ZoneGridViewer
        leftScan={leftScan}
        rightScan={rightScan}
        sync={activeSync}
        comparison={comparison}
        enabled={comparisonModel === "zone-grid" && Boolean(runState.result?.ready)}
      />

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
            <MetricCard label="Смещение аудио" value={activeSync.ready ? `${activeSync.offsetSeconds.toFixed(2)} сек` : "нет"} />
            <MetricCard label="Лучший момент" value={comparison.bestScore != null ? `${comparison.bestScore}%` : "-"} />
            <MetricCard label="Худший момент" value={comparison.worstScore != null ? `${comparison.worstScore}%` : "-"} />
            <MetricCard label="Длительность анализа" value={comparison.durationCompared ? `${comparison.durationCompared} сек` : "-"} />
            {comparisonModel === "openai-expert" && (
              <>
                <MetricCard label="OpenAI API" value={comparison.diagnostics?.openAiReady ? "ответ получен" : "нет ответа"} />
                <MetricCard label="OpenAI модель" value={comparison.diagnostics?.openAiModel || "-"} />
                <MetricCard
                  label="Уверенность OpenAI"
                  value={comparison.diagnostics?.openAiConfidence != null ? `${comparison.diagnostics.openAiConfidence}%` : "-"}
                />
                <MetricCard
                  label="Скан по OpenAI"
                  value={
                    comparison.diagnostics?.openAiTrackingQualityScore != null
                      ? `${comparison.diagnostics.openAiTrackingQualityScore}%`
                      : "-"
                  }
                />
                <MetricCard
                  label="Evidence gate"
                  value={comparison.diagnostics?.openAiEvidenceGateApplied ? "сработал" : "не сработал"}
                />
                {comparison.diagnostics?.openAiError && <MetricCard label="Ошибка OpenAI" value={comparison.diagnostics.openAiError} />}
              </>
            )}
            {comparison.diagnostics?.trackingOutliersSkipped > 0 && (
              <MetricCard label="Плохих кадров пропущено" value={comparison.diagnostics.trackingOutliersSkipped} />
            )}
            {comparisonModel === "activity" && comparison.diagnostics?.activityReference && comparison.diagnostics?.activityUser && (
              <>
                <MetricCard label="Активность эталона" value={`${comparison.diagnostics.activityReference.total.activity}%`} />
                <MetricCard label="Активность правого видео" value={`${comparison.diagnostics.activityUser.total.activity}%`} />
                <MetricCard
                  label="Совпадение активности"
                  value={comparison.diagnostics.activityMatch != null ? `${comparison.diagnostics.activityMatch}%` : "-"}
                />
                <MetricCard
                  label="Активность по фразе"
                  value={comparison.diagnostics.phraseMatch != null ? `${comparison.diagnostics.phraseMatch}%` : "-"}
                />
              </>
            )}
            {comparison.bodyParts && (
              <>
                <MetricCard label="Руки" value={`${comparison.bodyParts.arms ?? 0}%`} />
                <MetricCard label="Ноги" value={`${comparison.bodyParts.legs ?? 0}%`} />
                <MetricCard label="Корпус" value={`${comparison.bodyParts.torso ?? 0}%`} />
                <MetricCard label="Голова" value={`${comparison.bodyParts.head ?? 0}%`} />
                <MetricCard label="Ритм" value={`${comparison.bodyParts.rhythm ?? comparison.timingScore ?? 0}%`} />
              </>
            )}
          </div>

          <div className="verdict">
            <h2>Анализ правого видео относительно эталона</h2>
            <p>{comparison.verdict}</p>
            {comparison.worstMoment && (
              <p className="worst-moment">
                Самый слабый момент: эталон {formatSeconds(comparison.worstMoment.leftTime)}, правое видео{" "}
                {formatSeconds(comparison.worstMoment.rightTime)}, схожесть {comparison.worstMoment.score}%.
              </p>
            )}
          </div>

          {comparisonModel === "openai-expert" && (
            <div className={`ai-commentary ${comparison.diagnostics?.openAiReady ? "ready" : "error"}`}>
              <div>
                <p className="eyebrow">Ответ OpenAI эксперта</p>
                <h3>{comparison.diagnostics?.openAiReady ? "AI-комментарий по исполнению" : "OpenAI пока не ответил"}</h3>
              </div>
              {comparison.diagnostics?.openAiReady ? (
                <>
                  <p>{comparison.verdict}</p>
                  {comparison.diagnostics?.openAiReasoning && <p>{comparison.diagnostics.openAiReasoning}</p>}
                  {comparison.diagnostics?.openAiEvidenceGateApplied && comparison.diagnostics?.openAiEvidenceGateReason && (
                    <p>{comparison.diagnostics.openAiEvidenceGateReason}</p>
                  )}
                  {comparison.suggestions?.length > 0 && (
                    <div>
                      <strong>Что исправить:</strong>
                      {comparison.suggestions.slice(0, 5).map((item) => (
                        <p key={item}>{item}</p>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <p>{comparison.diagnostics?.openAiError || "Запустите полный анализ и проверьте OPENAI_API_KEY в Cloudflare."}</p>
              )}
            </div>
          )}

          <div className="angle-table">
            <div className="table-head">
              <span>Метрика</span>
              <span>Отклонение правого от эталона</span>
            </div>
            {(comparison.rows.length ? comparison.rows : activeSpecs).map((row) => (
              <div className="table-row" key={row.id}>
                <span>{row.title}</span>
                <span className={row.diff > 18 ? "bad" : row.diff > 9 ? "warn" : "good"}>
                  {formatMetricValue(row.diff, row.unit)}
                </span>
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
        onSave={() => saveLabExample(null, activeSync, "manual")}
        onExport={() => downloadJson("dmpa-lab-history.json", labHistory)}
        onClear={clearLabHistory}
        canSave={Boolean(runState.result?.ready)}
      />
        </>
      )}
    </main>
  );
}

function averageVisibility(landmarks) {
  const tracked = Object.keys(landmarkNames).map(Number);
  const visible = tracked.map((id) => landmarks[id]?.visibility ?? 0).filter((value) => value > 0);
  if (!visible.length) return 0;
  return visible.reduce((sum, value) => sum + value, 0) / visible.length;
}

class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error(error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <main className="app">
          <div className="global-error">
            Приложение поймало ошибку после действия. Обновите страницу и попробуйте снизить FPS сканирования до 2-3 кадров/сек.
            <br />
            Деталь: {this.state.error.message || String(this.state.error)}
          </div>
        </main>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")).render(
  <AppErrorBoundary>
    <App />
  </AppErrorBoundary>
);
