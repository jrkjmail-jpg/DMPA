const aliases = {
  head: ["head", "nose", 0],
  leftShoulder: ["leftShoulder", "left_shoulder", 11],
  rightShoulder: ["rightShoulder", "right_shoulder", 12],
  leftElbow: ["leftElbow", "left_elbow", 13],
  rightElbow: ["rightElbow", "right_elbow", 14],
  leftWrist: ["leftWrist", "left_wrist", 15],
  rightWrist: ["rightWrist", "right_wrist", 16],
  leftHip: ["leftHip", "left_hip", 23],
  rightHip: ["rightHip", "right_hip", 24],
  leftKnee: ["leftKnee", "left_knee", 25],
  rightKnee: ["rightKnee", "right_knee", 26],
  leftAnkle: ["leftAnkle", "left_ankle", 27],
  rightAnkle: ["rightAnkle", "right_ankle", 28]
};

const joints = Object.keys(aliases);
const compareJoints = ["head", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist", "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"];
const motionJoints = ["leftElbow", "rightElbow", "leftWrist", "rightWrist", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"];
const bones = [
  ["leftShoulder", "leftElbow", "arms"],
  ["leftElbow", "leftWrist", "arms"],
  ["rightShoulder", "rightElbow", "arms"],
  ["rightElbow", "rightWrist", "arms"],
  ["leftHip", "leftKnee", "legs"],
  ["leftKnee", "leftAnkle", "legs"],
  ["rightHip", "rightKnee", "legs"],
  ["rightKnee", "rightAnkle", "legs"],
  ["leftShoulder", "rightShoulder", "torso"],
  ["leftHip", "rightHip", "torso"]
];
const angleSpecs = [
  ["leftElbow", "leftShoulder", "leftElbow", "leftWrist", "arms"],
  ["rightElbow", "rightShoulder", "rightElbow", "rightWrist", "arms"],
  ["leftKnee", "leftHip", "leftKnee", "leftAnkle", "legs"],
  ["rightKnee", "rightHip", "rightKnee", "rightAnkle", "legs"],
  ["leftHip", "leftShoulder", "leftHip", "leftKnee", "torso"],
  ["rightHip", "rightShoulder", "rightHip", "rightKnee", "torso"]
];

export function compareSkeletons_2026_07_12(referenceSkeleton, userSkeleton, options = {}) {
  const maxFrames = Math.max(40, Math.min(260, options.maxFrames || 180));
  const referenceFrames = prepareSequence(referenceSkeleton, maxFrames);
  const userFrames = prepareSequence(userSkeleton, maxFrames);

  if (referenceFrames.length < 2 || userFrames.length < 2) {
    return emptyResult("Недостаточно кадров для эластичного сравнения.");
  }

  const alignment = dtwAlign(referenceFrames, userFrames);
  const poseScore = clampScore(100 * Math.exp(-alignment.averageCost * 2.35));
  const trajectoryScore = trajectoryScoreFor(alignment.path, referenceFrames, userFrames);
  const rangeScore = motionRangeScore(referenceFrames, userFrames);
  const rhythmScore = rhythmScoreFor(alignment.path, referenceFrames.length, userFrames.length);
  const bodyParts = bodyPartScores(alignment.path, referenceFrames, userFrames);
  const rawScore = clampScore(poseScore * 0.45 + trajectoryScore * 0.3 + rangeScore * 0.15 + rhythmScore * 0.1);
  const finalScore = Math.min(rawScore, compensationCeiling(poseScore, clampScore(trajectoryScore * 0.65 + rangeScore * 0.35)));
  const weakPoints = weakPointsFor({ poseScore, trajectoryScore, rangeScore, rhythmScore, bodyParts });
  const worst = worstMomentFor(alignment.path, referenceFrames, userFrames);

  return {
    ready: true,
    method: "12.07.2026",
    score: finalScore,
    finalScore,
    poseScore,
    boneDirectionScore: poseScore,
    motionScore: clampScore(trajectoryScore * 0.65 + rangeScore * 0.35),
    timingScore: rhythmScore,
    trajectoryScore,
    rangeScore,
    bodyParts,
    diagnostics: {
      averageTimeOffsetMs: Math.round(average(alignment.path.map(([i, j]) => Math.abs(referenceFrames[i].timestampMs - userFrames[j].timestampMs)))),
      elasticPathLength: alignment.path.length,
      weakPoints,
      missingJoints: [],
      confidence: confidenceFor(referenceFrames, userFrames)
    },
    rows: rowsFor({ poseScore, trajectoryScore, rangeScore, rhythmScore, bodyParts }),
    suggestions: weakPoints,
    framesCompared: alignment.path.length,
    bestScore: clampScore(100 * Math.exp(-alignment.bestCost * 2.35)),
    worstScore: clampScore(100 * Math.exp(-alignment.worstCost * 2.35)),
    durationCompared: Number(((referenceFrames.at(-1).timestampMs - referenceFrames[0].timestampMs) / 1000).toFixed(1)),
    worstMoment: worst,
    verdict: verdictFor(finalScore, weakPoints)
  };
}

function prepareSequence(input, maxFrames) {
  const frames = normalizeInput(input);
  const mapped = frames
    .map((frame, index) => ({
      timestampMs: frameTimeMs(frame, index),
      landmarks: landmarkMap(frame.joints || frame.landmarks || frame.points || frame),
      confidence: Number(frame.confidence || 0)
    }))
    .filter((frame) => frame.landmarks);
  const scale = stableScale(mapped);
  return sampleEvenly(mapped, maxFrames).map((frame) => {
    const normalized = normalizeFrame(frame.landmarks, scale);
    return {
      ...frame,
      normalized,
      angles: anglesFor(normalized),
      bones: bonesFor(normalized)
    };
  });
}

function normalizeInput(input) {
  const frames = Array.isArray(input) ? input : input?.frames;
  return Array.isArray(frames) ? frames : [];
}

function frameTimeMs(frame, index) {
  if (Number.isFinite(frame?.timestamp)) return Number(frame.timestamp);
  if (Number.isFinite(frame?.timestampMs)) return Number(frame.timestampMs);
  if (Number.isFinite(frame?.timeMs)) return Number(frame.timeMs);
  if (Number.isFinite(frame?.time)) return Number(frame.time) * 1000;
  return index * 200;
}

function landmarkMap(source) {
  if (!source) return null;
  const indexed = Array.isArray(source)
    ? Object.fromEntries(source.map((point, index) => (point ? [point.id ?? index, point] : null)).filter(Boolean))
    : source;
  const result = {};
  for (const joint of joints) result[joint] = findPoint(indexed, aliases[joint]);
  result.pelvis = midpoint(result.leftHip, result.rightHip);
  result.neck = midpoint(result.leftShoulder, result.rightShoulder);
  return result.pelvis && result.neck ? result : null;
}

function findPoint(source, names) {
  for (const name of names) {
    const point = source[name];
    if (isPoint(point)) return { x: Number(point.x), y: Number(point.y), z: Number(point.z || 0), visibility: Number(point.visibility || 0) };
  }
  return null;
}

function normalizeFrame(points, scale) {
  const pelvis = points.pelvis;
  const spine = vector(pelvis, points.neck);
  const rotation = -Math.atan2(spine.y, spine.x) - Math.PI / 2;
  return Object.fromEntries(
    Object.entries(points).map(([key, point]) => {
      if (!point) return [key, null];
      return [key, rotate2d({ x: (point.x - pelvis.x) / scale, y: (point.y - pelvis.y) / scale, z: (point.z - pelvis.z) / scale }, rotation)];
    })
  );
}

function stableScale(frames) {
  const values = frames
    .map((frame) => bodyScale(frame.landmarks))
    .filter((value) => Number.isFinite(value) && value > 0)
    .sort((a, b) => a - b);
  return values.length ? values[Math.floor(values.length / 2)] : 1;
}

function bodyScale(points) {
  const torso = distance(points.pelvis, points.neck);
  const thigh = average([distance(points.leftHip, points.leftKnee), distance(points.rightHip, points.rightKnee)]);
  const shin = average([distance(points.leftKnee, points.leftAnkle), distance(points.rightKnee, points.rightAnkle)]);
  return Math.max(0.0001, torso + thigh + shin || 1);
}

function dtwAlign(referenceFrames, userFrames) {
  const n = referenceFrames.length;
  const m = userFrames.length;
  const band = Math.max(8, Math.ceil(Math.max(n, m) * 0.22));
  const dp = Array.from({ length: n + 1 }, () => Array(m + 1).fill(Infinity));
  const back = Array.from({ length: n + 1 }, () => Array(m + 1).fill(null));
  dp[0][0] = 0;

  for (let i = 1; i <= n; i += 1) {
    const center = Math.round((i / n) * m);
    const from = Math.max(1, center - band);
    const to = Math.min(m, center + band);
    for (let j = from; j <= to; j += 1) {
      const cost = frameCost(referenceFrames[i - 1], userFrames[j - 1]);
      const choices = [
        [dp[i - 1][j], i - 1, j],
        [dp[i][j - 1], i, j - 1],
        [dp[i - 1][j - 1], i - 1, j - 1]
      ].sort((a, b) => a[0] - b[0]);
      dp[i][j] = cost + choices[0][0];
      back[i][j] = [choices[0][1], choices[0][2], cost];
    }
  }

  const path = [];
  const costs = [];
  let i = n;
  let j = m;
  while (i > 0 && j > 0 && back[i][j]) {
    const [previousI, previousJ, cost] = back[i][j];
    path.push([i - 1, j - 1]);
    costs.push(cost);
    i = previousI;
    j = previousJ;
  }
  path.reverse();
  costs.reverse();

  return {
    path,
    averageCost: average(costs),
    bestCost: costs.length ? Math.min(...costs) : 1,
    worstCost: costs.length ? Math.max(...costs) : 1
  };
}

function frameCost(a, b) {
  const pointCost = average(
    compareJoints.map((joint) => {
      const left = a.normalized[joint];
      const right = b.normalized[joint];
      return left && right ? Math.min(1.4, distance(left, right)) : 0.7;
    })
  );
  const angleCost = average(
    Object.keys(a.angles).map((key) =>
      Number.isFinite(a.angles[key]) && Number.isFinite(b.angles[key]) ? Math.min(1, Math.abs(a.angles[key] - b.angles[key]) / 135) : 0.5
    )
  );
  const boneCost = average(
    Object.keys(a.bones).map((key) => {
      const left = a.bones[key];
      const right = b.bones[key];
      return left && right ? (1 - dot(left, right)) / 2 : 0.5;
    })
  );
  return pointCost * 0.45 + angleCost * 0.35 + boneCost * 0.2;
}

function trajectoryScoreFor(path, referenceFrames, userFrames) {
  const scores = [];
  for (let index = 1; index < path.length; index += 1) {
    const [previousI, previousJ] = path[index - 1];
    const [i, j] = path[index];
    for (const joint of motionJoints) {
      const leftDelta = delta(referenceFrames[previousI].normalized[joint], referenceFrames[i].normalized[joint]);
      const rightDelta = delta(userFrames[previousJ].normalized[joint], userFrames[j].normalized[joint]);
      if (!leftDelta || length(leftDelta) < 0.01) continue;
      if (!rightDelta || length(rightDelta) < 0.005) {
        scores.push(0);
        continue;
      }
      const direction = Math.max(0, dot(normalize(leftDelta), normalize(rightDelta))) * 100;
      const amplitude = ratioScore(length(leftDelta), length(rightDelta));
      scores.push(direction * 0.65 + amplitude * 0.35);
    }
  }
  return scores.length ? clampScore(average(scores)) : 100;
}

function compensationCeiling(postureScore, motionEvidence) {
  const weak = Math.min(postureScore, motionEvidence);
  const strong = Math.max(postureScore, motionEvidence);
  if (weak < 55) return clampScore(weak);
  return clampScore(weak + (strong - weak) * 0.25);
}

function motionRangeScore(referenceFrames, userFrames) {
  const referenceRange = sequenceRange(referenceFrames);
  const userRange = sequenceRange(userFrames);
  return ratioScore(Math.max(0, referenceRange - 0.035), Math.max(0, userRange - 0.035));
}

function sequenceRange(frames) {
  return average(
    motionJoints.map((joint) => {
      const points = frames.map((frame) => frame.normalized[joint]).filter(Boolean);
      return pointSpread(points);
    })
  );
}

function rhythmScoreFor(path, referenceLength, userLength) {
  if (!path.length) return 0;
  const ideal = Math.max(referenceLength, userLength);
  const warpPenalty = Math.min(45, Math.abs(path.length - ideal) * 1.8);
  const monotonyPenalty = Math.min(30, repeatedStepRatio(path) * 40);
  return clampScore(100 - warpPenalty - monotonyPenalty);
}

function repeatedStepRatio(path) {
  let repeated = 0;
  for (let index = 1; index < path.length; index += 1) {
    const [prevI, prevJ] = path[index - 1];
    const [i, j] = path[index];
    if (i === prevI || j === prevJ) repeated += 1;
  }
  return path.length ? repeated / path.length : 0;
}

function bodyPartScores(path, referenceFrames, userFrames) {
  const parts = { arms: [], legs: [], torso: [], head: [] };
  for (const [i, j] of path) {
    const left = referenceFrames[i];
    const right = userFrames[j];
    for (const [a, b, part] of bones) {
      const key = `${a}-${b}`;
      if (left.bones[key] && right.bones[key]) parts[part].push(((dot(left.bones[key], right.bones[key]) + 1) / 2) * 100);
    }
    if (left.normalized.head && right.normalized.head) parts.head.push(clampScore(100 - distance(left.normalized.head, right.normalized.head) * 120));
  }
  return Object.fromEntries(Object.entries(parts).map(([part, values]) => [part, clampScore(average(values))]));
}

function rowsFor({ poseScore, trajectoryScore, rangeScore, rhythmScore, bodyParts }) {
  return [
    row("2026-07-12-pose", "12.07.2026: эластичное совпадение поз", poseScore),
    row("2026-07-12-trajectory", "12.07.2026: траектория движения", trajectoryScore),
    row("2026-07-12-range", "12.07.2026: амплитуда движения", rangeScore),
    row("2026-07-12-rhythm", "12.07.2026: эластичная синхронизация", rhythmScore),
    ...Object.entries(bodyParts).map(([part, score]) => row(`2026-07-12-${part}`, bodyPartTitle(part), score))
  ];
}

function row(id, title, score) {
  const rounded = clampScore(score);
  return { id, title, leftValue: rounded, rightValue: 100, diff: 100 - rounded, unit: "%", score: rounded };
}

function weakPointsFor({ poseScore, trajectoryScore, rangeScore, rhythmScore, bodyParts }) {
  const weak = [];
  if (poseScore < 72) weak.push(`Эластичное совпадение поз низкое: ${poseScore}%.`);
  if (trajectoryScore < 72) weak.push(`Траектория движения отличается: ${trajectoryScore}%.`);
  if (rangeScore < 72) weak.push(`Амплитуда движения отличается: ${rangeScore}%.`);
  if (rhythmScore < 72) weak.push(`Последовательность пришлось сильно растягивать по времени: ${rhythmScore}%.`);
  for (const [part, score] of Object.entries(bodyParts)) {
    if (score < 70) weak.push(`${bodyPartTitle(part)}: ${score}/100.`);
  }
  return weak.slice(0, 8);
}

function verdictFor(score, weakPoints) {
  if (score >= 88) return "12.07.2026: повторение очень близко к эталону с учетом нормализации тела и эластичной синхронизации.";
  if (score >= 74) return `12.07.2026: движение похоже, но есть заметные зоны отличий. ${weakPoints.slice(0, 2).join(" ")}`;
  if (score >= 55) return `12.07.2026: совпадение частичное. ${weakPoints.slice(0, 3).join(" ")}`;
  return `12.07.2026: движение существенно отличается от эталона. ${weakPoints.slice(0, 3).join(" ")}`;
}

function worstMomentFor(path, referenceFrames, userFrames) {
  let worst = null;
  for (const [i, j] of path) {
    const cost = frameCost(referenceFrames[i], userFrames[j]);
    if (!worst || cost > worst.cost) worst = { i, j, cost };
  }
  if (!worst) return null;
  return {
    leftTime: Number((referenceFrames[worst.i].timestampMs / 1000).toFixed(2)),
    rightTime: Number((userFrames[worst.j].timestampMs / 1000).toFixed(2)),
    score: clampScore(100 * Math.exp(-worst.cost * 2.35))
  };
}

function emptyResult(message) {
  return {
    ready: false,
    method: "12.07.2026",
    score: 0,
    finalScore: 0,
    poseScore: 0,
    boneDirectionScore: 0,
    motionScore: 0,
    timingScore: 0,
    trajectoryScore: 0,
    rangeScore: 0,
    bodyParts: { arms: 0, legs: 0, torso: 0, head: 0, rhythm: 0 },
    diagnostics: { averageTimeOffsetMs: 0, elasticPathLength: 0, weakPoints: [message], missingJoints: [], confidence: 0 },
    rows: [],
    suggestions: [message],
    framesCompared: 0,
    bestScore: 0,
    worstScore: 0,
    durationCompared: 0,
    worstMoment: null,
    verdict: message
  };
}

function confidenceFor(referenceFrames, userFrames) {
  return clampScore(((average(referenceFrames.map((frame) => frame.confidence)) + average(userFrames.map((frame) => frame.confidence))) / 2) * 100 || 90);
}

function anglesFor(points) {
  return Object.fromEntries(angleSpecs.map(([id, a, b, c]) => [id, angle(points[a], points[b], points[c])]));
}

function bonesFor(points) {
  return Object.fromEntries(
    bones.map(([a, b]) => {
      const direction = points[a] && points[b] ? normalize(vector(points[a], points[b])) : null;
      return [`${a}-${b}`, direction];
    })
  );
}

function sampleEvenly(items, maxItems) {
  if (items.length <= maxItems) return items;
  const last = items.length - 1;
  return Array.from({ length: maxItems }, (_, index) => items[Math.round((index / (maxItems - 1)) * last)]);
}

function pointSpread(points) {
  if (!points.length) return 0;
  return Math.hypot(range(points.map((p) => p.x)), range(points.map((p) => p.y)), range(points.map((p) => p.z || 0)) * 0.5);
}

function ratioScore(a, b) {
  if (a < 0.0001 && b < 0.0001) return 100;
  return clampScore((Math.min(a, b) / Math.max(a, b, 0.0001)) * 100);
}

function bodyPartTitle(part) {
  if (part === "arms") return "Руки";
  if (part === "legs") return "Ноги";
  if (part === "torso") return "Корпус";
  if (part === "head") return "Голова";
  return part;
}

function angle(a, b, c) {
  if (!a || !b || !c) return null;
  const ab = normalize(vector(b, a));
  const cb = normalize(vector(b, c));
  return (Math.acos(Math.max(-1, Math.min(1, dot(ab, cb)))) * 180) / Math.PI;
}

function delta(a, b) {
  if (!a || !b) return null;
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}

function vector(a, b) {
  if (!a || !b) return null;
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}

function midpoint(a, b) {
  return a && b ? { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: ((a.z || 0) + (b.z || 0)) / 2 } : null;
}

function rotate2d(point, angleValue) {
  const cos = Math.cos(angleValue);
  const sin = Math.sin(angleValue);
  return { x: point.x * cos - point.y * sin, y: point.x * sin + point.y * cos, z: point.z || 0 };
}

function normalize(value) {
  const size = length(value);
  return size ? { x: value.x / size, y: value.y / size, z: (value.z || 0) / size } : { x: 0, y: 0, z: 0 };
}

function distance(a, b) {
  if (!a || !b) return 0;
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function length(value) {
  return value ? Math.hypot(value.x || 0, value.y || 0, value.z || 0) : 0;
}

function dot(a, b) {
  return (a?.x || 0) * (b?.x || 0) + (a?.y || 0) * (b?.y || 0) + (a?.z || 0) * (b?.z || 0);
}

function range(values) {
  const clean = values.filter(Number.isFinite);
  return clean.length ? Math.max(...clean) - Math.min(...clean) : 0;
}

function average(values) {
  const clean = values.filter((value) => Number.isFinite(value));
  return clean.length ? clean.reduce((sum, value) => sum + value, 0) / clean.length : 0;
}

function clampScore(value) {
  return Math.max(0, Math.min(100, Math.round(Number.isFinite(value) ? value : 0)));
}

function isPoint(point) {
  return point && Number.isFinite(Number(point.x)) && Number.isFinite(Number(point.y));
}
