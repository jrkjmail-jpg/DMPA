const defaultOptions20260706 = {
  syncWindowMs: 300,
  useBodyNormalization: true,
  compareAngles: true,
  compareBoneDirections: true,
  compareVelocity: true
};

const jointAliases = {
  pelvis: ["pelvis", "hips", "hip", "таз"],
  spine: ["spine", "позвоночник"],
  chest: ["chest", "грудь"],
  neck: ["neck", "шея"],
  head: ["head", "голова", "nose", "нос", 0],
  leftShoulder: ["leftShoulder", "left_shoulder", "левое плечо", 11],
  rightShoulder: ["rightShoulder", "right_shoulder", "правое плечо", 12],
  leftElbow: ["leftElbow", "left_elbow", "левый локоть", 13],
  rightElbow: ["rightElbow", "right_elbow", "правый локоть", 14],
  leftWrist: ["leftWrist", "left_wrist", "leftHand", "left_hand", "левая кисть", "левая ладонь", 15, 17, 19, 21],
  rightWrist: ["rightWrist", "right_wrist", "rightHand", "right_hand", "правая кисть", "правая ладонь", 16, 18, 20, 22],
  leftHip: ["leftHip", "left_hip", "левое бедро", 23],
  rightHip: ["rightHip", "right_hip", "правое бедро", 24],
  leftKnee: ["leftKnee", "left_knee", "левое колено", 25],
  rightKnee: ["rightKnee", "right_knee", "правое колено", 26],
  leftAnkle: ["leftAnkle", "left_ankle", "leftFoot", "left_foot", "левая стопа", "левая ступня", 27, 29, 31],
  rightAnkle: ["rightAnkle", "right_ankle", "rightFoot", "right_foot", "правая стопа", "правая ступня", 28, 30, 32]
};

const joints = Object.keys(jointAliases);

const bones = [
  ["neck", "head", "head"],
  ["leftShoulder", "rightShoulder", "torso"],
  ["pelvis", "neck", "torso"],
  ["pelvis", "leftHip", "torso"],
  ["pelvis", "rightHip", "torso"],
  ["leftShoulder", "leftElbow", "arms"],
  ["leftElbow", "leftWrist", "arms"],
  ["rightShoulder", "rightElbow", "arms"],
  ["rightElbow", "rightWrist", "arms"],
  ["leftHip", "leftKnee", "legs"],
  ["leftKnee", "leftAnkle", "legs"],
  ["rightHip", "rightKnee", "legs"],
  ["rightKnee", "rightAnkle", "legs"]
];

const angleSpecs20260706 = [
  ["leftElbow", "leftShoulder", "leftElbow", "leftWrist", "arms"],
  ["rightElbow", "rightShoulder", "rightElbow", "rightWrist", "arms"],
  ["leftShoulder", "leftElbow", "leftShoulder", "leftHip", "arms"],
  ["rightShoulder", "rightElbow", "rightShoulder", "rightHip", "arms"],
  ["leftHip", "leftShoulder", "leftHip", "leftKnee", "torso"],
  ["rightHip", "rightShoulder", "rightHip", "rightKnee", "torso"],
  ["leftKnee", "leftHip", "leftKnee", "leftAnkle", "legs"],
  ["rightKnee", "rightHip", "rightKnee", "rightAnkle", "legs"],
  ["neck", "pelvis", "neck", "head", "head"]
];

const bodyPartJoints = {
  arms: ["leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist"],
  legs: ["leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"],
  torso: ["pelvis", "spine", "chest", "neck", "leftShoulder", "rightShoulder", "leftHip", "rightHip"],
  head: ["neck", "head"]
};

export function compareSkeletons_2026_07_06(referenceSkeleton, userSkeleton, options = {}) {
  const normalizedOptions = {
    ...defaultOptions20260706,
    ...(options || {}),
    syncWindowMs: Math.max(150, Math.min(600, Number(options?.syncWindowMs ?? defaultOptions20260706.syncWindowMs)))
  };
  const referenceFrames = normalizeSequence(referenceSkeleton);
  const userFrames = normalizeSequence(userSkeleton);

  if (!referenceFrames.length || !userFrames.length) {
    return emptyResult20260706(normalizedOptions, "Недостаточно кадров скелета для сравнения.");
  }

  const preparedReference = prepareFrames(referenceFrames, normalizedOptions);
  const preparedUser = prepareFrames(userFrames, normalizedOptions);
  const frameResults = [];
  const missingJoints = new Map();

  for (const referenceFrame of preparedReference) {
    const candidates = preparedUser.filter(
      (userFrame) => Math.abs(userFrame.timestampMs - referenceFrame.timestampMs) <= normalizedOptions.syncWindowMs
    );
    if (!candidates.length) {
      frameResults.push(unmatchedFrameResult(referenceFrame, normalizedOptions));
      continue;
    }
    let best = null;
    for (const candidate of candidates) {
      const result = comparePreparedFrames(referenceFrame, candidate, normalizedOptions);
      if (!best || result.frameScore > best.frameScore) best = result;
    }
    frameResults.push(best);
    for (const joint of best.missingJoints) {
      missingJoints.set(joint, (missingJoints.get(joint) || 0) + 1);
    }
  }

  const poseScore = average(frameResults.map((item) => item.poseScore));
  const boneDirectionScore = average(frameResults.map((item) => item.boneDirectionScore));
  const angleScore = average(frameResults.map((item) => item.angleScore));
  const velocityScore = average(frameResults.map((item) => item.velocityScore));
  const motionScore = normalizedOptions.compareVelocity
    ? angleScore * 0.625 + velocityScore * 0.375
    : angleScore;
  const timingScore = average(frameResults.map((item) => item.timingScore));
  const finalScore = clampScore(poseScore * 0.4 + boneDirectionScore * 0.25 + motionScore * 0.2 + timingScore * 0.15);
  const bodyParts = bodyPartScores(frameResults);
  const weakPoints = weakPointsFor({
    poseScore,
    boneDirectionScore,
    angleScore,
    velocityScore,
    timingScore,
    bodyParts,
    missingJoints
  });

  return {
    ready: true,
    method: "06.07.2026",
    score: finalScore,
    finalScore,
    poseScore: clampScore(poseScore),
    boneDirectionScore: clampScore(boneDirectionScore),
    motionScore: clampScore(motionScore),
    timingScore: clampScore(timingScore),
    angleScore: clampScore(angleScore),
    velocityScore: clampScore(velocityScore),
    bodyParts,
    diagnostics: {
      averageTimeOffsetMs: Math.round(average(frameResults.map((item) => Math.abs(item.timeOffsetMs)))),
      syncWindowMs: normalizedOptions.syncWindowMs,
      weakPoints,
      missingJoints: Array.from(missingJoints.entries()).map(([joint, count]) => ({ joint, frames: count })),
      confidence: confidenceFor(frameResults, missingJoints.size)
    },
    rows: rowsForResult({ poseScore, boneDirectionScore, angleScore, velocityScore, timingScore, bodyParts }),
    suggestions: weakPoints,
    framesCompared: frameResults.length,
    bestScore: clampScore(Math.max(...frameResults.map((item) => item.frameScore))),
    worstScore: clampScore(Math.min(...frameResults.map((item) => item.frameScore))),
    durationCompared: Number(((preparedReference.at(-1).timestampMs - preparedReference[0].timestampMs) / 1000).toFixed(1)),
    verdict: verdict20260706(finalScore, weakPoints)
  };
}

function normalizeSequence(input) {
  const frames = Array.isArray(input) ? input : input?.frames;
  if (!Array.isArray(frames)) return [];
  return frames
    .map((frame, index) => ({
      source: frame,
      timestampMs: frameTimeMs(frame, index),
      landmarks: landmarksToMap(frame.joints || frame.landmarks || frame.points || frame)
    }))
    .filter((frame) => Number.isFinite(frame.timestampMs) && frame.landmarks);
}

function frameTimeMs(frame, index) {
  if (Number.isFinite(frame?.timestamp)) return Number(frame.timestamp);
  if (Number.isFinite(frame?.timestampMs)) return Number(frame.timestampMs);
  if (Number.isFinite(frame?.timeMs)) return Number(frame.timeMs);
  if (Number.isFinite(frame?.time)) return Number(frame.time) * 1000;
  return index * 200;
}

function landmarksToMap(landmarks) {
  if (!landmarks) return null;
  const source = Array.isArray(landmarks)
    ? Object.fromEntries(landmarks.map((point, index) => [point?.id ?? index, point]))
    : landmarks;
  const mapped = {};
  for (const joint of joints) {
    mapped[joint] = findPoint(source, jointAliases[joint]);
  }
  mapped.pelvis ||= midpoint(mapped.leftHip, mapped.rightHip);
  mapped.neck ||= midpoint(mapped.leftShoulder, mapped.rightShoulder);
  mapped.chest ||= midpoint(mapped.leftShoulder, mapped.rightShoulder);
  mapped.spine ||= midpoint(mapped.pelvis, mapped.neck);
  mapped.head ||= findPoint(source, jointAliases.head) || mapped.neck;
  return mapped;
}

function findPoint(source, aliases) {
  for (const alias of aliases) {
    const point = source[alias];
    if (isPoint(point)) return point3(point);
  }
  return null;
}

function prepareFrames(frames, options) {
  const stableScale = options.useBodyNormalization ? stableBodyScale(frames) : 1;
  const prepared = frames.map((frame) => {
    const normalized = options.useBodyNormalization ? normalizeFrameJoints(frame.landmarks, stableScale) : frame.landmarks;
    return {
      ...frame,
      normalized,
      angles: anglesFor(normalized),
      boneDirections: boneDirectionsFor(normalized)
    };
  });
  for (let index = 0; index < prepared.length; index += 1) {
    prepared[index].velocity = velocityFor(prepared[index - 1], prepared[index]);
  }
  return prepared;
}

function stableBodyScale(frames) {
  const scales = frames
    .map((frame) => {
      const pelvis = frame.landmarks.pelvis || midpoint(frame.landmarks.leftHip, frame.landmarks.rightHip);
      const neck = frame.landmarks.neck || midpoint(frame.landmarks.leftShoulder, frame.landmarks.rightShoulder) || frame.landmarks.head;
      return pelvis ? bodyScale(frame.landmarks, pelvis, neck) : null;
    })
    .filter((value) => Number.isFinite(value) && value > 0)
    .sort((a, b) => a - b);
  if (!scales.length) return 1;
  return scales[Math.floor(scales.length / 2)];
}

function normalizeFrameJoints(points, stableScale) {
  const pelvis = points.pelvis || midpoint(points.leftHip, points.rightHip);
  if (!pelvis) return points;
  const neck = points.neck || midpoint(points.leftShoulder, points.rightShoulder) || points.head;
  const scale = stableScale || bodyScale(points, pelvis, neck);
  const spine = neck ? vector(pelvis, neck) : { x: 0, y: -1, z: 0 };
  const rotation = -Math.atan2(spine.y, spine.x) - Math.PI / 2;
  return Object.fromEntries(
    Object.entries(points).map(([joint, point]) => {
      if (!point) return [joint, null];
      const centered = {
        x: (point.x - pelvis.x) / scale,
        y: (point.y - pelvis.y) / scale,
        z: ((point.z || 0) - (pelvis.z || 0)) / scale
      };
      return [joint, rotate2d(centered, rotation)];
    })
  );
}

function bodyScale(points, pelvis, neck) {
  const torso = distance(pelvis, neck);
  const thigh = average([distance(points.leftHip, points.leftKnee), distance(points.rightHip, points.rightKnee)]);
  const shin = average([distance(points.leftKnee, points.leftAnkle), distance(points.rightKnee, points.rightAnkle)]);
  return Math.max(0.0001, torso + thigh + shin || 1);
}

function comparePreparedFrames(referenceFrame, userFrame, options) {
  const pose = compareJointPositions(referenceFrame.normalized, userFrame.normalized);
  const bone = options.compareBoneDirections ? compareBoneDirections(referenceFrame.boneDirections, userFrame.boneDirections) : perfectComponent();
  const angles = options.compareAngles ? compareAngles(referenceFrame.angles, userFrame.angles) : perfectComponent();
  const velocity = options.compareVelocity ? compareVelocity(referenceFrame.velocity, userFrame.velocity) : perfectComponent();
  const timeOffsetMs = userFrame.timestampMs - referenceFrame.timestampMs;
  const timingScore = clampScore(100 - (Math.abs(timeOffsetMs) / options.syncWindowMs) * 100);
  const frameScore = clampScore(pose.score * 0.35 + bone.score * 0.25 + angles.score * 0.25 + velocity.score * 0.15);

  return {
    frameScore,
    poseScore: pose.score,
    boneDirectionScore: bone.score,
    angleScore: angles.score,
    velocityScore: velocity.score,
    timingScore,
    timeOffsetMs,
    bodyParts: mergeBodyPartComponents([pose.parts, bone.parts, angles.parts, velocity.parts]),
    missingJoints: [...pose.missingJoints, ...bone.missingJoints, ...angles.missingJoints, ...velocity.missingJoints]
  };
}

function unmatchedFrameResult(referenceFrame, options) {
  return {
    frameScore: 0,
    poseScore: 0,
    boneDirectionScore: 0,
    angleScore: 0,
    velocityScore: 0,
    timingScore: 0,
    timeOffsetMs: options.syncWindowMs,
    bodyParts: { arms: 0, legs: 0, torso: 0, head: 0, rhythm: 0 },
    missingJoints: Object.keys(referenceFrame.normalized || {})
  };
}

function compareJointPositions(left, right) {
  const parts = {};
  const missingJoints = [];
  for (const [part, partJoints] of Object.entries(bodyPartJoints)) {
    const scores = [];
    for (const joint of partJoints) {
      const a = left[joint];
      const b = right[joint];
      if (!a || !b) {
        missingJoints.push(joint);
        continue;
      }
      scores.push(clampScore(100 - distance(a, b) * 90));
    }
    parts[part] = averageOrNull(scores);
  }
  return componentFromParts(parts, missingJoints);
}

function compareBoneDirections(left, right) {
  const parts = {};
  const missingJoints = [];
  for (const [a, b, part] of bones) {
    if (!left[`${a}-${b}`] || !right[`${a}-${b}`]) {
      missingJoints.push(`${a}-${b}`);
      continue;
    }
    const similarity = dot(left[`${a}-${b}`], right[`${a}-${b}`]);
    const score = clampScore(((similarity + 1) / 2) * 100);
    parts[part] = [...(parts[part] || []), score];
  }
  return componentFromParts(mapValues(parts, averageOrNull), missingJoints);
}

function compareAngles(left, right) {
  const parts = {};
  const missingJoints = [];
  for (const [id, , , , part] of angleSpecs20260706) {
    if (!Number.isFinite(left[id]) || !Number.isFinite(right[id])) {
      missingJoints.push(id);
      continue;
    }
    const score = clampScore(100 - Math.abs(left[id] - right[id]) * 1.75);
    parts[part] = [...(parts[part] || []), score];
  }
  return componentFromParts(mapValues(parts, averageOrNull), missingJoints);
}

function compareVelocity(left, right) {
  const parts = {};
  const missingJoints = [];
  for (const [part, partJoints] of Object.entries(bodyPartJoints)) {
    const scores = [];
    for (const joint of partJoints) {
      const a = left[joint];
      const b = right[joint];
      if (!a || !b) {
        missingJoints.push(`velocity-${joint}`);
        continue;
      }
      const magnitudePenalty = Math.min(55, Math.abs(length(a) - length(b)) * 60);
      const directionPenalty = length(a) > 0.01 || length(b) > 0.01 ? (1 - dot(normalize(a), normalize(b))) * 25 : 0;
      scores.push(clampScore(100 - magnitudePenalty - directionPenalty));
    }
    parts[part] = averageOrNull(scores);
  }
  return componentFromParts(parts, missingJoints);
}

function componentFromParts(parts, missingJoints) {
  const values = Object.values(parts).filter(Number.isFinite);
  return {
    score: values.length ? clampScore(average(values)) : 0,
    parts,
    missingJoints
  };
}

function perfectComponent() {
  return { score: 100, parts: { arms: 100, legs: 100, torso: 100, head: 100 }, missingJoints: [] };
}

function bodyPartScores(frameResults) {
  const result = {};
  for (const part of ["arms", "legs", "torso", "head"]) {
    result[part] = clampScore(average(frameResults.map((item) => item.bodyParts[part]).filter(Number.isFinite)));
  }
  result.rhythm = clampScore(average(frameResults.map((item) => item.timingScore)));
  return result;
}

function mergeBodyPartComponents(components) {
  const result = {};
  for (const part of ["arms", "legs", "torso", "head"]) {
    result[part] = average(
      components
        .map((component) => component[part])
        .filter(Number.isFinite)
    );
  }
  return result;
}

function rowsForResult(result) {
  return [
    { id: "2026-pose", title: "06.07.2026: относительная поза", leftValue: result.poseScore, rightValue: 100, diff: 100 - result.poseScore, unit: "%", score: result.poseScore },
    {
      id: "2026-bones",
      title: "06.07.2026: направления костей",
      leftValue: result.boneDirectionScore,
      rightValue: 100,
      diff: 100 - result.boneDirectionScore,
      unit: "%",
      score: result.boneDirectionScore
    },
    { id: "2026-angles", title: "06.07.2026: углы суставов", leftValue: result.angleScore, rightValue: 100, diff: 100 - result.angleScore, unit: "%", score: result.angleScore },
    { id: "2026-motion", title: "06.07.2026: скорость движения", leftValue: result.velocityScore, rightValue: 100, diff: 100 - result.velocityScore, unit: "%", score: result.velocityScore },
    { id: "2026-rhythm", title: "06.07.2026: ритм и синхронность", leftValue: result.timingScore, rightValue: 100, diff: 100 - result.timingScore, unit: "%", score: result.timingScore },
    ...Object.entries(result.bodyParts).map(([part, score]) => ({
      id: `2026-part-${part}`,
      title: bodyPartTitle(part),
      leftValue: score,
      rightValue: 100,
      diff: 100 - score,
      unit: "%",
      score
    }))
  ];
}

function weakPointsFor({ poseScore, boneDirectionScore, angleScore, velocityScore, timingScore, bodyParts, missingJoints }) {
  const weak = [];
  if (poseScore < 75) weak.push(`Относительная поза проседает до ${Math.round(poseScore)}%.`);
  if (boneDirectionScore < 75) weak.push(`Направления рук, ног или корпуса отличаются: ${Math.round(boneDirectionScore)}%.`);
  if (angleScore < 75) weak.push(`Углы суставов отличаются: ${Math.round(angleScore)}%.`);
  if (velocityScore < 75) weak.push(`Движение идет с другой скоростью или в другом направлении: ${Math.round(velocityScore)}%.`);
  if (timingScore < 75) weak.push(`Есть заметное раннее или позднее движение: ${Math.round(timingScore)}%.`);
  for (const [part, score] of Object.entries(bodyParts)) {
    if (score < 70) weak.push(`${bodyPartTitle(part)}: ${score}/100.`);
  }
  if (missingJoints.size) weak.push(`В части кадров отсутствуют суставы: ${Array.from(missingJoints.keys()).slice(0, 5).join(", ")}.`);
  return weak.slice(0, 8);
}

function confidenceFor(frameResults, missingJointKinds) {
  const matchedShare = frameResults.filter((item) => item.frameScore > 0).length / Math.max(1, frameResults.length);
  return clampScore(matchedShare * 100 - Math.min(35, missingJointKinds * 2));
}

function verdict20260706(score, weakPoints) {
  if (score >= 86) return "06.07.2026: движение близко к эталону после нормализации роста, комплекции и положения тела.";
  if (score >= 70) return `06.07.2026: движение в целом похоже, но есть зоны для исправления. ${weakPoints.slice(0, 2).join(" ")}`;
  if (score >= 52) return `06.07.2026: совпадение частичное. ${weakPoints.slice(0, 3).join(" ")}`;
  return `06.07.2026: движение сильно отличается от эталона. ${weakPoints.slice(0, 3).join(" ")}`;
}

function emptyResult20260706(options, message) {
  return {
    ready: false,
    method: "06.07.2026",
    score: 0,
    finalScore: 0,
    poseScore: 0,
    boneDirectionScore: 0,
    motionScore: 0,
    timingScore: 0,
    bodyParts: { arms: 0, legs: 0, torso: 0, head: 0, rhythm: 0 },
    diagnostics: { averageTimeOffsetMs: 0, syncWindowMs: options.syncWindowMs, weakPoints: [message], missingJoints: [], confidence: 0 },
    rows: [],
    suggestions: [message],
    framesCompared: 0,
    verdict: message
  };
}

function anglesFor(points) {
  return Object.fromEntries(angleSpecs20260706.map(([id, a, b, c]) => [id, angle3(points[a], points[b], points[c])]));
}

function boneDirectionsFor(points) {
  return Object.fromEntries(
    bones.map(([a, b]) => {
      const direction = points[a] && points[b] ? normalize(vector(points[a], points[b])) : null;
      return [`${a}-${b}`, direction];
    })
  );
}

function velocityFor(previous, current) {
  if (!previous) return Object.fromEntries(joints.map((joint) => [joint, { x: 0, y: 0, z: 0 }]));
  const seconds = Math.max(0.001, (current.timestampMs - previous.timestampMs) / 1000);
  return Object.fromEntries(
    joints.map((joint) => {
      const a = previous.normalized[joint];
      const b = current.normalized[joint];
      return [joint, a && b ? scaleVector(vector(a, b), 1 / seconds) : null];
    })
  );
}

function angle3(a, b, c) {
  if (!a || !b || !c) return null;
  const ab = normalize(vector(b, a));
  const cb = normalize(vector(b, c));
  return Math.acos(Math.max(-1, Math.min(1, dot(ab, cb)))) * (180 / Math.PI);
}

function midpoint(a, b) {
  if (!a || !b) return null;
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: ((a.z || 0) + (b.z || 0)) / 2 };
}

function isPoint(point) {
  return point && Number.isFinite(point.x) && Number.isFinite(point.y);
}

function point3(point) {
  return { x: Number(point.x), y: Number(point.y), z: Number(point.z || 0) };
}

function vector(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}

function scaleVector(a, scale) {
  return { x: a.x * scale, y: a.y * scale, z: a.z * scale };
}

function rotate2d(point, angleValue) {
  const cos = Math.cos(angleValue);
  const sin = Math.sin(angleValue);
  return {
    x: point.x * cos - point.y * sin,
    y: point.x * sin + point.y * cos,
    z: point.z
  };
}

function distance(a, b) {
  if (!a || !b) return null;
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function length(a) {
  return Math.hypot(a?.x || 0, a?.y || 0, a?.z || 0);
}

function normalize(a) {
  const value = length(a);
  return value ? { x: a.x / value, y: a.y / value, z: (a.z || 0) / value } : { x: 0, y: 0, z: 0 };
}

function dot(a, b) {
  return (a?.x || 0) * (b?.x || 0) + (a?.y || 0) * (b?.y || 0) + (a?.z || 0) * (b?.z || 0);
}

function average(values) {
  const clean = values.filter((value) => Number.isFinite(value));
  return clean.length ? clean.reduce((sum, value) => sum + value, 0) / clean.length : 0;
}

function averageOrNull(values) {
  const clean = values.filter((value) => Number.isFinite(value));
  return clean.length ? average(clean) : null;
}

function mapValues(object, mapper) {
  return Object.fromEntries(Object.entries(object).map(([key, value]) => [key, mapper(value)]));
}

function clampScore(value) {
  return Math.max(0, Math.min(100, Math.round(Number.isFinite(value) ? value : 0)));
}

function bodyPartTitle(part) {
  if (part === "arms") return "Руки";
  if (part === "legs") return "Ноги";
  if (part === "torso") return "Корпус";
  if (part === "head") return "Голова";
  return "Ритм";
}
