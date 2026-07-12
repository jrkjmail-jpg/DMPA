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
const torsoFitJoints = ["leftShoulder", "rightShoulder", "leftHip", "rightHip", "pelvis", "neck"];
const limbTemporalGraceFrames = { arms: 2, legs: 1, torso: 0, head: 0 };
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
const stableBoneSpecs = [
  ["leftUpperArm", "leftShoulder", "leftElbow", 1],
  ["leftForearm", "leftElbow", "leftWrist", 1],
  ["rightUpperArm", "rightShoulder", "rightElbow", 1],
  ["rightForearm", "rightElbow", "rightWrist", 1],
  ["leftThigh", "leftHip", "leftKnee", 1],
  ["leftShin", "leftKnee", "leftAnkle", 1],
  ["rightThigh", "rightHip", "rightKnee", 1],
  ["rightShin", "rightKnee", "rightAnkle", 1],
  ["leftTorso", "leftShoulder", "leftHip", 2],
  ["rightTorso", "rightShoulder", "rightHip", 2],
  ["shoulders", "leftShoulder", "rightShoulder", 2],
  ["hips", "leftHip", "rightHip", 2]
];
const angleSpecs = [
  ["leftElbow", "leftShoulder", "leftElbow", "leftWrist", "arms"],
  ["rightElbow", "rightShoulder", "rightElbow", "rightWrist", "arms"],
  ["leftKnee", "leftHip", "leftKnee", "leftAnkle", "legs"],
  ["rightKnee", "rightHip", "rightKnee", "rightAnkle", "legs"],
  ["leftHip", "leftShoulder", "leftHip", "leftKnee", "torso"],
  ["rightHip", "rightShoulder", "rightHip", "rightKnee", "torso"]
];

export function compareSkeletons_2026_07_13(referenceSkeleton, userSkeleton, options = {}) {
  const maxFrames = Math.max(40, Math.min(260, options.maxFrames || 180));
  const referencePrepared = prepareSequence(referenceSkeleton, maxFrames);
  const userPrepared = prepareSequence(userSkeleton, maxFrames);
  const referenceFrames = referencePrepared.frames;
  const userFrames = userPrepared.frames;

  if (referenceFrames.length < 2 || userFrames.length < 2) {
    return emptyResult("Недостаточно кадров для хореографического сравнения.");
  }

  const alignment = dtwAlign(referenceFrames, userFrames);
  const poseScore = clampScore(100 * Math.exp(-alignment.robustCost * 1.85));
  const frameHitScore = frameHitScoreFor(alignment.costs);
  const keyPoseScore = keyPoseScoreFor(alignment.path, referenceFrames, userFrames);
  const anglePatternScore = anglePatternScoreFor(alignment.path, referenceFrames, userFrames);
  const trajectoryScore = robustTrajectoryScoreFor(alignment.path, referenceFrames, userFrames);
  const rangeScore = motionRangeScore(referenceFrames, userFrames);
  const rhythmScore = rhythmScoreFor(alignment.path, referenceFrames.length, userFrames.length);
  const bodyParts = bodyPartScores(alignment.path, referenceFrames, userFrames);
  const activityGate = choreographyActivityGate(referenceFrames, userFrames, rangeScore);
  const trackingQualityGate = trackingQualityGateFor(referencePrepared, userPrepared);
  const evidenceGate = choreographyEvidenceGate({ trajectoryScore, rangeScore, keyPoseScore, frameHitScore, anglePatternScore, activityGate });
  const phraseScore = clampScore(anglePatternScore * 0.34 + keyPoseScore * 0.26 + poseScore * 0.18 + frameHitScore * 0.1 + rhythmScore * 0.12);
  const motionScore = clampScore(anglePatternScore * 0.35 + trajectoryScore * 0.22 + rangeScore * 0.25 + bodyParts.arms * 0.1 + bodyParts.legs * 0.08);
  const rawScore = clampScore(phraseScore * 0.56 + motionScore * 0.24 + bodyParts.torso * 0.08 + rhythmScore * 0.12);
  const stableExecutionScore = stableExecutionScoreFor({ bodyParts, rangeScore, rhythmScore, keyPoseScore, poseScore });
  const finalScore = Math.min(Math.max(rawScore, stableExecutionScore), activityGate.ceiling, evidenceGate.ceiling);
  const weakPoints = weakPointsFor({
    poseScore,
    trajectoryScore,
    rangeScore,
    rhythmScore,
    bodyParts,
    keyPoseScore,
    frameHitScore,
    activityGate,
    evidenceGate,
    trackingQualityGate
  });
  const worst = worstMomentFor(alignment.path, referenceFrames, userFrames);

  return {
    ready: true,
    method: "13.07.2026",
    score: finalScore,
    finalScore,
    poseScore,
    boneDirectionScore: poseScore,
    motionScore,
    timingScore: rhythmScore,
    trajectoryScore,
    rangeScore,
    keyPoseScore,
    frameHitScore,
    anglePatternScore,
    phraseScore,
    stableExecutionScore,
    bodyParts,
    diagnostics: {
      averageTimeOffsetMs: Math.round(average(alignment.path.map(([i, j]) => Math.abs(referenceFrames[i].timestampMs - userFrames[j].timestampMs)))),
      elasticPathLength: alignment.path.length,
      trackingOutliersSkipped: referencePrepared.skippedFrames + userPrepared.skippedFrames,
      referenceOutliersSkipped: referencePrepared.skippedFrames,
      userOutliersSkipped: userPrepared.skippedFrames,
      limbTemporalGraceFrames,
      activityGate,
      evidenceGate,
      trackingQualityGate,
      weakPoints,
      missingJoints: [],
      confidence: confidenceFor(referenceFrames, userFrames)
    },
    rows: rowsFor({
      poseScore,
      trajectoryScore,
      rangeScore,
      rhythmScore,
      bodyParts,
      keyPoseScore,
      frameHitScore,
      anglePatternScore,
      phraseScore,
      stableExecutionScore,
      trackingQualityGate
    }),
    suggestions: weakPoints,
    framesCompared: alignment.path.length,
    bestScore: clampScore(100 * Math.exp(-alignment.bestCost * 1.85)),
    worstScore: clampScore(100 * Math.exp(-alignment.worstCost * 1.85)),
    durationCompared: Number(((referenceFrames.at(-1).timestampMs - referenceFrames[0].timestampMs) / 1000).toFixed(1)),
    worstMoment: worst,
    verdict: verdictFor(finalScore, weakPoints)
  };
}

export function filterSkeletonFrames_2026_07_13(input) {
  const frames = normalizeInput(input);
  const mapped = frames
    .map((frame, index) => mapRawFrameForQuality(frame, index))
    .filter((frame) => frame.landmarks)
    .map(addQualityMeasurements);
  return filterTrackingOutliers(mapped).frames.map((frame) => frame.source);
}

function prepareSequence(input, maxFrames) {
  const frames = normalizeInput(input);
  const mapped = frames
    .map((frame, index) => mapRawFrameForQuality(frame, index))
    .filter((frame) => frame.landmarks)
    .map(addQualityMeasurements);
  const filtered = filterTrackingOutliers(mapped);
  const scale = stableScale(filtered.frames);
  const preparedFrames = sampleEvenly(filtered.frames, maxFrames).map((frame) => {
    const normalized = normalizeFrame(frame.landmarks, scale);
    return {
      ...frame,
      normalized,
      angles: anglesFor(normalized),
      bones: bonesFor(normalized)
    };
  });
  return {
    frames: preparedFrames,
    skippedFrames: filtered.skippedFrames,
    inputFrames: mapped.length
  };
}

function mapRawFrameForQuality(frame, index) {
  return {
    source: frame,
    timestampMs: frameTimeMs(frame, index),
    landmarks: landmarkMap(frame.joints || frame.landmarks || frame.points || frame),
    confidence: Number(frame.confidence || 0)
  };
}

function addQualityMeasurements(frame) {
  return {
    ...frame,
    bodyScale: bodyScale(frame.landmarks),
    spread: skeletonSpread(frame.landmarks),
    visibility: landmarkVisibility(frame.landmarks),
    boneLengths: stableBoneLengths(frame.landmarks)
  };
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

function filterTrackingOutliers(frames) {
  if (frames.length < 6) return { frames, skippedFrames: 0 };
  const medianScale = median(frames.map((frame) => frame.bodyScale).filter((value) => value > 0));
  const medianSpread = median(frames.map((frame) => frame.spread).filter((value) => value > 0));
  const medianBoneLengths = Object.fromEntries(
    stableBoneSpecs.map(([id]) => [id, median(frames.map((frame) => frame.boneLengths[id]).filter((value) => value > 0))])
  );
  const medianStep = median(
    frames
      .slice(1)
      .map((frame, index) => normalizedFrameJump(frames[index].landmarks, frame.landmarks, medianScale))
      .filter(Number.isFinite)
  );
  const medianMaxStep = median(
    frames
      .slice(1)
      .map((frame, index) => normalizedMaxFrameJump(frames[index].landmarks, frame.landmarks, medianScale))
      .filter(Number.isFinite)
  );
  const kept = frames.filter((frame, index) => {
    const scaleRatio = frame.bodyScale / Math.max(medianScale, 0.000001);
    const spreadRatio = frame.spread / Math.max(medianSpread, 0.000001);
    const previousJump = index > 0 ? normalizedFrameJump(frames[index - 1].landmarks, frame.landmarks, medianScale) : 0;
    const nextJump = index < frames.length - 1 ? normalizedFrameJump(frame.landmarks, frames[index + 1].landmarks, medianScale) : 0;
    const previousMaxJump = index > 0 ? normalizedMaxFrameJump(frames[index - 1].landmarks, frame.landmarks, medianScale) : 0;
    const nextMaxJump = index < frames.length - 1 ? normalizedMaxFrameJump(frame.landmarks, frames[index + 1].landmarks, medianScale) : 0;
    const jumpLimit = Math.max(1.1, medianStep * 5.5);
    const maxJumpLimit = Math.max(2.2, medianMaxStep * 6.5);
    const isolatedJump = previousJump > jumpLimit && nextJump > jumpLimit;
    const impossibleJointJump = previousMaxJump > maxJumpLimit && nextMaxJump > maxJumpLimit;
    const boneBreaks = skeletonBoneBreaks(frame, medianBoneLengths);
    return (
      frame.visibility >= 0.28 &&
      scaleRatio >= 0.42 &&
      scaleRatio <= 2.35 &&
      spreadRatio >= 0.35 &&
      spreadRatio <= 2.75 &&
      boneBreaks <= 1 &&
      !isolatedJump &&
      !impossibleJointJump
    );
  });
  if (kept.length < Math.max(4, frames.length * 0.35)) return { frames, skippedFrames: 0 };
  return { frames: kept, skippedFrames: frames.length - kept.length };
}

function stableBoneLengths(points) {
  return Object.fromEntries(stableBoneSpecs.map(([id, from, to]) => [id, distance(points[from], points[to])]));
}

function skeletonBoneBreaks(frame, medianBoneLengths) {
  let breaks = 0;
  for (const [id, , , weight] of stableBoneSpecs) {
    const lengthValue = frame.boneLengths[id];
    const medianValue = medianBoneLengths[id];
    if (!Number.isFinite(lengthValue) || !Number.isFinite(medianValue) || medianValue <= 0.000001) continue;
    const ratioValue = lengthValue / medianValue;
    if (ratioValue > 2.2 || ratioValue < 0.32) breaks += weight;
  }
  return breaks;
}

function skeletonSpread(points) {
  const selected = compareJoints.map((joint) => points[joint]).filter(Boolean);
  if (!selected.length) return 0;
  return Math.hypot(range(selected.map((point) => point.x)), range(selected.map((point) => point.y)), range(selected.map((point) => point.z || 0)) * 0.5);
}

function landmarkVisibility(points) {
  const values = compareJoints.map((joint) => points[joint]?.visibility).filter(Number.isFinite);
  return values.length ? average(values) : 1;
}

function normalizedFrameJump(previous, current, scale) {
  const jumps = compareJoints
    .map((joint) => distance(previous[joint], current[joint]) / Math.max(scale, 0.000001))
    .filter(Number.isFinite);
  return jumps.length ? average(jumps) : 0;
}

function normalizedMaxFrameJump(previous, current, scale) {
  const jumps = compareJoints
    .map((joint) => distance(previous[joint], current[joint]) / Math.max(scale, 0.000001))
    .filter(Number.isFinite);
  return jumps.length ? Math.max(...jumps) : 0;
}

function fitTorsoToReference(referencePoints, userPoints) {
  const pairs = torsoFitJoints
    .map((joint) => [referencePoints[joint], userPoints[joint]])
    .filter(([reference, user]) => reference && user);
  if (pairs.length < 2) return userPoints;

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

  const fitted = Object.fromEntries(
    Object.entries(userPoints).map(([joint, point]) => {
      if (!point) return [joint, null];
      const x = point.x - userCenter.x;
      const y = point.y - userCenter.y;
      return [
        joint,
        {
          ...point,
          x: referenceCenter.x + (x * cos - y * sin) * scale,
          y: referenceCenter.y + (x * sin + y * cos) * scale,
          z: (point.z || 0) * scale
        }
      ];
    })
  );
  for (const joint of torsoFitJoints) {
    if (referencePoints[joint] && fitted[joint]) fitted[joint] = { ...fitted[joint], ...referencePoints[joint] };
  }
  return fitted;
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
      const cost = frameCost(referenceFrames[i - 1], userFrames[j - 1], {
        userFrames,
        userIndex: j - 1
      });
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
    robustCost: robustAverage(costs, 0.12),
    costs,
    bestCost: costs.length ? Math.min(...costs) : 1,
    worstCost: costs.length ? Math.max(...costs) : 1
  };
}

function frameCost(a, b, context = null) {
  const rightNormalized = fitTorsoToReference(a.normalized, b.normalized);
  const rightAngles = anglesFor(rightNormalized);
  const rightBones = bonesFor(rightNormalized);
  const pointCost = average(
    compareJoints.map((joint) => {
      const left = a.normalized[joint];
      const right = bestFittedJoint(a, rightNormalized, context, joint);
      return left && right ? Math.min(1.4, distance(left, right)) : 0.7;
    })
  );
  const angleCost = average(
    angleSpecs.map(([key, p1, p2, p3, part]) => {
      const leftAngle = a.angles[key];
      const rightAngle = bestFittedAngle(a, context, key, [p1, p2, p3], part, rightAngles[key]);
      return Number.isFinite(leftAngle) && Number.isFinite(rightAngle) ? Math.min(1, Math.abs(leftAngle - rightAngle) / 135) : 0.5;
    })
  );
  const boneCost = average(
    bones.map(([from, to, part]) => {
      const key = `${from}-${to}`;
      const left = a.bones[key];
      const right = bestFittedBone(a, context, key, from, to, part, rightBones[key]);
      return left && right ? (1 - dot(left, right)) / 2 : 0.5;
    })
  );
  return pointCost * 0.45 + angleCost * 0.35 + boneCost * 0.2;
}

function bestFittedJoint(referenceFrame, fallbackFitted, context, joint) {
  const part = partForJoint(joint);
  const candidates = fittedCandidates(referenceFrame, context, part);
  if (!candidates.length) return fallbackFitted[joint];
  const reference = referenceFrame.normalized[joint];
  if (!reference) return fallbackFitted[joint];
  return minBy(candidates, (candidate) => distance(reference, candidate[joint]))?.[joint] || fallbackFitted[joint];
}

function bestFittedAngle(referenceFrame, context, key, points, part, fallbackAngle) {
  const candidates = fittedCandidates(referenceFrame, context, part);
  if (!candidates.length) return fallbackAngle;
  const leftAngle = referenceFrame.angles[key];
  if (!Number.isFinite(leftAngle)) return fallbackAngle;
  const best = minBy(candidates, (candidate) => {
    const value = angle(candidate[points[0]], candidate[points[1]], candidate[points[2]]);
    return Number.isFinite(value) ? Math.abs(leftAngle - value) : Infinity;
  });
  return best ? angle(best[points[0]], best[points[1]], best[points[2]]) : fallbackAngle;
}

function bestFittedBone(referenceFrame, context, key, from, to, part, fallbackBone) {
  const candidates = fittedCandidates(referenceFrame, context, part);
  if (!candidates.length) return fallbackBone;
  const leftBone = referenceFrame.bones[key];
  if (!leftBone) return fallbackBone;
  const best = minBy(candidates, (candidate) => {
    const rightBone = candidate[from] && candidate[to] ? normalize(vector(candidate[from], candidate[to])) : null;
    return rightBone ? (1 - dot(leftBone, rightBone)) / 2 : Infinity;
  });
  return best?.[from] && best?.[to] ? normalize(vector(best[from], best[to])) : fallbackBone;
}

function fittedCandidates(referenceFrame, context, part) {
  const window = limbTemporalGraceFrames[part] || 0;
  if (!context?.userFrames?.length || !Number.isFinite(context.userIndex) || window <= 0) return [];
  const candidates = [];
  for (let offset = -window; offset <= window; offset += 1) {
    const frame = context.userFrames[context.userIndex + offset];
    if (frame?.normalized) candidates.push(fitTorsoToReference(referenceFrame.normalized, frame.normalized));
  }
  return candidates;
}

function partForJoint(joint) {
  if (["leftElbow", "rightElbow", "leftWrist", "rightWrist", "leftShoulder", "rightShoulder"].includes(joint)) return "arms";
  if (["leftKnee", "rightKnee", "leftAnkle", "rightAnkle"].includes(joint)) return "legs";
  if (joint === "head") return "head";
  return "torso";
}

function robustTrajectoryScoreFor(path, referenceFrames, userFrames) {
  const scores = [];
  for (let index = 1; index < path.length; index += 1) {
    const [previousI, previousJ] = path[index - 1];
    const [i, j] = path[index];
    const previousUser = fitTorsoToReference(referenceFrames[previousI].normalized, userFrames[previousJ].normalized);
    const currentUser = fitTorsoToReference(referenceFrames[i].normalized, userFrames[j].normalized);
    for (const joint of motionJoints) {
      const leftDelta = delta(referenceFrames[previousI].normalized[joint], referenceFrames[i].normalized[joint]);
      const rightDelta = delta(previousUser[joint], currentUser[joint]);
      if (!leftDelta || length(leftDelta) < 0.01) continue;
      if (!rightDelta || length(rightDelta) < 0.005) {
        scores.push(0);
        continue;
      }
      const direction = Math.max(0, dot(normalize(leftDelta), normalize(rightDelta))) * 100;
      const amplitude = ratioScore(length(leftDelta), length(rightDelta));
      scores.push(direction * 0.45 + amplitude * 0.35 + Math.max(direction, amplitude) * 0.2);
    }
  }
  return scores.length ? clampScore(robustAverage(scores, 0.14)) : 100;
}

function motionRangeScore(referenceFrames, userFrames) {
  const referenceRange = sequenceRange(referenceFrames);
  const userRange = sequenceRange(userFrames);
  return ratioScore(Math.max(0, referenceRange - 0.035), Math.max(0, userRange - 0.035));
}

function frameHitScoreFor(costs) {
  if (!costs.length) return 0;
  const scores = costs.map((cost) => 100 * Math.exp(-cost * 2.1));
  return clampScore(robustAverage(scores, 0.1));
}

function keyPoseScoreFor(path, referenceFrames, userFrames) {
  const keyIndexes = keyPoseIndexes(referenceFrames);
  if (!keyIndexes.length) return clampScore(100 * Math.exp(-robustAverage(path.map(([i, j]) => frameCost(referenceFrames[i], userFrames[j])), 0.1) * 1.85));
  const pathByReference = new Map(path.map(([i, j]) => [i, j]));
  const scores = [];
  for (const index of keyIndexes) {
    const userIndex = pathByReference.get(index);
    if (!Number.isFinite(userIndex)) continue;
    const localScores = [];
    for (let shift = -2; shift <= 2; shift += 1) {
      const frame = userFrames[userIndex + shift];
      if (!frame) continue;
      localScores.push(100 * Math.exp(-frameCost(referenceFrames[index], frame) * 2.0));
    }
    if (localScores.length) scores.push(Math.max(...localScores));
  }
  return scores.length ? clampScore(robustAverage(scores, 0.08)) : 0;
}

function keyPoseIndexes(frames) {
  if (frames.length <= 8) return frames.map((_, index) => index);
  const scores = frames.map((frame, index) => {
    const previous = frames[Math.max(0, index - 1)];
    const next = frames[Math.min(frames.length - 1, index + 1)];
    const velocity = average(motionJoints.map((joint) => length(delta(previous.normalized[joint], next.normalized[joint]))));
    const shape = average(["leftWrist", "rightWrist", "leftAnkle", "rightAnkle"].map((joint) => distance(frame.normalized[joint], frame.normalized.pelvis)));
    return velocity * 0.65 + shape * 0.35;
  });
  const sorted = [...scores].sort((a, b) => b - a);
  const threshold = sorted[Math.max(0, Math.min(sorted.length - 1, Math.floor(sorted.length * 0.34)))];
  const minGap = Math.max(2, Math.floor(frames.length / 18));
  const selected = [];
  scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score)
    .forEach(({ score, index }) => {
      if (score < threshold && selected.length >= 8) return;
      if (selected.every((existing) => Math.abs(existing - index) >= minGap)) selected.push(index);
    });
  return selected.sort((a, b) => a - b).slice(0, 24);
}

function anglePatternScoreFor(path, referenceFrames, userFrames) {
  const poseScores = [];
  const motionScores = [];
  for (let index = 0; index < path.length; index += 1) {
    const [i, j] = path[index];
    const left = referenceFrames[i];
    const right = bestAngleFrameFor(referenceFrames[i], userFrames, j);
    if (!right) continue;
    for (const [key, , , , part] of angleSpecs) {
      const leftAngle = left.angles[key];
      const rightAngle = right.angles[key];
      if (!Number.isFinite(leftAngle) || !Number.isFinite(rightAngle)) continue;
      const tolerance = part === "arms" ? 52 : part === "legs" ? 44 : 36;
      poseScores.push(100 * Math.exp(-Math.pow(Math.abs(leftAngle - rightAngle) / tolerance, 2)));
    }
    if (index > 0) {
      const [previousI, previousJ] = path[index - 1];
      const previousLeft = referenceFrames[previousI];
      const previousRight = bestAngleFrameFor(referenceFrames[previousI], userFrames, previousJ) || userFrames[previousJ];
      for (const [key, , , , part] of angleSpecs) {
        const leftDelta = left.angles[key] - previousLeft.angles[key];
        const rightDelta = right.angles[key] - previousRight.angles[key];
        if (!Number.isFinite(leftDelta) || !Number.isFinite(rightDelta) || Math.abs(leftDelta) < 1.2) continue;
        const sameDirection = Math.sign(leftDelta) === Math.sign(rightDelta) ? 100 : 32;
        const amplitude = ratioScore(Math.abs(leftDelta), Math.abs(rightDelta));
        motionScores.push(sameDirection * 0.45 + amplitude * 0.55 + (part === "arms" ? 4 : 0));
      }
    }
  }
  const pose = poseScores.length ? robustAverage(poseScores, 0.12) : 0;
  const motion = motionScores.length ? robustAverage(motionScores, 0.16) : 0;
  return clampScore(pose * 0.72 + motion * 0.28);
}

function bestAngleFrameFor(referenceFrame, userFrames, userIndex) {
  const candidates = [];
  for (let offset = -2; offset <= 2; offset += 1) {
    const frame = userFrames[userIndex + offset];
    if (frame?.angles) candidates.push(frame);
  }
  if (!candidates.length) return null;
  return minBy(candidates, (frame) =>
    average(
      angleSpecs
        .map(([key]) => {
          const left = referenceFrame.angles[key];
          const right = frame.angles[key];
          return Number.isFinite(left) && Number.isFinite(right) ? Math.abs(left - right) : null;
        })
        .filter(Number.isFinite)
    )
  );
}

function choreographyActivityGate(referenceFrames, userFrames, rangeScore) {
  const referenceActivity = sequenceActivity(referenceFrames);
  const userActivity = sequenceActivity(userFrames);
  const activityRatio = referenceActivity > 0.0001 ? userActivity / referenceActivity : 1;
  let ceiling = 100;
  if (referenceActivity > 0.06 && activityRatio < 0.18) ceiling = 28;
  else if (referenceActivity > 0.06 && activityRatio < 0.35) ceiling = 48;
  else if (rangeScore < 42 && activityRatio < 0.55) ceiling = 68;
  return {
    referenceActivity: Number(referenceActivity.toFixed(3)),
    userActivity: Number(userActivity.toFixed(3)),
    activityRatio: Number(activityRatio.toFixed(2)),
    ceiling
  };
}

function choreographyEvidenceGate({ trajectoryScore, rangeScore, keyPoseScore, frameHitScore, anglePatternScore, activityGate }) {
  let ceiling = 100;
  const reason = [];
  if (trajectoryScore < 38 && rangeScore < 25) {
    ceiling = 32;
    reason.push("движение почти не повторяет траекторию и амплитуду эталона");
  } else if (trajectoryScore < 50) {
    const strongChoreographyEvidence = rangeScore >= 80 && keyPoseScore >= 70 && frameHitScore >= 72;
    if (rangeScore < 80) ceiling = Math.min(ceiling, 80);
    else if (!strongChoreographyEvidence && anglePatternScore < 76) ceiling = Math.min(ceiling, 82);
    else ceiling = Math.min(ceiling, 92);
    reason.push("траектория движения не подтверждает ту же хореографическую фразу");
  } else if (trajectoryScore < 62 && Math.min(keyPoseScore, frameHitScore, anglePatternScore) < 72) {
    ceiling = Math.min(ceiling, 74);
    reason.push("ключевые позы и траектория одновременно слабые");
  }
  if (activityGate?.ceiling < 60 && rangeScore < 45) {
    ceiling = Math.min(ceiling, activityGate.ceiling);
    reason.push("активность ученика заметно ниже эталона");
  }
  return { ceiling, reason };
}

function stableExecutionScoreFor({ bodyParts, rangeScore, rhythmScore, keyPoseScore, poseScore }) {
  return clampScore(
    (bodyParts.arms || 0) * 0.24 +
      (bodyParts.legs || 0) * 0.18 +
      (bodyParts.torso || 0) * 0.08 +
      rangeScore * 0.22 +
      rhythmScore * 0.18 +
      keyPoseScore * 0.07 +
      poseScore * 0.03
  );
}

function trackingQualityGateFor(referencePrepared, userPrepared) {
  const referenceRatio = referencePrepared.inputFrames ? referencePrepared.skippedFrames / referencePrepared.inputFrames : 0;
  const userRatio = userPrepared.inputFrames ? userPrepared.skippedFrames / userPrepared.inputFrames : 0;
  const worstRatio = Math.max(referenceRatio, userRatio);
  let ceiling = 100;
  if (worstRatio >= 0.25) ceiling = 55;
  else if (worstRatio >= 0.18) ceiling = 65;
  else if (worstRatio >= 0.1) ceiling = 76;
  return {
    ceiling,
    referenceSkippedRatio: Number(referenceRatio.toFixed(2)),
    userSkippedRatio: Number(userRatio.toFixed(2)),
    worstSkippedRatio: Number(worstRatio.toFixed(2))
  };
}

function sequenceActivity(frames) {
  const values = [];
  for (let index = 1; index < frames.length; index += 1) {
    values.push(average(motionJoints.map((joint) => length(delta(frames[index - 1].normalized[joint], frames[index].normalized[joint])))));
  }
  return robustAverage(values, 0.08);
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
    const fittedRight = fitTorsoToReference(left.normalized, right.normalized);
    for (const [a, b, part] of bones) {
      const key = `${a}-${b}`;
      const partFitted = bestFittedFrameForPart(left, userFrames, j, part) || fittedRight;
      const fittedBones = bonesFor(partFitted);
      if (left.bones[key] && fittedBones[key]) parts[part].push(((dot(left.bones[key], fittedBones[key]) + 1) / 2) * 100);
    }
    const headFitted = bestFittedFrameForPart(left, userFrames, j, "head") || fittedRight;
    if (left.normalized.head && headFitted.head) parts.head.push(clampScore(100 - distance(left.normalized.head, headFitted.head) * 120));
  }
  return Object.fromEntries(Object.entries(parts).map(([part, values]) => [part, clampScore(average(values))]));
}

function bestFittedFrameForPart(referenceFrame, userFrames, userIndex, part) {
  const candidates = fittedCandidates(referenceFrame, { userFrames, userIndex }, part);
  if (!candidates.length) return null;
  const partJoints = compareJoints.filter((joint) => partForJoint(joint) === part);
  return minBy(candidates, (candidate) =>
    average(partJoints.map((joint) => distance(referenceFrame.normalized[joint], candidate[joint])).filter(Number.isFinite))
  );
}

function rowsFor({
  poseScore,
  trajectoryScore,
  rangeScore,
  rhythmScore,
  bodyParts,
  keyPoseScore,
  frameHitScore,
  anglePatternScore,
  phraseScore,
  stableExecutionScore,
  trackingQualityGate
}) {
  return [
    row("2026-07-13-phrase", "13.07.2026: хореографическая фраза", phraseScore),
    row("2026-07-13-stable-execution", "13.07.2026: устойчивое исполнение", stableExecutionScore),
    row("2026-07-13-angle-pattern", "13.07.2026: рисунок углов", anglePatternScore),
    row("2026-07-13-key-poses", "13.07.2026: ключевые позы", keyPoseScore),
    row("2026-07-13-pose", "13.07.2026: форма тела", poseScore),
    row("2026-07-13-frame-hit", "13.07.2026: попадание в кадры", frameHitScore),
    row("2026-07-13-trajectory", "13.07.2026: траектория без микрошумов", trajectoryScore),
    row("2026-07-13-range", "13.07.2026: амплитуда движения", rangeScore),
    row("2026-07-13-rhythm", "13.07.2026: музыкальная синхронность", rhythmScore),
    row("2026-07-13-tracking-quality", "13.07.2026: надежность скана", trackingQualityGate?.ceiling ?? 100),
    ...Object.entries(bodyParts).map(([part, score]) => row(`2026-07-13-${part}`, bodyPartTitle(part), score))
  ];
}

function row(id, title, score) {
  const rounded = clampScore(score);
  return { id, title, leftValue: rounded, rightValue: 100, diff: 100 - rounded, unit: "%", score: rounded };
}

function weakPointsFor({
  poseScore,
  trajectoryScore,
  rangeScore,
  rhythmScore,
  bodyParts,
  keyPoseScore,
  frameHitScore,
  activityGate,
  evidenceGate,
  trackingQualityGate
}) {
  const weak = [];
  if (trackingQualityGate?.ceiling < 90) {
    weak.push(`Скан требует осторожности: отброшено до ${Math.round((trackingQualityGate.worstSkippedRatio || 0) * 100)}% кадров.`);
  }
  if (activityGate?.ceiling < 90) weak.push(`Активность движения ниже эталона: потолок оценки ${activityGate.ceiling}%.`);
  if (evidenceGate?.ceiling < 90) weak.push(`Двигательная фраза ограничила оценку: ${evidenceGate.reason.join(", ")}.`);
  if (keyPoseScore < 74) weak.push(`Ключевые позы читаются слабее эталона: ${keyPoseScore}%.`);
  if (poseScore < 70) weak.push(`Форма тела отличается: ${poseScore}%.`);
  if (frameHitScore < 70) weak.push(`Много кадров не попадают в близкую позу: ${frameHitScore}%.`);
  if (trajectoryScore < 62) weak.push(`Траектория движения отличается: ${trajectoryScore}%.`);
  if (rangeScore < 64) weak.push(`Амплитуда движения отличается: ${rangeScore}%.`);
  if (rhythmScore < 72) weak.push(`Последовательность пришлось сильно растягивать по времени: ${rhythmScore}%.`);
  for (const [part, score] of Object.entries(bodyParts)) {
    if (score < 70) weak.push(`${bodyPartTitle(part)}: ${score}/100.`);
  }
  return weak.slice(0, 8);
}

function verdictFor(score, weakPoints) {
  if (score >= 88) return "13.07.2026: хореографическая фраза очень близко повторяет эталон с учетом корпуса, ключевых поз и музыкальной синхронизации.";
  if (score >= 74) return `13.07.2026: повторение похоже на эталон, но есть зоны для уточнения. ${weakPoints.slice(0, 2).join(" ")}`;
  if (score >= 55) return `13.07.2026: хореография распознана частично. ${weakPoints.slice(0, 3).join(" ")}`;
  return `13.07.2026: движение существенно отличается от эталона. ${weakPoints.slice(0, 3).join(" ")}`;
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
    method: "13.07.2026",
    score: 0,
    finalScore: 0,
    poseScore: 0,
    boneDirectionScore: 0,
    motionScore: 0,
    timingScore: 0,
    trajectoryScore: 0,
    rangeScore: 0,
    keyPoseScore: 0,
    frameHitScore: 0,
    phraseScore: 0,
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

function robustAverage(values, trimRatio = 0.1) {
  const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!clean.length) return 0;
  const trim = Math.min(Math.floor(clean.length * trimRatio), Math.floor((clean.length - 1) / 2));
  const trimmed = clean.slice(trim, clean.length - trim);
  return average(trimmed.length ? trimmed : clean);
}

function median(values) {
  const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
  return clean.length ? clean[Math.floor(clean.length / 2)] : 0;
}

function minBy(items, scoreForItem) {
  let best = null;
  let bestScore = Infinity;
  for (const item of items) {
    const score = scoreForItem(item);
    if (score < bestScore) {
      best = item;
      bestScore = score;
    }
  }
  return best;
}

function averagePoint(points) {
  const clean = points.filter(Boolean);
  return clean.length
    ? {
        x: average(clean.map((point) => point.x)),
        y: average(clean.map((point) => point.y)),
        z: average(clean.map((point) => point.z || 0))
      }
    : null;
}

function clampScore(value) {
  return Math.max(0, Math.min(100, Math.round(Number.isFinite(value) ? value : 0)));
}

function isPoint(point) {
  return point && Number.isFinite(Number(point.x)) && Number.isFinite(Number(point.y));
}
