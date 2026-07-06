import test from "node:test";
import assert from "node:assert/strict";
import { compareSkeletons_2026_07_06 } from "./skeletonComparison20260706.mjs";

test("identical skeleton scores close to 100", () => {
  const reference = makeSequence();
  const result = compareSkeletons_2026_07_06(reference, reference);
  assert.ok(result.finalScore >= 98, `expected >= 98, got ${result.finalScore}`);
});

test("same pose with taller person keeps a high score", () => {
  const reference = makeSequence();
  const user = makeSequence({ scale: 1.55, offsetX: 4.2, offsetY: -2.4 });
  const result = compareSkeletons_2026_07_06(reference, user);
  assert.ok(result.finalScore >= 92, `expected >= 92, got ${result.finalScore}`);
});

test("200 ms late movement is penalized softly", () => {
  const reference = makeSequence();
  const user = makeSequence({ delayMs: 200 });
  const result = compareSkeletons_2026_07_06(reference, user, { syncWindowMs: 300 });
  assert.ok(result.finalScore >= 80, `expected >= 80, got ${result.finalScore}`);
  assert.ok(result.timingScore > 20 && result.timingScore < 60, `expected soft timing penalty, got ${result.timingScore}`);
});

test("1000 ms late movement gets low timing score", () => {
  const reference = makeSequence();
  const user = makeSequence({ delayMs: 1000 });
  const result = compareSkeletons_2026_07_06(reference, user, { syncWindowMs: 300 });
  assert.ok(result.timingScore <= 25, `expected low timing score, got ${result.timingScore}`);
});

test("unmatched worst moment keeps missing right time explicit", () => {
  const reference = makeSequence();
  const user = makeSequence({ delayMs: 1000 });
  const result = compareSkeletons_2026_07_06(reference, user, { syncWindowMs: 300 });
  assert.equal(result.worstMoment.rightTime, null);
  assert.equal(typeof result.worstMoment.leftTime, "number");
});

test("matching arms and wrong legs produce high arms score and low legs score", () => {
  const reference = makeSequence();
  const user = makeSequence({ wrongLegs: true });
  const result = compareSkeletons_2026_07_06(reference, user);
  assert.ok(result.bodyParts.arms >= 90, `expected high arms score, got ${result.bodyParts.arms}`);
  assert.ok(result.bodyParts.legs <= 72, `expected low legs score, got ${result.bodyParts.legs}`);
});

test("different arm and leg lengths do not strongly reduce the score", () => {
  const reference = makeSequence();
  const user = makeSequence({ armScale: 1.35, legScale: 0.8 });
  const result = compareSkeletons_2026_07_06(reference, user);
  assert.ok(result.finalScore >= 84, `expected >= 84, got ${result.finalScore}`);
});

test("dance against a standing person is not scored as highly similar", () => {
  const reference = makeSequence();
  const user = makeSequence({ freeze: true });
  const result = compareSkeletons_2026_07_06(reference, user);
  assert.ok(result.finalScore <= 25, `expected <= 25, got ${result.finalScore}`);
  assert.ok(result.activityScore <= 45, `expected low activity score, got ${result.activityScore}`);
  assert.equal(result.diagnostics.activity.staticMismatch, true);
});

test("dance against a standing person with tracking jitter stays low", () => {
  const reference = makeSequence();
  const user = makeSequence({ freeze: true, jitter: 0.025 });
  const result = compareSkeletons_2026_07_06(reference, user);
  assert.ok(result.finalScore <= 30, `expected <= 30, got ${result.finalScore}`);
  assert.equal(result.diagnostics.motionRange.staticRangeMismatch, true);
});

test("dance against a standing person with stronger tracking jitter stays low", () => {
  const reference = makeSequence();
  const user = makeSequence({ freeze: true, jitter: 0.055 });
  const result = compareSkeletons_2026_07_06(reference, user);
  assert.ok(result.finalScore <= 45, `expected <= 45, got ${result.finalScore}`);
  assert.equal(result.diagnostics.motionRange.staticRangeMismatch, true);
});

test("different moving choreography does not score as a close match", () => {
  const reference = makeSequence();
  const user = makeSequence({ oppositeDance: true });
  const result = compareSkeletons_2026_07_06(reference, user);
  assert.ok(result.finalScore <= 65, `expected <= 65, got ${result.finalScore}`);
});

test("missing joint does not break calculation and appears in diagnostics", () => {
  const reference = makeSequence();
  const user = makeSequence();
  delete user.frames[2].joints.leftWrist;
  const result = compareSkeletons_2026_07_06(reference, user);
  assert.equal(result.ready, true);
  assert.ok(result.diagnostics.missingJoints.length > 0, "expected missing joint diagnostics");
  assert.ok(result.diagnostics.confidence < 100, `expected confidence penalty, got ${result.diagnostics.confidence}`);
});

test("worst moment keeps UI-compatible time fields", () => {
  const reference = makeSequence();
  const user = makeSequence({ wrongLegs: true });
  const result = compareSkeletons_2026_07_06(reference, user);
  assert.equal(typeof result.worstMoment.leftTime, "number");
  assert.equal(typeof result.worstMoment.rightTime, "number");
});

function makeSequence(options = {}) {
  const frames = [0, 200, 400, 600, 800, 1000].map((timestamp) => ({
    timestamp: timestamp + (options.delayMs || 0),
    joints: makePose(options.freeze ? 0 : timestamp / 1000, { ...options, timestamp })
  }));
  return { frames };
}

function makePose(seconds, options = {}) {
  const scale = options.scale || 1;
  const armScale = options.armScale || 1;
  const legScale = options.legScale || 1;
  const offsetX = options.offsetX || 0;
  const offsetY = options.offsetY || 0;
  const lift = Math.sin(seconds * Math.PI * 1.5) * 0.22;
  const legMove = Math.cos(seconds * Math.PI) * 0.08;
  const armDirection = options.oppositeDance ? -1 : 1;
  const legDirection = options.oppositeDance ? -1 : 1;

  const pelvis = p(0, 0);
  const neck = p(0, -1.05);
  const leftShoulder = p(-0.38, -0.95);
  const rightShoulder = p(0.38, -0.95);
  const leftHip = p(-0.26, 0.05);
  const rightHip = p(0.26, 0.05);
  const leftElbow = p(-0.64 * armScale, -0.7 - lift * armDirection);
  const rightElbow = p(0.64 * armScale, -0.7 + lift * 0.75 * armDirection);
  const leftWrist = p(-0.85 * armScale, -0.45 - lift * 1.7 * armDirection);
  const rightWrist = p(0.85 * armScale, -0.45 + lift * 1.35 * armDirection);
  const leftKnee = p(options.wrongLegs ? -0.65 : -0.28, 0.72 * legScale + legMove);
  const rightKnee = p(options.wrongLegs ? 0.65 : 0.28, 0.72 * legScale - legMove * 0.4 * legDirection);
  const leftAnkle = p(options.wrongLegs ? -0.82 : -0.32, 1.36 * legScale + legMove * 0.8 * legDirection);
  const rightAnkle = p(options.wrongLegs ? 0.82 : 0.32, 1.36 * legScale - legMove * 0.35 * legDirection);

  return {
    pelvis,
    spine: midpoint(pelvis, neck),
    chest: midpoint(leftShoulder, rightShoulder),
    neck,
    head: p(0, -1.35),
    leftShoulder,
    rightShoulder,
    leftElbow,
    rightElbow,
    leftWrist,
    rightWrist,
    leftHip,
    rightHip,
    leftKnee,
    rightKnee,
    leftAnkle,
    rightAnkle
  };

  function p(x, y) {
    const jitter = options.jitter || 0;
    const seed = (options.timestamp || 0) / 200 + x * 7 + y * 11;
    return {
      x: x * scale + offsetX + Math.sin(seed) * jitter,
      y: y * scale + offsetY + Math.cos(seed * 1.3) * jitter,
      z: 0
    };
  }
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: 0 };
}
