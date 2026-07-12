import test from "node:test";
import assert from "node:assert/strict";
import { compareSkeletons_2026_07_12 } from "./skeletonComparison20260712.mjs";

test("same choreography with body scale, camera offset, and phase shift scores high", () => {
  const reference = makeDanceSequence();
  const user = makeDanceSequence({ scale: 1.35, offsetX: 2.1, offsetY: -1.4, phase: 0.16, timeScale: 1.08, amplitude: 0.92 });
  const result = compareSkeletons_2026_07_12(reference, user);
  assert.ok(result.finalScore >= 78, `expected >= 78, got ${result.finalScore}`);
  assert.ok(result.trajectoryScore >= 70, `expected trajectory >= 70, got ${result.trajectoryScore}`);
});

test("identical choreography scores near 100", () => {
  const reference = makeDanceSequence();
  const result = compareSkeletons_2026_07_12(reference, reference);
  assert.ok(result.finalScore >= 98, `expected >= 98, got ${result.finalScore}`);
});

test("standing person against dancing reference scores low", () => {
  const reference = makeDanceSequence();
  const user = makeDanceSequence({ freeze: true, jitter: 0.02 });
  const result = compareSkeletons_2026_07_12(reference, user);
  assert.ok(result.finalScore <= 45, `expected <= 45, got ${result.finalScore}`);
});

test("different moving choreography is not treated as excellent", () => {
  const reference = makeDanceSequence();
  const user = makeDanceSequence({ differentDance: true });
  const result = compareSkeletons_2026_07_12(reference, user);
  assert.ok(result.finalScore <= 70, `expected <= 70, got ${result.finalScore}`);
});

function makeDanceSequence(options = {}) {
  const frames = Array.from({ length: 42 }, (_, index) => {
    const time = index * 0.16;
    return {
      time,
      timestamp: Math.round(time * 1000),
      confidence: 0.96,
      landmarks: makePose(time * (options.timeScale || 1) + (options.phase || 0), options)
    };
  });
  return { frames };
}

function makePose(time, options = {}) {
  const scale = options.scale || 1;
  const offsetX = options.offsetX || 0;
  const offsetY = options.offsetY || 0;
  const amplitude = options.freeze ? 0 : options.amplitude ?? 1;
  const direction = options.oppositeDance ? -1 : 1;
  const arm = options.differentDance
    ? Math.cos(time * Math.PI * 2.2) * 0.32 * amplitude
    : Math.sin(time * Math.PI * 1.35) * 0.26 * amplitude * direction;
  const cross = options.differentDance
    ? Math.sin(time * Math.PI * 1.7) * 0.28 * amplitude
    : Math.cos(time * Math.PI * 0.9) * 0.18 * amplitude;
  const leg = options.differentDance
    ? Math.cos(time * Math.PI * 2.0 + 0.7) * 0.18 * amplitude
    : Math.sin(time * Math.PI * 1.05 + 0.4) * 0.12 * amplitude * direction;

  const point = (x, y) => {
    const jitter = options.jitter || 0;
    const seed = time * 8 + x * 5 + y * 7;
    return {
      x: x * scale + offsetX + Math.sin(seed) * jitter,
      y: y * scale + offsetY + Math.cos(seed * 1.2) * jitter,
      z: 0,
      visibility: 0.95
    };
  };

  const landmarks = [];
  landmarks[0] = point(0.02 + cross * 0.15, -1.34);
  landmarks[11] = point(-0.38, -0.96);
  landmarks[12] = point(0.38, -0.96);
  landmarks[13] = point(-0.64 - cross * 0.35, -0.72 - arm);
  landmarks[14] = point(0.64 + cross * 0.3, -0.72 + arm * 0.82);
  landmarks[15] = point(-0.86 - cross * 0.6, -0.48 - arm * 1.55);
  landmarks[16] = point(0.86 + cross * 0.55, -0.48 + arm * 1.35);
  landmarks[23] = point(-0.25, 0.04);
  landmarks[24] = point(0.25, 0.04);
  landmarks[25] = point(-0.3 - leg * 0.2, 0.72 + leg);
  landmarks[26] = point(0.3 + leg * 0.25, 0.72 - leg * 0.7);
  landmarks[27] = point(-0.34 - leg * 0.45, 1.34 + leg * 1.1);
  landmarks[28] = point(0.34 + leg * 0.4, 1.34 - leg * 0.85);
  return landmarks;
}
