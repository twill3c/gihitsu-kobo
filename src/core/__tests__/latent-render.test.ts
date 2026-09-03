// latent-render.test.ts — 潜在補間・乱数・描画変換(F-05 / N-05)
//
// 期待値の出所: lerp の端点・中点は定義式から。乱数の分布は緩い統計帯
// (64 次元 × 100 本 = 6,400 標本の平均 ±0.05・分散 [0.9,1.1] — 標準正規なら
// 平均の標準誤差 ≈ 0.0125 なので 4σ 帯。厳密検定ではなく実装取り違えの網)。

import { describe, expect, it } from "vitest";
import { lerpLatent, mulberry32, randomLatent } from "../latent";
import { logitToProb, pixelsToRgba } from "../render";
import { LATENT_DIM } from "../model";

describe("F-05: randomLatent", () => {
  it("決定的: 同 seed → 同ベクトル、異 seed → 異なる", () => {
    expect(randomLatent(7)).toEqual(randomLatent(7));
    expect(randomLatent(7)).not.toEqual(randomLatent(8));
    expect(randomLatent(7).length).toBe(LATENT_DIM);
  });

  it("全値が有限(Box-Muller の log(0) を作らない — N-05)", () => {
    for (let s = 0; s < 50; s++) {
      for (const v of randomLatent(s)) expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("標準正規の緩い帯: 平均 ±0.05・分散 [0.9, 1.1](6,400 標本)", () => {
    let sum = 0;
    let sq = 0;
    const n = 100 * LATENT_DIM;
    for (let s = 1000; s < 1100; s++) {
      for (const v of randomLatent(s)) {
        sum += v;
        sq += v * v;
      }
    }
    const mean = sum / n;
    const varc = sq / n - mean * mean;
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(varc).toBeGreaterThan(0.9);
    expect(varc).toBeLessThan(1.1);
  });

  it("mulberry32 は [0,1) を返す", () => {
    const r = mulberry32(1);
    for (let i = 0; i < 1000; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});

describe("F-05: lerpLatent", () => {
  const a = Float64Array.from([0, 1, -2]);
  const b = Float64Array.from([4, 1, 2]);

  it("端点と中点(定義式)", () => {
    expect(lerpLatent(a, b, 0)).toEqual(a);
    expect(lerpLatent(a, b, 1)).toEqual(b);
    expect(Array.from(lerpLatent(a, b, 0.5))).toEqual([2, 1, 0]);
  });

  it("t は [0,1] へクランプ、非有限 t と長さ不一致は RangeError(N-05)", () => {
    expect(lerpLatent(a, b, -3)).toEqual(a);
    expect(lerpLatent(a, b, 9)).toEqual(b);
    expect(() => lerpLatent(a, b, Number.NaN)).toThrow(RangeError);
    expect(() => lerpLatent(a, Float64Array.from([1]), 0.5)).toThrow(RangeError);
  });
});

describe("描画変換", () => {
  it("pixelsToRgba: 0 → 紙(255)、1 → 墨(0)、値域外はクランプ", () => {
    const px = new Array<number>(784).fill(0);
    px[0] = 1;
    px[1] = 0.5;
    px[2] = 2; // クランプ → 墨
    const rgba = pixelsToRgba(px);
    expect(rgba[0]).toBe(0);
    expect([rgba[4], rgba[5], rgba[6], rgba[7]]).toEqual([128, 128, 128, 255]);
    expect(rgba[8]).toBe(0);
    expect(rgba[3 * 4]).toBe(255);
    expect(rgba.length).toBe(784 * 4);
  });

  it("縁: 長さ不正・非有限値は RangeError(N-05)", () => {
    expect(() => pixelsToRgba([0.5])).toThrow(RangeError);
    const px = new Array<number>(784).fill(0);
    px[10] = Number.NaN;
    expect(() => pixelsToRgba(px)).toThrow(RangeError);
  });

  it("logitToProb は sigmoid(0 → 0.5、±∞や NaN は RangeError)", () => {
    expect(logitToProb(0)).toBe(0.5);
    expect(logitToProb(2)).toBeCloseTo(1 / (1 + Math.exp(-2)), 15);
    expect(() => logitToProb(Number.POSITIVE_INFINITY)).toThrow(RangeError);
  });
});
