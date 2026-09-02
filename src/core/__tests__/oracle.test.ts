// oracle.test.ts — 二実装照合(T-001 / T-002)
//
// 正 = fixtures.json(train_gan.py が丸め済み重みを numpy float64 で読み戻して再計算。
// 入力 z / 画像も配布形 7 桁丸め — HC-139)。TS は写し。
// 許容 1e-9 の出所: SPEC G-01/G-02(較正: Python 自己検算 5.0e-13 / 4.6e-13、
// 記録値の丸め 12 桁 = 床 5e-13 — 2026-09-03)

import { describe, expect, it } from "vitest";
import generatorRaw from "../model/generator.json";
import discriminatorRaw from "../model/discriminator.json";
import fixturesRaw from "../model/fixtures.json";
import { discriminate, generate, loadGenerator, loadDiscriminator } from "../model";

const TOL = 1e-9;

const g = loadGenerator(generatorRaw as Record<string, unknown>);
const d = loadDiscriminator(discriminatorRaw as Record<string, unknown>);
const fx = fixturesRaw as unknown as {
  z: number[][];
  pixels: number[][];
  d_inputs: number[][];
  d_logits: number[];
};

describe("G-01: Generator ピクセル照合", () => {
  it("フィクスチャが空でない(走査対象の実在 — HC-041)", () => {
    expect(fx.z.length).toBe(64);
    expect(fx.pixels.length).toBe(64);
  });

  it("64 潜在点 × 784 値が記録値と < 1e-9 で一致", () => {
    let maxErr = 0;
    for (let i = 0; i < fx.z.length; i++) {
      const out = generate(g, fx.z[i]);
      expect(out.length).toBe(784);
      for (let p = 0; p < 784; p++) {
        const e = Math.abs(out[p] - fx.pixels[i][p]);
        if (e > maxErr) maxErr = e;
      }
    }
    expect(maxErr).toBeLessThan(TOL);
  });
});

describe("G-02: Discriminator ロジット照合", () => {
  it("フィクスチャが空でない", () => {
    expect(fx.d_inputs.length).toBe(32);
    expect(fx.d_logits.length).toBe(32);
  });

  it("32 画像のロジットが記録値と < 1e-9 で一致", () => {
    let maxErr = 0;
    for (let i = 0; i < fx.d_inputs.length; i++) {
      const e = Math.abs(discriminate(d, fx.d_inputs[i]) - fx.d_logits[i]);
      if (e > maxErr) maxErr = e;
    }
    expect(maxErr).toBeLessThan(TOL);
  });

  it("陽性対照: 重みを 1 箇所だけ壊すと照合が落ちる(照合自体のテスト — HC-041)", () => {
    const broken = loadDiscriminator(discriminatorRaw as Record<string, unknown>);
    broken.fcB = Float64Array.from([broken.fcB[0] + 1e-6]);
    const e = Math.abs(discriminate(broken, fx.d_inputs[0]) - fx.d_logits[0]);
    expect(e).toBeGreaterThan(TOL);
  });
});
