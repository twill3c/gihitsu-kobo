// model.test.ts — 形状・決定論(T-003 / T-004)と縁(T-005 / T-006)
//
// 形状の正は SPEC F-02(2026-09-03 確定)。壊した入力で投げることの陽性対照つき(HC-041)。

import { describe, expect, it } from "vitest";
import generatorRaw from "../model/generator.json";
import discriminatorRaw from "../model/discriminator.json";
import { discriminate, generate, LATENT_DIM, loadGenerator, loadDiscriminator } from "../model";

const g = loadGenerator(generatorRaw as Record<string, unknown>);
const d = loadDiscriminator(discriminatorRaw as Record<string, unknown>);

describe("G-03: 形状", () => {
  it("配布 JSON が F-02 の形状で読み込める(平坦化後の総数で確認)", () => {
    expect(g.denseW.length).toBe(64 * 3136);
    expect(g.ct1K.length).toBe(4 * 4 * 32 * 64);
    expect(g.ct2K.length).toBe(4 * 4 * 1 * 32);
    expect(d.c1K.length).toBe(4 * 4 * 1 * 32);
    expect(d.c2K.length).toBe(4 * 4 * 32 * 64);
    expect(d.fcW.length).toBe(3136);
  });

  it("陽性対照: 形状を壊した JSON は読み込みで投げる", () => {
    const broken = JSON.parse(JSON.stringify(generatorRaw)) as Record<string, unknown>;
    (broken["dense_b"] as number[]).pop();
    expect(() => loadGenerator(broken)).toThrow(RangeError);
    const wrongLatent = { ...(generatorRaw as Record<string, unknown>), latent_dim: 2 };
    expect(() => loadGenerator(wrongLatent)).toThrow(RangeError);
    const missing = { ...(discriminatorRaw as Record<string, unknown>) };
    delete missing["fc_w"];
    expect(() => loadDiscriminator(missing)).toThrow(RangeError);
  });
});

describe("G-03: 決定論(T-004)", () => {
  it("同一 z の generate 2 回が深い等値", () => {
    const z = Array.from({ length: LATENT_DIM }, (_, i) => Math.sin(i + 1)); // 決定的な z
    expect(generate(g, z)).toEqual(generate(g, z));
  });

  it("出力は [0,1](sigmoid の値域 — F-02)", () => {
    const z = Array.from({ length: LATENT_DIM }, (_, i) => Math.cos(i));
    const out = generate(g, z);
    for (const v of out) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});

describe("N-05: 縁は正常系(T-005 / T-006)", () => {
  it("z の長さ不正・非有限値は RangeError", () => {
    expect(() => generate(g, [0, 1])).toThrow(RangeError);
    const z = new Array<number>(LATENT_DIM).fill(0);
    z[3] = Number.NaN;
    expect(() => generate(g, z)).toThrow(RangeError);
    z[3] = Number.POSITIVE_INFINITY;
    expect(() => generate(g, z)).toThrow(RangeError);
  });

  it("discriminate 入力の長さ不正・非有限値は RangeError", () => {
    expect(() => discriminate(d, [0.5])).toThrow(RangeError);
    const img = new Array<number>(784).fill(0.5);
    img[100] = Number.NEGATIVE_INFINITY;
    expect(() => discriminate(d, img)).toThrow(RangeError);
  });

  it("対照: 正常な入力では投げない", () => {
    const z = new Array<number>(LATENT_DIM).fill(0.1);
    expect(() => generate(g, z)).not.toThrow();
    const img = new Array<number>(784).fill(0.5);
    expect(() => discriminate(d, img)).not.toThrow();
  });
});
