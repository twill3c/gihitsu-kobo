// timing.test.ts — N-04(T-009): 生成・採点がポインタ操作に追随できる速さであること
//
// 実測(2026-09-03・vitest 内): generate 平均 ≈ 15ms / discriminate ≈ 11ms
// (oracle.test.ts の 64 回 963ms / 32 回 350ms からの導出と本テストの実測が同水準)。
// 閾値 60ms は実測の 4 倍マージン — CI・低速機のゆらぎ吸収のためで、体感要求は
// 「モーフ 30fps(33ms)・単発更新に停滞感がない」水準(SPEC N-04)。

import { describe, expect, it } from "vitest";
import generatorRaw from "../model/generator.json";
import discriminatorRaw from "../model/discriminator.json";
import { discriminate, generate, LATENT_DIM, loadGenerator, loadDiscriminator } from "../model";

const g = loadGenerator(generatorRaw as Record<string, unknown>);
const d = loadDiscriminator(discriminatorRaw as Record<string, unknown>);

function meanMs(fn: () => void, n: number): number {
  fn(); // ウォームアップ(JIT)
  const t0 = performance.now();
  for (let i = 0; i < n; i++) fn();
  return (performance.now() - t0) / n;
}

describe("N-04: 所要時間", () => {
  it("generate 平均 < 60ms(実測 ≈ 15ms — 2026-09-03)", () => {
    const z = Array.from({ length: LATENT_DIM }, (_, i) => Math.sin(i * 1.7));
    expect(meanMs(() => generate(g, z), 20)).toBeLessThan(60);
  });

  it("discriminate 平均 < 60ms(実測 ≈ 11ms — 2026-09-03)", () => {
    const img = new Array<number>(784).fill(0).map((_, i) => (i % 29) / 29);
    expect(meanMs(() => discriminate(d, img), 20)).toBeLessThan(60);
  });
});
