// meta.test.ts — Python 側実測の写しの検査(T-007 / T-008)
//
// meta.json は train_gan.py が学習直後に numpy forward で計測した値。
// 閾値の出所: SPEC §4(較正実測 — 2026-09-03: g04 1.639e-06 / g04_d 6.991e-06、
// auc_trained 0.7153 / auc_untrained 0.5354)

import { describe, expect, it } from "vitest";
import metaRaw from "../model/meta.json";

const meta = metaRaw as unknown as {
  latent_dim: number;
  epochs: number;
  snapshot_epochs: number;
  g04_max_abs_err: number;
  g04_d_max_abs_err: number;
  auc_trained: number;
  auc_untrained: number;
  auc_n: number;
};

describe("G-04: BN 畳み込みの等価性(Python 側実測の検査)", () => {
  it("Generator: TF 推論との最大絶対誤差 < 5e-6", () => {
    expect(meta.g04_max_abs_err).toBeGreaterThan(0); // 恒等比較ではないことの対照
    expect(meta.g04_max_abs_err).toBeLessThan(5e-6);
  });

  it("Discriminator: ロジットの最大絶対誤差 < 1e-4", () => {
    expect(meta.g04_d_max_abs_err).toBeGreaterThan(0);
    expect(meta.g04_d_max_abs_err).toBeLessThan(1e-4);
  });
});

describe("G-05: 目利きの弁別力と対照", () => {
  it("学習済み AUC > 0.65(実測 0.7153 — 2026-09-03)", () => {
    expect(meta.auc_n).toBeGreaterThanOrEqual(500);
    expect(meta.auc_trained).toBeGreaterThan(0.65);
  });

  it("対照: 未学習 AUC < 0.60(実測 0.5354 — 2026-09-03)。学習との差 > 0.10", () => {
    expect(meta.auc_untrained).toBeLessThan(0.6);
    expect(meta.auc_trained - meta.auc_untrained).toBeGreaterThan(0.1);
  });
});

describe("整合: meta とモデル定数", () => {
  it("latent_dim=64・snapshot_epochs は epochs と一致(F-04 のスライダーが参照)", () => {
    expect(meta.latent_dim).toBe(64);
    expect(meta.snapshot_epochs).toBe(meta.epochs);
  });
});
