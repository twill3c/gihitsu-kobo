// models.ts — 配布重みの読込(モジュール評価時に一度だけ)と再輸出
//
// UI レイヤはここを通して core を使う(依存方向: lib → core)。

import generatorRaw from "@/core/model/generator.json";
import discriminatorRaw from "@/core/model/discriminator.json";
import fixturesRaw from "@/core/model/fixtures.json";
import metaRaw from "@/core/model/meta.json";
import { loadGenerator, loadDiscriminator } from "@/core/model";

export const generatorModel = loadGenerator(generatorRaw as Record<string, unknown>);
export const discriminatorModel = loadDiscriminator(discriminatorRaw as Record<string, unknown>);

const fx = fixturesRaw as unknown as { d_inputs: number[][] };

/** 目利きの採点に使う本物の札(KMNIST テスト集合の先頭 16 枚 — fixtures.json と同一物) */
export const realSamples: number[][] = fx.d_inputs.slice(0, 16);

export const meta = metaRaw as unknown as {
  epochs: number;
  snapshot_epochs: number;
  auc_trained: number;
  auc_untrained: number;
  trained_at: string;
  tensorflow: string;
};
