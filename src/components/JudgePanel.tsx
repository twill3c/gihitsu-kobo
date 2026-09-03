"use client";

// JudgePanel — F-06 目利きの採点。本物の札(KMNIST テスト集合)と偽筆師の札を並べ、
// 目利き(Discriminator)の見立て(sigmoid(logit))をその場で出す。
// AUC(SPEC G-05)が示すとおり、均衡に達した目利きは完璧には見抜けない —
// それ自体が競い合いの証拠なので、外れる様子ごと見せる。

import { useMemo, useState } from "react";
import { discriminate, generate } from "@/core/model";
import { randomLatent } from "@/core/latent";
import { discriminatorModel, generatorModel, realSamples } from "@/lib/models";
import { GlyphCanvas } from "./GlyphCanvas";
import { ProbBar } from "./ProbBar";

export function JudgePanel() {
  const [realIdx, setRealIdx] = useState(0);
  const [fakeSeed, setFakeSeed] = useState(101);

  const real = realSamples[realIdx];
  const fake = useMemo(() => generate(generatorModel, randomLatent(fakeSeed)), [fakeSeed]);
  const realLogit = useMemo(() => discriminate(discriminatorModel, real), [real]);
  const fakeLogit = useMemo(() => discriminate(discriminatorModel, fake), [fake]);

  return (
    <div style={{ display: "flex", gap: "2rem", flexWrap: "wrap" }}>
      <figure style={{ margin: 0 }} data-testid="judge-real">
        <GlyphCanvas pixels={real} size={140} label="本物のくずし字" />
        <figcaption style={{ fontSize: "0.85rem", marginTop: "0.25rem" }}>
          本物(KMNIST テスト集合)
        </figcaption>
        <div style={{ marginTop: "0.5rem" }}>
          <ProbBar logit={realLogit} label="目利きの見立て" />
        </div>
        <button
          type="button"
          onClick={() => setRealIdx((i) => (i + 1) % realSamples.length)}
          style={{ marginTop: "0.5rem" }}
        >
          別の本物を出す
        </button>
      </figure>
      <figure style={{ margin: 0 }} data-testid="judge-fake">
        <GlyphCanvas pixels={fake} size={140} label="偽筆師の生成した文字" />
        <figcaption style={{ fontSize: "0.85rem", marginTop: "0.25rem" }}>
          偽筆師の作(いま生成)
        </figcaption>
        <div style={{ marginTop: "0.5rem" }}>
          <ProbBar logit={fakeLogit} label="目利きの見立て" />
        </div>
        <button type="button" onClick={() => setFakeSeed((s) => s + 1)} style={{ marginTop: "0.5rem" }}>
          別の偽物を描かせる
        </button>
      </figure>
    </div>
  );
}
