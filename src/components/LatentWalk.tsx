"use client";

// LatentWalk — F-05 潜在の散歩。二つの潜在点の線形補間で文字が連続変形する。
// 生成(generate)はポインタ操作・アニメの各フレームで実行(実測 ≈ 15ms — N-04)。

import { useEffect, useMemo, useRef, useState } from "react";
import { generate, discriminate } from "@/core/model";
import { lerpLatent, randomLatent } from "@/core/latent";
import { discriminatorModel, generatorModel } from "@/lib/models";
import { GlyphCanvas } from "./GlyphCanvas";
import { ProbBar } from "./ProbBar";

export function LatentWalk() {
  const [seedA, setSeedA] = useState(11);
  const [seedB, setSeedB] = useState(23);
  const [t, setT] = useState(0.0);
  const [walking, setWalking] = useState(false);
  const raf = useRef(0);

  const zA = useMemo(() => randomLatent(seedA), [seedA]);
  const zB = useMemo(() => randomLatent(seedB), [seedB]);
  // 端点はシードが変わったときだけ生成(モーフ中の毎フレーム再生成を避ける — N-04)
  const pxA = useMemo(() => generate(generatorModel, zA), [zA]);
  const pxB = useMemo(() => generate(generatorModel, zB), [zB]);
  const pixels = useMemo(() => generate(generatorModel, lerpLatent(zA, zB, t)), [zA, zB, t]);
  const logit = useMemo(() => discriminate(discriminatorModel, pixels), [pixels]);

  // 散歩: t を 0 ⇄ 1 に往復させる
  useEffect(() => {
    if (!walking) return;
    let dir = 1;
    let cur = 0;
    const step = () => {
      cur += dir * 0.02;
      if (cur >= 1) {
        cur = 1;
        dir = -1;
      } else if (cur <= 0) {
        cur = 0;
        dir = 1;
      }
      setT(Number(cur.toFixed(3)));
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current);
  }, [walking]);

  return (
    <div>
      <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
        <GlyphCanvas pixels={pxA} size={84} label="出発点の文字" />
        <GlyphCanvas pixels={pixels} size={168} label="いま生成している文字" />
        <GlyphCanvas pixels={pxB} size={84} label="到着点の文字" />
      </div>
      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", marginTop: "0.5rem", maxWidth: 448 }}>
        <button type="button" onClick={() => setWalking((w) => !w)} data-testid="walk-toggle">
          {walking ? "止める" : "歩かせる"}
        </button>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={t}
          onChange={(ev) => {
            setWalking(false);
            setT(Number(ev.target.value));
          }}
          aria-label="出発点と到着点のあいだの位置"
          style={{ flex: 1 }}
          data-testid="walk-slider"
        />
      </div>
      <div style={{ display: "flex", gap: "0.75rem", marginTop: "0.5rem" }}>
        <button type="button" onClick={() => setSeedA((s) => s + 1)}>
          出発点を振り直す
        </button>
        <button type="button" onClick={() => setSeedB((s) => s + 1)}>
          到着点を振り直す
        </button>
      </div>
      <div style={{ marginTop: "0.75rem" }}>
        <ProbBar logit={logit} label="この文字への目利きの見立て" />
      </div>
    </div>
  );
}
