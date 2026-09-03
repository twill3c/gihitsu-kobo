"use client";

// SnapshotPlayer — F-04 学習の巻き戻しスライダー。
// epoch ごとの固定潜在 8×8 格子(public/snapshots/epoch_XXX.png)をスライダーと再生で行き来する。
// 画像は事前計算(train_gan.py の出力)。ブラウザでは選ぶだけで、生成はしない。

import { useEffect, useRef, useState } from "react";

export function SnapshotPlayer({ epochs }: { epochs: number }) {
  const [epoch, setEpoch] = useState(epochs);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  // 再生: 1 → epochs まで進んで止まる
  useEffect(() => {
    if (!playing) return;
    timer.current = setInterval(() => {
      setEpoch((e) => {
        if (e >= epochs) {
          setPlaying(false);
          return e;
        }
        return e + 1;
      });
    }, 220);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [playing, epochs]);

  return (
    <div>
      {/* 全 epoch を DOM に置き、表示だけ切り替える(スライダー操作でネットワーク待ちを作らない) */}
      <div style={{ position: "relative", maxWidth: 448, aspectRatio: "1" }}>
        {Array.from({ length: epochs }, (_, i) => i + 1).map((e) => (
          // eslint-disable-next-line @next/next/no-img-element -- 静的エクスポートでは next/image の最適化サーバが無い。事前計算 PNG をそのまま出す
          <img
            key={e}
            src={`/snapshots/epoch_${String(e).padStart(3, "0")}.png`}
            alt={e === epoch ? `第 ${e} epoch の生成標本 64 枚` : ""}
            aria-hidden={e !== epoch}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              imageRendering: "pixelated",
              visibility: e === epoch ? "visible" : "hidden",
            }}
          />
        ))}
      </div>
      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", marginTop: "0.5rem", maxWidth: 448 }}>
        <button type="button" onClick={() => setPlaying((p) => !p)} data-testid="play-toggle">
          {playing ? "止める" : "初めから見る"}
        </button>
        <input
          type="range"
          min={1}
          max={epochs}
          value={epoch}
          onChange={(ev) => {
            setPlaying(false);
            setEpoch(Number(ev.target.value));
          }}
          aria-label="epoch を選ぶ"
          style={{ flex: 1 }}
          data-testid="epoch-slider"
        />
        <span style={{ minWidth: "5.5rem", textAlign: "right" }} data-testid="epoch-label">
          第 {epoch} / {epochs} 期
        </span>
      </div>
      <p style={{ fontSize: "0.85rem", color: "#6b6250" }}>
        64 枚とも同じ潜在ベクトルから描いている。変わっていくのは偽筆師の腕だけ。
      </p>
    </div>
  );
}
