"use client";

// ProbBar — 目利き(Discriminator)の見立てを示す横棒。
// 表示する数は sigmoid(logit) = Discriminator が「本物」と見立てる確率そのもの(HC-079:
// 記号は仕様の述語に対応させる。良し悪しの色は付けず、見立ての強さだけを長さで見せる)。

import { logitToProb } from "@/core/render";

export function ProbBar({ logit, label }: { logit: number; label: string }) {
  const p = logitToProb(logit);
  return (
    <div style={{ maxWidth: 448 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          flexWrap: "wrap",
          columnGap: "1rem",
          fontSize: "0.85rem",
        }}
      >
        <span>{label}</span>
        <span data-testid="prob-value" style={{ whiteSpace: "nowrap" }}>
          本物らしさ {(p * 100).toFixed(1)}%(logit {logit.toFixed(2)})
        </span>
      </div>
      <div style={{ height: 10, background: "#e4dbc8", borderRadius: 5, overflow: "hidden" }}>
        <div
          className="prob-fill"
          style={{ width: `${(p * 100).toFixed(1)}%`, height: "100%", background: "#4a4336" }}
        />
      </div>
    </div>
  );
}
