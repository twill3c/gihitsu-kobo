"use client";

// GlyphCanvas — 784 ピクセル([0,1])を 28×28 → 拡大表示する canvas。
// Canvas API は effect 内のみ(HC-002)。

import { useEffect, useRef } from "react";
import { pixelsToRgba } from "@/core/render";

export function GlyphCanvas({
  pixels,
  size = 140,
  label,
}: {
  pixels: ArrayLike<number>;
  size?: number;
  label?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(new ImageData(pixelsToRgba(pixels), 28, 28), 0, 0);
  }, [pixels]);

  return (
    <canvas
      ref={ref}
      width={28}
      height={28}
      role="img"
      aria-label={label ?? "生成された文字"}
      style={{
        width: size,
        height: size,
        imageRendering: "pixelated",
        border: "1px solid #c9bfa8",
        background: "#fff",
      }}
    />
  );
}
