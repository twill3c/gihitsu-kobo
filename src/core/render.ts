// render.ts — ピクセル列の描画用変換(純関数部)
//
// generate の出力([0,1]・784)を RGBA へ写す。Canvas API は触らない(N-02 / HC-002)。
// 非有限値は RangeError(N-05)— generate の値域保証の外から来た入力を黙って描かない。

/** [0,1] の 784 値 → 墨色グレースケール RGBA(白地に黒 v=1 が濃い墨) */
export function pixelsToRgba(pixels: ArrayLike<number>): Uint8ClampedArray<ArrayBuffer> {
  if (pixels.length !== 784) throw new RangeError(`長さ ${pixels.length} ≠ 784`);
  const out = new Uint8ClampedArray(new ArrayBuffer(784 * 4));
  for (let i = 0; i < 784; i++) {
    const v = pixels[i];
    if (!Number.isFinite(v)) throw new RangeError(`pixels[${i}] が非有限値`);
    // KMNIST は黒地に白筆。紙に墨で見せるため反転する(1 → 墨、0 → 紙)
    const ink = Math.round(255 * (1 - Math.min(1, Math.max(0, v))));
    out[i * 4] = ink;
    out[i * 4 + 1] = ink;
    out[i * 4 + 2] = ink;
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** ロジット → 「本物らしさ」確率(Discriminator の sigmoid 出力そのもの) */
export function logitToProb(logit: number): number {
  if (!Number.isFinite(logit)) throw new RangeError("logit が非有限値");
  return 1 / (1 + Math.exp(-logit));
}
