// latent.ts — 潜在ベクトルの生成・補間(F-05)
//
// 乱数は seed 決定的(mulberry32 + Box-Muller)。UI の「振り直し」も seed を進めるだけなので
// 同じ seed からは同じ散歩が再現できる(テスト可能性のため乱数源を core に置く)。

import { LATENT_DIM } from "./model";

/** mulberry32 — 32bit seed の決定的 PRNG([0,1)) */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller による標準正規乱数の潜在ベクトル。同じ seed → 同じベクトル */
export function randomLatent(seed: number): Float64Array {
  const rand = mulberry32(seed);
  const z = new Float64Array(LATENT_DIM);
  for (let i = 0; i < LATENT_DIM; i += 2) {
    // u1 は 0 を避ける(log(0) = -Inf を作らない — N-05)
    const u1 = 1 - rand();
    const u2 = rand();
    const r = Math.sqrt(-2 * Math.log(u1));
    z[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < LATENT_DIM) z[i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  return z;
}

/** 線形補間 z = a·(1-t) + b·t。t は [0,1] へクランプ、非有限 t は RangeError(N-05) */
export function lerpLatent(a: Float64Array, b: Float64Array, t: number): Float64Array {
  if (!Number.isFinite(t)) throw new RangeError("t が非有限値");
  if (a.length !== b.length) throw new RangeError(`長さ不一致 ${a.length} ≠ ${b.length}`);
  const tc = t < 0 ? 0 : t > 1 ? 1 : t;
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * (1 - tc) + b[i] * tc;
  return out;
}
