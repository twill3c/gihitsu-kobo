// model.ts — 配布重み JSON の読込・形状検査(G-03)と generate / discriminate(F-03)
//
// 形状の正は SPEC F-02。JSON の実形状が宣言と食い違えば読み込みで投げる(黙って進まない)。
// 入力の縁(長さ不正・非有限値)は RangeError(N-05 / T-005 / T-006)。

import { conv2dSame2, convTransposeSame2, dense, leakyRelu, relu, sigmoid } from "./forward";

// F-02 で確定した層構成(2026-09-03)
export const LATENT_DIM = 64;
const G_DENSE_OUT = 7 * 7 * 64;
const LEAKY_ALPHA = 0.2;

export interface GeneratorModel {
  denseW: Float64Array; // (64, 3136) row-major
  denseB: Float64Array; // (3136,)
  ct1K: Float64Array;   // (4,4,32,64)
  ct1B: Float64Array;   // (32,)
  ct2K: Float64Array;   // (4,4,1,32)
  ct2B: Float64Array;   // (1,)
}

export interface DiscriminatorModel {
  c1K: Float64Array; // (4,4,1,32)
  c1B: Float64Array; // (32,)
  c2K: Float64Array; // (4,4,32,64)
  c2B: Float64Array; // (64,)
  fcW: Float64Array; // (3136, 1)
  fcB: Float64Array; // (1,)
}

type Nested = number | Nested[];

/** ネスト配列の形状を測りつつ row-major に平坦化する。ラグ配列(行長の不揃い)は投げる */
export function flattenChecked(v: Nested, expectShape: number[], name: string): Float64Array {
  const out = new Float64Array(expectShape.reduce((a, b) => a * b, 1));
  let n = 0;
  const walk = (node: Nested, depth: number): void => {
    if (depth === expectShape.length) {
      if (typeof node !== "number") throw new RangeError(`${name}: 深さ ${depth} に数値以外`);
      out[n++] = node;
      return;
    }
    if (!Array.isArray(node) || node.length !== expectShape[depth]) {
      const got = Array.isArray(node) ? node.length : typeof node;
      throw new RangeError(`${name}: 深さ ${depth} の長さ ${got} ≠ ${expectShape[depth]}`);
    }
    for (const child of node) walk(child, depth + 1);
  };
  walk(v, 0);
  return out;
}

function field(raw: Record<string, unknown>, key: string): Nested {
  if (!(key in raw)) throw new RangeError(`重み JSON に ${key} が無い`);
  return raw[key] as Nested;
}

export function loadGenerator(raw: Record<string, unknown>): GeneratorModel {
  if (raw["latent_dim"] !== LATENT_DIM) {
    throw new RangeError(`latent_dim ${String(raw["latent_dim"])} ≠ ${LATENT_DIM}`);
  }
  return {
    denseW: flattenChecked(field(raw, "dense_w"), [LATENT_DIM, G_DENSE_OUT], "dense_w"),
    denseB: flattenChecked(field(raw, "dense_b"), [G_DENSE_OUT], "dense_b"),
    ct1K: flattenChecked(field(raw, "ct1_k"), [4, 4, 32, 64], "ct1_k"),
    ct1B: flattenChecked(field(raw, "ct1_b"), [32], "ct1_b"),
    ct2K: flattenChecked(field(raw, "ct2_k"), [4, 4, 1, 32], "ct2_k"),
    ct2B: flattenChecked(field(raw, "ct2_b"), [1], "ct2_b"),
  };
}

export function loadDiscriminator(raw: Record<string, unknown>): DiscriminatorModel {
  return {
    c1K: flattenChecked(field(raw, "c1_k"), [4, 4, 1, 32], "c1_k"),
    c1B: flattenChecked(field(raw, "c1_b"), [32], "c1_b"),
    c2K: flattenChecked(field(raw, "c2_k"), [4, 4, 32, 64], "c2_k"),
    c2B: flattenChecked(field(raw, "c2_b"), [64], "c2_b"),
    fcW: flattenChecked(field(raw, "fc_w"), [3136, 1], "fc_w"),
    fcB: flattenChecked(field(raw, "fc_b"), [1], "fc_b"),
  };
}

function assertFiniteVector(x: ArrayLike<number>, len: number, name: string): void {
  if (x.length !== len) throw new RangeError(`${name} の長さ ${x.length} ≠ ${len}`);
  for (let i = 0; i < len; i++) {
    if (!Number.isFinite(x[i])) throw new RangeError(`${name}[${i}] が非有限値`);
  }
}

/** z(64 次元)→ 784 ピクセル([0,1]・28×28 row-major)。numpy np_generate の写し */
export function generate(m: GeneratorModel, z: ArrayLike<number>): Float64Array {
  assertFiniteVector(z, LATENT_DIM, "z");
  const zArr = Float64Array.from(z as ArrayLike<number>);
  let h = relu(dense(zArr, m.denseW, m.denseB, LATENT_DIM, G_DENSE_OUT)); // (7,7,64) 相当
  h = relu(convTransposeSame2(h, 7, 7, 64, m.ct1K, 4, 4, 32, m.ct1B));    // (14,14,32)
  h = sigmoid(convTransposeSame2(h, 14, 14, 32, m.ct2K, 4, 4, 1, m.ct2B)); // (28,28,1)
  return h;
}

/** 784 ピクセル([0,1])→ ロジット。numpy np_discriminate の写し */
export function discriminate(m: DiscriminatorModel, img: ArrayLike<number>): number {
  assertFiniteVector(img, 784, "img");
  const x = Float64Array.from(img as ArrayLike<number>);
  let h = leakyRelu(conv2dSame2(x, 28, 28, 1, m.c1K, 4, 4, 32, m.c1B), LEAKY_ALPHA); // (14,14,32)
  h = leakyRelu(conv2dSame2(h, 14, 14, 32, m.c2K, 4, 4, 64, m.c2B), LEAKY_ALPHA);    // (7,7,64)
  let acc = 0;
  for (let i = 0; i < 3136; i++) acc += h[i] * m.fcW[i];
  return acc + m.fcB[0]; // numpy 側と同順: dot の後に bias
}
