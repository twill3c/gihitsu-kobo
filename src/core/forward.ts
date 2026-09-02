// forward.ts — 手書き forward 演算(F-03)
//
// train_gan.py の numpy float64 forward の写し。演算順・パディング規則・
// 平坦化順(row-major)を Python 側と揃える。照合は G-01/G-02(< 1e-9)。
// 重みのメモリ配置は TF のまま: convT カーネル (kh,kw,cout,cin)、conv カーネル (kh,kw,cin,cout)。

export function relu(x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
  return out;
}

export function leakyRelu(x: Float64Array, alpha: number): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : alpha * x[i];
  return out;
}

export function sigmoid(x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = 1 / (1 + Math.exp(-x[i]));
  return out;
}

/** dense: out[j] = Σ_i x[i]·w[i·cols + j] + b[j](w は (rows, cols) row-major) */
export function dense(x: Float64Array, w: Float64Array, b: Float64Array, rows: number, cols: number): Float64Array {
  const out = new Float64Array(b);
  for (let i = 0; i < rows; i++) {
    const xi = x[i];
    const base = i * cols;
    for (let j = 0; j < cols; j++) out[j] += xi * w[base + j];
  }
  return out;
}

/**
 * TF Conv2DTranspose(strides=2, padding='same') と等価。
 * x: (h,w,cin) row-major、k: (kh,kw,cout,cin)。出力 (2h,2w,cout)。
 * scatter-add で全出力 ((h-1)·2+kh) を作り、pad_beg=(kh-2)/2 で切り出す(train_gan.py と同順)。
 */
export function convTransposeSame2(
  x: Float64Array, h: number, w: number, cin: number,
  k: Float64Array, kh: number, kw: number, cout: number,
  b: Float64Array,
): Float64Array {
  const s = 2;
  const fullH = (h - 1) * s + kh;
  const fullW = (w - 1) * s + kw;
  const full = new Float64Array(fullH * fullW * cout);
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      const xBase = (i * w + j) * cin;
      for (let a = 0; a < kh; a++) {
        for (let bb = 0; bb < kw; bb++) {
          const oBase = ((i * s + a) * fullW + (j * s + bb)) * cout;
          const kBase = (a * kw + bb) * cout * cin;
          for (let c = 0; c < cout; c++) {
            let acc = 0;
            const kcBase = kBase + c * cin;
            for (let d = 0; d < cin; d++) acc += k[kcBase + d] * x[xBase + d];
            full[oBase + c] += acc;
          }
        }
      }
    }
  }
  const pbH = (kh - s) >> 1;
  const pbW = (kw - s) >> 1;
  const oh = h * s;
  const ow = w * s;
  const out = new Float64Array(oh * ow * cout);
  for (let i = 0; i < oh; i++) {
    for (let j = 0; j < ow; j++) {
      const src = ((i + pbH) * fullW + (j + pbW)) * cout;
      const dst = (i * ow + j) * cout;
      for (let c = 0; c < cout; c++) out[dst + c] = full[src + c] + b[c];
    }
  }
  return out;
}

/**
 * TF Conv2D(strides=2, padding='same') と等価。
 * x: (h,w,cin) row-major、k: (kh,kw,cin,cout)。出力 (h/2,w/2,cout)。
 * pad_total = (h/2-1)·2 + kh - h、pad_beg = pad_total/2(train_gan.py と同順)。
 */
export function conv2dSame2(
  x: Float64Array, h: number, w: number, cin: number,
  k: Float64Array, kh: number, kw: number, cout: number,
  b: Float64Array,
): Float64Array {
  const s = 2;
  const oh = Math.floor(h / s);
  const ow = Math.floor(w / s);
  const padH = Math.max((oh - 1) * s + kh - h, 0);
  const padW = Math.max((ow - 1) * s + kw - w, 0);
  const pbH = padH >> 1;
  const pbW = padW >> 1;
  const out = new Float64Array(oh * ow * cout);
  for (let i = 0; i < oh; i++) {
    for (let j = 0; j < ow; j++) {
      const oBase = (i * ow + j) * cout;
      for (let a = 0; a < kh; a++) {
        const xi = i * s + a - pbH;
        if (xi < 0 || xi >= h) continue;
        for (let bb = 0; bb < kw; bb++) {
          const xj = j * s + bb - pbW;
          if (xj < 0 || xj >= w) continue;
          const xBase = (xi * w + xj) * cin;
          const kBase = (a * kw + bb) * cin * cout;
          for (let c = 0; c < cin; c++) {
            const xv = x[xBase + c];
            const kcBase = kBase + c * cout;
            for (let d = 0; d < cout; d++) out[oBase + d] += xv * k[kcBase + d];
          }
        }
      }
      for (let d = 0; d < cout; d++) out[oBase + d] += b[d];
    }
  }
  return out;
}
