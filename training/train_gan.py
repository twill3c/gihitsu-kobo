# train_gan.py — KMNIST DCGAN 学習パイプライン(F-01/F-02)
#
# 手元専用(N-03)。出力はすべて丸め済み JSON / PNG の静的成果物。
# 契約(SPEC §4): 照合の正は「丸め済み重みを numpy float64 で読み戻した forward」。
# TF(float32)との比較は G-04(BN 畳み込みの等価性)にのみ使い、許容誤差は別に持つ。

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LATENT_DIM = 64
SEED = 42
ROUND_DECIMALS = 7


# ---------------------------------------------------------------- データ

def load_kmnist(data_dir: Path):
    train = np.load(data_dir / "kmnist-train-imgs.npz")["arr_0"]  # (60000,28,28) uint8
    test = np.load(data_dir / "kmnist-test-imgs.npz")["arr_0"]    # (10000,28,28) uint8
    return train, test


# ---------------------------------------------------------------- モデル

def build_models(tf):
    from tensorflow.keras import layers

    g = tf.keras.Sequential(
        [
            layers.Input(shape=(LATENT_DIM,)),
            layers.Dense(7 * 7 * 64, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(32, 4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(1, 4, strides=2, padding="same", activation="sigmoid"),
        ],
        name="generator",
    )
    d = tf.keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, 4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(64, 4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dense(1),
        ],
        name="discriminator",
    )
    return g, d


# ---------------------------------------------------------------- BN 畳み込み

def fold_generator(g):
    """BatchNorm を線形層へ畳み込み、推論等価な重み辞書を返す(float64)。

    dense(バイアス無) + BN: W' = W * s, b' = beta - mean * s  (s = gamma/sqrt(var+eps))
    convT(バイアス無) + BN: カーネルの出力チャネル軸に s を掛ける。
    """
    # 位置 index は Input の数え方で壊れる(loop_001 GEN-LOGIC)。型で引き当て、数を検算する
    from tensorflow.keras import layers as kl

    denses = [l for l in g.layers if isinstance(l, kl.Dense)]
    bns = [l for l in g.layers if isinstance(l, kl.BatchNormalization)]
    cts = [l for l in g.layers if isinstance(l, kl.Conv2DTranspose)]
    assert (len(denses), len(bns), len(cts)) == (1, 2, 2), (denses, bns, cts)
    (dw,) = [w for w in denses[0].weights if "kernel" in w.name]
    bn1, bn2 = bns
    ct1, ct2 = cts

    def bn_params(bn):
        gamma, beta, mean, var = [v.numpy().astype(np.float64) for v in bn.weights]
        s = gamma / np.sqrt(var + bn.epsilon)
        return s, beta - mean * s

    s1, b1 = bn_params(bn1)
    dense_w = dw.numpy().astype(np.float64) * s1  # (64, 3136) * (3136,)

    s2, b2 = bn_params(bn2)
    ct1_k = ct1.weights[0].numpy().astype(np.float64)  # (4,4,out=32,in=64)
    ct1_k = ct1_k * s2[None, None, :, None]

    ct2_k = ct2.weights[0].numpy().astype(np.float64)  # (4,4,out=1,in=32)
    ct2_b = ct2.weights[1].numpy().astype(np.float64)

    return {
        "latent_dim": LATENT_DIM,
        "dense_w": dense_w,   # (64, 3136)
        "dense_b": b1,        # (3136,)
        "ct1_k": ct1_k,       # (4,4,32,64)
        "ct1_b": b2,          # (32,)
        "ct2_k": ct2_k,       # (4,4,1,32)
        "ct2_b": ct2_b,       # (1,)
    }


def extract_discriminator(d):
    ws = [w.numpy().astype(np.float64) for w in d.weights]
    return {
        "c1_k": ws[0],  # (4,4,1,32)
        "c1_b": ws[1],
        "c2_k": ws[2],  # (4,4,32,64)
        "c2_b": ws[3],
        "fc_w": ws[4],  # (3136,1)
        "fc_b": ws[5],
    }


def round_weights(wd):
    return {k: (np.round(v, ROUND_DECIMALS) if isinstance(v, np.ndarray) else v) for k, v in wd.items()}


# ---------------------------------------------------------------- numpy forward(照合の正)

def np_conv2d_transpose_same_s2(x, k, b):
    """TF の Conv2DTranspose(strides=2, padding='same') と等価な forward。

    x: (H,W,Cin), k: (kh,kw,Cout,Cin) — TF の重み配置のまま。出力 (2H,2W,Cout)。
    scatter-add で全出力 ((H-1)*2+kh) を作り、TF の 'same' 規則
    pad_beg = (kh - 2) // 2 で切り出す。等価性は G-04 で TF 出力と照合して確かめる。
    """
    h, w, cin = x.shape
    kh, kw, cout, _ = k.shape
    s = 2
    full_h, full_w = (h - 1) * s + kh, (w - 1) * s + kw
    out = np.zeros((full_h, full_w, cout), dtype=np.float64)
    # (kh,kw,cout,cin) x (cin,) の縮約を全入力画素で行う
    for i in range(h):
        for j in range(w):
            out[i * s : i * s + kh, j * s : j * s + kw, :] += np.einsum(
                "abcd,d->abc", k, x[i, j, :]
            )
    pb_h = (kh - s) // 2
    pb_w = (kw - s) // 2
    return out[pb_h : pb_h + h * s, pb_w : pb_w + w * s, :] + b


def np_conv2d_same_s2(x, k, b):
    """TF の Conv2D(strides=2, padding='same') と等価な forward。

    x: (H,W,Cin), k: (kh,kw,Cin,Cout)。出力 (H/2,W/2,Cout)。
    """
    h, w, cin = x.shape
    kh, kw, _, cout = k.shape
    s = 2
    oh, ow = h // s, w // s
    pad_h = max((oh - 1) * s + kh - h, 0)
    pad_w = max((ow - 1) * s + kw - w, 0)
    pb_h, pb_w = pad_h // 2, pad_w // 2
    xp = np.zeros((h + pad_h, w + pad_w, cin), dtype=np.float64)
    xp[pb_h : pb_h + h, pb_w : pb_w + w, :] = x
    out = np.empty((oh, ow, cout), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            patch = xp[i * s : i * s + kh, j * s : j * s + kw, :]
            out[i, j, :] = np.einsum("abc,abcd->d", patch, k)
    return out + b


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def np_generate(wd, z):
    """丸め済み重み(float64)による Generator forward。z: (latent_dim,) → (784,)"""
    h = np.maximum(z @ wd["dense_w"] + wd["dense_b"], 0.0)
    h = h.reshape(7, 7, 64)
    h = np.maximum(np_conv2d_transpose_same_s2(h, wd["ct1_k"], wd["ct1_b"]), 0.0)
    h = sigmoid(np_conv2d_transpose_same_s2(h, wd["ct2_k"], wd["ct2_b"]))
    return h.reshape(784)


def np_discriminate(wd, img784):
    """丸め済み重み(float64)による Discriminator forward。→ ロジット(スカラー)"""
    x = img784.reshape(28, 28, 1)
    x = np_conv2d_same_s2(x, wd["c1_k"], wd["c1_b"])
    x = np.where(x > 0, x, 0.2 * x)
    x = np_conv2d_same_s2(x, wd["c2_k"], wd["c2_b"])
    x = np.where(x > 0, x, 0.2 * x)
    return float(x.reshape(-1) @ wd["fc_w"][:, 0] + wd["fc_b"][0])


# ---------------------------------------------------------------- 学習

def train(tf, g, d, train_imgs, epochs, batch_size, snapshot_dir, grid_z):
    x = (train_imgs.astype(np.float32) / 255.0)[..., None]  # [0,1] — G 出力 sigmoid と揃える
    ds = (
        tf.data.Dataset.from_tensor_slices(x)
        .shuffle(60000, seed=SEED, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
    )
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    @tf.function
    def step(real):
        n = tf.shape(real)[0]
        z = tf.random.normal((n, LATENT_DIM))
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = g(z, training=True)
            real_logit = d(real, training=True)
            fake_logit = d(fake, training=True)
            d_loss = bce(tf.ones_like(real_logit), real_logit) + bce(
                tf.zeros_like(fake_logit), fake_logit
            )
            g_loss = bce(tf.ones_like(fake_logit), fake_logit)
        d_opt.apply_gradients(zip(dt.gradient(d_loss, d.trainable_variables), d.trainable_variables))
        g_opt.apply_gradients(zip(gt.gradient(g_loss, g.trainable_variables), g.trainable_variables))
        return d_loss, g_loss

    times = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        d_l = g_l = 0.0
        nb = 0
        for real in ds:
            dl, gl = step(real)
            d_l += float(dl)
            g_l += float(gl)
            nb += 1
        dt_s = time.time() - t0
        times.append(dt_s)
        save_snapshot(g, grid_z, snapshot_dir / f"epoch_{epoch:03d}.png")
        print(f"epoch {epoch}: d_loss={d_l/nb:.4f} g_loss={g_l/nb:.4f} ({dt_s:.1f}s)", flush=True)
    return times


def save_snapshot(g, grid_z, path):
    from PIL import Image

    imgs = g(grid_z, training=False).numpy()  # (64,28,28,1) in [0,1]
    sheet = np.zeros((8 * 28, 8 * 28), dtype=np.uint8)
    for idx in range(64):
        r, c = divmod(idx, 8)
        sheet[r * 28 : (r + 1) * 28, c * 28 : (c + 1) * 28] = np.clip(
            imgs[idx, :, :, 0] * 255.0, 0, 255
        ).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet, mode="L").save(path, optimize=True)


# ---------------------------------------------------------------- 出力・照合

def dump_json(obj, path):
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=default, separators=(",", ":"))
    return path.stat().st_size


def readback(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v) for k, v in raw.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE.parent / "src" / "core" / "model")
    ap.add_argument("--snapshots", type=Path, default=HERE.parent / "public" / "snapshots")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    import tensorflow as tf

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    print(f"tensorflow {tf.__version__} / numpy {np.__version__}", flush=True)

    train_imgs, test_imgs = load_kmnist(args.data)
    g, d = build_models(tf)

    rng = np.random.default_rng(SEED)
    grid_z = tf.constant(rng.standard_normal((64, LATENT_DIM)).astype(np.float32))

    times = train(tf, g, d, train_imgs, args.epochs, args.batch_size, args.snapshots, grid_z)
    print(f"epoch time: mean {np.mean(times):.1f}s / total {sum(times):.1f}s", flush=True)

    # --- 重みの畳み込み → 丸め → 書き出し → 読み戻し
    g_wd = round_weights(fold_generator(g))
    d_wd = round_weights(extract_discriminator(d))
    g_size = dump_json(g_wd, args.out / "generator.json")
    d_size = dump_json(d_wd, args.out / "discriminator.json")
    print(f"generator.json {g_size/1024:.0f} KB / discriminator.json {d_size/1024:.0f} KB", flush=True)

    g_rb = readback(args.out / "generator.json")
    d_rb = readback(args.out / "discriminator.json")

    # --- G-04: BN 畳み込みの等価性(丸め済み numpy forward vs TF 推論モード)
    z_check = rng.standard_normal((16, LATENT_DIM))
    tf_out = g(tf.constant(z_check.astype(np.float32)), training=False).numpy().reshape(16, 784)
    np_out = np.stack([np_generate(g_rb, z) for z in z_check])
    g04_err = float(np.max(np.abs(tf_out.astype(np.float64) - np_out)))
    print(f"G-04 max|TF - numpy(folded,rounded)| = {g04_err:.3e}", flush=True)

    d_imgs = np.concatenate([test_imgs[:8].astype(np.float64).reshape(8, 784) / 255.0, np_out[:8]])
    tf_logits = d(tf.constant(d_imgs.reshape(-1, 28, 28, 1).astype(np.float32)), training=False).numpy()[:, 0]
    np_logits = np.array([np_discriminate(d_rb, im) for im in d_imgs])
    d_err = float(np.max(np.abs(tf_logits.astype(np.float64) - np_logits)))
    print(f"G-04(D) max|TF - numpy(rounded)| = {d_err:.3e}", flush=True)

    # --- 照合フィクスチャ(正 = numpy float64 読み戻し forward)
    # 入力(z・画像)は先に配布形(丸め済み)へ落とし、出力はその実物から導出する。
    # 丸め前の中間値から計算すると照合が原理的に不成立になる(loop_001 GEN-LOGIC ×2)
    fx_z = np.round(rng.standard_normal((64, LATENT_DIM)), ROUND_DECIMALS)
    fx_pixels = np.stack([np_generate(g_rb, z) for z in fx_z])
    fx_imgs = np.round(
        np.concatenate([test_imgs[:16].astype(np.float64).reshape(16, 784) / 255.0, fx_pixels[:16]]),
        ROUND_DECIMALS,
    )
    fx_logits = np.array([np_discriminate(d_rb, im) for im in fx_imgs])
    fixtures = {
        "seed": SEED,
        "z": fx_z,
        "pixels": np.round(fx_pixels, 12),
        "d_inputs": fx_imgs,
        "d_logits": np.round(fx_logits, 12),
        "g04_max_abs_err": g04_err,
        "g04_d_max_abs_err": d_err,
    }
    fx_size = dump_json(fixtures, args.out / "fixtures.json")
    print(f"fixtures.json {fx_size/1024:.0f} KB", flush=True)

    # 検算: 書き出した実物(読み戻し)だけから再計算し、記録値と一致することを確かめる。
    # 許容 1e-12 は記録側の丸め(12 桁)のみに由来する
    fx_rb = readback(args.out / "fixtures.json")
    g_self = float(np.max(np.abs(np.stack([np_generate(g_rb, z) for z in fx_rb["z"]]) - fx_rb["pixels"])))
    d_self = float(np.max(np.abs(np.array([np_discriminate(d_rb, im) for im in fx_rb["d_inputs"]]) - fx_rb["d_logits"])))
    print(f"fixture self-check G max err = {g_self:.3e} / D max err = {d_self:.3e}", flush=True)
    assert g_self < 1e-12 and d_self < 1e-12, (g_self, d_self)


if __name__ == "__main__":
    main()
