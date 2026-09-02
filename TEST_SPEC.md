# TEST_SPEC.md — gihitsu-kobo(偽筆工房)

<!-- scaffold template v1.24.0 から展開(2026-09-02) -->

## 実行規約

- `npm run verify:fast`(typecheck + lint + vitest)を stage 3–5 の判定に使用。完了条件は `npm run verify`(+ next build)
- フィクスチャ更新(モデル資産の再学習)は専用コミットで行い、理由をループログに記す
- モデル資産(`src/core/model/*.json`)は train_gan.py だけが生成する。テストはこれを読み取り専用で使う

## 期待値の出所(HC-016)

| 出所 | 書き方 |
|---|---|
| SPEC の条項 | 条項 ID を書く。SPEC の保証粒度を超える期待値を書かない |
| 実測 | 実測日と実測値をコメントに残す |

## オラクルの出所

| フィクスチャ | 出所 | 性格 |
|---|---|---|
| `src/core/model/fixtures.json` | train_gan.py が丸め済み重みを numpy float64 で読み戻して再計算(HC-139 の導出契約: 入力も配布形から) | 二実装照合の正。TS は写し |
| `src/core/model/generator.json` / `discriminator.json` | train_gan.py(BN 畳み込み+ 7 桁丸め) | 配布重み。形状は F-02 が正 |
| `src/core/model/meta.json` | train_gan.py が学習直後に numpy forward で計測(AUC・G-04 誤差) | Python 側実測の写し。閾値の根拠は SPEC §4 の較正記録 |

## ケース一覧

| ID | 対応要求 | ケース | 期待 |
|---|---|---|---|
| T-001 | G-01 | フィクスチャ 64 潜在点の generate | 各 784 値が記録値と最大絶対誤差 < 1e-9 |
| T-002 | G-02 | フィクスチャ 32 画像の discriminate | ロジットが記録値と最大絶対誤差 < 1e-9 |
| T-003 | G-03 | 重み JSON の形状 | F-02 の層構成から導出される形状と一致(latent_dim=64, dense 64×3136, ct1 4×4×32×64, ct2 4×4×1×32, c1 4×4×1×32, c2 4×4×32×64, fc 3136×1) |
| T-004 | G-03 | 決定論 | 同一 z の generate 2 回が深い等値 |
| T-005 | N-05 | 縁: z の長さ不正・非有限値(NaN/±Inf) | RangeError を投げる(黙って NaN を返さない) |
| T-006 | N-05 | 縁: discriminate 入力の長さ不正・非有限値 | RangeError を投げる |
| T-007 | G-04 | meta.json の BN 畳み込み等価性 | g04_max_abs_err < 5e-6・g04_d_max_abs_err < 1e-4(較正実測 6.2e-07 / 1.1e-05 — 2026-09-03) |
| T-008 | G-05 | meta.json の弁別力と対照 | auc_trained > 閾値(本学習の実測後に確定)・auc_untrained は 0.5 近傍の帯(対照: 未学習では分けられない) |
| T-009 | N-04 | generate / discriminate 1 回の平均所要時間 | < 60ms(実測 ≈ 15ms / 11ms — 2026-09-03。閾値は 4 倍マージン) |
