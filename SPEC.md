# SPEC.md — gihitsu-kobo(偽筆工房)

<!-- scaffold template v1.24.0 から展開(2026-09-02)。以後このファイルはプロジェクトが育てる -->

## 1. 目的

KMNIST(くずし字)で DCGAN を学習し、偽筆師(Generator)と目利き(Discriminator)の
競い合いをブラウザで見せる解剖台。目玉は**学習の巻き戻しスライダー** —— epoch ごとの
生成標本を並べ、砂嵐が文字になっていく過程をその場で再生する。
学習はフリート初の TensorFlow(training/・手元専用)。ブラウザには丸め済み重み JSON と
事前計算した標本画像だけを載せ、生成・採点は TypeScript 手書き forward で行う。
正しさの正本は**二実装照合オラクル** —— 丸め済み重みを numpy float64 で読み戻した
forward が正、TS 側が写し(senzai-niwa / tegaki-yomi の契約と同じ)。
kuzushi-yomi(読む)⇔ 本作(産む)、senzai-niwa(VAE で産む)⇔ 本作(GAN で産む)の姉妹作。
ブラウザ完結・静的エクスポート・cron ゼロ・関数ゼロ・課金経路ゼロ。

## 2. 機能要求

| ID | 要求 | 優先度 |
|---|---|---|
| F-01 | 学習パイプライン(`training/train_gan.py`・TensorFlow・シード固定): KMNIST で DCGAN を学習し、(a) 丸め済み Generator 重み JSON(BatchNorm は推論等価に畳み込んで出力)、(b) 丸め済み Discriminator 重み JSON、(c) 照合フィクスチャ(丸め済み重みを numpy float64 で読み戻して再計算した潜在点 64 件の 784 ピクセル値と、実画像/生成画像それぞれの判定ロジット)、(d) epoch ごとの固定潜在 8×8 格子の生成標本(画像アセット)を出力する | must |
| F-02 | モデル(L0 較正で確定 — 2026-09-03): Generator: z(64)→Dense(3136, bias 無)→BN→ReLU→Reshape(7,7,64)→Conv2DTranspose(32, 4×4, s2, same, bias 無)→BN→ReLU→Conv2DTranspose(1, 4×4, s2, same)→sigmoid([0,1])。Discriminator: 28×28×1→Conv2D(32, 4×4, s2, same)→LeakyReLU(0.2)→Conv2D(64, 4×4, s2, same)→LeakyReLU(0.2)→Flatten→Dense(1) ロジット。配布重みは BN 畳み込み済み・小数 7 桁丸め | must |
| F-03 | TS forward(純関数): 丸め済み重み JSON を読み、generate(z) → 784 ピクセル([0,1])と discriminate(image) → ロジットを実装(dense / conv2dTranspose / conv2d / LeakyReLU を手書き) | must |
| F-04 | 学習巻き戻しスライダー: epoch 列の生成標本(同一潜在格子)をスライダーで行き来し、学習過程を再生できる | must |
| F-05 | 潜在の散歩: 潜在 2 点間の線形補間で生成文字が連続変形するアニメ。点は振り直し可能 | must |
| F-06 | 目利きの採点: 生成画像と本物の KMNIST 標本を並べ、Discriminator のロジット(=目利きの見立て)をその場で表示 | must |
| F-07 | 通し勝負: 本物/偽物をまぜた札を目利きに通し、見抜いた率・騙された率を集計表示 | should(後続ループ) |
| F-08 | フッタ: MIT License・GitHub・歩き方・設計図・App Menu の 5 リンク(フリート共通規約)。実装済み — 本番 gihitsu-kobo.vercel.app 稼働・app-menu cat12 掲載(実測 2026-09-03) | must(公開ループ) |

## 3. 非機能要求

| ID | 要求 | 検証方法 |
|---|---|---|
| N-01 | 静的エクスポート(`output: "export"`)。生成・採点は全てブラウザ内・外部通信なし | `next build` が out/ を生成 |
| N-02 | `src/core` は純関数のみ。カバレッジ lines/functions/statements ≥ 90%, branches ≥ 85%(重み等のデータ資産は対象外) | eslint 境界規則 + vitest coverage 閾値 |
| N-03 | TensorFlow は手元専用(training/)。Vercel へ渡るのは静的成果物(JSON・画像)のみ | リポジトリ構成 + out/ の内容 |
| N-04 | 生成 1 回(generate forward)がポインタ操作に追随できる速さ: 平均 < 60ms(実測 ≈ 15ms、discriminate ≈ 11ms — 2026-09-03。閾値は 4 倍マージン) | timing テスト(T-009) |
| N-05 | 縁(非有限値入力・潜在範囲外)は正常系として仕様化 | core テスト |
| N-06 | KMNIST は CC BY-SA 4.0(CODH「KMNIST データセット」)。NOTICE に出典を明記 | NOTICE ファイル |

## 4. 品質基準

`npm run verify` green を全ループの完了条件とする(内訳は AGENTS.md §3)。

二実装照合ゲート(丸め済み重みを numpy float64 で読み戻した forward が正、TS が写し):

| ID | ゲート | 基準 |
|---|---|---|
| G-01 | Generator ピクセル照合 | フィクスチャ 64 潜在点で TS generate の 784 値が記録値と最大絶対誤差 < 1e-9 で一致(記録値の丸めは 12 桁 = 床 5e-13。較正: Python 自己検算 5.0e-13 — 2026-09-03) |
| G-02 | Discriminator ロジット照合 | フィクスチャ 32 画像(実 16 + 生成 16)の TS discriminate ロジットが記録値と最大絶対誤差 < 1e-9 で一致(較正: Python 自己検算 4.6e-13 — 2026-09-03) |
| G-03 | 形状・決定論 | 重み JSON の形状検査。同一 z → 深い等値 |
| G-04 | BatchNorm 畳み込みの等価性 | 畳み込み+丸め後の numpy forward が TF 推論モード出力と一致: Generator < 5e-6・Discriminator ロジット < 1e-4(Python 側検査。較正実測: 6.2e-07 / 1.1e-05 — 2026-09-03。TF は float32・oneDNN のため床が高い) |
| G-05 | 目利きの弁別力 | meta.json の auc_trained > 0.65(実測 0.7153 — 2026-09-03・実/生成各 500)。対照: auc_untrained < 0.60(実測 0.5354)。均衡に達した GAN では D は完全には分けられないので 1.0 近傍を要求しない |

**フィクスチャ導出の契約(HC-139)**: 入力(z・画像)を先に配布形(7 桁丸め)へ落とし、
出力は書き出して読み戻した実物からのみ導出する。書き出し直後に「読み戻しだけからの再計算 =
記録値(< 1e-12)」の自己検算を train_gan.py が assert する。

### L0 較正の実測記録欄(loop_001・2026-09-03)

- TensorFlow 2.21.0 / numpy 2.5.2 / Python 3.12.13(uv venv・Windows CPU)
- 1 epoch の学習時間(CPU): 12.8〜16.9 秒(batch 128・468 step)。30 epoch ≈ 7〜9 分の見込み
- 丸め済み重み JSON: generator.json 2,409 KB / discriminator.json 375 KB / fixtures.json 949 KB
- G-04 実測: Generator 6.2e-07・Discriminator ロジット 1.1e-05(TF float32 との比較)
- 丸め: 重み・フィクスチャ入力 7 桁、フィクスチャ記録値 12 桁。自己検算 5.0e-13 / 4.6e-13(= 12 桁丸めの床)
- epoch 標本: 8×8 格子 224×224 グレースケール PNG、約 27〜34 KB/epoch(30 epoch ≈ 1 MB の見込み)
- **学習はビット再現しない**(同一シードの 3 回で epoch1 d_loss = 0.3465 / 0.3088 / 0.3007)。
  oneDNN の浮動小数順序が原因とみられる(TF 起動時の告知)。オラクルの正は丸め済み重みの
  読み戻し forward なので照合には影響しない(HC-073: 一致する量と揃える規則を分ける)。
  ゆえに「学習の再現」はゲートにしない。配布重みはコミットされた実物が正

## 5. スコープ外

- ブラウザ内での学習・再学習(学習は手元の TensorFlow のみ)
- TensorFlow.js 等の推論ランタイム導入(forward は TS 手書き。二実装照合を成立させるため)
- 条件付き生成(クラス指定)・拡散モデル・StyleGAN 系の大型化
- 生成品質の定量主張(FID 等)。見せるのは競い合いの構造であって SOTA ではない
- 描いた絵の判定(Discriminator に任意手書き入力を通す機能は当面外)
