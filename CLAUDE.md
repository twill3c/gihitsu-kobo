# CLAUDE.md

@AGENTS.md

上記ハーネスがこのリポジトリの正本ルール。要点のみ再掲する:

- 仕様の正本は SPEC.md。変更は スペック → テスト → 実装 の順。
- すべてのタスクは 7 段階ループプロトコル(AGENTS.md 末尾の共通規律)で進め、
  `python harness/looplog.py append` で `logs/loops/{loop_id}.jsonl` に記録する。
- 完了条件は `npm run verify` green + `looplog.py validate` 合格。
- `src/core` は純関数のみ・カバレッジ 90% 以上を維持。
- 正しさの正本は二実装照合オラクル(丸め済み重み読み戻しの numpy forward が正、
  TS generate/discriminate が写し — G-01/G-02)。モデル資産は training/train_gan.py
  だけが生成する。照合ゲート G-xx の数値は較正実験の証拠付きでのみ変更できる
  (緩和は人間の承認が必要)。
- TensorFlow は手元専用(training/)。Vercel へ渡るのは静的成果物のみ。
- scaffold ブロック(AGENTS.md 末尾)と `.wt/gate.json` の上限は直接編集しない。
