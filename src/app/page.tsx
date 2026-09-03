import { SnapshotPlayer } from "@/components/SnapshotPlayer";
import { LatentWalk } from "@/components/LatentWalk";
import { JudgePanel } from "@/components/JudgePanel";
import { meta } from "@/lib/models";
import { FOOTER_LINKS } from "@/lib/links";

const section: React.CSSProperties = {
  background: "#fffdf7",
  border: "1px solid #dfd5bd",
  borderRadius: 8,
  padding: "1.25rem 1.5rem",
  marginTop: "1.5rem",
};

export default function Home() {
  return (
    <main style={{ padding: "2rem 1rem 4rem", maxWidth: "46rem", margin: "0 auto" }}>
      <h1 style={{ marginBottom: "0.25rem" }}>偽筆工房</h1>
      <p style={{ color: "#6b6250", marginTop: 0 }}>くずし字 DCGAN の解剖台</p>
      <p>
        くずし字(KMNIST)を贋作する<strong>偽筆師</strong>(Generator)と、それを見抜こうとする
        <strong>目利き</strong>(Discriminator)を競わせて学習させた。ここに並ぶ文字はすべて、
        実在しない「偽のくずし字」——偽筆師がいまブラウザの中で描いている。
      </p>

      <section style={section} aria-labelledby="h-rewind">
        <h2 id="h-rewind">一、学習の巻き戻し</h2>
        <p>
          砂嵐が {meta.epochs} 期かけて文字になっていく。偽筆師の練習帳を最初のページからめくり直す。
        </p>
        <SnapshotPlayer epochs={meta.snapshot_epochs} />
      </section>

      <section style={section} aria-labelledby="h-walk">
        <h2 id="h-walk">二、潜在の散歩</h2>
        <p>
          偽筆師は 64 個の数(潜在ベクトル)から一枚を描く。二つの潜在点のあいだを歩くと、
          一つの文字が別の文字へ連続に化けていく。
        </p>
        <LatentWalk />
      </section>

      <section style={section} aria-labelledby="h-judge">
        <h2 id="h-judge">三、目利きの採点</h2>
        <p>
          本物と偽物を目利きに見せる。見立ては「本物らしさ」の確率(sigmoid(logit))で出る。
          学習を終えた目利きの弁別力は AUC {meta.auc_trained.toFixed(3)}
          (未学習の対照は {meta.auc_untrained.toFixed(3)})——完璧に見抜けないのは、
          偽筆師との競い合いが均衡に達した証でもある。
        </p>
        <JudgePanel />
      </section>

      <section style={section} aria-labelledby="h-notes">
        <h2 id="h-notes">仕掛けの注記</h2>
        <ul style={{ paddingLeft: "1.25rem", lineHeight: 1.9 }}>
          <li>
            学習は手元の TensorFlow {meta.tensorflow}(CPU・{meta.trained_at})。
            このページに TensorFlow は載っていない——生成も採点も TypeScript の手書き forward。
          </li>
          <li>
            正しさは二実装照合で担保する: 配布した丸め済み重みを Python(numpy float64)で
            読み戻した計算が正、TS は写し。64 潜在点 × 784 ピクセルと 32 枚のロジットが
            最大絶対誤差 1e-9 未満で一致することをテストが確かめている。
          </li>
          <li>データは KMNIST(CODH・CC BY-SA 4.0)。派生する生成画像にも同条件が及ぶ。</li>
        </ul>
      </section>

      <footer className="footer">
        {FOOTER_LINKS.map((l, i) => (
          <span key={l.href}>
            {i > 0 && " ・ "}
            <a href={l.href} target="_blank" rel="noreferrer">
              {l.label}
            </a>
            {l.label === "MIT License" && " © 2026 坂田哲朗"}
          </span>
        ))}
      </footer>
    </main>
  );
}
