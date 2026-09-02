import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "偽筆工房 — くずし字 DCGAN の解剖台",
  description:
    "偽筆師(Generator)と目利き(Discriminator)の競い合いをブラウザで見る。学習巻き戻し・潜在の散歩・目利きの採点。",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
