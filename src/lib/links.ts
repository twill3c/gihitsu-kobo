// フッタリンクの正本(F-08)。歩き方・設計図はアーティファクト(要共有リンク)。
// 規約適合は src/core/footerRule.ts + footer.test.ts(T-010/T-011)で検査する。

import type { FooterLink } from "@/core/footerRule";

export const FOOTER_LINKS: readonly FooterLink[] = [
  {
    label: "MIT License",
    href: "https://github.com/twill3c/gihitsu-kobo/blob/main/LICENSE",
  },
  { label: "GitHub", href: "https://github.com/twill3c/gihitsu-kobo" },
  {
    label: "偽筆工房の歩き方",
    href: "https://claude.ai/code/artifact/0b987f81-7007-4f26-9c2c-dea8cb3225bd",
  },
  {
    label: "偽筆工房 設計図",
    href: "https://claude.ai/code/artifact/5725e2c1-1653-4a7d-b57e-18827794c6a4",
  },
  { label: "App Menu", href: "https://app-menu-amber.vercel.app" },
] as const;
