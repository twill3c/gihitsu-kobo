// フリート共通フッタ規約の検査規則(F-08 / T-010)。
// 規約: 5 項目・並び固定 — MIT License / GitHub / 歩き方 / 設計図 / App Menu。
// 表示コードではなくリンクの正本データ(src/lib/links.ts)に当てる。

export interface FooterLink {
  label: string;
  href: string;
}

/** 規約違反の一覧を返す(適合なら空配列) */
export function validateFooterLinks(
  links: readonly FooterLink[],
  repo: string,
): string[] {
  const errs: string[] = [];
  if (links.length !== 5) {
    errs.push(`項目数が ${links.length}(規約は 5)`);
    return errs;
  }
  const [license, github, walk, blueprint, menu] = links;
  if (license.label !== "MIT License") {
    errs.push(`1 番目のラベルが ${license.label}(規約は MIT License)`);
  }
  if (license.href !== `https://github.com/twill3c/${repo}/blob/main/LICENSE`) {
    errs.push(`MIT License の行き先が不正: ${license.href}`);
  }
  if (github.label !== "GitHub") {
    errs.push(`2 番目のラベルが ${github.label}(規約は GitHub)`);
  }
  if (github.href !== `https://github.com/twill3c/${repo}`) {
    errs.push(`GitHub の行き先が不正: ${github.href}`);
  }
  for (const l of [walk, blueprint]) {
    if (!l.href.startsWith("https://claude.ai/code/artifact/")) {
      errs.push(`解説の行き先がアーティファクトでない: ${l.label} → ${l.href}`);
    }
  }
  if (menu.label !== "App Menu") {
    errs.push(`5 番目のラベルが ${menu.label}(規約は App Menu)`);
  }
  if (menu.href !== "https://app-menu-amber.vercel.app") {
    errs.push(`App Menu の行き先が不正: ${menu.href}`);
  }
  return errs;
}
