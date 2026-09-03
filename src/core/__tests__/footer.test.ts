// footer.test.ts — フリート共通フッタ規約(F-08 / T-010 / T-011)
//
// 出所: フリート規約(koho-lens 正本)= 5 項目・この並び:
//   MIT License(→ 本リポジトリの LICENSE)・GitHub(→ 本リポジトリ)・
//   歩き方・設計図(→ claude.ai/code/artifact/)・App Menu(→ app-menu-amber.vercel.app)
// 検査規則は表示コードではなく src/lib/links.ts の正本データに対して書く(HC-068)

import { describe, expect, it } from "vitest";
import { validateFooterLinks } from "@/core/footerRule";
import { FOOTER_LINKS } from "@/lib/links";

describe("footer links(F-08)", () => {
  it("T-010: FOOTER_LINKS が規約 5 項目・並び・行き先に適合(違反 0 件)", () => {
    expect(validateFooterLinks(FOOTER_LINKS, "gihitsu-kobo")).toEqual([]);
  });

  it("T-011: 陽性対照(HC-041)— 壊し方ごとに検査器が撃つ", () => {
    const base = FOOTER_LINKS.map((l) => ({ ...l }));

    expect(validateFooterLinks(base.slice(0, 4), "gihitsu-kobo").length).toBeGreaterThan(0);

    const swapped = [base[1], base[0], base[2], base[3], base[4]];
    expect(validateFooterLinks(swapped, "gihitsu-kobo").length).toBeGreaterThan(0);

    const wrongLicense = base.map((l, i) =>
      i === 0 ? { ...l, href: "https://opensource.org/licenses/MIT" } : l,
    );
    expect(validateFooterLinks(wrongLicense, "gihitsu-kobo").length).toBeGreaterThan(0);

    const wrongRepo = base.map((l, i) =>
      i === 1 ? { ...l, href: "https://github.com/someone-else/gihitsu-kobo" } : l,
    );
    expect(validateFooterLinks(wrongRepo, "gihitsu-kobo").length).toBeGreaterThan(0);

    const wrongArtifact = base.map((l, i) =>
      i === 2 ? { ...l, href: "https://example.com/how-to" } : l,
    );
    expect(validateFooterLinks(wrongArtifact, "gihitsu-kobo").length).toBeGreaterThan(0);

    const wrongMenu = base.map((l, i) =>
      i === 4 ? { ...l, href: "https://app-menu.vercel.app" } : l,
    );
    expect(validateFooterLinks(wrongMenu, "gihitsu-kobo").length).toBeGreaterThan(0);
  });
});
