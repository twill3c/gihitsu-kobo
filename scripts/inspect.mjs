// 実ブラウザ検品(HC-041 / HC-078 / HC-138)。out/ を配信して:
//   1. コンソールエラー 0・ページエラー 0・外部への通信 0(N-01)
//   2. F-04: スライダー操作で表示中の epoch 標本(可視 img)が切り替わる(到達の証拠つき)
//   3. F-05: 散歩スライダー操作で中央 canvas の画素が実際に変わる(toDataURL 差分)
//   4. F-06: 本物/偽物の見立てが出ており、「別の偽物」で canvas と見立てが変わる
//   5. 幾何: 3 幅(1280/800/390)で横溢れなし・縦 16,000px 未満・主要 canvas が可視域内
//   6. スクリーンショットを .loop/shots/ に保存(目視は人とエージェントの義務)
// 検品器の規範: 失敗は終了コードで知らせる。操作は届いた証拠(変化)を確かめてから判定する。
// 実行: node scripts/inspect.mjs  (先に npm run build)

import { createServer } from "node:http";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { extname, join } from "node:path";
import { chromium } from "playwright";

const OUT = "out";
const SHOTS = ".loop/shots";
if (!existsSync(OUT)) {
  console.error("out/ が無い。先に npm run build を走らせること");
  process.exit(2);
}
mkdirSync(SHOTS, { recursive: true });

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".txt": "text/plain",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
};

const server = createServer((req, res) => {
  let path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (path.endsWith("/")) path += "index.html";
  let file = join(OUT, path);
  if (!existsSync(file)) file = join(OUT, path + ".html");
  if (!existsSync(file)) {
    res.writeHead(404);
    res.end("not found");
    return;
  }
  res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
  res.end(readFileSync(file));
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const origin = `http://127.0.0.1:${server.address().port}`;

const problems = [];
const fail = (msg) => {
  problems.push(msg);
  console.error(`  x ${msg}`);
};
const pass = (msg) => console.log(`  o ${msg}`);

const canvasData = (page, label) =>
  page.evaluate((l) => {
    const el = document.querySelector(`canvas[aria-label="${l}"]`);
    return el ? el.toDataURL() : null;
  }, label);

const browser = await chromium.launch();
const WIDTHS = [
  [1280, 900],
  [800, 900],
  [390, 844],
];

for (const [w, h] of WIDTHS) {
  console.log(`-- ${w}x${h}`);
  const page = await browser.newPage({ viewport: { width: w, height: h } });
  const consoleErrors = [];
  const external = [];
  page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
  page.on("pageerror", (e) => consoleErrors.push(String(e)));
  page.on("request", (r) => {
    if (!r.url().startsWith(origin)) external.push(r.url());
  });

  await page.goto(`${origin}/`, { waitUntil: "networkidle" });

  // --- F-04: 巻き戻しスライダー(可視 img が 1 枚で、操作で切り替わる)
  const slider = page.getByTestId("epoch-slider");
  await slider.scrollIntoViewIfNeeded();
  const visibleCount = () =>
    page.evaluate(
      () =>
        [...document.querySelectorAll('img[src^="/snapshots/"]')].filter(
          (el) => el.style.visibility === "visible",
        ).length,
    );
  const visibleSrc = () =>
    page.evaluate(
      () =>
        [...document.querySelectorAll('img[src^="/snapshots/"]')].find(
          (el) => el.style.visibility === "visible",
        )?.getAttribute("src") ?? null,
    );
  if ((await visibleCount()) !== 1) fail("F-04: 可視の epoch 標本が 1 枚でない");
  const srcBefore = await visibleSrc();
  await slider.fill("3");
  const labelAfter = await page.getByTestId("epoch-label").textContent();
  const srcAfter = await visibleSrc();
  if (srcBefore === srcAfter && !labelAfter?.includes("3")) {
    fail("F-04: スライダー操作が届いていない(標本もラベルも変わらない)");
  } else if (srcAfter !== "/snapshots/epoch_003.png") {
    fail(`F-04: 表示中の標本が epoch_003 でない(${srcAfter})`);
  } else {
    pass("F-04: スライダーで標本が切り替わる");
  }

  // --- F-05: 散歩スライダーで中央 canvas の画素が変わる(到達の証拠 = 画素差分)
  const walk = page.getByTestId("walk-slider");
  await walk.scrollIntoViewIfNeeded();
  const centerLabel = "いま生成している文字";
  const before = await canvasData(page, centerLabel);
  await walk.fill("1");
  await page.waitForTimeout(300);
  const after = await canvasData(page, centerLabel);
  if (!before || !after) fail("F-05: 中央 canvas が見つからない");
  else if (before === after) fail("F-05: スライダー操作で画素が変わらない(操作が届いていない)");
  else pass("F-05: 散歩で生成画素が変わる");

  // --- F-06: 見立てが出ており、「別の偽物」で canvas と見立てが変わる
  const fakeFig = page.getByTestId("judge-fake");
  await fakeFig.scrollIntoViewIfNeeded();
  const probTexts = await page.getByTestId("prob-value").allTextContents();
  if (probTexts.length < 3 || probTexts.some((t) => !/本物らしさ \d+\.\d%/.test(t))) {
    fail(`F-06: 見立ての表示が欠けている(${probTexts.length} 件)`);
  } else {
    pass(`F-06: 見立て ${probTexts.length} 件が表示`);
  }
  const fakeLabel = "偽筆師の生成した文字";
  const fakeBefore = await canvasData(page, fakeLabel);
  await fakeFig.getByRole("button", { name: "別の偽物を描かせる" }).click();
  await page.waitForTimeout(300);
  const fakeAfter = await canvasData(page, fakeLabel);
  if (fakeBefore === fakeAfter) fail("F-06: 「別の偽物」で画素が変わらない");
  else pass("F-06: 「別の偽物」で生成が変わる");

  // --- 幾何(HC-078 / HC-138)
  const geo = await page.evaluate(() => ({
    scrollW: document.documentElement.scrollWidth,
    clientW: document.documentElement.clientWidth,
    scrollH: document.documentElement.scrollHeight,
  }));
  if (geo.scrollW > geo.clientW) fail(`横溢れ: scrollWidth ${geo.scrollW} > ${geo.clientW}`);
  else pass("横溢れなし");
  if (geo.scrollH >= 16000) fail(`縦伸びすぎ: ${geo.scrollH}px`);
  else pass(`縦 ${geo.scrollH}px`);

  const canvasBoxOk = await page.evaluate(() => {
    const el = document.querySelector('canvas[aria-label="いま生成している文字"]');
    if (!el) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.left >= 0 && r.right <= document.documentElement.clientWidth;
  });
  if (!canvasBoxOk) fail("中央 canvas が可視域から外れている");
  else pass("中央 canvas は可視域内");

  // --- F-08: 固定フッタの高さが body の逃げ(padding-bottom)に収まる(hanshoku の教訓)
  const foot = await page.evaluate(() => {
    const els = [...document.querySelectorAll("footer")];
    // フッタは要素名でなく中身(MIT License と App Menu)で選ぶ(フリート規約検品の教訓)
    const f = els.find((e) => e.innerText.includes("MIT License") && e.innerText.includes("App Menu"));
    if (!f) return null;
    const cs = getComputedStyle(f);
    return {
      fixed: cs.position === "fixed" && cs.bottom === "0px",
      height: f.getBoundingClientRect().height,
      escape: parseFloat(getComputedStyle(document.body).paddingBottom),
      items: [...f.querySelectorAll("a")].length,
      seps: (f.innerText.match(/・/g) ?? []).length,
    };
  });
  if (!foot) fail("F-08: 規約フッタ(MIT License + App Menu)が見つからない");
  else {
    if (!foot.fixed) fail("F-08: フッタが position:fixed; bottom:0 でない");
    if (foot.items !== 5) fail(`F-08: リンクが ${foot.items} 個(規約は 5)`);
    if (foot.seps !== 4) fail(`F-08: 区切りの・が ${foot.seps} 個(規約は項目間 4)`);
    if (foot.height > foot.escape) {
      fail(`F-08: フッタ高 ${foot.height}px > 逃げ ${foot.escape}px(最下部本文が隠れる)`);
    }
    if (!problems.some((p) => p.startsWith("F-08"))) pass(`F-08: フッタ規約 OK(高 ${foot.height}px ≤ 逃げ ${foot.escape}px)`);
  }

  if (consoleErrors.length) fail(`コンソールエラー: ${consoleErrors.join(" / ")}`);
  else pass("コンソールエラー 0");
  if (external.length) fail(`外部通信: ${external.join(" / ")}`);
  else pass("外部通信 0(N-01)");

  await page.screenshot({ path: `${SHOTS}/w${w}.png`, fullPage: true });
  await page.close();
}

await browser.close();
server.close();

if (problems.length) {
  console.error(`検品 NG: ${problems.length} 件`);
  process.exit(1);
}
console.log("検品 OK(スクリーンショットの目視を忘れない — .loop/shots/)");
