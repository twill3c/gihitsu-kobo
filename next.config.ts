import type { NextConfig } from "next";

// 静的エクスポート(N-01)— サーバ API を持たず、out/ のみで動作する
const nextConfig: NextConfig = {
  output: "export",
};

export default nextConfig;
