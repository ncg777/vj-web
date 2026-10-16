import { defineConfig } from "vite";

export default defineConfig({
  publicDir: false,
  build: {
    ssr: true,
    target: "node20",
    outDir: "dist-mcp/server",
    rollupOptions: { input: { index: "mcp/index.ts", server: "mcp/server.ts" } },
  },
});
