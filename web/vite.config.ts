import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath } from "node:url";

// Public address used in link previews (Open Graph). Override at build time with
// SITE_URL=https://your-domain.org when the site moves to a custom domain.
const SITE_URL = (process.env.SITE_URL || "https://glucolab.onrender.com").replace(/\/+$/, "");

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    { name: "site-url", transformIndexHtml: (html: string) => html.replaceAll("__SITE_URL__", SITE_URL) },
  ],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 1600,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (!id.includes("node_modules")) return undefined;
          if (/[\\/](react|react-dom|react-router|react-router-dom|scheduler|@tanstack)[\\/]/.test(id)) return "react";
          if (/[\\/](recharts|d3-[^\\/]+|victory-vendor)[\\/]/.test(id)) return "charts";
          if (/[\\/]three[\\/]/.test(id)) return "three";
          if (/[\\/]3dmol[\\/]/.test(id)) return "3dmol";
          return undefined;
        },
      },
    },
  },
});
