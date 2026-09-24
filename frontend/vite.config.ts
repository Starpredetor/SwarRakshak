import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The AudioWorklet is served from public/worklets/ as a plain file so it
    // is not bundled -- worklets load in a separate realm and must be a real
    // URL at runtime.
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
