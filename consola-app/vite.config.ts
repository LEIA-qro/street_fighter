import path from "path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"

// El build sale a ../web/app y lo sirve el hub (tools/leia_hub.py): el
// toolchain de node solo vive en la Mac, pero el ARTEFACTO es estatico y
// viaja con el repo -- asi ningun rig necesita bun para ver la consola.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/app/",
  build: { outDir: "../web/app", emptyOutDir: true },
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: { proxy: { "/api": "http://127.0.0.1:8099" } },
})
