import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import App from "./App"

// La consola es dark-first (se mira durante horas); el modo claro existe en
// los tokens y llegara como toggle -- por ahora el sistema decide.
if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
  document.documentElement.classList.add("dark")
}
createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>)
