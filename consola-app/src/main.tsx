import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import App from "./App"
import { aplicarTema, leerTemaGuardado } from "./components/tema"

// El tema guardado (leia-theme: light | dark | system) se aplica ANTES de
// montar para que no haya destello; el toggle del header lo maneja despues.
aplicarTema(leerTemaGuardado())

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>)
