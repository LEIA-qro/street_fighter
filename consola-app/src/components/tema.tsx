// Tema claro/oscuro/sistema, persistido en localStorage ("leia-theme").
// main.tsx aplica el valor guardado ANTES de montar para evitar el destello;
// aqui vive el ciclo y la escucha del sistema.
import { useEffect, useState } from "react";
import { Monitor, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";

export type Tema = "light" | "dark" | "system";
const CLAVE = "leia-theme";

export function leerTemaGuardado(): Tema {
  try {
    const v = localStorage.getItem(CLAVE);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

export function aplicarTema(tema: Tema) {
  const oscuro =
    tema === "dark" ||
    (tema === "system" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", oscuro);
}

const CICLO: Tema[] = ["system", "dark", "light"];
const VISTA: Record<Tema, { Icono: typeof Sun; nombre: string }> = {
  system: { Icono: Monitor, nombre: "Sistema" },
  dark: { Icono: Moon, nombre: "Oscuro" },
  light: { Icono: Sun, nombre: "Claro" },
};

export function BotonTema() {
  const [tema, setTema] = useState<Tema>(leerTemaGuardado);

  useEffect(() => {
    aplicarTema(tema);
    try {
      localStorage.setItem(CLAVE, tema);
    } catch { /* almacenamiento bloqueado: el tema vive solo esta sesion */ }
    if (tema !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const alCambiar = () => aplicarTema("system");
    mq.addEventListener("change", alCambiar);
    return () => mq.removeEventListener("change", alCambiar);
  }, [tema]);

  const { Icono, nombre } = VISTA[tema];
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() =>
        setTema(CICLO[(CICLO.indexOf(tema) + 1) % CICLO.length])
      }
      className="h-7 gap-1.5 px-2 font-mono text-[11px] uppercase tracking-wide text-muted-foreground"
      title="Cambiar tema (sistema / oscuro / claro)"
    >
      <Icono size={14} aria-hidden />
      {nombre}
    </Button>
  );
}
