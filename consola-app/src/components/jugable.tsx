// JUGABLE — arma un enfrentamiento y produce el comando exacto para copiar.
// Honesto: esta pantalla NO lanza procesos aun; eso llega con la fase de jobs.
import { useState } from "react";
import { Copy, Crown, Info } from "lucide-react";
import { toast } from "sonner";
import type { Estado } from "@/lib/api";
import { pc } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Dato, SinDato, Titulo, fechaCorta } from "@/components/comunes";
import { cn } from "@/lib/utils";

const RIVALES = [
  "RANDOM", "RYU", "KEN", "CHUNLI", "GUILE", "BLANKA", "ZANGIEF",
  "DHALSIM", "EHONDA", "BALROG", "VEGA", "SAGAT", "MBISON",
] as const;

type ModoP2 = "humano" | "cpu" | "apex" | "sb3";

const MODOS: { id: ModoP2; nombre: string; detalle: string }[] = [
  { id: "humano", nombre: "Humano", detalle: "pad en puerto 2 (rig BizHawk)" },
  { id: "cpu", nombre: "CPU del juego", detalle: "dificultad 1–8, cualquier maquina" },
  { id: "apex", nombre: "Checkpoint Ape-X", detalle: "otro checkpoint del run" },
  { id: "sb3", nombre: "Modelo clasico", detalle: "SB3 / era anterior" },
];

export function Jugable({ estado }: { estado: Estado }) {
  const campeon = estado.campeon;
  const [modo, setModo] = useState<ModoP2>("humano");
  const [rival, setRival] = useState<string>("RANDOM");
  const [dificultad, setDificultad] = useState<string>("8");

  const comando =
    modo === "cpu"
      ? `.venv/bin/python tools/watch_es.py --rainbow-ckpt benchmarks/apex_milestones/apex_escalera_best.pt --difficulty ${dificultad} --desync-max 30 --speed 0.5`
      : `.venv\\Scripts\\python.exe src\\scripts\\stand_leia.py --opponent ${rival}`;

  const copiar = async () => {
    try {
      await navigator.clipboard.writeText(comando);
      toast.success("Comando copiado al portapapeles");
    } catch {
      toast.error("No se pudo copiar — selecciona el texto a mano");
    }
  };

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <Titulo>Enfrentamiento</Titulo>
        <div className="grid gap-3 lg:grid-cols-2">
          {/* -- P1: el campeon, fijo -------------------------------- */}
          <Card className="border-l-2 border-l-champion">
            <CardContent className="space-y-3 p-4">
              <div className="flex items-center justify-between">
                <div className="dlabel">P1 · fijo</div>
                <span className="inline-flex items-center gap-1.5 rounded-sm bg-state-champion-subtle px-1.5 py-0.5 font-mono text-[11px] font-medium uppercase tracking-wide text-state-champion-fg">
                  <Crown size={11} aria-hidden />
                  campeon
                </span>
              </div>
              {campeon ? (
                <div className="grid grid-cols-3 gap-3">
                  <Dato label="version" sub={campeon.archivo ?? "—"}>
                    v{campeon.weights_version}
                  </Dato>
                  <Dato label="wr media" sub="banco greedy n=48 · rounds de apertura">
                    {pc(campeon.wr_media)}
                  </Dato>
                  <Dato label="coronado" sub="hora local">
                    <span className="text-sm">{fechaCorta(campeon.t)}</span>
                  </Dato>
                </div>
              ) : (
                <SinDato>sin campeon coronado todavia</SinDato>
              )}
            </CardContent>
          </Card>

          {/* -- P2: a elegir ---------------------------------------- */}
          <Card>
            <CardContent className="space-y-4 p-4">
              <div className="dlabel">P2 · rival</div>
              <div className="grid grid-cols-2 gap-2">
                {MODOS.map((op) => (
                  <button
                    key={op.id}
                    type="button"
                    onClick={() => setModo(op.id)}
                    aria-pressed={modo === op.id}
                    className={cn(
                      "rounded-md border px-3 py-2 text-left transition-colors",
                      modo === op.id
                        ? "border-primary bg-accent"
                        : "border-border hover:bg-accent",
                    )}
                  >
                    <div className="text-[13px] font-semibold">{op.nombre}</div>
                    <div className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                      {op.detalle}
                    </div>
                  </button>
                ))}
              </div>

              {modo === "cpu" ? (
                <div className="space-y-1.5">
                  <div className="dlabel">dificultad del juego</div>
                  <Select value={dificultad} onValueChange={setDificultad}>
                    <SelectTrigger className="w-44 font-mono text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {Array.from({ length: 8 }, (_, i) => String(i + 1)).map(
                        (d) => (
                          <SelectItem key={d} value={d} className="font-mono text-xs">
                            nivel {d}
                          </SelectItem>
                        ),
                      )}
                    </SelectContent>
                  </Select>
                </div>
              ) : (
                <div className="space-y-1.5">
                  <div className="dlabel">personaje rival</div>
                  <Select value={rival} onValueChange={setRival}>
                    <SelectTrigger className="w-44 font-mono text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {RIVALES.map((r) => (
                        <SelectItem key={r} value={r} className="font-mono text-xs">
                          {r}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      {/* -- comando ----------------------------------------------------- */}
      <section className="space-y-3">
        <Titulo
          extra={
            <span className="font-mono text-[11px] text-muted-foreground">
              {modo === "cpu"
                ? "corre en cualquier maquina"
                : "rig Windows / BizHawk"}
            </span>
          }
        >
          Comando
        </Titulo>
        <Card>
          <CardContent className="space-y-3 p-4">
            <div className="flex items-start gap-3">
              <pre className="tnum min-w-0 flex-1 overflow-x-auto rounded-md bg-muted p-3 font-mono text-xs leading-relaxed">
                {comando}
              </pre>
              <Button variant="outline" size="sm" onClick={copiar} className="shrink-0">
                <Copy size={14} aria-hidden />
                Copiar
              </Button>
            </div>
            {(modo === "apex" || modo === "sb3") && (
              <p className="font-mono text-[11px] text-muted-foreground">
                el flag del checkpoint P2 ({modo === "apex" ? "Ape-X" : "SB3"}) lo
                anade la fase de jobs — hoy el stand lo pide en su propia UI
              </p>
            )}
            <p className="flex items-center gap-1.5 border-t border-border pt-3 font-mono text-[11px] text-warning-fg">
              <Info size={13} aria-hidden />
              esta pantalla NO lanza procesos aun: copia el comando y corre a mano.
              Lo hara la fase de jobs.
            </p>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
