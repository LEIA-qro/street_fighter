/* CONSOLA LEIA — mini-plan de diseno
 * ==================================
 * TIPOGRAFIA (una escala, no ad-hoc):
 *   10px  .dlabel     — etiquetas de dato: IBM Plex Mono, uppercase, tracking .12em
 *   11px  text-[11px] — letra chica de instrumento (n, IC, notas), mono
 *   12px  text-xs     — celdas de tabla y datos densos, mono
 *   13px  text-[13px] — cuerpo, IBM Plex Sans
 *   15px  .stitle     — titulares de seccion, Chakra Petch (SOLO ahi y el logo)
 *   20px  text-xl     — valores de los Datos del titular, mono tabular
 *   28px  text-[28px] — el numero heroe de cada acta, mono tabular
 * ESPACIADO (base 4px, sin excepciones):
 *   p-4 en toda tarjeta · gap-3 entre tarjetas de una rejilla · space-y-3
 *   titulo→contenido · space-y-6 entre SECCIONES (el blanco separa secciones,
 *   no infla tarjetas) · px-4/px-6 de pagina.
 * COLOR (champion-chrome, valores intocables):
 *   --primary       barras de medida (banco del campeon) y foco/tab activa
 *   --state-*       chips y franjas izquierdas de estado (subtle=fondo, fg=texto)
 *   --champion      LA FIRMA: solo la identidad del campeon — tarjeta vigente en
 *                   Modelos, chip P1 en Jugable, marca "vigente" en coronaciones.
 *                   Jamas hovers, links ni decoracion.
 *   -fg slots       todo color usado como TEXTO sobre superficie (warning-fg etc.)
 *   Tinta sobre relleno cromatico: siempre el -foreground del token, nunca a mano.
 * EL CROMADO (identidad Champion Chrome, sin tocar un hex pinneado):
 *   derivados color-mix en index.css — --chrome-tint (lavado cian de la
 *   cabecera), --chrome-line (linea luminosa bajo el header y en las reglas
 *   de titulo), --chrome-card-line (borde de tarjeta con sangre cian).
 *   El cian existe ESTRUCTURALMENTE: logo en degradado primary→secondary,
 *   tab activa en relleno primary, tick+regla de cada titulo de seccion,
 *   cabeceras de tarjeta en .dlabel-chrome, barras y sparklines en primary.
 * ESTADO POR FORMA ademas de color: icono distinto por estado en EstadoChip,
 * franja izquierda en filas, atenuacion en mudas — legible en gris.
 */
import { useEffect, useRef, useState } from "react";
import { Radio } from "lucide-react";
import { leerEstado, edad, type Estado } from "@/lib/api";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Toaster } from "@/components/ui/sonner";
import { EstadoChip } from "@/components/comunes";
import { Config } from "@/components/config";
import { Flota, semaforoGlobal } from "@/components/flota";
import { Jugable } from "@/components/jugable";
import { Modelos } from "@/components/modelos";
import { BotonTema } from "@/components/tema";

const POLL_MS = 5000;

export default function App() {
  const [estado, setEstado] = useState<Estado | null>(null);
  const [errorHub, setErrorHub] = useState<string | null>(null);
  // Reloj del servidor extrapolado: estado.ahora + segundos locales desde el
  // fetch. Evita que un desfase de reloj cliente/hub invente frescura.
  const [tick, setTick] = useState(0);
  const fetchLocal = useRef<number>(0);

  useEffect(() => {
    let vivo = true;
    const pedir = async () => {
      try {
        const e = await leerEstado();
        if (!vivo) return;
        fetchLocal.current = Date.now() / 1000;
        setEstado(e);
        setErrorHub(null);
      } catch (err) {
        if (vivo) setErrorHub(err instanceof Error ? err.message : String(err));
      }
    };
    pedir();
    const p = setInterval(pedir, POLL_MS);
    const t = setInterval(() => setTick((n) => n + 1), 1000);
    return () => {
      vivo = false;
      clearInterval(p);
      clearInterval(t);
    };
  }, []);
  void tick; // el tick solo fuerza re-render para que la frescura avance

  const ahoraServidor = estado
    ? estado.ahora + (Date.now() / 1000 - fetchLocal.current)
    : Date.now() / 1000;
  const frescuraS =
    estado?.muestra != null
      ? Math.max(0, ahoraServidor - Date.parse(estado.muestra.ts) / 1000)
      : null;
  const global = estado ? semaforoGlobal(estado, frescuraS) : null;
  const run = estado?.plano.run;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Toaster position="bottom-right" />
      <Tabs defaultValue="flota">
        <header className="chrome-header sticky top-0 z-10">
          <div className="mx-auto flex min-h-14 max-w-6xl flex-wrap items-center gap-x-4 gap-y-1 py-1.5 px-4 md:px-6">
            <div className="flex items-baseline gap-2">
              <span className="stitle chrome-logo text-base">LEIA</span>
              <span className="dlabel hidden sm:block">consola de operacion</span>
            </div>
            <TabsList className="h-8">
              <TabsTrigger value="flota" className="font-mono text-xs uppercase tracking-wide data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                Flota
              </TabsTrigger>
              <TabsTrigger value="jugable" className="font-mono text-xs uppercase tracking-wide data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                Jugable
              </TabsTrigger>
              <TabsTrigger value="modelos" className="font-mono text-xs uppercase tracking-wide data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                Modelos
              </TabsTrigger>
              <TabsTrigger value="config" className="font-mono text-xs uppercase tracking-wide data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                Config
              </TabsTrigger>
            </TabsList>
            <div className="ml-auto flex items-center gap-2">
              <BotonTema />
              {run?.id && (
                <span className="hidden items-center gap-1.5 font-mono text-[11px] text-muted-foreground md:inline-flex">
                  <Radio size={13} aria-hidden />
                  {run.id}
                </span>
              )}
              {errorHub ? (
                <EstadoChip estado="alarm">hub caido · {errorHub}</EstadoChip>
              ) : global ? (
                <EstadoChip
                  estado={
                    frescuraS !== null && frescuraS > 180
                      ? "degraded"
                      : global.sem
                  }
                >
                  {frescuraS === null
                    ? "sin muestra"
                    : `muestra hace ${edad(frescuraS)}`}
                </EstadoChip>
              ) : (
                <EstadoChip estado="unknown">conectando…</EstadoChip>
              )}
            </div>
          </div>
        </header>

        <main className="mx-auto max-w-6xl px-4 py-6 md:px-6">
          {(() => {
            const placeholder = (
              <div className="rounded-md border border-dashed border-border px-4 py-16 text-center font-mono text-xs text-muted-foreground">
                {errorHub
                  ? `sin conexion con el hub (${errorHub}) — reintentando cada ${POLL_MS / 1000} s`
                  : "leyendo /api/state…"}
              </div>
            );
            return (
              <>
                <TabsContent value="flota">
                  {estado ? <Flota estado={estado} frescuraS={frescuraS} /> : placeholder}
                </TabsContent>
                <TabsContent value="jugable">
                  {estado ? <Jugable estado={estado} /> : placeholder}
                </TabsContent>
                <TabsContent value="modelos">
                  {estado ? <Modelos estado={estado} ahora={ahoraServidor} /> : placeholder}
                </TabsContent>
                <TabsContent value="config">
                  <Config />
                </TabsContent>
              </>
            );
          })()}
        </main>
      </Tabs>
    </div>
  );
}
