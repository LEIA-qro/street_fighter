// FLOTA — responde "¿vive y aprende?" en 3 segundos.
import { Bird, TriangleAlert } from "lucide-react";
import type { Estado, Maquina } from "@/lib/api";
import { edad, nf, pc } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dato,
  EstadoChip,
  LeyendaEscalera,
  PistaNivel,
  SinDato,
  Spark,
  Titulo,
  nivelesDe,
  type Semaforo,
} from "@/components/comunes";

/** Normaliza una banda de dificultad ("4,5,6,7,8" | 8) a tokens ordenados. */
function banda(v: string | number | null | undefined): string | null {
  if (v === null || v === undefined || v === "") return null;
  return String(v)
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean)
    .sort((a, b) => Number(a) - Number(b))
    .join(",");
}

const FRESCURA_MAX_S = 180;

// ---- semaforos -------------------------------------------------------------

function estadoMaquina(m: Maquina | undefined): {
  sem: Semaforo;
  texto: string;
} {
  if (!m) return { sem: "unknown", texto: "sin registro" };
  if (m.estado === "vivo") return { sem: "running", texto: "viva" };
  // MUDA: esperada y callada. Nunca reporto -> unknown (no alarm).
  if (m.age_s === null) return { sem: "unknown", texto: "nunca reporto" };
  return {
    sem: m.critico ? "alarm" : "degraded",
    texto: `muda · hace ${edad(m.age_s)}`,
  };
}

export function semaforoGlobal(
  estado: Estado,
  frescuraS: number | null,
): { sem: Semaforo; texto: string } {
  const m = estado.muestra;
  if (!m || frescuraS === null)
    return { sem: "unknown", texto: "sin muestra del hub" };
  if (frescuraS > FRESCURA_MAX_S)
    return { sem: "degraded", texto: `muestra rancia · ${edad(frescuraS)}` };
  const minimo = estado.plano.umbrales?.minimo_maquinas ?? 1;
  const criticoMudo = m.maquinas.some(
    (q) => q.critico && q.estado !== "vivo" && q.age_s !== null,
  );
  if (m.vivas < minimo || criticoMudo)
    return {
      sem: "alarm",
      texto: criticoMudo ? "canario mudo" : `vivas < ${minimo}`,
    };
  if (m.vivas < m.esperadas)
    return { sem: "degraded", texto: `${m.vivas}/${m.esperadas} vivas` };
  return { sem: "running", texto: "flota completa" };
}

// ---- pantalla --------------------------------------------------------------

export function Flota({
  estado,
  frescuraS,
}: {
  estado: Estado;
  frescuraS: number | null;
}) {
  const m = estado.muestra;
  const campeon = estado.campeon;
  const global = semaforoGlobal(estado, frescuraS);

  // Pelea completa mas reciente (dato de archivo: se pinta aunque no haya muestra).
  const gauntlets = Object.values(estado.gauntlets);
  const ultimo = gauntlets.length
    ? gauntlets.reduce((a, b) =>
        Date.parse(a.fecha) >= Date.parse(b.fecha) ? a : b,
      )
    : null;

  // Escalera: union de niveles presentes en campeon y en la ventana viva.
  const bancoNiveles = new Map(campeon ? nivelesDe(campeon) : []);
  const ventanaNiveles = new Map(
    m
      ? Object.entries(m.por_nivel).map(
          ([k, v]) => [Number(k), v] as [number, number],
        )
      : [],
  );
  const niveles = [
    ...new Set([...bancoNiveles.keys(), ...ventanaNiveles.keys()]),
  ].sort((a, b) => a - b);

  // Tendencias desde historia[] (~180 puntos submuestreados por el hub).
  const historia = estado.historia;
  const spanH =
    historia.length > 1
      ? (Date.parse(historia[historia.length - 1].ts) -
          Date.parse(historia[0].ts)) /
        3600000
      : 0;
  const etiquetaSpan = `${spanH.toFixed(1)} h · n=${historia.length}`;
  const serieGrads = historia
    .map((h) => h.grads_per_s)
    .filter((v): v is number => v !== undefined);
  const maxEsperadas = Math.max(
    m?.esperadas ?? 0,
    ...historia.map((h) => h.esperadas ?? 0),
  );
  const serieVivas = historia
    .map((h) => h.vivas)
    .filter((v): v is number => v !== undefined);
  const serieWr = historia
    .map((h) => h.wr)
    .filter((v): v is number => v !== undefined);

  // Censo: una fila por maquina ESPERADA, cruzada con la muestra.
  const esperadas = estado.plano.expected ?? [];
  const porId = new Map((m?.maquinas ?? []).map((q) => [q.id, q]));

  return (
    <div className="space-y-6">
      {/* -- titular ---------------------------------------------------- */}
      <section className="space-y-3">
        <Titulo extra={<EstadoChip estado={global.sem}>{global.texto}</EstadoChip>}>
          Signos vitales
        </Titulo>
        <div className="grid gap-3 md:grid-cols-3">
          <Card>
            <CardContent className="space-y-3 p-4">
              <div className="dlabel dlabel-chrome">Learner</div>
              {m ? (
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                  <Dato label="grad steps" sub="acumulados">
                    {nf(m.grad_steps)}
                  </Dato>
                  <Dato
                    label="grads/s"
                    sub={
                      m.batch !== undefined
                        ? `batch ${m.batch}${m.batch_supuesto ? " (supuesto)" : ""}`
                        : "ventana 60 s"
                    }
                  >
                    {m.grads_per_s?.toFixed(1) ?? "—"}
                  </Dato>
                  <Dato label="buffer" sub="transiciones">
                    {nf(m.buffer)}
                  </Dato>
                </div>
              ) : (
                <SinDato>sin muestra del hub — el learner no se puede leer</SinDato>
              )}
              {serieGrads.length > 1 && (
                <div className="flex items-end justify-between gap-3 border-t border-border pt-2">
                  <div className="dlabel">grads/s · {etiquetaSpan}</div>
                  <Spark datos={serieGrads} />
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardContent className="space-y-3 p-4">
              <div className="dlabel dlabel-chrome">Flota</div>
              {m ? (
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                  <Dato label="vivas" sub={`de ${m.esperadas} esperadas`}>
                    <span
                      className={
                        m.vivas < m.esperadas ? "text-warning-fg" : undefined
                      }
                    >
                      {m.vivas}/{m.esperadas}
                    </span>
                  </Dato>
                  <Dato label="trans/s" sub="ventana 60 s">
                    {nf(Math.round(m.trans_per_s ?? 0))}
                  </Dato>
                  <Dato label="replay ratio" sub="grads·batch / trans">
                    {m.replay_ratio?.toFixed(2) ?? "—"}
                  </Dato>
                </div>
              ) : (
                <SinDato>sin muestra del hub — censo ciego</SinDato>
              )}
              {serieVivas.length > 1 && (
                <div className="flex items-end justify-between gap-3 border-t border-border pt-2">
                  <div className="dlabel">
                    vivas 0–{maxEsperadas} · {etiquetaSpan}
                  </div>
                  <Spark datos={serieVivas} dominio={[0, maxEsperadas]} />
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardContent className="space-y-3 p-4">
              <div className="dlabel dlabel-chrome">Peleas completas · mas reciente</div>
              {ultimo ? (
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                  <Dato
                    label={`gauntlet lvl ${ultimo.dificultad}`}
                    sub={`${ultimo.peleas_ganadas}/${ultimo.peleas_totales} peleas`}
                  >
                    {pc(ultimo.peleas_ganadas / ultimo.peleas_totales, 0)}
                  </Dato>
                  <Dato
                    label="IC 95%"
                    sub={`n=${ultimo.peleas_totales} peleas`}
                  >
                    {ultimo.ic95
                      ? `${pc(ultimo.ic95[0], 0)}–${pc(ultimo.ic95[1], 0)}`
                      : "—"}
                  </Dato>
                  <Dato
                    label="checkpoint"
                    sub={ultimo.fecha.replace("T", " ")}
                  >
                    <span className="block truncate text-sm">{ultimo.checkpoint}</span>
                  </Dato>
                </div>
              ) : (
                <SinDato>sin gauntlets registrados</SinDato>
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      {/* -- censo ------------------------------------------------------ */}
      <section className="space-y-3">
        <Titulo
          extra={
            m ? (
              <span className="tnum font-mono text-[11px] text-muted-foreground">
                muestra de hace {frescuraS === null ? "—" : edad(frescuraS)}
              </span>
            ) : undefined
          }
        >
          Censo de maquinas
        </Titulo>
        <Card>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="dlabel w-36 pl-4">maquina</TableHead>
                  <TableHead className="dlabel w-44">estado</TableHead>
                  <TableHead className="dlabel hidden md:table-cell">rol</TableHead>
                  <TableHead className="dlabel hidden md:table-cell">banda lvl</TableHead>
                  <TableHead className="dlabel hidden lg:table-cell">actor</TableHead>
                  <TableHead className="dlabel text-right">procs</TableHead>
                  <TableHead className="dlabel text-right">steps/s</TableHead>
                  <TableHead className="dlabel hidden pr-4 text-right sm:table-cell">
                    ultimo reporte
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {esperadas.map((e) => {
                  const q = porId.get(e.id);
                  const viva = q?.estado === "vivo";
                  const { sem, texto } = estadoMaquina(q);
                  const critico = e.critico || q?.critico;
                  return (
                    <TableRow
                      key={e.id}
                      className={cn(
                        "border-l-2",
                        sem === "running" && "border-l-state-running",
                        sem === "degraded" && "border-l-state-degraded",
                        sem === "alarm" && "border-l-state-alarm",
                        sem === "unknown" && "border-l-state-unknown",
                        sem === "stopped" && "border-l-state-stopped",
                        !viva && "text-muted-foreground",
                      )}
                    >
                      <TableCell className="pl-4">
                        <div className="font-mono text-[13px] font-semibold text-foreground">
                          {e.id}
                        </div>
                        <div className="mt-0.5 flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
                          <span>{e.host ?? "—"}</span>
                          {critico && (
                            <span className="inline-flex items-center gap-1 font-semibold text-warning-fg">
                              <Bird size={12} aria-hidden />
                              CANARIO
                            </span>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        <EstadoChip estado={sem}>{texto}</EstadoChip>
                      </TableCell>
                      <TableCell className="hidden font-mono text-xs md:table-cell">
                        {e.rol ?? "—"}
                        {e.duenio ? (
                          <span className="text-muted-foreground"> · {e.duenio}</span>
                        ) : null}
                      </TableCell>
                      <TableCell className="tnum hidden font-mono text-xs md:table-cell">
                        {(() => {
                          const real = banda(q?.difficulty);
                          const esperado = banda(e.difficulty);
                          // Actor viejo (sin wire de difficulty): la esperada sola.
                          if (real === null)
                            return (
                              <span className="text-muted-foreground">
                                {esperado ?? "—"}
                              </span>
                            );
                          if (esperado !== null && real !== esperado)
                            return (
                              <span className="inline-flex items-center gap-1.5 font-semibold text-state-degraded-fg">
                                <TriangleAlert size={12} aria-hidden />
                                {real} ≠ {esperado}
                              </span>
                            );
                          return <span>{real}</span>;
                        })()}
                      </TableCell>
                      <TableCell className="hidden font-mono text-xs lg:table-cell">
                        {q?.actor ?? "—"}
                      </TableCell>
                      <TableCell className="tnum text-right font-mono text-xs">
                        {q?.procs ?? "—"}
                        <span className="text-muted-foreground">/{e.procs ?? "—"}</span>
                      </TableCell>
                      <TableCell className="tnum text-right font-mono text-xs">
                        {viva ? nf(Math.round(q!.steps_per_s)) : "—"}
                      </TableCell>
                      <TableCell className="tnum hidden pr-4 text-right font-mono text-xs sm:table-cell">
                        {q ? edad(q.age_s) : "—"}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
            {!m && (
              <div className="border-t border-border p-3">
                <SinDato>
                  sin muestra del hub: el censo muestra solo lo ESPERADO del plano
                </SinDato>
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      {/* -- escalera --------------------------------------------------- */}
      <section className="space-y-3">
        <Titulo
          extra={
            campeon ? (
              <span className="tnum font-mono text-[11px] text-muted-foreground">
                campeon v{campeon.weights_version} · media {pc(campeon.wr_media)}
              </span>
            ) : undefined
          }
        >
          Escalera por nivel
        </Titulo>
        <Card>
          <CardContent className="space-y-4 p-4">
            {niveles.length ? (
              <>
                {serieWr.length > 1 && (
                  <div className="flex items-end justify-between gap-3 border-b border-border pb-3">
                    <div className="dlabel">
                      wr ventana viva (todos los niveles) · n≈200 · {etiquetaSpan}
                    </div>
                    <div className="flex items-end gap-2">
                      <Spark datos={serieWr} />
                      <span className="tnum font-mono text-xs">
                        {pc(serieWr[serieWr.length - 1])}
                      </span>
                    </div>
                  </div>
                )}
                <div className="space-y-2">
                  {niveles.map((n) => {
                    const banco = bancoNiveles.get(n) ?? null;
                    const ventana = ventanaNiveles.get(n) ?? null;
                    return (
                      <div key={n} className="flex items-center gap-3">
                        <div className="dlabel w-12 shrink-0">lvl {n}</div>
                        <PistaNivel banco={banco} ventana={ventana} />
                        <div className="tnum w-14 shrink-0 text-right font-mono text-xs">
                          {banco === null ? "—" : pc(banco, 1)}
                        </div>
                        <div className="tnum hidden w-14 shrink-0 text-right font-mono text-xs text-muted-foreground sm:block">
                          {ventana === null ? "—" : pc(ventana, 1)}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="flex items-center justify-between gap-4 border-t border-border pt-3">
                  <LeyendaEscalera />
                  {!m && (
                    <span className="font-mono text-[11px] text-warning-fg">
                      sin muestra: la ventana viva no se puede pintar
                    </span>
                  )}
                </div>
              </>
            ) : (
              <SinDato>
                sin campeon ni muestra: no hay niveles que pintar
              </SinDato>
            )}
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
