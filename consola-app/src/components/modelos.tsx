// MODELOS — el comparador: campeon vigente, actas de peleas completas
// (gauntlets) y el historial de coronaciones.
import { Crown } from "lucide-react";
import type { Campeon, Estado, Gauntlet } from "@/lib/api";
import { edad, pc } from "@/lib/api";
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
  SinDato,
  Titulo,
  fechaCorta,
  nivelesDe,
} from "@/components/comunes";

// El color de un rival lo decide su win rate en peleas completas:
// <60% alarma, <95% degradado, resto normal. Los problematicos SALTAN.
function claseRival(wr: number): string {
  if (wr < 0.6) return "bg-state-alarm-subtle text-state-alarm-fg";
  if (wr < 0.95) return "bg-state-degraded-subtle text-state-degraded-fg";
  return "bg-state-running-subtle text-state-running-fg";
}

function desfaseDe(g: Gauntlet, campeon: Campeon | null): string {
  const mt = /v(\d+)/.exec(g.checkpoint);
  if (!mt || !campeon) return "—";
  const d = campeon.weights_version - Number(mt[1]);
  return d === 0 ? "al dia" : d > 0 ? `−${d} versiones` : `+${-d} versiones`;
}

function ActaGauntlet({
  g,
  campeon,
  ahora,
}: {
  g: Gauntlet;
  campeon: Campeon | null;
  ahora: number;
}) {
  // Desglose por rival AGREGADO sobre rivales[] (con repeticiones, un rival
  // puede aparecer varias veces).
  const porRival = new Map<string, { ganadas: number; n: number }>();
  for (const r of g.rivales) {
    const acc = porRival.get(r.rival) ?? { ganadas: 0, n: 0 };
    acc.n += 1;
    if (r.gano) acc.ganadas += 1;
    porRival.set(r.rival, acc);
  }
  const rivales = [...porRival.entries()].sort(
    (a, b) =>
      a[1].ganadas / a[1].n - b[1].ganadas / b[1].n ||
      a[0].localeCompare(b[0]),
  );
  const wr = g.peleas_totales ? g.peleas_ganadas / g.peleas_totales : null;
  const haceS = ahora - Date.parse(g.fecha) / 1000;

  return (
    <Card>
      <CardContent className="space-y-4 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="dlabel">
            Gauntlet · dificultad {g.dificultad} · peleas completas al mejor de{" "}
            {g.rounds_para_ganar ?? 2}
          </div>
          <span className="tnum font-mono text-[11px] text-muted-foreground">
            {g.fecha.replace("T", " ")} · hace {edad(haceS)}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div>
            <div className="dlabel">peleas ganadas</div>
            <div className="tnum mt-1 font-mono text-[28px] font-semibold leading-none">
              {pc(wr, 0)}
            </div>
            <div className="tnum mt-1 font-mono text-[11px] text-muted-foreground">
              {g.peleas_ganadas}/{g.peleas_totales} peleas · n={g.peleas_totales}
            </div>
          </div>
          <Dato label="IC 95%" sub={`sobre n=${g.peleas_totales} peleas`}>
            {g.ic95 ? `${pc(g.ic95[0], 0)}–${pc(g.ic95[1], 0)}` : "—"}
          </Dato>
          <Dato label="checkpoint" sub={`desfase: ${desfaseDe(g, campeon)}`}>
            <span className="block truncate text-sm">{g.checkpoint}</span>
          </Dato>
          <Dato
            label="condiciones"
            sub={`desync ≤ ${g.desync_max ?? "—"} · semilla ${g.semilla ?? "—"}`}
          >
            <span className="text-sm">
              ×{g.repeticiones ?? 1} rep{(g.repeticiones ?? 1) > 1 ? "s" : ""}
            </span>
          </Dato>
        </div>

        <div className="space-y-1.5 border-t border-border pt-3">
          <div className="dlabel">por rival · peleas completas ganadas</div>
          <div className="flex flex-wrap gap-1.5">
            {rivales.map(([nombre, r]) => (
              <span
                key={nombre}
                className={cn(
                  "tnum inline-flex items-baseline gap-1.5 rounded-sm px-2 py-1 font-mono text-[11px] font-medium",
                  claseRival(r.ganadas / r.n),
                )}
              >
                {nombre}
                <span className="font-semibold">
                  {r.ganadas}/{r.n}
                </span>
              </span>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function Modelos({
  estado,
  ahora,
}: {
  estado: Estado;
  ahora: number;
}) {
  const campeon = estado.campeon;
  const niveles = campeon ? nivelesDe(campeon) : [];
  const gauntlets = Object.values(estado.gauntlets).sort(
    (a, b) => b.dificultad - a.dificultad,
  );
  const coronaciones = [...estado.coronaciones].sort((a, b) => b.t - a.t);
  const nivelesCoro = [
    ...new Set(coronaciones.flatMap((c) => nivelesDe(c).map(([n]) => n))),
  ].sort((a, b) => a - b);

  return (
    <div className="space-y-6">
      {/* -- campeon vigente: LA unica tarjeta con el lima --------------- */}
      <section className="space-y-3">
        <Titulo>Campeon vigente</Titulo>
        {campeon ? (
          <Card className="border-l-2 border-l-champion">
            <CardContent className="space-y-4 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="flex items-center gap-3">
                  <span className="inline-flex items-center gap-1.5 rounded-sm bg-state-champion-subtle px-2 py-1 font-mono text-xs font-semibold uppercase tracking-wide text-state-champion-fg">
                    <Crown size={13} aria-hidden />
                    v{campeon.weights_version}
                  </span>
                  <span className="font-mono text-xs text-muted-foreground">
                    {campeon.archivo ?? "—"}
                  </span>
                </div>
                <span className="tnum font-mono text-[11px] text-muted-foreground">
                  coronado {fechaCorta(campeon.t)} · hace {edad(ahora - campeon.t)}
                </span>
              </div>

              <div className="flex flex-wrap items-end gap-6">
                <div>
                  <div className="dlabel">wr media</div>
                  <div className="tnum mt-1 font-mono text-[28px] font-semibold leading-none">
                    {pc(campeon.wr_media)}
                  </div>
                  <div className="mt-1 font-mono text-[11px] text-muted-foreground">
                    banco greedy n=48 por nivel · rounds de apertura
                  </div>
                </div>
                <div className="min-w-56 flex-1 space-y-1">
                  {niveles.map(([n, v]) => (
                    <div key={n} className="flex items-center gap-2">
                      <div className="dlabel w-10 shrink-0">lvl {n}</div>
                      <div className="h-2 flex-1 rounded-[2px] bg-muted">
                        <div
                          className="h-full rounded-[2px] bg-primary"
                          style={{ width: `${v * 100}%` }}
                        />
                      </div>
                      <div className="tnum w-13 shrink-0 text-right font-mono text-[11px]">
                        {pc(v, 1)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        ) : (
          <SinDato>sin campeon coronado todavia</SinDato>
        )}
      </section>

      {/* -- actas de peleas completas ---------------------------------- */}
      <section className="space-y-3">
        <Titulo
          extra={
            <span className="font-mono text-[11px] text-muted-foreground">
              peleas completas, no rounds — medida distinta al banco
            </span>
          }
        >
          Actas de gauntlet
        </Titulo>
        {gauntlets.length ? (
          <div className="space-y-3">
            {gauntlets.map((g) => (
              <ActaGauntlet
                key={g.dificultad}
                g={g}
                campeon={campeon}
                ahora={ahora}
              />
            ))}
          </div>
        ) : (
          <SinDato>sin gauntlets registrados todavia</SinDato>
        )}
      </section>

      {/* -- coronaciones ----------------------------------------------- */}
      <section className="space-y-3">
        <Titulo
          extra={
            <span className="tnum font-mono text-[11px] text-muted-foreground">
              {coronaciones.length} coronaciones · banco greedy n=48 · rounds de
              apertura
            </span>
          }
        >
          Historial de coronaciones
        </Titulo>
        <Card>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="dlabel pl-4">version</TableHead>
                  <TableHead className="dlabel text-right">media</TableHead>
                  {nivelesCoro.map((n) => (
                    <TableHead
                      key={n}
                      className="dlabel hidden text-right lg:table-cell"
                    >
                      L{n}
                    </TableHead>
                  ))}
                  <TableHead className="dlabel pr-4 text-right">hace</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {coronaciones.map((c) => {
                  const vigente = campeon?.weights_version === c.weights_version;
                  const porNivel = new Map(nivelesDe(c));
                  return (
                    <TableRow
                      key={`${c.weights_version}-${c.t}`}
                      className={cn(
                        vigente && "border-l-2 border-l-champion bg-state-champion-subtle/40",
                      )}
                    >
                      <TableCell className="pl-4">
                        <span className="tnum font-mono text-xs font-semibold">
                          v{c.weights_version}
                        </span>
                        {vigente && (
                          <span className="ml-2 inline-flex items-center gap-1 font-mono text-[10px] font-semibold uppercase tracking-wide text-state-champion-fg">
                            <Crown size={11} aria-hidden />
                            vigente
                          </span>
                        )}
                      </TableCell>
                      <TableCell className="tnum text-right font-mono text-xs font-semibold">
                        {pc(c.wr_media)}
                      </TableCell>
                      {nivelesCoro.map((n) => {
                        const v = porNivel.get(n);
                        return (
                          <TableCell
                            key={n}
                            className="tnum hidden text-right font-mono text-[11px] text-muted-foreground lg:table-cell"
                          >
                            {v === undefined ? "—" : pc(v, 0)}
                          </TableCell>
                        );
                      })}
                      <TableCell className="tnum pr-4 text-right font-mono text-xs text-muted-foreground">
                        {edad(ahora - c.t)}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
