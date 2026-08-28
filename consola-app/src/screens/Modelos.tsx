// MODELOS -- el comparador. Junta las tres fuentes de verdad que hoy viven
// separadas: las coronaciones del selector (rounds, n=48/nivel), las actas de
// gauntlet (PELEAS completas, con n e IC), y el campeón vigente. Cada número
// lleva su n al lado: un porcentaje sin su n es una opinión.
import type { Estado } from "@/lib/api"
import { edad, pc } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

export default function Modelos({ d }: { d: Estado }) {
  const c = d.campeon
  const gaunt = Object.values(d.gauntlets ?? {}).sort((a, b) => b.dificultad - a.dificultad)
  const cors = [...(d.coronaciones ?? [])].reverse()
  const nivelesDe = (r: Record<string, unknown>) =>
    Object.keys(r).filter(k => k.startsWith("wr_lvl")).map(k => +k.slice(6)).sort((a, b) => a - b)

  return (
    <div className="space-y-8">
      {c && (
        <Card className="border-champion gap-2 py-4">
          <CardHeader className="px-5">
            <CardTitle className="font-display text-[17px] text-champion dark:text-champion">
              v{c.weights_version} · {pc(c.wr_media)} de rounds de apertura
            </CardTitle>
            <p className="font-mono text-xs text-muted-foreground">
              {c.archivo} · coronado hace {edad(d.ahora - c.t)} · n=48 por nivel · greedy + desfase
            </p>
          </CardHeader>
          <CardContent className="px-5">
            <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${nivelesDe(c).length}, 1fr)` }}>
              {nivelesDe(c).map(l => {
                const v = c[`wr_lvl${l}`] as number
                return (
                  <div key={l} className="text-center">
                    <div className="font-mono text-[10px] text-muted-foreground">LVL {l}</div>
                    <div className="font-mono text-lg font-semibold tnum">{Math.round(v * 100)}</div>
                    <div className="h-[3px] rounded-sm bg-muted"><div className="h-full rounded-sm bg-champion" style={{ width: `${v * 100}%` }} /></div>
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}

      <section>
        <h2 className="mb-1 font-display text-[13px] font-semibold uppercase tracking-[.13em] text-muted-foreground">Peleas completas por dificultad</h2>
        <p className="mb-3 max-w-[74ch] text-xs text-muted-foreground">
          Al mejor de tres, con desfase aleatorio. <strong className="text-foreground">El número que contesta “¿le gana al juego?”</strong> — los rounds de arriba corren por encima.
        </p>
        {gaunt.length === 0 && <Card className="px-5 py-4 text-sm italic text-muted-foreground">sin actas de gauntlet · <code className="font-mono not-italic">tools/grabar_gauntlet.py --difficulty N --repeticiones 30</code></Card>}
        <div className="grid gap-3">
          {gaunt.map(a => {
            const agg: Record<string, [number, number]> = {}
            a.rivales.forEach(r => { agg[r.rival] = agg[r.rival] ?? [0, 0]; agg[r.rival][1]++; if (r.gano) agg[r.rival][0]++ })
            const orden = Object.entries(agg).sort((x, y) => x[1][0] / x[1][1] - y[1][0] / y[1][1])
            return (
              <Card key={a.dificultad} className="gap-2 px-5 py-4">
                <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                  <span className="font-mono text-[10px] uppercase tracking-[.14em] text-muted-foreground">nivel {a.dificultad}</span>
                  <span className="font-mono text-3xl font-semibold tnum">{(a.peleas_ganadas / a.peleas_totales * 100).toFixed(1)}<span className="text-sm text-muted-foreground">%</span></span>
                  <span className="font-mono text-sm text-muted-foreground tnum">{a.peleas_ganadas}/{a.peleas_totales}</span>
                  <span className="font-mono text-xs text-muted-foreground tnum">
                    IC95 {a.ic95 ? `${(a.ic95[0] * 100).toFixed(0)}–${(a.ic95[1] * 100).toFixed(0)}%` : "—"} · desfase ≤{a.desync_max ?? 0}
                    {!a.desync_max && <Badge variant="outline" className="ml-2 border-warning text-warning-fg">determinista</Badge>}
                  </span>
                </div>
                <div className="text-xs text-muted-foreground">{a.checkpoint} · {a.fecha?.replace("T", " ")}</div>
                <div className="mt-2 grid gap-1.5 sm:grid-cols-3 lg:grid-cols-4">
                  {orden.map(([riv, [g, t]]) => {
                    const p = g / t
                    const tono = p < 0.6 ? "border-state-alarm text-state-alarm-fg" : p < 0.95 ? "border-state-degraded" : "border-border"
                    const barra = p < 0.6 ? "bg-state-alarm" : p < 0.95 ? "bg-state-degraded" : "bg-success"
                    return (
                      <div key={riv} className={`rounded-md border px-2.5 py-1.5 font-mono text-[11.5px] ${tono}`}>
                        <div className="flex justify-between font-medium"><span>{riv}</span><span className="tnum">{g}/{t}</span></div>
                        <div className="mt-1 h-[3px] rounded-sm bg-muted"><div className={`h-full rounded-sm ${barra}`} style={{ width: `${p * 100}%` }} /></div>
                      </div>
                    )
                  })}
                </div>
              </Card>
            )
          })}
        </div>
      </section>

      <section>
        <h2 className="mb-3 font-display text-[13px] font-semibold uppercase tracking-[.13em] text-muted-foreground">Historial de coronaciones</h2>
        <Card className="overflow-hidden p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="font-mono text-[10px] uppercase tracking-[.12em]">versión</TableHead>
                <TableHead className="text-right font-mono text-[10px] uppercase tracking-[.12em]">media</TableHead>
                <TableHead className="hidden font-mono text-[10px] uppercase tracking-[.12em] md:table-cell">por nivel</TableHead>
                <TableHead className="text-right font-mono text-[10px] uppercase tracking-[.12em]">cuándo</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {cors.map(r => (
                <TableRow key={String(r.t)}>
                  <TableCell className="font-mono">v{r.weights_version}{c?.weights_version === r.weights_version && <Badge className="ml-2 bg-champion text-champion-foreground">vigente</Badge>}</TableCell>
                  <TableCell className="text-right font-mono tnum">{pc(r.wr_media)}</TableCell>
                  <TableCell className="hidden font-mono text-xs text-muted-foreground tnum md:table-cell">
                    {nivelesDe(r).map(l => Math.round((r[`wr_lvl${l}`] as number) * 100)).join(" · ")}
                  </TableCell>
                  <TableCell className="text-right font-mono text-xs text-muted-foreground tnum">{edad(d.ahora - r.t)}</TableCell>
                </TableRow>
              ))}
              {cors.length === 0 && <TableRow><TableCell colSpan={4} className="italic text-muted-foreground">todavía ninguna</TableCell></TableRow>}
            </TableBody>
          </Table>
        </Card>
      </section>
    </div>
  )
}
