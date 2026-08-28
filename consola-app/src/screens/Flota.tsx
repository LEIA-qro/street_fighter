// FLOTA -- el censo contra fleet.json y la escalera. Reglas heredadas del
// diseño que ya pasó revisión: filas fantasma (una máquina esperada que calla
// NO desaparece), frescura visible en cada dato, desconocido nunca verde, y
// las dos escaleras (banco vs ventana viva) dibujadas DISTINTO porque son
// medidas distintas.
import type { Estado } from "@/lib/api"
import { edad, nf } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

function Tile({ k, v, sub, tono }: { k: string; v: React.ReactNode; sub: string; tono: "ok" | "mal" | "tibio" | "" }) {
  const franja = { ok: "border-l-state-running", mal: "border-l-state-alarm", tibio: "border-l-state-degraded", "": "border-l-border" }[tono]
  return (
    <Card className={`border-l-[3px] ${franja} gap-2 py-4`}>
      <CardHeader className="px-4"><CardTitle className="font-mono text-[10px] font-medium uppercase tracking-[.14em] text-muted-foreground">{k}</CardTitle></CardHeader>
      <CardContent className="px-4">
        <div className="font-mono text-3xl font-semibold tnum leading-none">{v}</div>
        <div className="mt-1 text-xs text-muted-foreground">{sub}</div>
      </CardContent>
    </Card>
  )
}

export default function Flota({ d }: { d: Estado }) {
  const m = d.muestra
  const esperadas = d.plano.expected ?? []
  const gl = Object.values(d.gauntlets ?? {}).sort((a, b) => b.dificultad - a.dificultad)[0]
  const c = d.campeon
  const niveles = [...new Set([
    ...Object.keys(m?.por_nivel ?? {}),
    ...Object.keys(c ?? {}).filter(k => k.startsWith("wr_lvl")).map(k => k.slice(6)),
  ])].map(Number).filter(n => !isNaN(n)).sort((a, b) => a - b)

  return (
    <div className="space-y-8">
      <div className="grid gap-3 md:grid-cols-3">
        <Tile k="learner" tono={m ? "ok" : "mal"}
          v={m ? nf(m.grad_steps) : "—"}
          sub={m ? `${m.grads_per_s ?? "—"} grads/s · buffer ${nf(m.buffer)}` : "el hub no logra medir"} />
        <Tile k="flota" tono={m ? (m.vivas >= m.esperadas ? "ok" : m.vivas >= (d.plano.umbrales?.minimo_maquinas ?? 2) ? "tibio" : "mal") : "mal"}
          v={<>{m?.vivas ?? "—"}<span className="text-base text-muted-foreground">/{m?.esperadas ?? esperadas.length}</span></>}
          sub={m ? `${nf(Math.round(m.trans_per_s ?? 0))} trans/s · ratio ${m.replay_ratio ?? "—"}` : "—"} />
        <Tile k={`peleas completas · lvl ${gl?.dificultad ?? "—"}`} tono={gl ? (gl.peleas_ganadas / gl.peleas_totales >= 0.9 ? "ok" : "tibio") : ""}
          v={gl ? `${(gl.peleas_ganadas / gl.peleas_totales * 100).toFixed(1)}%` : "—"}
          sub={gl ? `${gl.peleas_ganadas}/${gl.peleas_totales} · IC95 ${gl.ic95 ? `${(gl.ic95[0] * 100).toFixed(0)}–${(gl.ic95[1] * 100).toFixed(0)}%` : "—"}` : "sin gauntlet grabado"} />
      </div>

      <section>
        <h2 className="mb-1 font-display text-[13px] font-semibold uppercase tracking-[.13em] text-muted-foreground">Censo de máquinas</h2>
        <p className="mb-3 max-w-[72ch] text-xs text-muted-foreground">
          Lo que <strong className="text-foreground">debería</strong> correr según <code className="font-mono">fleet.json</code>, contra lo que el learner reporta. Una máquina que calla no desaparece: sale muda, con desde cuándo.
        </p>
        <div className="grid gap-2">
          {(m?.maquinas ?? esperadas.map(e => ({ id: e.id, estado: "MUDA" as const, actor: null, procs: null, steps_per_s: 0, age_s: null, critico: !!e.critico }))).map(x => {
            const e = esperadas.find(q => q.id === x.id)
            const vivo = x.estado === "vivo"
            const nunca = x.age_s === null
            return (
              <Card key={x.id} className={`grid grid-cols-[1.5fr_auto_auto] items-center gap-x-5 gap-y-1 border-l-[3px] px-4 py-3 md:grid-cols-[1.6fr_.8fr_.7fr_.7fr_.8fr] ${vivo ? "border-l-state-running" : nunca ? "border-l-state-unknown" : "border-l-state-alarm bg-muted"}`}>
                <div className="min-w-0">
                  <span className="text-[15px] font-semibold">{x.id}</span>
                  {x.critico && <Badge variant="outline" className="ml-2 border-warning font-mono text-[9px] tracking-[.12em] text-warning-fg">CANARIO</Badge>}
                  <div className="truncate text-xs text-muted-foreground">{e?.host} · {e?.duenio}</div>
                </div>
                <Dato v={e?.difficulty ?? "—"} u="niveles" oculta />
                <Dato v={x.procs ?? "—"} u="procs" />
                <Dato v={vivo ? Math.round(x.steps_per_s) : "—"} u="steps/s" />
                <Dato v={edad(x.age_s)} u={vivo ? "visto" : "callada"} />
              </Card>
            )
          })}
        </div>
      </section>

      <section>
        <h2 className="mb-1 font-display text-[13px] font-semibold uppercase tracking-[.13em] text-muted-foreground">Escalera por nivel</h2>
        <p className="mb-3 max-w-[72ch] text-xs text-muted-foreground">
          Barra = <strong className="text-foreground">banco del campeón</strong> (greedy + desfase, n=48). Marca = <strong className="text-foreground">ventana viva</strong> (n≈200, con exploración). <em>Rounds de apertura</em> las dos — no confundir con peleas.
        </p>
        <Card className="gap-0 px-5 py-4">
          {niveles.map(l => {
            const banco = (c?.[`wr_lvl${l}`] as number | undefined) ?? null
            const viva = m?.por_nivel?.[String(l)] ?? null
            return (
              <div key={l} className="grid grid-cols-[52px_1fr_60px_60px] items-center gap-3 border-b border-border/50 py-1.5 last:border-0">
                <span className="font-mono text-[11px] tracking-[.08em] text-muted-foreground">LVL {l}</span>
                <div className="relative h-4 overflow-hidden rounded-[3px] bg-muted">
                  {banco !== null && <div className="absolute inset-y-0 left-0 rounded-[3px] bg-primary" style={{ width: `${banco * 100}%` }} />}
                  {viva !== null && <div className="absolute -inset-y-0.5 w-0.5 bg-foreground" style={{ left: `calc(${viva * 100}% - 1px)` }} />}
                </div>
                <Dato v={banco !== null ? (banco * 100).toFixed(1) : "—"} u="banco" />
                <Dato v={viva !== null ? (viva * 100).toFixed(1) : "—"} u="viva" oculta />
              </div>
            )
          })}
          {niveles.length === 0 && <p className="py-3 text-sm italic text-muted-foreground">sin datos por nivel todavía</p>}
        </Card>
      </section>
    </div>
  )
}

function Dato({ v, u, oculta }: { v: React.ReactNode; u: string; oculta?: boolean }) {
  return (
    <div className={`text-right font-mono text-sm tnum ${oculta ? "hidden md:block" : ""}`}>
      {v}
      <span className="block text-[10px] uppercase tracking-[.08em] text-muted-foreground">{u}</span>
    </div>
  )
}
