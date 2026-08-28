import { useEffect, useState } from "react"
import { leerEstado, edad, type Estado } from "@/lib/api"
import Flota from "@/screens/Flota"
import Jugable from "@/screens/Jugable"
import Modelos from "@/screens/Modelos"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Toaster } from "@/components/ui/sonner"

export default function App() {
  const [d, setD] = useState<Estado | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let vivo = true
    const tick = () => leerEstado()
      .then(x => { if (vivo) { setD(x); setErr(null) } })
      .catch(e => { if (vivo) setErr(String(e)) })
    tick()
    const id = setInterval(tick, 5000)
    return () => { vivo = false; clearInterval(id) }
  }, [])

  const m = d?.muestra
  const edadM = m && d ? d.ahora - Date.parse(m.ts) / 1000 : null
  const rancio = edadM !== null && edadM > 180

  return (
    <div className="mx-auto max-w-[1200px] px-5 pb-20">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-border py-6">
        <div>
          <h1 className="font-display text-[26px] font-bold leading-none">
            CONSOLA <span className="text-primary">LEIA</span>
          </h1>
          <p className="mt-1 text-xs text-muted-foreground">{d?.plano.run?.descripcion ?? "…"}</p>
        </div>
        <div className="text-right font-mono text-xs">
          {err ? <span className="text-state-alarm-fg">● hub inalcanzable</span>
            : m ? <span className={rancio ? "text-state-degraded-fg" : "text-state-running-fg"}>● medido hace {edad(edadM)}</span>
            : <span className="text-state-alarm-fg">● sin contacto con el learner</span>}
          {d && <div className="text-muted-foreground">hub arriba hace {edad(d.ahora - d.hub_desde)}</div>}
        </div>
      </header>

      <Tabs defaultValue="flota" className="mt-5">
        <TabsList className="font-mono">
          <TabsTrigger value="flota">Flota</TabsTrigger>
          <TabsTrigger value="jugable">Jugable</TabsTrigger>
          <TabsTrigger value="modelos">Modelos</TabsTrigger>
        </TabsList>
        <TabsContent value="flota" className="mt-5">{d && <Flota d={d} />}</TabsContent>
        <TabsContent value="jugable" className="mt-5">{d && <Jugable d={d} />}</TabsContent>
        <TabsContent value="modelos" className="mt-5">{d && <Modelos d={d} />}</TabsContent>
      </Tabs>
      {!d && !err && <p className="py-10 text-sm italic text-muted-foreground">cargando…</p>}
      {err && !d && <p className="py-10 text-sm text-state-alarm-fg">No llego al hub: arranca <code className="font-mono">tools/leia_hub.py --serve 8099</code></p>}
      <Toaster position="bottom-right" />
    </div>
  )
}
