// JUGABLE -- el quién-contra-quién. La sección que en el Gradio viejo era un
// mar de dropdowns repetidos, colapsada al patrón del plan: DOS tarjetas de
// jugador con el mismo componente parametrizado, y el resto por revelado
// progresivo. v1 honesto: esta pantalla ARMA el comando exacto y lo copia --
// lanzar procesos remotos llega con la fase de jobs del hub, y decir que un
// botón "lanza" cuando no lanza sería otro botón mentiroso de los que
// acabamos de podar.
import { useMemo, useState } from "react"
import type { Estado } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { toast } from "sonner"

const RIVALES = ["RANDOM", "RYU", "KEN", "CHUNLI", "GUILE", "BLANKA", "ZANGIEF", "DHALSIM", "EHONDA", "BALROG", "VEGA", "SAGAT", "MBISON"] as const
type TipoP2 = "humano" | "cpu" | "apex" | "sb3"

export default function Jugable({ d }: { d: Estado }) {
  const campeon = d.campeon
  const [p2, setP2] = useState<TipoP2>("humano")
  const [rival, setRival] = useState<string>("RANDOM")
  const [nivel, setNivel] = useState("8")

  const comando = useMemo(() => {
    if (p2 === "humano")
      return `.venv\\Scripts\\python.exe src\\scripts\\stand_leia.py --opponent ${rival}`
    if (p2 === "cpu")
      return `.venv/bin/python tools/watch_es.py --rainbow-ckpt benchmarks/apex_milestones/apex_escalera_best.pt --difficulty ${nivel} --desync-max 30 --speed 0.5`
    if (p2 === "apex")
      return `.venv\\Scripts\\python.exe src\\scripts\\stand_leia.py --p2-type apex --opponent ${rival}`
    return `.venv\\Scripts\\python.exe src\\scripts\\stand_leia.py --p2-type sb3 --opponent ${rival}`
  }, [p2, rival, nivel])

  const rig = p2 === "cpu" ? "stable-retro · corre en CUALQUIER máquina (la ventana del emulador sale local)" : "BizHawk · corre en un rig Windows (desktop / laptop del equipo)"

  return (
    <div className="space-y-6">
      <p className="max-w-[74ch] text-sm text-muted-foreground">
        Arma el enfrentamiento y copia el comando listo. <strong className="text-foreground">P1 siempre es el campeón</strong> (Ryu, como entrenó); tú eliges quién lo reta.
      </p>

      <div className="grid gap-3 md:grid-cols-2">
        <Card className="border-champion gap-2 py-4">
          <CardHeader className="px-5">
            <CardTitle className="flex items-baseline justify-between font-display text-base">
              <span>P1 · La IA</span>
              <Badge className="bg-champion font-mono text-champion-foreground">campeón</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="px-5 font-mono text-sm">
            {campeon ? (
              <>
                <div className="text-lg font-semibold">v{campeon.weights_version}</div>
                <div className="text-xs text-muted-foreground">
                  {String(campeon.archivo ?? "apex_escalera_best.pt")} · {(campeon.wr_media * 100).toFixed(1)}% rounds (n=48/nivel)
                </div>
              </>
            ) : <span className="italic text-muted-foreground">sin campeón coronado</span>}
          </CardContent>
        </Card>

        <Card className="gap-2 py-4">
          <CardHeader className="px-5"><CardTitle className="font-display text-base">P2 · El retador</CardTitle></CardHeader>
          <CardContent className="space-y-3 px-5">
            <Select value={p2} onValueChange={v => setP2(v as TipoP2)}>
              <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="humano">🎮 Humano (pad en puerto 2)</SelectItem>
                <SelectItem value="cpu">🕹️ CPU del juego (elige nivel)</SelectItem>
                <SelectItem value="apex">🤖 Otro Ape-X (.pt)</SelectItem>
                <SelectItem value="sb3">📦 Modelo clásico (PPO/DQN)</SelectItem>
              </SelectContent>
            </Select>

            {p2 === "cpu" ? (
              <Select value={nivel} onValueChange={setNivel}>
                <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {[1, 2, 3, 4, 5, 6, 7, 8].map(n => (
                    <SelectItem key={n} value={String(n)}>Nivel {n}{n === 8 ? " · HARD" : ""}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <Select value={rival} onValueChange={setRival}>
                <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {RIVALES.map(r => <SelectItem key={r} value={r}>{r === "RANDOM" ? "🎲 Aleatorio por round" : r}</SelectItem>)}
                </SelectContent>
              </Select>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="gap-3 py-4">
        <CardHeader className="px-5">
          <CardTitle className="font-mono text-[10px] font-medium uppercase tracking-[.14em] text-muted-foreground">El comando</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 px-5">
          <code className="block overflow-x-auto rounded-md bg-muted p-3 font-mono text-xs whitespace-pre">{comando}</code>
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={() => { navigator.clipboard.writeText(comando); toast.success("Comando copiado") }}>Copiar comando</Button>
            <span className="text-xs text-muted-foreground">{rig}</span>
          </div>
          <Separator />
          <p className="text-xs text-muted-foreground">
            Esta pantalla <strong className="text-foreground">no lanza procesos todavía</strong> — arma el comando exacto y verificado. El botón de lanzar llega con la fase de jobs del hub; hasta entonces, un botón que dijera “lanzar” sin lanzar sería un botón mentiroso.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
