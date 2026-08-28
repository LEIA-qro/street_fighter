// CONFIG — editor del plano de control (fleet.json) contra /api/fleet.
// El documento viaja COMPLETO en el PUT: lo que la consola no edita
// (_comentario, schema, claves desconocidas) se preserva tal cual.
import { useCallback, useEffect, useMemo, useState } from "react";
import { Plus, RotateCcw, Save, Trash2, TriangleAlert } from "lucide-react";
import { toast } from "sonner";
import {
  guardarFleet,
  leerFleet,
  type FleetDoc,
  type FleetMaquina,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SinDato, Titulo } from "@/components/comunes";

// Campos numericos editados como TEXTO (para poder teclear "0." sin pelear
// con el parser); se convierten y validan al guardar.
type Borrador = {
  doc: FleetDoc; // el documento original, con sus claves desconocidas
  maquinas: {
    original: FleetMaquina; // claves no editables preservadas
    id: string;
    host: string;
    duenio: string;
    rol: string;
    procs: string;
    difficulty: string;
    critico: boolean;
    notas: string;
  }[];
  umbrales: { clave: string; texto: string }[];
};

function aBorrador(doc: FleetDoc): Borrador {
  return {
    doc,
    maquinas: (doc.expected ?? []).map((m) => ({
      original: m,
      id: m.id ?? "",
      host: m.host ?? "",
      duenio: m.duenio ?? "",
      rol: m.rol ?? "",
      procs: m.procs === undefined ? "" : String(m.procs),
      difficulty: m.difficulty ?? "",
      critico: Boolean(m.critico),
      notas: m.notas ?? "",
    })),
    umbrales: Object.entries(doc.umbrales ?? {}).map(([clave, v]) => ({
      clave,
      texto: String(v),
    })),
  };
}

/** Reconstruye el documento completo. Lanza Error con mensaje local si un
 *  numero no parsea — mejor fallar aqui que mandar NaN al hub. */
function aDocumento(b: Borrador): FleetDoc {
  const expected: FleetMaquina[] = b.maquinas.map((m, i) => {
    const fila: FleetMaquina = { ...m.original, id: m.id.trim(), host: m.host.trim() };
    const pon = (k: keyof FleetMaquina, v: string) => {
      const limpio = v.trim();
      if (limpio) (fila as Record<string, unknown>)[k] = limpio;
      else delete (fila as Record<string, unknown>)[k];
    };
    pon("duenio", m.duenio);
    pon("rol", m.rol);
    pon("difficulty", m.difficulty);
    pon("notas", m.notas);
    if (m.procs.trim() === "") delete fila.procs;
    else {
      const n = Number(m.procs);
      if (!Number.isFinite(n))
        throw new Error(`maquina ${i + 1}: procs "${m.procs}" no es numero`);
      fila.procs = n;
    }
    if (m.critico) fila.critico = true;
    else delete fila.critico;
    return fila;
  });
  const umbrales: Record<string, number> = {};
  for (const u of b.umbrales) {
    const n = Number(u.texto);
    if (u.texto.trim() === "" || !Number.isFinite(n))
      throw new Error(`umbral ${u.clave}: "${u.texto}" no es numero`);
    umbrales[u.clave] = n;
  }
  return { ...b.doc, expected, umbrales };
}

const CAMPO =
  "h-7 rounded-sm border-input bg-background px-2 font-mono text-xs";

export function Config() {
  const [borrador, setBorrador] = useState<Borrador | null>(null);
  const [base, setBase] = useState<string>(""); // JSON del ultimo doc servido
  const [errorCarga, setErrorCarga] = useState<string | null>(null);
  const [errorGuardar, setErrorGuardar] = useState<string | null>(null);
  const [guardando, setGuardando] = useState(false);

  const cargar = useCallback(async () => {
    setErrorCarga(null);
    try {
      const doc = await leerFleet();
      setBorrador(aBorrador(doc));
      setBase(JSON.stringify(doc));
      setErrorGuardar(null);
    } catch (e) {
      setErrorCarga(e instanceof Error ? e.message : String(e));
    }
  }, []);
  useEffect(() => {
    cargar();
  }, [cargar]);

  const sucio = useMemo(() => {
    if (!borrador) return false;
    try {
      return JSON.stringify(aDocumento(borrador)) !== base;
    } catch {
      return true; // hay un campo no parseable: hay cambios (invalidos)
    }
  }, [borrador, base]);

  const editaMaquina = (
    i: number,
    campo: keyof Borrador["maquinas"][number],
    valor: string | boolean,
  ) =>
    setBorrador((b) => {
      if (!b) return b;
      const maquinas = b.maquinas.slice();
      maquinas[i] = { ...maquinas[i], [campo]: valor };
      return { ...b, maquinas };
    });

  const guardar = async () => {
    if (!borrador) return;
    setErrorGuardar(null);
    let doc: FleetDoc;
    try {
      doc = aDocumento(borrador);
    } catch (e) {
      setErrorGuardar(e instanceof Error ? e.message : String(e));
      return;
    }
    setGuardando(true);
    try {
      await guardarFleet(doc);
      toast.success("Plano guardado — el hub vigila la nueva configuracion");
      setBase(JSON.stringify(doc));
      setBorrador(aBorrador(doc));
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setErrorGuardar(msg);
      toast.error(`El hub rechazo el plano: ${msg}`);
    } finally {
      setGuardando(false);
    }
  };

  if (errorCarga)
    return (
      <SinDato>
        no se pudo leer /api/fleet ({errorCarga}) —{" "}
        <button type="button" className="underline" onClick={cargar}>
          reintentar
        </button>
      </SinDato>
    );
  if (!borrador) return <SinDato>leyendo /api/fleet…</SinDato>;

  return (
    <div className="space-y-6">
      {/* La advertencia va ANTES que el formulario a proposito. */}
      <div className="flex items-start gap-2 rounded-md border border-l-2 border-border border-l-warning bg-card p-4">
        <TriangleAlert size={15} className="mt-0.5 shrink-0 text-warning-fg" aria-hidden />
        <p className="font-mono text-[11px] leading-relaxed text-warning-fg">
          Este plano alimenta las alarmas del censo: lo que declares aqui es lo
          que el hub VIGILA. Dar de baja una maquina la saca de la vigilancia;
          los umbrales mueven cuando suena la alarma. Se guarda atomico con
          respaldo .bak en el hub.
        </p>
      </div>

      {/* -- maquinas esperadas ----------------------------------------- */}
      <section className="space-y-3">
        <Titulo
          extra={
            <Button
              variant="outline"
              size="sm"
              className="h-7 gap-1.5 px-2 font-mono text-[11px] uppercase tracking-wide"
              onClick={() =>
                setBorrador((b) =>
                  b
                    ? {
                        ...b,
                        maquinas: [
                          ...b.maquinas,
                          {
                            original: { id: "" },
                            id: "",
                            host: "",
                            duenio: "",
                            rol: "actor",
                            procs: "",
                            difficulty: "",
                            critico: false,
                            notas: "",
                          },
                        ],
                      }
                    : b,
                )
              }
            >
              <Plus size={13} aria-hidden />
              Alta de maquina
            </Button>
          }
        >
          Maquinas esperadas
        </Titulo>
        <div className="space-y-3">
          {borrador.maquinas.length === 0 && (
            <SinDato>
              expected[] vacio — el hub lo rechazara al guardar: da de alta al
              menos una maquina
            </SinDato>
          )}
          {borrador.maquinas.map((m, i) => (
            <Card key={i}>
              <CardContent className="space-y-3 p-4">
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  {(
                    [
                      ["id", "id", "sss"],
                      ["host", "host", "SSS"],
                      ["duenio", "duenio", "—"],
                      ["rol", "rol", "actor"],
                    ] as const
                  ).map(([campo, etiqueta, ph]) => (
                    <div key={campo} className="space-y-1">
                      <Label htmlFor={`m-${i}-${campo}`} className="dlabel">
                        {etiqueta}
                      </Label>
                      <Input
                        id={`m-${i}-${campo}`}
                        className={CAMPO}
                        placeholder={ph}
                        value={m[campo]}
                        onChange={(e) => editaMaquina(i, campo, e.target.value)}
                      />
                    </div>
                  ))}
                </div>
                <div className="grid items-end gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <div className="space-y-1">
                    <Label htmlFor={`m-${i}-procs`} className="dlabel">
                      procs esperados
                    </Label>
                    <Input
                      id={`m-${i}-procs`}
                      className={CAMPO}
                      inputMode="numeric"
                      placeholder="40"
                      value={m.procs}
                      onChange={(e) => editaMaquina(i, "procs", e.target.value)}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label htmlFor={`m-${i}-diff`} className="dlabel">
                      difficulty (banda)
                    </Label>
                    <Input
                      id={`m-${i}-diff`}
                      className={CAMPO}
                      placeholder="4,5,6,7,8"
                      value={m.difficulty}
                      onChange={(e) =>
                        editaMaquina(i, "difficulty", e.target.value)
                      }
                    />
                  </div>
                  <div className="flex items-center gap-2 pb-1">
                    <Checkbox
                      id={`m-${i}-critico`}
                      checked={m.critico}
                      onCheckedChange={(v) =>
                        editaMaquina(i, "critico", v === true)
                      }
                    />
                    <Label
                      htmlFor={`m-${i}-critico`}
                      className="dlabel cursor-pointer"
                    >
                      critica (canario)
                    </Label>
                  </div>
                  <div className="flex justify-end pb-0.5">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 gap-1.5 px-2 font-mono text-[11px] uppercase tracking-wide text-destructive-fg hover:text-destructive-fg"
                      onClick={() =>
                        setBorrador((b) =>
                          b
                            ? {
                                ...b,
                                maquinas: b.maquinas.filter((_, j) => j !== i),
                              }
                            : b,
                        )
                      }
                    >
                      <Trash2 size={13} aria-hidden />
                      Baja
                    </Button>
                  </div>
                </div>
                <div className="space-y-1">
                  <Label htmlFor={`m-${i}-notas`} className="dlabel">
                    notas
                  </Label>
                  <Input
                    id={`m-${i}-notas`}
                    className={CAMPO}
                    placeholder="contexto operativo de la maquina"
                    value={m.notas}
                    onChange={(e) => editaMaquina(i, "notas", e.target.value)}
                  />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* -- umbrales ---------------------------------------------------- */}
      <section className="space-y-3">
        <Titulo>Umbrales de alarma</Titulo>
        <Card>
          <CardContent className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-3">
            {borrador.umbrales.map((u, i) => (
              <div key={u.clave} className="space-y-1">
                <Label htmlFor={`u-${u.clave}`} className="dlabel">
                  {u.clave}
                </Label>
                <Input
                  id={`u-${u.clave}`}
                  className={CAMPO}
                  inputMode="decimal"
                  value={u.texto}
                  onChange={(e) =>
                    setBorrador((b) => {
                      if (!b) return b;
                      const umbrales = b.umbrales.slice();
                      umbrales[i] = { ...umbrales[i], texto: e.target.value };
                      return { ...b, umbrales };
                    })
                  }
                />
              </div>
            ))}
            {borrador.umbrales.length === 0 && (
              <SinDato>el documento no trae umbrales</SinDato>
            )}
          </CardContent>
        </Card>
      </section>

      {/* -- runs (solo lectura) ----------------------------------------- */}
      <section className="space-y-3">
        <Titulo
          extra={
            <span className="font-mono text-[11px] text-muted-foreground">
              solo lectura — se administran desde el repo
            </span>
          }
        >
          Runs
        </Titulo>
        <Card>
          <CardContent className="divide-y divide-border p-0">
            {(borrador.doc.runs ?? []).map((r, i) => (
              <div key={r.id ?? i} className="flex flex-wrap items-baseline gap-x-4 gap-y-1 p-4">
                <span className="font-mono text-[13px] font-semibold">
                  {r.id ?? r.nombre ?? `run ${i + 1}`}
                </span>
                {r.activa && (
                  <span className="rounded-sm bg-state-running-subtle px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wide text-state-running-fg">
                    activa
                  </span>
                )}
                <span className="font-mono text-[11px] text-muted-foreground">
                  {r.descripcion ?? "—"}
                </span>
                <span className="tnum ml-auto font-mono text-[11px] text-muted-foreground">
                  {r.learner ?? "—"}
                </span>
              </div>
            ))}
            {!borrador.doc.runs?.length && (
              <div className="p-4">
                <SinDato>sin runs declaradas en el plano</SinDato>
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      {/* -- barra de guardado ------------------------------------------- */}
      <div className="flex flex-wrap items-center gap-3 border-t border-chrome-line pt-4">
        <Button
          onClick={guardar}
          disabled={!sucio || guardando}
          className="gap-1.5 font-mono text-xs uppercase tracking-wide"
        >
          <Save size={14} aria-hidden />
          {guardando ? "Guardando…" : "Guardar plano"}
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={!sucio || guardando}
          onClick={cargar}
          className="h-8 gap-1.5 font-mono text-[11px] uppercase tracking-wide"
        >
          <RotateCcw size={13} aria-hidden />
          Descartar cambios
        </Button>
        <span className="font-mono text-[11px] text-muted-foreground">
          {sucio ? "hay cambios sin guardar" : "sin cambios"}
        </span>
        {errorGuardar && (
          <span className="font-mono text-[11px] text-destructive-fg">
            {errorGuardar}
          </span>
        )}
      </div>
    </div>
  );
}
