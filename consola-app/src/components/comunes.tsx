// Piezas compartidas de la consola. Todo estado del dominio pasa por aqui:
// el color NUNCA viaja solo — cada estado tiene forma (icono) ademas de tinte.
import type { ReactNode } from "react";
import {
  Circle,
  CircleDashed,
  CirclePause,
  Crown,
  OctagonX,
  TriangleAlert,
} from "lucide-react";
import { cn } from "@/lib/utils";

export type Semaforo =
  | "running"
  | "stopped"
  | "degraded"
  | "alarm"
  | "champion"
  | "unknown";

const ICONO: Record<Semaforo, typeof Circle> = {
  running: Circle,
  stopped: CirclePause,
  degraded: TriangleAlert,
  alarm: OctagonX,
  champion: Crown,
  unknown: CircleDashed,
};

const CHIP: Record<Semaforo, string> = {
  running: "bg-state-running-subtle text-state-running-fg",
  stopped: "bg-state-stopped-subtle text-state-stopped-fg",
  degraded: "bg-state-degraded-subtle text-state-degraded-fg",
  alarm: "bg-state-alarm-subtle text-state-alarm-fg",
  champion: "bg-state-champion-subtle text-state-champion-fg",
  unknown: "bg-state-unknown-subtle text-state-unknown-fg",
};

/** Chip de estado: fondo -subtle, texto -fg, icono distinto por estado
 *  (forma ademas de color: hay daltonicos en los equipos). */
export function EstadoChip({
  estado,
  children,
  className,
}: {
  estado: Semaforo;
  children: ReactNode;
  className?: string;
}) {
  const Icono = ICONO[estado];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-sm px-1.5 py-0.5 font-mono text-[11px] font-medium uppercase tracking-wide",
        CHIP[estado],
        className,
      )}
    >
      <Icono
        size={11}
        strokeWidth={2.5}
        fill={estado === "running" ? "currentColor" : "none"}
        aria-hidden
      />
      {children}
    </span>
  );
}

/** Etiqueta + valor de instrumento. El sub lleva la letra chica obligatoria:
 *  que mide, con que n. Un numero sin su n es una opinion. */
export function Dato({
  label,
  children,
  sub,
  className,
}: {
  label: string;
  children: ReactNode;
  sub?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <div className="dlabel">{label}</div>
      <div className="tnum mt-1 font-mono text-xl font-medium leading-none">
        {children}
      </div>
      {sub ? (
        <div className="tnum mt-1 font-mono text-[11px] leading-tight text-muted-foreground">
          {sub}
        </div>
      ) : null}
    </div>
  );
}

/** Titular de seccion (Chakra Petch — solo aqui y en el logotipo). */
export function Titulo({
  children,
  extra,
}: {
  children: ReactNode;
  extra?: ReactNode;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <h2 className="stitle text-foreground">{children}</h2>
      {extra}
    </div>
  );
}

/** Pista de la escalera: barra = campeon (banco greedy), marca = ventana viva.
 *  Dos medidas distintas sobre la misma pista, imposibles de confundir:
 *  la barra es un relleno solido; la marca es un rombo con borde. */
export function PistaNivel({
  banco,
  ventana,
}: {
  banco: number | null;
  ventana: number | null;
}) {
  return (
    <div className="relative h-3 w-full min-w-24 rounded-[2px] bg-muted">
      {banco !== null && (
        <div
          className="absolute inset-y-0 left-0 rounded-[2px] bg-primary"
          style={{ width: `${Math.max(0, Math.min(1, banco)) * 100}%` }}
        />
      )}
      {ventana !== null && (
        <div
          className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rotate-45 border-2 border-foreground bg-background"
          style={{ left: `${Math.max(0, Math.min(1, ventana)) * 100}%` }}
        />
      )}
    </div>
  );
}

/** Leyenda de la escalera — obligatoria donde haya pistas. */
export function LeyendaEscalera({ nVentana }: { nVentana?: number }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px] text-muted-foreground">
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-2 w-5 rounded-[2px] bg-primary" />
        campeon · banco greedy n=48
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block size-2 rotate-45 border-2 border-foreground bg-background" />
        ventana viva · n≈{nVentana ?? 200} con exploracion
      </span>
      <span>ambas: rounds de apertura</span>
    </div>
  );
}

/** Aviso de dato faltante (sin muestra del hub, etc.). */
export function SinDato({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-md border border-dashed border-border px-4 py-6 text-center font-mono text-xs text-muted-foreground">
      {children}
    </div>
  );
}

/** Extrae los niveles de un campeon/coronacion (wr_lvlN) sin asumir cuantos. */
export function nivelesDe(obj: Record<string, unknown>): [number, number][] {
  return Object.entries(obj)
    .filter(([k, v]) => /^wr_lvl\d+$/.test(k) && typeof v === "number")
    .map(([k, v]) => [Number(k.slice(6)), v as number] as [number, number])
    .sort((a, b) => a[0] - b[0]);
}

export function fechaCorta(tSeg: number): string {
  return new Date(tSeg * 1000).toLocaleString("es-MX", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}
