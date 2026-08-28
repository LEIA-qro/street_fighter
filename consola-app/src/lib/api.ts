// El contrato con el hub (tools/leia_hub.py::/api/state). Tipos a mano y
// honestos: lo que el hub no manda es `null`, no un default inventado.
export interface Maquina {
  id: string; estado: "vivo" | "MUDA"; actor: string | null;
  procs: number | null; steps_per_s: number; age_s: number | null;
  critico: boolean;
}
export interface Muestra {
  ts: string; grad_steps: number; buffer: number;
  wr_recent200: number | null; por_nivel: Record<string, number>;
  maquinas: Maquina[]; vivas: number; esperadas: number;
  grads_per_s?: number; trans_per_s?: number; replay_ratio?: number | null;
  batch?: number; batch_supuesto?: boolean;
}
export interface RivalPelea { rival: string; rounds_propios: number; rounds_rival: number; gano: boolean; pasos: number; }
export interface Gauntlet {
  fecha: string; checkpoint: string; dificultad: number;
  peleas_ganadas: number; peleas_totales: number; repeticiones?: number;
  desync_max?: number; ic95?: [number, number]; semilla?: number;
  rivales: RivalPelea[];
}
export interface Campeon { weights_version: number; wr_media: number; t: number; archivo?: string; [k: string]: unknown; }
export interface Estado {
  muestra: Muestra | null;
  plano: { run?: { id?: string; descripcion?: string }; runs?: { id: string; activa?: boolean; descripcion?: string }[];
           expected?: { id: string; host?: string; duenio?: string; rol?: string; difficulty?: string; procs?: number; critico?: boolean }[];
           umbrales?: Record<string, number> };
  campeon: Campeon | null;
  coronaciones: Campeon[];
  gauntlets: Record<string, Gauntlet>;
  historia: { ts: string; grads_per_s?: number; trans_per_s?: number; vivas?: number; wr?: number }[];
  hub_desde: number; ahora: number;
}
export async function leerEstado(): Promise<Estado> {
  const r = await fetch("/api/state");
  if (!r.ok) throw new Error(`hub ${r.status}`);
  return r.json();
}
export function edad(s: number | null | undefined): string {
  if (s === null || s === undefined) return "nunca";
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  if (s < 172800) return `${(s / 3600).toFixed(1)} h`;
  return `${(s / 86400).toFixed(1)} d`;
}
export const pc = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined ? "—" : `${(v * 100).toFixed(d)}%`;
export const nf = (v: number | null | undefined) =>
  v === null || v === undefined ? "—" : v.toLocaleString("es-MX");
