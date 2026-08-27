# validate_forged_states.py -- does poking 0xFE45 REALLY change the CPU?
#
# forge_states.py mints missing difficulty tiers by poking the difficulty
# byte inside a donor snapshot. The byte provably persists (menu_probe RAM
# diffs + the forge probe), but persistence is not causation: if the fight
# scene copied difficulty into AI-local parameters at load time, a poked
# state would LOOK like lvl7 in RAM and PLAY like its lvl1 donor. This tool
# settles it behaviorally, on opponents where the authentic ladder is deep
# enough to hold ground truth at both ends (CHUNLI 1..8, ZANGIEF/DHALSIM
# 1..7 as of 2026-08-27):
#
#   arms per opponent (K episodes each, fixed policy, desync perturbation):
#     A_lo  authentic lvl lo          A_hi  authentic lvl hi
#     F_hi  forged lo->hi (FORGEVAL)  F_lo  forged hi->lo (FORGEVAL)
#
#   comparisons (two-sided permutation tests; fitness is the deciding
#   metric, win-rate enters verdicts via the equivalence gap tolerance --
#   p_wins is reported for the reader, not consulted by the verdict):
#     control     A_lo vs A_hi  must DIFFER  (else the probe policy cannot
#                                             discriminate: INCONCLUSIVE)
#     equiv_hi    F_hi vs A_hi  must MATCH   (forged plays like its label)
#     discrim_hi  F_hi vs A_lo  must DIFFER  (forged stopped playing like
#                                             its donor)
#     equiv_lo    F_lo vs A_lo  must MATCH   (both directions, so a result
#     discrim_lo  F_lo vs A_hi  must DIFFER   cannot be a one-way artifact)
#
# Per-opponent verdict: FAIL if an equivalence breaks hard (p < .01 or the
# win-rate gap to the label exceeds max(0.15, a third of the control gap));
# INCONCLUSIVE if the control or a discrimination is not significant; PASS
# otherwise. Global: FAIL if any opponent FAILs; PASS if none fail and >=2
# PASS. The verdict lands in benchmarks/forge_validation/report.json, which
# forge_states --fill reads as its gate.
#
# Known limitations (reviewed 2026-08-27; acceptable for this gate, revisit
# if a verdict is ever CLOSE): the design leans conservative -- with n=40
# per arm a working poke can spuriously FAIL (the wr tolerance floor is
# ~1.3 standard errors at mid win-rates, and 6 equivalence tests compound),
# while the anti-cosmetic direction rests on the discrimination tests, so a
# HALF-working poke could scrape past per-opponent equivalence;
# equivalence-by-nonsignificance compares means only (a forged arm matching
# the label's mean but not its distribution shape passes); and desync draws
# are unpaired across arms (pairing them by episode would buy power free).
# The 2026-08-27 PASS was nowhere near any of these edges: equiv deltas
# <=0.10 wr with p>=0.14, discrimination gaps 0.52-0.85 wr at p<1e-4.
#
#     .venv/bin/python tools/forge_states.py --forge-validation
#     .venv/bin/python tools/validate_forged_states.py [--eps 40 --procs 5]

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import numpy as np  # noqa: E402

import bench_12rivals as bench  # noqa: E402 -- reuses its episode machinery
from farm_states import OPP_NAMES, state_name  # noqa: E402
from forge_states import (  # noqa: E402
    INTEGRATION_DIR, VALIDATION_OPPS, VALIDATION_REPORT, authentic_levels,
    read_manifest,
)

DEFAULT_CKPT = os.path.join(REPO_ROOT, "benchmarks", "apex_milestones",
                            "apex_curriculum_best.pt")

ALPHA_DIFFER = 0.01   # significance demanded of control/discrimination
ALPHA_MATCH = 0.01    # an equivalence p BELOW this is a hard behavioral break


def perm_pvalue(a, b, n_perm=20000, seed=0):
    """Two-sided permutation test on the difference of means."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    observed = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    sims = rng.permuted(np.tile(pool, (n_perm, 1)), axis=1)
    diffs = np.abs(sims[:, :len(a)].mean(axis=1) - sims[:, len(a):].mean(axis=1))
    return float((1 + np.sum(diffs >= observed - 1e-12)) / (n_perm + 1))


def arm_stats(rows):
    wins = [w for _f, w, _s in rows]
    fits = [f for f, _w, _s in rows]
    return {"episodes": len(rows), "win_rate": float(np.mean(wins)),
            "fitness": float(np.mean(fits)),
            "fitness_std": float(np.std(fits)),
            "mean_steps": float(np.mean([s for _f, _w, s in rows]))}


def compare(rows_a, rows_b, seed):
    fits = lambda rows: [f for f, _w, _s in rows]  # noqa: E731
    wins = lambda rows: [w for _f, w, _s in rows]  # noqa: E731
    return {
        "p_fitness": perm_pvalue(fits(rows_a), fits(rows_b), seed=seed),
        "p_wins": perm_pvalue(wins(rows_a), wins(rows_b), seed=seed + 1),
        "d_win_rate": float(np.mean(wins(rows_a)) - np.mean(wins(rows_b))),
        "d_fitness": float(np.mean(fits(rows_a)) - np.mean(fits(rows_b))),
    }


def judge_opponent(comps, control_wr_gap):
    """PASS / FAIL / INCONCLUSIVE from the five comparisons of one opponent."""
    if comps["control"]["p_fitness"] >= ALPHA_DIFFER:
        return "INCONCLUSIVE", "el control no discrimina (A_lo ~ A_hi)"
    wr_tolerance = max(0.15, abs(control_wr_gap) / 3.0)
    for tag in ("equiv_hi", "equiv_lo"):
        c = comps[tag]
        if c["p_fitness"] < ALPHA_MATCH or abs(c["d_win_rate"]) > wr_tolerance:
            return "FAIL", (f"{tag}: el forjado NO juega como su etiqueta "
                            f"(p={c['p_fitness']:.4f}, "
                            f"d_wr={c['d_win_rate']:+.3f}, "
                            f"tolerancia {wr_tolerance:.3f})")
    for tag in ("discrim_hi", "discrim_lo"):
        if comps[tag]["p_fitness"] >= ALPHA_DIFFER:
            return "INCONCLUSIVE", (f"{tag}: el forjado no se distingue de su "
                                    f"donante (p={comps[tag]['p_fitness']:.4f})")
    return "PASS", "forjado ~ etiqueta y forjado != donante, en ambas direcciones"


def judge_global(verdicts):
    if any(v == "FAIL" for v, _ in verdicts.values()):
        return "FAIL"
    if sum(1 for v, _ in verdicts.values() if v == "PASS") >= 2:
        return "PASS"
    return "INCONCLUSIVE"


def build_matrix(opps):
    manifest = read_manifest()
    levels = authentic_levels(manifest)
    opp_ids = {n: i for i, n in OPP_NAMES.items()}
    matrix = {}
    for opp in opps:
        lo, hi = levels[opp][0], levels[opp][-1]
        arms = {
            "A_lo": state_name(opp_ids[opp], lo),
            "A_hi": state_name(opp_ids[opp], hi),
            "F_hi": f"FORGEVAL_{opp}_lvl{hi}_from{lo}",
            "F_lo": f"FORGEVAL_{opp}_lvl{lo}_from{hi}",
        }
        for name in arms.values():
            if name.startswith("FORGEVAL_") and not os.path.exists(
                    os.path.join(INTEGRATION_DIR, name + ".state")):
                raise SystemExit(f"[valida] falta {name}.state -- corre "
                                 "tools/forge_states.py --forge-validation")
        matrix[opp] = {"lo": lo, "hi": hi, "arms": arms}
    return matrix


def main():
    ap = argparse.ArgumentParser(
        description="Valida conductualmente los estados forjados por poke")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--opps", default=",".join(VALIDATION_OPPS))
    ap.add_argument("--eps", type=int, default=40, help="episodios por brazo")
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--nice", type=int, default=12)
    ap.add_argument("--desync-max", type=int, default=30)
    ap.add_argument("--out", default=VALIDATION_REPORT)
    args = ap.parse_args()

    matrix = build_matrix([o for o in args.opps.split(",") if o])
    arm_states, arm_index = [], {}
    for opp, spec in matrix.items():
        for arm, state in spec["arms"].items():
            arm_index[(opp, arm)] = len(arm_states)
            arm_states.append(state)
    tasks = [(state, idx, ep)
             for idx, state in enumerate(arm_states)
             for ep in range(args.eps)]
    print(f"[valida] {len(matrix)} rivales x 4 brazos x {args.eps} eps = "
          f"{len(tasks)} episodios | ckpt {os.path.basename(args.ckpt)} "
          f"desync<={args.desync_max} procs={args.procs}", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.procs, initializer=bench._init_rainbow,
                  initargs=(args.ckpt, args.nice, 0.0,
                            args.desync_max)) as pool:
        rows = pool.map(bench._episode_rainbow, tasks)

    by_arm = {}
    for state, _ep, fit, win, steps in rows:
        by_arm.setdefault(state, []).append((fit, win, steps))

    report = {"date": time.strftime("%Y-%m-%d %H:%M"),
              "ckpt": os.path.basename(args.ckpt),
              "eps_per_arm": args.eps, "desync_max": args.desync_max,
              "opponents": {}, "verdicts": {}}
    verdicts = {}
    for opp, spec in matrix.items():
        arm_rows = {arm: by_arm.get(state, [])
                    for arm, state in spec["arms"].items()}
        comps = {
            "control": compare(arm_rows["A_lo"], arm_rows["A_hi"], seed=11),
            "equiv_hi": compare(arm_rows["F_hi"], arm_rows["A_hi"], seed=13),
            "discrim_hi": compare(arm_rows["F_hi"], arm_rows["A_lo"], seed=17),
            "equiv_lo": compare(arm_rows["F_lo"], arm_rows["A_lo"], seed=19),
            "discrim_lo": compare(arm_rows["F_lo"], arm_rows["A_hi"], seed=23),
        }
        verdict, why = judge_opponent(comps, comps["control"]["d_win_rate"])
        verdicts[opp] = (verdict, why)
        report["verdicts"][opp] = {"verdict": verdict, "why": why}
        report["opponents"][opp] = {
            "levels": {"lo": spec["lo"], "hi": spec["hi"]},
            "arms": {arm: arm_stats(r) for arm, r in arm_rows.items()},
            "comparisons": comps,
            "verdict": verdict, "why": why,
        }
        a = report["opponents"][opp]["arms"]
        print(f"\n[valida] {opp} (lvl{spec['lo']} vs lvl{spec['hi']}):")
        for arm in ("A_lo", "F_lo", "A_hi", "F_hi"):
            print(f"  {arm}: wr={a[arm]['win_rate']:.3f} "
                  f"fit={a[arm]['fitness']:+.3f} "
                  f"steps={a[arm]['mean_steps']:.0f}")
        for tag, c in comps.items():
            print(f"  {tag:10s}: p_fit={c['p_fitness']:.4f} "
                  f"p_win={c['p_wins']:.4f} d_wr={c['d_win_rate']:+.3f}")
        print(f"  => {verdict}: {why}")

    report["verdict"] = judge_global(verdicts)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"\n[valida] VEREDICTO GLOBAL: {report['verdict']} "
          f"({len(tasks)} eps en {time.time() - t0:.0f}s) -> {args.out}")


if __name__ == "__main__":
    main()
