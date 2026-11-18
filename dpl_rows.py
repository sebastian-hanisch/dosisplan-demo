"""Zeilen der vorgerechneten Tabellen: je Phantom, Höchstdosis M und Kombination (d, p) alle Werte der drei Wege und der Strahlpläne.

Dieses Modul wird von `tools/dpl_make_table.py` (Erzeugung und `--check`) und von den Tests benutzt, nicht von der App zur Laufzeit.
Die Zeile hält den exakten Wert als Zielwert des LP zur gefundenen MIP-Menge S (`solve_set`): dadurch liegt der exakte Strahlplan in der App als
LP-Teil vor (S aus der Tabelle, Strahlplan live), und Wert und Strahlplan stimmen auf LP-Genauigkeit überein."""

import numpy as np

import dpl_constants as C
import dpl_solvers as S

MIP_STATUS = {S.OPTIMAL: 0, S.LIMIT: 1, S.INFEASIBLE: 2, S.ERROR: 3}


def _r(x, nd: int = 10):
    return None if x is None else round(float(x), nd)


def k_of(p: float, n_organ: int) -> int:
    """Erlaubte Zahl k von Organvoxeln über d: floor(p * Organvoxel), mindestens 1."""
    return max(1, int(np.floor(p * n_organ + 1e-9)))


def build_row(prob: S.Problem, gap: str, i: int, ci: int, cap: float, free: S.Solution, probe: bool = True,
              time_limit: float = C.MIP_TIME_LIMIT) -> dict:
    d, p = C.COMBOS[ci]
    k = k_of(p, prob.n_o)
    row = dict(gap=gap, i=i, ci=ci, M=cap, d=d, p=p, k=k, nO=prob.n_o)
    if not free.feasible:                       # Tumorfenster bei dieser Organ-Höchstdosis nicht erreichbar: keine Zeile mit Werten
        row.update(f_free=None)
        return row
    row.update(f_free=_r(free.f), ex_free=prob.over(free.x, d), exc_free=_r(prob.excess(free.x, d)), mo_free=_r(prob.organ_dose(free.x).mean()))
    mip = prob.solve_mip(d, k, cap, time_limit=time_limit)
    row["mip_status"] = MIP_STATUS[mip.status]
    plan = prob.solve_set(d, mip.chosen, cap) if mip.feasible else None
    if plan is not None and plan.feasible:
        row.update(f_mip=_r(plan.f), S=list(mip.chosen), ex_mip=prob.over(plan.x, d), mo_mip=_r(prob.organ_dose(plan.x).mean()))
    else:
        row.update(f_mip=None, S=[], ex_mip=None, mo_mip=None)
    lpr = prob.solve_relax(d, k, cap)
    row["f_lpr"] = _r(lpr.f)
    for name, m in (("cvk", k), ("cv1", k + 1)):
        cv = prob.solve_cvar(d, m, cap)
        row["f_" + name] = _r(cv.f)
        row["ex_" + name] = prob.over(cv.x, d) if cv.feasible else None
        row["mo_" + name] = _r(prob.organ_dose(cv.x).mean()) if cv.feasible else None
    row["f_maxd"] = _r(prob.solve_strict(d).f)
    row.update(wit=None, Sb=None, ex_mid=None, f_b=None)
    if probe and plan is not None and plan.feasible and mip.status == S.OPTIMAL:
        pr = S.nonconvex_probe(prob, d, k, cap, mip.chosen, time_limit)
        if pr is not None:
            row.update(wit=int(pr["over_mid"] > k), Sb=list(pr["chosen_b"]), ex_mid=int(pr["over_mid"]), f_b=_r(pr["f_b"]),
                       ov_a=len(pr["over_a"]), ov_b=len(pr["over_b"]))
    return row


def build_curve(prob: S.Problem, gap: str, i: int, d: float = C.CURVE_D, cap: float = C.ORGAN_CAP, time_limit: float = 60.0) -> dict:
    """Kostenkurve über k = 1 bis Zahl der Organvoxel bei fester Grenzdosis d: exakter Wert, Relaxation, CVaR(k+1) und CVaR(k)."""
    free = prob.solve_free(cap)
    strict = prob.solve_strict(d)
    out = dict(gap=gap, i=i, nO=prob.n_o, d=d, f_free=_r(free.f), ex_free=prob.over(free.x, d), f_strict=_r(strict.f),
               k=[], f_mip=[], f_lpr=[], f_cv1=[], f_cvk=[])
    for k in range(1, prob.n_o + 1):
        m = prob.solve_mip(d, k, cap, time_limit=time_limit)
        plan = prob.solve_set(d, m.chosen, cap) if m.feasible else None
        out["k"].append(k)
        out["f_mip"].append(_r(plan.f) if plan is not None and plan.feasible else None)
        out["f_lpr"].append(_r(prob.solve_relax(d, k, cap).f))
        out["f_cv1"].append(_r(prob.solve_cvar(d, k + 1, cap).f))
        out["f_cvk"].append(_r(prob.solve_cvar(d, k, cap).f))
    return out
