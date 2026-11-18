"""Kennzahlen, Zusammenfassungen über die vorgerechneten Tabellen und Verdikte (reine Funktionen).

Begriffe (wie in App und README): Grenzdosis d, Höchstdosis M, k = erlaubte Zahl Organvoxel über d, "Organvoxel über d" (Zähleinheit),
Kosten der Grenze = (f_exakt - f_ohne Grenze) / f_ohne Grenze, Lücke der Relaxation = (f_exakt - f_Relaxation) / f_exakt,
Lücke der CVaR-Restriktion = (f_CVaR - f_exakt) / f_exakt, aktiv = die Grenze kostet mehr als 1e-5 relativ,
Anteil der Kosten in der Relaxation = (f_Relaxation - f_ohne Grenze) / (f_exakt - f_ohne Grenze).
Zähleinheit der Verteilungen: Zeilen = Phantom x Kombination (d, p) bei fester Höchstdosis M."""

from dataclasses import dataclass, field

import numpy as np

import dpl_constants as C
import dpl_data as D
import dpl_model as M
import dpl_solvers as S
import dpl_stats as ST

TOL_ORDER = 1e-5          # relative Toleranz für Reihenfolgen und Monotonie (die MIP-Werte sind auf 1e-6 relativ genau)
SHARE_MIN_COST = 1e-5     # unterhalb dieser relativen Kosten ist der Anteil der Relaxation nicht definiert


@dataclass(frozen=True)
class Settings:
    gap: str = "nah"
    seed: int = 0
    combo: int = 1
    cap: float = 1.2


@dataclass
class Analysis:
    settings: Settings
    row: dict
    n_organ: int
    d: float
    k: int
    cap: float
    f: dict = field(default_factory=dict)        # Zielwerte je Weg (None = unzulässig)
    dose: dict = field(default_factory=dict)     # Dosis je Körpervoxel je Strahlplan
    over: dict = field(default_factory=dict)     # Organvoxel über d je Strahlplan
    organ_sorted: dict = field(default_factory=dict)   # Organdosen absteigend sortiert je Strahlplan
    excess_free: float = 0.0
    allowed_excess: float = 0.0


# ---------------------------------------------------------------- Einzelkennzahlen

def cost_rel(f_exact: float, f_free: float) -> float:
    """Kosten der Grenze relativ zum Strahlplan ohne Grenze."""
    return (f_exact - f_free) / f_free


def gap_lower(f_exact: float, f_lpr: float) -> float:
    """Lücke der LP-Relaxation (untere Schranke) relativ zum exakten Wert."""
    return (f_exact - f_lpr) / f_exact


def gap_upper(f_exact: float, f_cvar: float) -> float:
    """Lücke der CVaR-Restriktion (obere Schranke) relativ zum exakten Wert."""
    return (f_cvar - f_exact) / f_exact


def lpr_share(f_free: float, f_exact: float, f_lpr: float) -> float | None:
    """Anteil der Dosis-Volumen-Kosten, den die Relaxation sieht; ohne Kosten nicht definiert."""
    if f_exact - f_free <= SHARE_MIN_COST * f_free:
        return None
    return (f_lpr - f_free) / (f_exact - f_free)


def is_active(r: dict) -> bool:
    """Zeile mit bewiesenem exaktem Wert, in der die Grenze mehr als 1e-5 relativ kostet."""
    return r.get("mip_status") == 0 and r["f_mip"] - r["f_free"] > C.ACTIVE_REL * r["f_free"]


def k_label(k: int) -> str:
    return f"{k} Organvoxel" if k != 1 else "1 Organvoxel"


# ---------------------------------------------------------------- Live-Analyse einer Zelle

def problem_of(gap: str, i: int) -> S.Problem:
    ph = D.phantom(gap, i)
    return S.Problem(M.standard_matrix(), ph.tumor, ph.organ)


def analyse(settings: Settings) -> Analysis:
    """Rechnet die LP-Teile der gewählten Zelle live (Strahlplan ohne Grenze, Relaxation, CVaR in zwei Formen, exakter Strahlplan zur vorgerechneten Menge S)."""
    row = D.get_row(settings.gap, settings.seed, settings.combo, settings.cap)
    prob = problem_of(settings.gap, settings.seed)
    d, k, cap = row["d"], row["k"], row["M"]
    a = Analysis(settings, row, prob.n_o, d, k, cap)
    if row.get("f_free") is None:
        return a
    plans = {"free": prob.solve_free(cap), "lpr": prob.solve_relax(d, k, cap), "cvk": prob.solve_cvar(d, k, cap), "cv1": prob.solve_cvar(d, k + 1, cap)}
    if row.get("f_mip") is not None:
        plans["exact"] = prob.solve_set(d, row["S"], cap)
    for name, sol in plans.items():
        a.f[name] = sol.f
        if sol.feasible and sol.x is not None:
            a.dose[name] = M.standard_matrix() @ sol.x
            a.over[name] = prob.over(sol.x, d)
            a.organ_sorted[name] = np.sort(prob.organ_dose(sol.x))[::-1]
    a.excess_free = prob.excess(plans["free"].x, d)
    a.allowed_excess = k * (cap - d)
    return a


# ---------------------------------------------------------------- Zusammenfassungen über Zeilen

def _stat(values, groups, seed):
    mean, lo, hi = ST.mean_ci(values, groups, seed)
    return dict(n=len(values), mean=mean, lo=lo, hi=hi, median=ST.median(values), max=float(max(values)) if len(values) else None)


def _grp(rs):
    return [(r["gap"], r["i"]) for r in rs]


def summarize(rs: list[dict]) -> dict:
    """Alle Kennzahlen über eine Menge von Zeilen (Prozentwerte als Prozent, Anteile als Anteil). Bootstrap über Phantome."""
    out = dict(n=len(rs))
    proven = [r for r in rs if r.get("mip_status") == 0]
    out["n_proven"] = len(proven)
    out["n_unproven"] = sum(1 for r in rs if r.get("mip_status") == 1)
    out["n_mip_infeasible"] = sum(1 for r in rs if r.get("mip_status") == 2)
    act = [r for r in proven if is_active(r)]
    inact = [r for r in proven if not is_active(r)]
    out["n_active"] = len(act)
    out["n_inactive"] = len(inact)
    out["n_inactive_free_ok"] = sum(1 for r in inact if r["ex_free"] <= r["k"])
    out["cost"] = _stat([100 * cost_rel(r["f_mip"], r["f_free"]) for r in act], _grp(act), 1101)
    out["gap_lpr"] = _stat([100 * gap_lower(r["f_mip"], r["f_lpr"]) for r in act], _grp(act), 1102)
    out["n_gap_lpr_gt1"] = sum(1 for r in act if gap_lower(r["f_mip"], r["f_lpr"]) > 0.01)
    shares = [(lpr_share(r["f_free"], r["f_mip"], r["f_lpr"]), r) for r in act]
    shares = [(s, r) for s, r in shares if s is not None]
    sv = [s for s, _ in shares]
    out["share"] = dict(n=len(sv), median=ST.median(sv), mean=float(np.mean(sv)) if sv else None, max=max(sv) if sv else None,
                        n_lt001=sum(1 for s in sv if s < 0.01), n_gt05=sum(1 for s in sv if s > 0.5))
    out["n_lpr_eq_free"] = sum(1 for r in act if abs(r["f_lpr"] - r["f_free"]) <= 1e-9 * r["f_free"])
    out["n_free_excess_within"] = sum(1 for r in act if r["exc_free"] <= r["k"] * (r["M"] - r["d"]) + 1e-9)
    out["allowed_excess_mean"] = float(np.mean([r["k"] * (r["M"] - r["d"]) for r in act])) if act else None
    for name, seed in (("cvk", 1103), ("cv1", 1104)):
        feas = [r for r in act if r["f_" + name] is not None]
        out[name + "_infeasible"] = len(act) - len(feas)
        out[name + "_infeasible_nah"] = sum(1 for r in act if r["f_" + name] is None and r["gap"] == "nah")
        out[name + "_gap"] = _stat([100 * gap_upper(r["f_mip"], r["f_" + name]) for r in feas], _grp(feas), seed)
        out[name + "_n_gt1"] = sum(1 for r in feas if gap_upper(r["f_mip"], r["f_" + name]) > 0.01)
    both = [r for r in act if r["f_cvk"] is not None and r["f_cv1"] is not None]
    out["n_both"] = len(both)
    out["cv1_gap_on_cvk_rows"] = _stat([100 * gap_upper(r["f_mip"], r["f_cv1"]) for r in both], _grp(both), 1105) if both else None
    out["only_cvk_infeasible"] = sum(1 for r in act if r["f_cvk"] is None and r["f_cv1"] is not None)
    out["only_cv1_infeasible"] = sum(1 for r in act if r["f_cv1"] is None and r["f_cvk"] is not None)
    tol = lambda x: TOL_ORDER * max(1.0, abs(x))  # noqa: E731
    out["order_violations"] = sum(1 for r in both if not (r["f_lpr"] <= r["f_mip"] + tol(r["f_mip"]) and r["f_mip"] <= r["f_cv1"] + tol(r["f_mip"])
                                                          and r["f_cv1"] <= r["f_cvk"] + tol(r["f_cv1"])))
    out["ex"] = dict(free=float(np.mean([r["ex_free"] for r in both])) if both else None, mip=float(np.mean([r["ex_mip"] for r in both])) if both else None,
                     cv1=float(np.mean([r["ex_cv1"] for r in both])) if both else None, cvk=float(np.mean([r["ex_cvk"] for r in both])) if both else None,
                     k=float(np.mean([r["k"] for r in both])) if both else None)
    out["n_cvk_le_km1"] = sum(1 for r in act if r["f_cvk"] is not None and r["ex_cvk"] <= r["k"] - 1)
    out["n_cv1_le_k"] = sum(1 for r in act if r["f_cv1"] is not None and r["ex_cv1"] <= r["k"])
    out["n_cvk_plans"] = sum(1 for r in act if r["f_cvk"] is not None)
    out["n_cv1_plans"] = sum(1 for r in act if r["f_cv1"] is not None)
    out["n_strict_infeasible"] = sum(1 for r in act if r["f_maxd"] is None)
    wit = [r for r in act if r.get("wit") is not None]
    out["wit_n"] = len(wit)
    out["wit_viol"] = sum(1 for r in wit if r["wit"] == 1)
    out["ex_free_mean_act"] = float(np.mean([r["ex_free"] for r in act])) if act else None
    out["ex_mip_mean_act"] = float(np.mean([r["ex_mip"] for r in act])) if act else None
    return out


def summary_of(cap: float = C.ORGAN_CAP, gap: str | None = None, ci: int | None = None, second: bool = False) -> dict:
    """Zusammenfassung der ersten (oder zweiten) Menge von Zufalls-Seeds, optional je Abstandsklasse und Kombination."""
    key = (cap, gap, ci, second)
    if key not in _SUMMARY_CACHE:
        data = D.seeds2() if second else D.table()
        _SUMMARY_CACHE[key] = summarize(D.rows(cap=cap, gap=gap, ci=ci, data=data))
    return _SUMMARY_CACHE[key]


_SUMMARY_CACHE: dict = {}


def percent(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",") + " %"


def num(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


# ---------------------------------------------------------------- Kennzahlen der gewählten Zelle und Verteilungen

def cell_facts(a: Analysis) -> dict:
    """Kennzahlen der gewählten Zelle (ein Phantom!) aus der Live-Analyse und der Tabellenzeile."""
    r = a.row
    f = a.f
    facts = dict(i=a.settings.seed, gap=a.settings.gap, d=a.d, k=a.k, cap=a.cap, n_organ=a.n_organ, mip_status=r.get("mip_status"), reachable=r.get("f_free") is not None)
    if not facts["reachable"]:
        return facts
    facts.update(f_free=f["free"], ex_free=a.over.get("free"))
    exact = f.get("exact")
    facts["has_exact"] = exact is not None
    facts["active"] = bool(exact is not None and r["mip_status"] == 0 and exact - f["free"] > C.ACTIVE_REL * f["free"])
    facts["cost"] = 100 * cost_rel(exact, f["free"]) if exact is not None else None
    facts["ex_exact"] = a.over.get("exact")
    facts["share"] = lpr_share(f["free"], exact, f["lpr"]) if exact is not None else None
    facts["gap_lpr"] = 100 * gap_lower(exact, f["lpr"]) if exact is not None else None
    facts["gap_cvk"] = 100 * gap_upper(exact, f["cvk"]) if exact is not None and f.get("cvk") is not None else None
    facts["gap_cv1"] = 100 * gap_upper(exact, f["cv1"]) if exact is not None and f.get("cv1") is not None else None
    facts["ex_cvk"], facts["ex_cv1"] = a.over.get("cvk"), a.over.get("cv1")
    return facts


def dist_points(gap: str, ci: int, cap: float) -> list[dict]:
    """Je Phantom (30 je Abstandsklasse) eine Zeile mit Kosten und Lücken in Prozent (für die Streudiagramme)."""
    pts = []
    for r in D.rows(cap=cap, gap=gap, ci=ci):
        if r.get("mip_status") != 0:
            continue
        act = is_active(r)
        p = dict(i=r["i"], active=act, cost=100 * cost_rel(r["f_mip"], r["f_free"]), gap_lpr=100 * gap_lower(r["f_mip"], r["f_lpr"]),
                 exc=r["exc_free"], allowed=r["k"] * (r["M"] - r["d"]))
        p["cvk"] = 100 * gap_upper(r["f_mip"], r["f_cvk"]) if r["f_cvk"] is not None else None
        p["cv1"] = 100 * gap_upper(r["f_mip"], r["f_cv1"]) if r["f_cv1"] is not None else None
        pts.append(p)
    return pts


def verdict_cost(facts: dict, dist: dict) -> tuple[str, str]:
    """Kernfrage 1: Was kostet die Grenze, und was ändert sie am Strahlplan? `dist` = summarize() über die 30 Phantome dieses Abstands, dieser Kombination und Höchstdosis."""
    who = f"Phantom {facts['i']}, Organ {facts['gap']}, d = {num(facts['d'])}, höchstens k = {facts['k']} von {facts['n_organ']} Organvoxeln"
    if not facts["reachable"]:
        return "warning", f"{who}: Bei der Höchstdosis M = {num(facts['cap'])} ist das Tumorfenster 1,0 bis 1,2 für dieses Phantom nicht erreichbar; es gibt keinen Strahlplan."
    if facts["mip_status"] == 2:
        return "warning", (f"{who}: Die Grenze ist nicht erfüllbar (die exakte Aufgabe ist unzulässig); der Strahlplan ohne Grenze hat {facts['ex_free']} Organvoxel über d. "
                           f"Von den 30 Phantomen dieses Abstands sind in dieser Kombination {dist['n_mip_infeasible']} unzulässig.")
    if facts["mip_status"] != 0:
        return "warning", f"{who}: Der exakte Wert ist nicht bewiesen (Zeitlimit der vorgerechneten MIP); die Zahlen sind eine Obergrenze des exakten Werts."
    tail = (f" Über die 30 Phantome dieses Abstands: Grenze aktiv in {dist['n_active']} von {dist['n_proven']}, Median der Kosten {percent(dist['cost']['median'], 2)}."
            if dist["n_active"] else f" Über die 30 Phantome dieses Abstands: Grenze aktiv in 0 von {dist['n_proven']}.")
    if not facts["active"]:
        return "info", (f"{who}: Der Strahlplan ohne Grenze hat {facts['ex_free']} Organvoxel über d (erlaubt {facts['k']}); die Grenze kostet hier höchstens 0,001 % der Zielgröße, sie ist nicht aktiv."
                        + tail)
    return "success", (f"{who}: Ohne Grenze liegen {facts['ex_free']} Organvoxel über d, im exakten Strahlplan {facts['ex_exact']}. Die Grenze kostet {percent(facts['cost'], 2)} der Zielgröße "
                       f"(ein Phantom, kein Befund)." + tail)


def verdict_relax(facts: dict, dist: dict) -> tuple[str, str]:
    """Kernfrage 2: Was sieht die LP-Schranke davon?"""
    if not facts["reachable"] or facts["mip_status"] != 0 or not facts["has_exact"]:
        return "info", "Ohne exakten Wert gibt es keine Lücke der Relaxation zu zeigen."
    if not facts["active"]:
        return "info", "Die Grenze kostet hier nichts, also gibt es nichts, was die LP-Relaxation sehen könnte (Lücke 0)."
    sh = facts["share"]
    med = dist["share"]["median"]
    tail = (f" Über die aktiven Zeilen dieser Kombination ist der mittlere Anteil {num(dist['share']['mean'], 2)} (Median {num(med, 2)}; unter 0,01 in {dist['share']['n_lt001']} von {dist['share']['n']})."
            if dist["share"]["n"] else "")
    return "info", (f"Die LP-Relaxation zeigt in diesem Phantom {num(100 * sh, 0)} % der Kosten der Grenze; ihr Wert liegt {percent(facts['gap_lpr'], 2)} unter dem exakten Wert." + tail)


def verdict_cvar(facts: dict, dist: dict) -> tuple[str, str]:
    """Kernfrage 3: Wie weit reicht die konvexe Näherung?"""
    if not facts["reachable"] or facts["mip_status"] != 0 or not facts["has_exact"]:
        return "info", "Ohne exakten Wert gibt es keinen Vergleich der CVaR-Restriktionen."
    if not facts["active"]:
        return "info", "Die Grenze kostet hier nichts; die CVaR-Restriktionen sind strenger als nötig, der Vergleich zeigt nur dann etwas, wenn die Grenze aktiv ist."
    def one(name, gap, ex):
        return f"{name} ist unzulässig, obwohl die Aufgabe lösbar ist" if gap is None else f"{name} zahlt {percent(gap, 2)} mehr als exakt ({ex} Organvoxel über d)"
    text = (f"{one('CVaR(k), Form der Literatur,', facts['gap_cvk'], facts['ex_cvk'])}; {one('CVaR(k+1), eigene Variante,', facts['gap_cv1'], facts['ex_cv1'])} "
            f"(erlaubt k = {facts['k']}; Zeile ein Phantom). Über die aktiven Zeilen dieser Kombination: CVaR(k) unzulässig in {dist['cvk_infeasible']} von {dist['n_active']}, CVaR(k+1) in {dist['cv1_infeasible']}.")
    kind = "warning" if facts["gap_cvk"] is None or facts["gap_cv1"] is None else "info"
    return kind, text


# ---------------------------------------------------------------- Tabellen für die Experimente

def gap_table(cap: float = C.ORGAN_CAP, second: bool = False) -> list[dict]:
    """Je Abstandsklasse (über alle drei Kombinationen) die Kennzahlen für Experiment 2."""
    out = []
    for g in C.GAP_NAMES:
        s = summary_of(cap, g, None, second)
        out.append(dict(gap=g, n=s["n"], n_proven=s["n_proven"], n_active=s["n_active"], n_inactive_free_ok=s["n_inactive_free_ok"], cost=s["cost"], gap_lpr=s["gap_lpr"],
                        cv1_gap=s["cv1_gap"], cvk_gap=s["cvk_gap"]))
    return out


def cap_table(second: bool = False) -> list[dict]:
    """Je Höchstdosis M die Kennzahlen für Experiment 3 (alle Abstände, alle Kombinationen)."""
    out = []
    for cap in (C.CAPS if not second else (C.ORGAN_CAP,)):
        s = summary_of(cap, None, None, second)
        out.append(dict(cap=cap, n_active=s["n_active"], share=s["share"], gap_lpr=s["gap_lpr"], allowed=s["allowed_excess_mean"], cv1_infeasible=s["cv1_infeasible"],
                        n_lpr_eq_free=s["n_lpr_eq_free"], n_free_excess_within=s["n_free_excess_within"]))
    return out


def curve_summary(tol: float = TOL_ORDER) -> dict:
    """Kennzahlen der 30 vorgerechneten Kostenkurven (d = 0,3, M = 1,2): Monotonie in k, k*, Lücken nach k/nO, größte Lücken."""
    cs = D.curves()["curves"]
    out = dict(n_curves=len(cs), steps=0, viol={"f_mip": 0, "f_lpr": 0, "f_cv1": 0, "f_cvk": 0}, kstar_eq=0, kstar_le=0, rows=[], strict_infeasible=0)
    bins = [(0.0, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 1.01)]
    acc = {b: dict(lpr=[], lpr_share=[], cv1=[], cv1_inf=0, cvk=[], cvk_inf=0, n=0) for b in bins}
    remaining = {0.25: [], 0.5: []}
    biggest = dict(lpr=(0.0, None), cv1=(0.0, None))
    for c in cs:
        out["steps"] += len(c["k"]) - 1
        for key in out["viol"]:
            v = [x for x in c[key] if x is not None]
            out["viol"][key] += sum(1 for p, q in zip(v, v[1:]) if q > p + tol * max(1.0, abs(p)))
        if c["f_strict"] is None:
            out["strict_infeasible"] += 1
        ff = c["f_free"]
        act_k = [k for k, v in zip(c["k"], c["f_mip"]) if v is not None and v - ff > C.ACTIVE_REL * ff]
        kstar = (max(act_k) + 1) if act_k else 1
        out["kstar_eq"] += int(kstar == c["ex_free"])
        out["kstar_le"] += int(kstar <= c["ex_free"])
        cost1 = (c["f_mip"][0] - ff) / ff if c["f_mip"][0] is not None else None
        for k, fm, fl, f1, fk in zip(c["k"], c["f_mip"], c["f_lpr"], c["f_cv1"], c["f_cvk"]):
            if fm is None or k >= kstar:
                continue
            frac = k / c["nO"]
            b = next(bb for bb in bins if bb[0] < frac <= bb[1])
            acc[b]["n"] += 1
            gl = gap_lower(fm, fl)
            acc[b]["lpr"].append(100 * gl)
            sh = lpr_share(ff, fm, fl)
            if sh is not None:
                acc[b]["lpr_share"].append(sh)
            if f1 is None:
                acc[b]["cv1_inf"] += 1
            else:
                acc[b]["cv1"].append(100 * gap_upper(fm, f1))
            if fk is None:
                acc[b]["cvk_inf"] += 1
            else:
                acc[b]["cvk"].append(100 * gap_upper(fm, fk))
            if 100 * gl > biggest["lpr"][0]:
                biggest["lpr"] = (100 * gl, (c["gap"], c["i"], k, c["nO"]))
            if f1 is not None and 100 * gap_upper(fm, f1) > biggest["cv1"][0]:
                biggest["cv1"] = (100 * gap_upper(fm, f1), (c["gap"], c["i"], k, c["nO"]))
        if cost1 is not None and cost1 > C.ACTIVE_REL:
            for q in remaining:
                k_q = max(1, int(np.floor(q * c["nO"])))
                fm = c["f_mip"][k_q - 1]
                if fm is not None:
                    remaining[q].append(((fm - ff) / ff) / cost1)
    out["bins"] = [dict(lo=b[0], hi=b[1], n=v["n"], lpr_mean=float(np.mean(v["lpr"])) if v["lpr"] else None, lpr_median=ST.median(v["lpr"]),
                        share_median=ST.median(v["lpr_share"]), cv1_inf=v["cv1_inf"], cvk_inf=v["cvk_inf"],
                        cv1_mean=float(np.mean(v["cv1"])) if v["cv1"] else None, cvk_mean=float(np.mean(v["cvk"])) if v["cvk"] else None) for b, v in acc.items()]
    out["n_active_steps"] = sum(v["n"] for v in acc.values())
    out["cv1_inf_total"] = sum(v["cv1_inf"] for v in acc.values())
    out["cvk_inf_total"] = sum(v["cvk_inf"] for v in acc.values())
    out["remaining"] = {q: ST.median(v) for q, v in remaining.items()}
    out["biggest"] = biggest
    return out


def h3_results(d: float, k: int) -> dict:
    """Handinstanz auf dem 6 x 6-Gitter (drei Winkel 0, 60, 120 Grad, sechs Strahlen je Winkel, 24 Körpervoxel; Tumor 8, 9, 11; Organ 6, 14, 17, 22, 23)."""
    a = M.make_matrix(6, (0, 60, 120), 6)
    prob = S.Problem(a, (8, 9, 11), (6, 14, 17, 22, 23))
    free = prob.solve_free()
    exact = S.solve_enumeration(prob, d, k)
    return dict(prob=prob, free=free, exact=exact, lpr=prob.solve_relax(d, k), cvk=prob.solve_cvar(d, k), cv1=prob.solve_cvar(d, k + 1), d=d, k=k)
