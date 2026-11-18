"""Löser für die Dosis-Volumen-Grenze: Strahlplan ohne Grenze, exakte MIP, LP-Relaxation, CVaR-Restriktion in zwei Formen, Strahlpläne zu festen Mengen.

Aufgabe: min c.x mit c = mittlere Dosis aller Nicht-Tumor-Voxel, Tumorfenster 1,0 <= D <= 1,2, Organ-Höchstdosis D_v <= M, x >= 0, und
"höchstens k Organvoxel mit D_v > d". Drei Wege zur selben Grenze:

* exakt (MIP):      D_v <= d + (M - d) z_v, z_v in {0, 1}, Summe z <= k
* LP-Relaxation:    z_v in [0, 1] (untere Schranke)
* CVaR-Restriktion: Mittel der m größten Organdosen <= d (Rockafellar und Uryasev: zeta + Summe w / m <= d, w_v >= D_v - zeta, w >= 0).
                    m = k ist die Form der Literatur (Kishimoto/Yamashita 2018, Gl. (3); Engberg u. a. 2017), m = k + 1 ist eine eigene,
                    schwächere Variante dieser Demo. Beide sind Restriktionen (obere Schranken).

Nur numpy und scipy (HiGHS über `linprog` und `milp`). Herkunft: für diese Demo neu geschrieben.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

import dpl_constants as C

OPTIMAL, INFEASIBLE, LIMIT, ERROR = "optimal", "infeasible", "limit", "error"


@dataclass(frozen=True)
class Solution:
    status: str                       # optimal, infeasible, limit (Lösung bekannt, nicht bewiesen), error
    f: float | None = None            # Zielwert
    x: np.ndarray | None = field(default=None, compare=False, repr=False)
    chosen: tuple = ()                # MIP: Organvoxel (lokale Indizes), die über d liegen dürfen (z_v = 1)

    @property
    def feasible(self) -> bool:
        return self.f is not None


class Problem:
    """Eine Instanz: Dosismatrix mit Tumor- und Organvoxeln (Indizes in der Reihenfolge der Körpervoxel)."""

    def __init__(self, a: np.ndarray, tumor, organ):
        self.a = np.asarray(a, dtype=float)
        self.tumor = np.asarray(tumor, dtype=int)
        self.organ = np.asarray(organ, dtype=int)
        self.at = self.a[self.tumor]
        self.ao = self.a[self.organ]
        non_tumor = np.ones(self.a.shape[0], dtype=bool)
        non_tumor[self.tumor] = False
        self.c = self.a[non_tumor].mean(axis=0)
        self.n_t, self.n_o, self.n_x = self.at.shape[0], self.ao.shape[0], self.a.shape[1]

    # ---- gemeinsame Bausteine
    def _window(self):
        return [-self.at, self.at], [-C.TUMOR_MIN * np.ones(self.n_t), C.TUMOR_MAX * np.ones(self.n_t)]

    @staticmethod
    def _lp(c, a_ub, b_ub, bounds) -> Solution:
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.status == 0:
            return Solution(OPTIMAL, float(res.fun), np.asarray(res.x))
        return Solution(INFEASIBLE if res.status == 2 else ERROR)

    def over(self, x, d: float, tol: float = C.TOL_COUNT) -> int:
        """Zahl der Organvoxel mit Dosis über d."""
        return int((self.ao @ x > d + tol).sum())

    def organ_dose(self, x) -> np.ndarray:
        return self.ao @ x

    def excess(self, x, d: float) -> float:
        """Gesamtüberschuss über d: Summe der Organdosen über d (Zielgröße der Überschuss-Form der Relaxation)."""
        return float(np.maximum(self.ao @ x - d, 0.0).sum())

    # ---- Strahlplan ohne Grenze und strenge Höchstdosis
    def solve_free(self, cap: float = C.ORGAN_CAP) -> Solution:
        """Strahlplan ohne Dosis-Volumen-Grenze: nur Tumorfenster und Organ-Höchstdosis `cap`."""
        rows, rhs = self._window()
        rows.append(self.ao)
        rhs.append(cap * np.ones(self.n_o))
        return self._lp(self.c, np.vstack(rows), np.concatenate(rhs), (0, None))

    def solve_strict(self, d: float) -> Solution:
        """Strenge Höchstdosis d für alle Organvoxel (k = 0); kann unzulässig sein."""
        return self.solve_free(cap=d)

    # ---- Strahlplan zu einer festen Menge S (LP): die Voxel in S dürfen bis `cap`, alle anderen höchstens d
    def solve_set(self, d: float, chosen, cap: float = C.ORGAN_CAP) -> Solution:
        limit = np.full(self.n_o, d)
        if len(chosen):
            limit[list(chosen)] = cap
        rows, rhs = self._window()
        rows.append(self.ao)
        rhs.append(np.minimum(limit, cap))
        sol = self._lp(self.c, np.vstack(rows), np.concatenate(rhs), (0, None))
        return Solution(sol.status, sol.f, sol.x, tuple(int(v) for v in chosen))

    # ---- exakt (MIP) und Relaxation (gleiche Form, z stetig)
    def _bigm(self, d: float, k: int, cap: float, relax: bool, forbid, time_limit: float):
        n_t, n_o, n_x = self.n_t, self.n_o, self.n_x
        coef = cap - d
        rows = [np.hstack([self.at, np.zeros((n_t, n_o))]), np.hstack([self.at, np.zeros((n_t, n_o))]),
                np.hstack([self.ao, -coef * np.eye(n_o)]), np.hstack([np.zeros((1, n_x)), np.ones((1, n_o))])]
        lo = np.concatenate([C.TUMOR_MIN * np.ones(n_t), np.full(n_t, -np.inf), np.full(n_o, -np.inf), [-np.inf]])
        hi = np.concatenate([np.full(n_t, np.inf), C.TUMOR_MAX * np.ones(n_t), d * np.ones(n_o), [float(k)]])
        ub = np.concatenate([np.full(n_x, np.inf), np.ones(n_o)])
        if forbid is not None and len(forbid):
            ub[n_x + np.asarray(forbid, dtype=int)] = 0.0
        integrality = np.concatenate([np.zeros(n_x), np.zeros(n_o) if relax else np.ones(n_o)])
        res = milp(np.concatenate([self.c, np.zeros(n_o)]), constraints=LinearConstraint(np.vstack(rows), lo, hi), integrality=integrality,
                   bounds=Bounds(np.zeros(n_x + n_o), ub), options=dict(time_limit=time_limit, mip_rel_gap=C.MIP_REL_GAP, disp=False))
        return res

    def solve_mip(self, d: float, k: int, cap: float = C.ORGAN_CAP, forbid=None, time_limit: float = C.MIP_TIME_LIMIT) -> Solution:
        """Exakte Lösung der Grenze. status optimal = bewiesen (relative Lücke <= 1e-6), limit = beste gefundene Lösung ohne Beweis, infeasible = unzulässig.
        `forbid`: Organvoxel (lokale Indizes), die nicht über d liegen dürfen. Ist cap <= d, gibt es keine Grenze (Strahlplan ohne Grenze)."""
        if cap <= d or self.n_o == 0:
            sol = self.solve_free(cap)
            return Solution(sol.status, sol.f, sol.x, ())
        res = self._bigm(d, k, cap, False, forbid, time_limit)
        if res.x is None:
            return Solution(INFEASIBLE if res.status == 2 else ERROR)
        z = res.x[self.n_x:]
        chosen = tuple(int(v) for v in np.nonzero(z > 0.5)[0])
        return Solution(OPTIMAL if res.status == 0 else LIMIT, float(res.fun), np.asarray(res.x[:self.n_x]), chosen)

    def solve_relax(self, d: float, k: int, cap: float = C.ORGAN_CAP) -> Solution:
        """LP-Relaxation der Big-M-Form (z_v in [0, 1]): untere Schranke des exakten Werts."""
        if cap <= d or self.n_o == 0:
            return self.solve_free(cap)
        res = self._bigm(d, k, cap, True, None, 120.0)
        if res.status != 0:
            return Solution(INFEASIBLE if res.status == 2 else ERROR)
        return Solution(OPTIMAL, float(res.fun), np.asarray(res.x[:self.n_x]))

    def solve_relax_excess(self, d: float, k: int, cap: float = C.ORGAN_CAP) -> Solution:
        """Zweiter Weg zur Relaxation: Überschuss-Form  Summe_v max(0, D_v - d) <= k (M - d)  (mit D_v <= M). Gleichwertig zur Big-M-Relaxation."""
        n_t, n_o, n_x = self.n_t, self.n_o, self.n_x
        a_ub = [np.hstack([-self.at, np.zeros((n_t, n_o))]), np.hstack([self.at, np.zeros((n_t, n_o))]),
                np.hstack([self.ao, -np.eye(n_o)]), np.hstack([np.zeros((1, n_x)), np.ones((1, n_o))]),
                np.hstack([self.ao, np.zeros((n_o, n_o))])]
        b_ub = [-C.TUMOR_MIN * np.ones(n_t), C.TUMOR_MAX * np.ones(n_t), d * np.ones(n_o), [k * (cap - d)], cap * np.ones(n_o)]
        sol = self._lp(np.concatenate([self.c, np.zeros(n_o)]), np.vstack(a_ub), np.concatenate(b_ub), (0, None))
        return Solution(sol.status, sol.f, None if sol.x is None else sol.x[:n_x])

    # ---- CVaR-Restriktion: Mittel der m größten Organdosen <= d
    def solve_cvar(self, d: float, m: int, cap: float = C.ORGAN_CAP) -> Solution:
        """Rockafellar und Uryasev: Variablen x, zeta (frei), w >= 0. Für 1 <= m <= Zahl der Organvoxel; sonst unzulässig gemeldet (m > n_o hat keinen Sinn)."""
        n_t, n_o, n_x = self.n_t, self.n_o, self.n_x
        if m < 1 or m > n_o:
            return Solution(INFEASIBLE)
        a_ub = [np.hstack([-self.at, np.zeros((n_t, 1 + n_o))]), np.hstack([self.at, np.zeros((n_t, 1 + n_o))]),
                np.hstack([self.ao, -np.ones((n_o, 1)), -np.eye(n_o)]), np.hstack([np.zeros((1, n_x)), np.ones((1, 1)), np.ones((1, n_o)) / m]),
                np.hstack([self.ao, np.zeros((n_o, 1 + n_o))])]
        b_ub = [-C.TUMOR_MIN * np.ones(n_t), C.TUMOR_MAX * np.ones(n_t), np.zeros(n_o), [d], cap * np.ones(n_o)]
        bounds = [(0, None)] * n_x + [(None, None)] + [(0, None)] * n_o
        sol = self._lp(np.concatenate([self.c, [0.0], np.zeros(n_o)]), np.vstack(a_ub), np.concatenate(b_ub), bounds)
        return Solution(sol.status, sol.f, None if sol.x is None else sol.x[:n_x])



def solve_cvar_pairs(p: Problem, d: float, m: int, cap: float = C.ORGAN_CAP) -> Solution:
    """Zweiter Weg zur CVaR-Restriktion (nur für kleine Organe): Mittel über jede m-Menge von Organvoxeln <= d, ohne Hilfsvariable."""
    from itertools import combinations
    if m < 1 or m > p.n_o:
        return Solution(INFEASIBLE)
    pair_rows = [p.ao[list(s)].mean(axis=0) for s in combinations(range(p.n_o), m)]
    rows, rhs = p._window()
    rows.append(p.ao)
    rhs.append(cap * np.ones(p.n_o))
    rows.append(np.array(pair_rows))
    rhs.append(d * np.ones(len(pair_rows)))
    return Problem._lp(p.c, np.vstack(rows), np.concatenate(rhs), (0, None))


def solve_enumeration(p: Problem, d: float, k: int, cap: float = C.ORGAN_CAP) -> Solution:
    """Orakel der exakten MIP für kleine Organe: Aufzählung aller Mengen S mit höchstens k Voxeln, die über d liegen dürfen; je Menge ein LP."""
    from itertools import combinations
    best = None
    for size in range(min(k, p.n_o) + 1):
        for s in combinations(range(p.n_o), size):
            sol = p.solve_set(d, s, cap)
            if sol.feasible and (best is None or sol.f < best.f):
                best = sol
    return best if best is not None else Solution(INFEASIBLE)


# ---------------------------------------------------------------- Zielkonflikt (Vorstufe): Pareto-Kurve und Preis

def min_organ_cap(p: Problem) -> float | None:
    """Kleinste mögliche Organ-Höchstdosis tau_min bei Tumorfenster [1; 1,2]."""
    n_t, n_o, n_x = p.n_t, p.n_o, p.n_x
    a_ub = np.vstack([np.hstack([-p.at, np.zeros((n_t, 1))]), np.hstack([p.at, np.zeros((n_t, 1))]), np.hstack([p.ao, -np.ones((n_o, 1))])])
    b_ub = np.concatenate([-C.TUMOR_MIN * np.ones(n_t), C.TUMOR_MAX * np.ones(n_t), np.zeros(n_o)])
    c = np.zeros(n_x + 1)
    c[-1] = 1.0
    sol = Problem._lp(c, a_ub, b_ub, (0, None))
    return sol.f


def pareto_point(p: Problem, tau: float) -> tuple[float, float] | None:
    """(Zielwert, Preis) bei Organ-Höchstdosis tau; Preis = -(Summe der Duale der Organzeilen) = Abnahme der Zielgröße je Einheit tau."""
    rows, rhs = p._window()
    rows.append(p.ao)
    rhs.append(tau * np.ones(p.n_o))
    res = linprog(p.c, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs), bounds=(0, None), method="highs")
    if res.status != 0:
        return None
    return float(res.fun), float(-np.asarray(res.ineqlin.marginals)[-p.n_o:].sum())


def pareto_curve(p: Problem, n_points: int = C.D1_POINTS):
    """Punkte (tau, f, Preis) von tau_min bis zum Knie (größte Organdosis des Strahlplans ohne Organgrenze); None, wenn das Tumorfenster nicht erreichbar ist."""
    tau_min = min_organ_cap(p)
    free = p.solve_free(C.ORGAN_CAP)
    if tau_min is None or not free.feasible:
        return None
    knee = max(float(p.organ_dose(free.x).max()), tau_min)
    taus = np.linspace(tau_min, knee, n_points) if knee > tau_min + 1e-9 else np.array([tau_min])
    pts = [(float(t),) + pareto_point(p, float(t)) for t in taus]
    return dict(tau_min=tau_min, knee=knee, points=pts)


# ---------------------------------------------------------------- Nichtkonvexitäts-Probe

def nonconvex_probe(p: Problem, d: float, k: int, cap: float, chosen, time_limit: float = C.MIP_TIME_LIMIT):
    """Zwei zulässige Strahlpläne mit disjunkten Überschreitungsmengen und ihr Mittelpunkt.
    Strahlplan a: LP zur MIP-Menge `chosen`. Strahlplan b: MIP, in dem die Voxel, die a über d liegen, nicht über d liegen dürfen.
    Rückgabe None, wenn a keine Überschreitung hat oder b nicht existiert, sonst dict mit Mengen, Strahlplänen und der Zahl der Organvoxel über d im Mittelpunkt."""
    pa = p.solve_set(d, chosen, cap)
    if not pa.feasible:
        return None
    over_a = tuple(int(v) for v in np.nonzero(p.organ_dose(pa.x) > d + C.TOL_COUNT)[0])
    if not over_a:
        return None
    mb = p.solve_mip(d, k, cap, forbid=over_a, time_limit=time_limit)
    if not mb.feasible:
        return None
    pb = p.solve_set(d, mb.chosen, cap)
    if not pb.feasible:
        return None
    mid = 0.5 * (pa.x + pb.x)
    over_b = tuple(int(v) for v in np.nonzero(p.organ_dose(pb.x) > d + C.TOL_COUNT)[0])
    return dict(xa=pa.x, xb=pb.x, mid=mid, f_a=pa.f, f_b=pb.f, chosen_b=mb.chosen, over_a=over_a, over_b=over_b, over_mid=p.over(mid, d))
