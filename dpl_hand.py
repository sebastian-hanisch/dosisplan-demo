"""Handinstanz in Brüchen: drei Strahlen, drei Organvoxel, ein Tumorvoxel, alle Werte exakt (Modul `fractions`, Eckenaufzählung statt Löser).

Sie zeigt in einer Rechnung, die man von Hand nachvollziehen kann, alle Wege zur Grenze "höchstens k = 1 von 3 Organvoxeln über d = 1/2":
Strahlplan ohne Grenze, exakt (Aufzählung der Mengen S), LP-Relaxation (Überschuss-Form), CVaR(k + 1) = CVaR(2) und CVaR(k) = CVaR(1), dazu der Mittelpunkt
zweier zulässiger Strahlpläne. Erfundene Zahlen, kein Bezug zu einer echten Dosis.
Herkunft: für diese Demo geschrieben; ein früheres Handbeispiel außerhalb des Repos (nicht importiert) lieferte die Zielwerte."""

import itertools
from fractions import Fraction as F

# Dosis je Einheit Strahlgewicht (Spalten = Strahlen 1 bis 3)
ORGAN = ((F(3, 4), F(3, 4), F(1, 2)),     # O1
         (F(3, 4), F(0), F(1)),           # O2
         (F(0), F(1, 4), F(1, 2)))        # O3
NORMAL = (F(0), F(3, 4), F(1))            # gesundes Gewebe N
TUMOR = (F(1), F(1), F(1))                # Tumorvoxel T: Dosis x1 + x2 + x3 in [1; 6/5]
D, CAP, K = F(1, 2), F(1), 1              # Grenzdosis d, Höchstdosis M (zugleich Big-M), höchstens k Organvoxel über d
N_X = 3
COST = tuple((sum(ORGAN[v][j] for v in range(3)) + NORMAL[j]) / 4 for j in range(N_X))   # mittlere Dosis der vier Nicht-Tumor-Voxel


def lp_exact(cost, cons, n=N_X):
    """min cost.x über {x : a.x <= b für (a, b) in cons} durch Aufzählung aller Ecken in Brüchen. Gibt (Wert, x) oder (None, None)."""
    best, arg = None, None
    for idx in itertools.combinations(range(len(cons)), n):
        m = [list(cons[i][0]) + [cons[i][1]] for i in idx]
        ok = True
        for col in range(n):
            piv = next((r for r in range(col, n) if m[r][col] != 0), None)
            if piv is None:
                ok = False
                break
            m[col], m[piv] = m[piv], m[col]
            pv = m[col][col]
            m[col] = [v / pv for v in m[col]]
            for r in range(n):
                if r != col and m[r][col] != 0:
                    f = m[r][col]
                    m[r] = [a - f * b for a, b in zip(m[r], m[col])]
        if not ok:
            continue
        x = [m[i][n] for i in range(n)]
        if all(sum(a * xi for a, xi in zip(ac, x)) <= bc for ac, bc in cons):
            v = sum(c * xi for c, xi in zip(cost, x))
            if best is None or v < best:
                best, arg = v, x
    return best, arg


def base_constraints():
    cons = [([F(-1) if i == j else F(0) for i in range(N_X)], F(0)) for j in range(N_X)]     # x >= 0
    cons.append(([-w for w in TUMOR], F(-1)))
    cons.append((list(TUMOR), F(6, 5)))
    cons += [(list(ORGAN[v]), CAP) for v in range(3)]
    return cons


def organ_dose(x):
    return [sum(ORGAN[v][j] * x[j] for j in range(N_X)) for v in range(3)]


def over_count(x, d=D):
    return sum(1 for dv in organ_dose(x) if dv > d)


def solve_all():
    """Alle Wege auf der Handinstanz; Rückgabe als Wörterbuch mit Brüchen."""
    out = {}
    out["free"] = lp_exact(COST, base_constraints())
    sets = {}
    for size in range(K + 1):
        for s in itertools.combinations(range(3), size):
            cons = base_constraints() + [(list(ORGAN[v]), D) for v in range(3) if v not in s]
            sets[s] = lp_exact(COST, cons)
    out["sets"] = sets
    feas = [(f, s, x) for s, (f, x) in sets.items() if f is not None]
    f_best, s_best, x_best = min(feas, key=lambda t: t[0])
    out["exact"] = (f_best, x_best, s_best)
    rel = base_constraints()
    for size in range(1, 4):
        for s in itertools.combinations(range(3), size):
            rel.append(([sum(ORGAN[v][j] for v in s) for j in range(N_X)], K * (CAP - D) + size * D))
    out["relax"] = lp_exact(COST, rel)
    out["cvar"] = {}
    for m in (K, K + 1):
        cons = base_constraints() + [([sum(ORGAN[v][j] for v in s) / m for j in range(N_X)], D) for s in itertools.combinations(range(3), m)]
        out["cvar"][m] = lp_exact(COST, cons)
    xa, xb = sets[(0,)][1], sets[(1,)][1]
    mid = [(p + q) / 2 for p, q in zip(xa, xb)]
    out["plans"] = (xa, xb, mid)
    out["excess_free"] = sum(max(F(0), dv - D) for dv in organ_dose(out["free"][1]))
    return out


def fmt(fr: F) -> str:
    """Bruch als Text, z. B. 19/48 oder 3."""
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"
