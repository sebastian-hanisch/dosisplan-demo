"""Orakel-Tests (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2): jede Kernroutine gegen einen anderen Rechenweg, schnell (unter 10 s).

(a) LP-Dualität: Primal = Dual (unabhängig gebautes Dual-LP), Schattenpreis der Pareto-Kurve zwischen linker und rechter Steigung.
(b) Exakte MIP gegen Aufzählung aller Mengen S auf 6 x 6-Gittern mit drei Winkeln; Kette Relaxation <= exakt <= CVaR(k+1) <= CVaR(k).
(d) Strahlenmatrix: Skalar-Schleife gegen Matrix, Summe des Streukerns gegen die Poisson-Summe.
(e) Überschuss-Form = Big-M-Relaxation auf Zeilen der vorgerechneten Tabelle.
(f) CVaR-Restriktion: Paar-Form = Rockafellar-Uryasev; Zählung (CVaR(k) höchstens k - 1, CVaR(k+1) höchstens k Organvoxel über d).
(h) M_v: die größte Dosis, die ein Organvoxel unter Tumorfenster und Höchstdosis erreichen kann, ist die Höchstdosis selbst (kein schärferes Big-M).
(i) Die vorgerechnete Tabelle gegen eine frische Rechnung der LP-Teile (Werte relativ 1e-5).
Referenzen: scipy HiGHS (`linprog`, `milp`) als zweiter Löser für dieselbe Mathematik, Aufzählung, Brüche (tests/test_hand_examples.py)."""

import math

import numpy as np
import pytest
from scipy.optimize import linprog

import dpl_constants as C
import dpl_data as D
import dpl_evaluation as E
import dpl_model as M
import dpl_solvers as S
from dpl_rng import SplitMix64


def _random_instances(n, seed=11):
    """n kleine Instanzen auf dem 6 x 6-Gitter: Tumor wie H3 (Voxel 8, 9, 11), Organ aus 4 bis 6 zufällig gewählten anderen Voxeln (SplitMix64)."""
    rng = SplitMix64(seed)
    a = M.make_matrix(6, (0, 60, 120), 6)
    out = []
    others = [v for v in range(24) if v not in (8, 9, 11)]
    for _ in range(n):
        pool = list(others)
        rng.shuffle(pool)
        out.append(S.Problem(a, (8, 9, 11), tuple(sorted(pool[: 4 + rng.below(3)]))))
    return out


# ---------------------------------------------------------------- (a) LP-Dualität
def _dual_value(p: S.Problem, cap: float) -> float:
    """Wert des Dual-LP, unabhängig vom Primal-Aufbau gebaut. Primal: min c x  s.t.  G x <= h, x >= 0. Dual: max -h y  s.t.  -G^T y <= c, y >= 0,
    als linprog: min h y unter -G^T y <= c, Dualwert = -(min h y)."""
    g = np.vstack([-p.at, p.at, p.ao])
    h = np.concatenate([-C.TUMOR_MIN * np.ones(p.n_t), C.TUMOR_MAX * np.ones(p.n_t), cap * np.ones(p.n_o)])
    res = linprog(h, A_ub=-g.T, b_ub=p.c, bounds=(0, None), method="highs")
    assert res.status == 0
    return -float(res.fun)


def test_primal_equals_dual_on_six_phantoms():
    for gap in C.GAP_NAMES:
        for i in (0, 1):
            ph = D.phantom(gap, i)
            p = S.Problem(M.standard_matrix(), ph.tumor, ph.organ)
            primal = p.solve_free()
            assert primal.feasible
            assert _dual_value(p, C.ORGAN_CAP) == pytest.approx(primal.f, rel=1e-8)


def test_shadow_price_lies_between_left_and_right_slope():
    for gap, i in (("nah", 0), ("mittel", 1), ("fern", 2)):
        ph = D.phantom(gap, i)
        p = S.Problem(M.standard_matrix(), ph.tumor, ph.organ)
        curve = S.pareto_curve(p, 11)
        pts = curve["points"]
        for j in range(1, len(pts) - 1):
            left = -(pts[j][1] - pts[j - 1][1]) / (pts[j][0] - pts[j - 1][0])
            right = -(pts[j + 1][1] - pts[j][1]) / (pts[j + 1][0] - pts[j][0])
            price = pts[j][2]
            lo, hi = min(left, right), max(left, right)
            assert lo - 1e-6 <= price <= hi + 1e-6, (gap, i, j, left, price, right)
        assert all(p1[2] >= p2[2] - 1e-6 for p1, p2 in zip(pts, pts[1:]))                  # der Preis fällt mit tau (konvexe Kurve)
        assert all(q1[1] >= q2[1] - 1e-9 for q1, q2 in zip(pts, pts[1:]))                  # die Zielgröße fällt mit tau


# ---------------------------------------------------------------- (b) exakt gegen Aufzählung
@pytest.mark.parametrize("d,k", [(0.3, 1), (0.3, 2), (0.5, 1), (0.5, 2)])
def test_mip_equals_enumeration_and_chain_of_bounds(d, k):
    n_cmp = n_both_infeasible = 0
    for p in _random_instances(8):
        enum = S.solve_enumeration(p, d, k)
        mip = p.solve_mip(d, k)
        if not enum.feasible:
            assert not mip.feasible
            n_both_infeasible += 1
            continue
        assert mip.status == S.OPTIMAL and mip.f == pytest.approx(enum.f, abs=1e-7)
        assert p.over(mip.x, d) <= k and p.over(enum.x, d) <= k
        rel = p.solve_relax(d, k)
        assert rel.f <= mip.f + 1e-8
        cv1, cvk = p.solve_cvar(d, k + 1), p.solve_cvar(d, k)
        if cv1.feasible:
            assert mip.f <= cv1.f + 1e-7 and p.over(cv1.x, d) <= k
        if cvk.feasible:
            assert cv1.feasible and cv1.f <= cvk.f + 1e-7                                    # CVaR(k) ist strenger als CVaR(k+1)
        n_cmp += 1
    assert n_cmp + n_both_infeasible == 8 and n_cmp >= 4                                    # genug Vergleiche, sonst wäre der Test leer


def test_mip_plan_from_the_found_set_has_the_mip_value():
    for p in _random_instances(6, seed=5):
        mip = p.solve_mip(0.3, 2)
        if mip.feasible:
            assert p.solve_set(0.3, mip.chosen).f == pytest.approx(mip.f, abs=1e-7)


# ---------------------------------------------------------------- (d) Strahlenmatrix
def test_streukern_sum_matches_the_poisson_sum():
    a = M.standard_matrix()
    # Summe der 16 Strahlen eines Winkels je Voxel, geteilt durch die Dämpfung = Summe der Gaußfaktoren = sigma sqrt(2 pi) / Breite (Poisson-Summe), bei 3 sigma abgeschnitten
    omega = 2 * C.BODY_RADIUS / C.N_BEAMS
    expected = C.SIGMA_REL * math.sqrt(2 * math.pi)
    xs, ys = M.voxel_centers()
    body = M.body_mask()
    x, y = xs[body], ys[body]
    c, s = M.cos_sin(0.0)
    lat = -x * s + y * c
    u0 = -np.sqrt(np.maximum(C.BODY_RADIUS ** 2 - lat ** 2, 0.0))
    damp = np.exp(-C.MU * np.maximum(x * c + y * s - u0, 0.0))
    ratio = a[:, :C.N_BEAMS].sum(axis=1) / damp
    inner = np.abs(lat) < C.BODY_RADIUS - 3 * C.SIGMA_REL * omega                              # Strahlen am Rand fehlen: nur innere Voxel
    assert inner.sum() > 300
    assert np.max(np.abs(ratio[inner] - expected)) < 0.02                                       # Abweichung wegen des Abschneidens bei 3 sigma (gemessen 0,011)


def test_scalar_loop_equals_matrix_on_random_entries():
    a = M.standard_matrix()
    xs, ys = M.voxel_centers()
    body = M.body_mask()
    x, y = xs[body], ys[body]
    rng = SplitMix64(3)
    for _ in range(300):
        v, j = rng.below(a.shape[0]), rng.below(a.shape[1])
        ang, beam = C.ANGLES_DEG[j // C.N_BEAMS], j % C.N_BEAMS
        assert a[v, j] == pytest.approx(M.beam_scalar(x[v], y[v], ang, beam, C.N_BEAMS), rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------- (e), (f) Überschuss-Form, CVaR in zwei Wegen und Zählung auf Zeilen der Tabelle
def _active_rows(n, cap=C.ORGAN_CAP):
    rows = [r for r in D.rows(cap=cap) if E.is_active(r)]
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def test_excess_form_equals_big_m_relaxation_on_20_rows():
    for r in _active_rows(20):
        p = E.problem_of(r["gap"], r["i"])
        a, b = p.solve_relax(r["d"], r["k"], r["M"]), p.solve_relax_excess(r["d"], r["k"], r["M"])
        assert a.f == pytest.approx(b.f, abs=1e-7)
        assert a.f == pytest.approx(r["f_lpr"], abs=1e-7)                                         # und gleich der eingefrorenen Zeile


def test_cvar_count_and_order_on_20_rows():
    n_k, n_k1 = 0, 0
    for r in _active_rows(20):
        p = E.problem_of(r["gap"], r["i"])
        ck, c1 = p.solve_cvar(r["d"], r["k"], r["M"]), p.solve_cvar(r["d"], r["k"] + 1, r["M"])
        if ck.feasible:
            assert p.over(ck.x, r["d"]) <= r["k"] - 1                                             # Mittel der k größten <= d: höchstens k - 1 Voxel über d
            assert c1.feasible and c1.f <= ck.f + 1e-5
            n_k += 1
        if c1.feasible:
            assert p.over(c1.x, r["d"]) <= r["k"]
            assert r["f_mip"] <= c1.f + 1e-5 * max(1.0, c1.f)
            n_k1 += 1
    assert n_k >= 8 and n_k1 >= 8


def test_cvar_matches_pair_form_on_small_instances():
    for p in _random_instances(6, seed=9):
        for m in (1, 2):
            a, b = p.solve_cvar(0.5, m), S.solve_cvar_pairs(p, 0.5, m)
            assert a.feasible == b.feasible
            if a.feasible:
                assert a.f == pytest.approx(b.f, abs=1e-8)


# ---------------------------------------------------------------- (h) kein schärferes Big-M
def test_largest_reachable_organ_dose_is_the_cap_on_three_phantoms():
    for gap, i in (("nah", 3), ("mittel", 4), ("fern", 5)):
        ph = D.phantom(gap, i)
        p = S.Problem(M.standard_matrix(), ph.tumor, ph.organ)
        a_ub = np.vstack([-p.at, p.at, p.ao])
        b_ub = np.concatenate([-C.TUMOR_MIN * np.ones(p.n_t), C.TUMOR_MAX * np.ones(p.n_t), C.ORGAN_CAP * np.ones(p.n_o)])
        for v in range(p.n_o):
            res = linprog(-p.ao[v], A_ub=a_ub, b_ub=b_ub, bounds=(0, None), method="highs")
            assert res.status == 0 and -res.fun == pytest.approx(C.ORGAN_CAP, abs=1e-7)


# ---------------------------------------------------------------- (i) Tabelle gegen frische Rechnung
def test_frozen_table_matches_a_fresh_computation_of_the_lp_parts():
    for r in _active_rows(8) + [x for x in D.rows(cap=0.6) if E.is_active(x)][:2]:
        p = E.problem_of(r["gap"], r["i"])
        assert p.solve_free(r["M"]).f == pytest.approx(r["f_free"], abs=1e-8)
        assert p.solve_relax(r["d"], r["k"], r["M"]).f == pytest.approx(r["f_lpr"], abs=1e-7)
        plan = p.solve_set(r["d"], r["S"], r["M"])
        assert plan.f == pytest.approx(r["f_mip"], abs=1e-8)
        assert p.over(plan.x, r["d"]) <= r["k"] and r["ex_mip"] <= r["k"]            # die Zahl über d hängt am Eckpunkt des LP (Gleichstände), garantiert ist nur höchstens k
        for name, m in (("f_cvk", r["k"]), ("f_cv1", r["k"] + 1)):
            cv = p.solve_cvar(r["d"], m, r["M"])
            assert (cv.f is None) == (r[name] is None)
            if cv.f is not None:
                assert cv.f == pytest.approx(r[name], abs=1e-7)


def test_frozen_exact_value_is_not_beaten_by_any_enumerated_neighbour_set():
    """Der eingefrorene exakte Wert ist optimal: kein LP zu einer anderen zulässigen Menge (Tausch eines Voxels) ist besser (Stichprobe, 1e-5 relativ)."""
    rows = _active_rows(4)
    for r in rows:
        p = E.problem_of(r["gap"], r["i"])
        S0 = set(r["S"])
        for out in list(S0)[:3]:
            for inn in range(p.n_o):
                if inn in S0:
                    continue
                alt = p.solve_set(r["d"], sorted((S0 - {out}) | {inn}), r["M"])
                if alt.feasible and len((S0 - {out}) | {inn}) <= r["k"]:
                    assert alt.f >= r["f_mip"] - 1e-5 * r["f_mip"]
                break
