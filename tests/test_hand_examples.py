"""Handbeispiele von Hand nachgerechnet (DEMO-PLAYBOOK §1b): jede Prüfung ein Wert, den man mit Papier nachvollzieht.

H0 Strahlenmatrix (5 x 5-Gitter), H1 Zielkonflikt mit zwei Strahlen (Brüche), H2 Dosis-Volumen-Grenze mit drei Strahlen und drei Organvoxeln (Brüche und
unabhängiger zweiter Rechenweg), H3 dasselbe auf einem 6 x 6-Gitter. Die Erwartungswerte stehen als Rechnung im Test, nicht als aus dem Code kopierte Zahl."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import dpl_hand as H
import dpl_model as M
import dpl_solvers as S


def _voxel_index(g, x, y):
    xs, ys = M.voxel_centers(g)
    body = M.body_mask(g)
    flat = [(xs[iy, ix], ys[iy, ix]) for iy in range(g) for ix in range(g) if body[iy, ix]]
    return next(i for i, (px, py) in enumerate(flat) if abs(px - x) < 1e-12 and abs(py - y) < 1e-12)


# ---------------------------------------------------------------- H0 Strahlenmatrix
def test_h0_body_voxel_counts():
    assert int(M.body_mask(5).sum()) == 21           # die vier Ecken (±0,8; ±0,8) liegen außerhalb des Radius 0,95
    assert int(M.body_mask(6).sum()) == 24
    assert int(M.body_mask(32).sum()) == 732


def test_h0_matrix_entries_by_hand():
    a = M.make_matrix(5, (0, 180, 90), 5)           # Spalten: Winkel 0 (0 bis 4), 180 (5 bis 9), 90 (10 bis 14)
    centre = _voxel_index(5, 0.0, 0.0)
    assert a[centre, 2] == pytest.approx(math.exp(-0.95), rel=1e-13)            # Mitte, 0 Grad, Mittelstrahl: Tiefe 0,95
    v = _voxel_index(5, 0.4, 0.0)
    assert a[v, 2] == pytest.approx(math.exp(-1.35), rel=1e-13)                 # (0,4; 0), 0 Grad: Tiefe 0,95 + 0,4
    assert a[v, 5 + 2] == pytest.approx(math.exp(-0.55), rel=1e-13)             # 180 Grad: Tiefe 0,95 - 0,4
    w = _voxel_index(5, 0.0, 0.4)
    assert a[w, 10 + 2] == pytest.approx(math.exp(-1.35), rel=1e-12)            # (0; 0,4), 90 Grad: Tiefe 1,35
    omega = 2 * 0.95 / 5
    sig = 0.6 * omega
    assert a[centre, 3] == pytest.approx(math.exp(-(omega ** 2) / (2 * sig ** 2)) * math.exp(-0.95), rel=1e-12)   # Nachbarstrahl: Gauß-Faktor exp(-1/(2 * 0,36))
    assert a[centre, 4] == 0.0                                                   # zwei Strahlbreiten entfernt: über 3 Sigma abgeschnitten


def test_h0_matrix_equals_scalar_formula():
    a = M.make_matrix(6, (0, 60, 120), 6)
    xs, ys = M.voxel_centers(6)
    body = M.body_mask(6)
    pts = [(xs[iy, ix], ys[iy, ix]) for iy in range(6) for ix in range(6) if body[iy, ix]]
    for vi, (x, y) in enumerate(pts):
        for ai, ang in enumerate((0, 60, 120)):
            for b in range(6):
                assert a[vi, ai * 6 + b] == pytest.approx(M.beam_scalar(x, y, ang, b, 6), rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------- H1 Zielkonflikt in Brüchen
def _h1(tau):
    cons = [([F(-1), F(0)], F(0)), ([F(0), F(-1)], F(0)), ([F(-1), F(-1)], F(-1)), ([F(1), F(0)], tau)]
    return H.lp_exact([F(1), F(2)], cons, 2)


@pytest.mark.parametrize("tau", [F(1, 5), F(1, 2), F(9, 10)])
def test_h1_optimum_is_two_minus_tau(tau):
    f, x = _h1(tau)
    assert f == 2 - tau and x == [tau, 1 - tau]                               # x1 = tau, x2 = 1 - tau: Kosten tau + 2 (1 - tau)


def test_h1_price_is_the_slope_minus_one_and_zero_beyond_the_knee():
    f1, f2 = _h1(F(1, 2))[0], _h1(F(3, 5))[0]
    assert (f2 - f1) / F(1, 10) == -1
    assert _h1(F(3, 2))[0] == 1                                               # jenseits des Knies (tau > 1) wirkt die Organgrenze nicht mehr: x = (1, 0), Wert 1


# ---------------------------------------------------------------- H2 in Brüchen
@pytest.fixture(scope="module")
def h2():
    return H.solve_all()


def test_h2_costs_per_beam_by_hand():
    assert H.COST == (F(3, 8), F(7, 16), F(3, 4))                              # (3/4 + 3/4 + 0 + 0)/4, (3/4 + 0 + 1/4 + 3/4)/4, (1/2 + 1 + 1/2 + 1)/4


def test_h2_plan_without_limit(h2):
    f, x = h2["free"]
    assert f == F(3, 8) and x == [F(1), F(0), F(0)]
    assert H.organ_dose(x) == [F(3, 4), F(3, 4), F(0)] and H.over_count(x) == 2     # zwei Organvoxel über d = 1/2, erlaubt ist eins


def test_h2_exact_by_enumeration_of_the_sets(h2):
    sets = h2["sets"]
    assert sets[()][0] is None and sets[(2,)][0] is None                          # S leer und S = {O3}: unzulässig
    assert sets[(0,)] == (F(2, 3) * F(3, 8) + F(1, 3) * F(7, 16), [F(2, 3), F(1, 3), F(0)])     # 19/48
    assert sets[(1,)] == (F(3, 4), [F(0), F(0), F(1)])
    f, x, s = h2["exact"]
    assert f == F(19, 48) and s == (0,) and H.over_count(x) == 1


def test_h2_relaxation_sees_nothing(h2):
    f, _ = h2["relax"]
    assert f == h2["free"][0] == F(3, 8)
    assert h2["excess_free"] == F(1, 4) + F(1, 4) == 1 * (H.CAP - H.D)            # Überschuss genau k (M - d)


def test_h2_cvar_forms(h2):
    f2, x2 = h2["cvar"][2]
    assert f2 == F(5, 12) and x2 == [F(1, 3), F(2, 3), F(0)]                      # eigene Variante CVaR(k+1) = CVaR(2)
    assert h2["cvar"][1][0] is None                                              # Literaturform CVaR(k) = CVaR(1): unzulässig, obwohl die Aufgabe lösbar ist
    f_free, f_rel, f_mip = h2["free"][0], h2["relax"][0], h2["exact"][0]
    assert f_free == f_rel < f_mip < f2
    assert (f_mip - f_rel) / f_mip == F(1, 19) == (f2 - f_mip) / f_mip           # beide Lücken 1/19 = 5,26 %


def test_h2_midpoint_of_two_feasible_plans_violates_the_limit(h2):
    xa, xb, mid = h2["plans"]
    assert mid == [F(1, 3), F(1, 6), F(1, 2)]
    assert H.organ_dose(mid) == [F(5, 8), F(3, 4), F(7, 24)]
    assert H.over_count(xa) == 1 and H.over_count(xb) == 1 and H.over_count(mid) == 2 > H.K


def test_h2_second_path_with_the_solver(mini_h2, h2):
    p = mini_h2
    kw = dict(d=0.5, cap=1.0)
    assert S.solve_enumeration(p, k=1, **kw).f == pytest.approx(float(F(19, 48)), abs=1e-9)
    assert p.solve_mip(k=1, **kw).f == pytest.approx(19 / 48, abs=1e-9)
    assert p.solve_relax(k=1, **kw).f == pytest.approx(3 / 8, abs=1e-9)
    assert p.solve_relax_excess(k=1, **kw).f == pytest.approx(3 / 8, abs=1e-9)
    assert p.solve_cvar(m=2, **kw).f == pytest.approx(5 / 12, abs=1e-9)
    assert S.solve_cvar_pairs(p, m=2, **kw).f == pytest.approx(5 / 12, abs=1e-9)
    assert not p.solve_cvar(m=1, **kw).feasible and not S.solve_cvar_pairs(p, m=1, **kw).feasible


# ---------------------------------------------------------------- H3 auf dem 6 x 6-Gitter
def test_h3_values(mini_h3):
    p = mini_h3
    assert p.n_o == 5 and p.n_t == 3 and p.a.shape == (24, 18)
    free = p.solve_free()
    assert free.f == pytest.approx(0.46274, abs=5e-6) and p.over(free.x, 0.3) == 2
    exact = S.solve_enumeration(p, 0.3, 1)
    assert exact.f == pytest.approx(0.50795, abs=5e-6)
    assert p.solve_mip(0.3, 1).f == pytest.approx(exact.f, abs=1e-7)
    assert p.solve_relax(0.3, 1).f == pytest.approx(0.49042, abs=5e-6)
    assert p.solve_cvar(0.3, 2).f == pytest.approx(0.78814, abs=5e-6)
    assert not p.solve_cvar(0.3, 1).feasible                                     # Literaturform bei k = 1 = strenge Höchstdosis: unzulässig
    assert (exact.f - free.f) / free.f == pytest.approx(0.0977, abs=5e-4)        # die Grenze kostet 9,8 %


def test_h3_counter_cases_at_the_same_phantom(mini_h3):
    p = mini_h3
    free = p.solve_free().f
    e5 = S.solve_enumeration(p, 0.5, 1).f
    assert e5 == pytest.approx(0.48040, abs=5e-6)
    assert p.solve_relax(0.5, 1).f == pytest.approx(e5, abs=1e-8)                # die Schranke ist hier exakt: Lücke 0
    assert p.solve_cvar(0.5, 1).f == pytest.approx(0.52439, abs=5e-6) and p.solve_cvar(0.5, 2).f == pytest.approx(0.52251, abs=5e-6)
    assert S.solve_enumeration(p, 0.3, 2).f == pytest.approx(free, abs=1e-8)     # k = 2: der Strahlplan ohne Grenze erfüllt die Grenze schon
    assert p.solve_cvar(0.3, 2).f == pytest.approx(0.78814, abs=5e-6)            # die CVaR-Form kostet trotzdem
    assert p.over(p.solve_cvar(0.5, 1).x, 0.5) == 0                              # CVaR(k) mit k = 1: höchstens k - 1 = 0 über d


def test_h3_two_ways_agree(mini_h3):
    p = mini_h3
    assert p.solve_relax(0.3, 1).f == pytest.approx(p.solve_relax_excess(0.3, 1).f, abs=1e-8)
    for m in (1, 2):
        a, b = p.solve_cvar(0.5, m), S.solve_cvar_pairs(p, 0.5, m)
        assert a.f == pytest.approx(b.f, abs=1e-8)
