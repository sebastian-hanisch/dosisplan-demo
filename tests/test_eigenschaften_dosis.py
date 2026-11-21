"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Beziehungen, die gelten müssen, ohne dass man das richtige Ergebnis kennt.

Jede Eigenschaft an 20 bis 50 Instanzen mit festem Zufalls-Seed (SplitMix64), plus Randfälle. Alle Aussagen sind Aussagen über die Aufgabe (LP und MIP),
nicht über das heutige Verhalten des Codes: Zielfunktion mal c ändert den Strahlplan nicht, Umnummerieren der Strahlen ändert den Wert nicht,
mehr erlaubte Voxel (k) oder höhere Grenzdosis (d) verschlechtern nie, ein zusätzlicher Einstrahlwinkel verschlechtert nie, die Reihenfolge der Schranken."""

import numpy as np
import pytest

import dpl_constants as C
import dpl_data as D
import dpl_model as M
import dpl_solvers as S
from dpl_rng import SplitMix64


def _instances(n, seed=21):
    rng = SplitMix64(seed)
    a = M.make_matrix(6, (0, 60, 120), 6)
    others = [v for v in range(24) if v not in (8, 9, 11)]
    out = []
    for _ in range(n):
        pool = list(others)
        rng.shuffle(pool)
        out.append(S.Problem(a, (8, 9, 11), tuple(sorted(pool[: 4 + rng.below(3)]))))
    return out


def test_scaling_the_objective_scales_the_value_and_keeps_the_plan():
    for p in _instances(20):
        base = p.solve_mip(0.3, 2)
        if not base.feasible:
            continue
        q = S.Problem(p.a, p.tumor, p.organ)
        q.c = 3.5 * p.c
        scaled = q.solve_mip(0.3, 2)
        assert scaled.f == pytest.approx(3.5 * base.f, rel=1e-7)
        assert p.solve_set(0.3, scaled.chosen).f == pytest.approx(base.f, rel=1e-6)          # der Strahlplan zur gefundenen Menge ist für beide optimal


def test_renumbering_the_beams_does_not_change_the_values():
    rng = SplitMix64(4)
    for p in _instances(20, seed=22):
        perm = list(range(p.n_x))
        rng.shuffle(perm)
        q = S.Problem(p.a[:, perm], p.tumor, p.organ)
        assert q.solve_free().f == pytest.approx(p.solve_free().f, abs=1e-9)
        for d, k in ((0.3, 2), (0.5, 1)):
            a, b = p.solve_mip(d, k), q.solve_mip(d, k)
            assert a.feasible == b.feasible
            if a.feasible:
                assert a.f == pytest.approx(b.f, abs=1e-8)
            assert p.solve_relax(d, k).f == pytest.approx(q.solve_relax(d, k).f, abs=1e-8)


def test_values_do_not_increase_when_k_grows():
    """Exakt, Relaxation und CVaR(k+1) fallen oder bleiben gleich, wenn k wächst; die CVaR-Form mit Tail-Größe k ebenfalls (Mittel der k größten Dosen fällt mit k)."""
    for p in _instances(10):
        prev = {}
        for k in range(1, p.n_o):
            cur = {"mip": p.solve_mip(0.3, k).f, "lpr": p.solve_relax(0.3, k).f, "cv1": p.solve_cvar(0.3, k + 1).f, "cvk": p.solve_cvar(0.3, k).f}
            for name, v in cur.items():
                if v is not None and prev.get(name) is not None:
                    assert v <= prev[name] + 1e-7, (name, k)
                if v is None and prev.get(name) is not None:
                    pytest.fail(f"{name} wird bei wachsendem k unzulässig")
            prev = cur


def test_exact_value_does_not_increase_when_d_grows():
    for p in _instances(12):
        vals = [p.solve_mip(d, 2).f for d in (0.2, 0.3, 0.4, 0.5)]
        feas = [v for v in vals if v is not None]
        assert all(b <= a + 1e-8 for a, b in zip(feas, feas[1:]))
        assert all(v is not None for v in vals[vals.index(next(v for v in vals if v is not None)):])     # einmal zulässig, bei größerem d weiter zulässig


def test_limit_is_empty_for_k_at_least_the_organ_size_and_for_d_at_least_the_cap():
    for p in _instances(15):
        free = p.solve_free().f
        assert p.solve_mip(0.3, p.n_o).f == pytest.approx(free, abs=1e-8)
        assert p.solve_mip(0.3, p.n_o + 3).f == pytest.approx(free, abs=1e-8)
        assert p.solve_mip(C.ORGAN_CAP, 1).f == pytest.approx(free, abs=1e-8)                   # d = M: kein Voxel kann über d liegen
        assert p.solve_mip(1.5, 1).f == pytest.approx(free, abs=1e-8)                           # d > M: die Grenze ist leer


def test_k_zero_is_the_strict_maximum_dose():
    for p in _instances(15):
        strict = p.solve_strict(0.3)
        mip0 = p.solve_mip(0.3, 0)
        assert strict.feasible == mip0.feasible
        if strict.feasible:
            assert mip0.f == pytest.approx(strict.f, abs=1e-8)
            assert p.over(strict.x, 0.3) == 0


def test_an_additional_beam_angle_never_makes_the_value_worse():
    """120 Proben in der früheren Messung; hier 24 Proben auf den 32 x 32-Phantomen: Wert mit n + 1 Winkeln höchstens so groß wie mit n Winkeln."""
    rng = SplitMix64(8)
    a = M.standard_matrix()
    n_tested = 0
    for gap in C.GAP_NAMES:
        for i in (0, 1):
            ph = D.phantom(gap, i)
            for _ in range(4):
                angles = list(range(C.N_ANGLES))
                rng.shuffle(angles)
                n = 4 + rng.below(3)
                small, big = angles[:n], angles[: n + 1]
                cols = lambda sel: [ang * C.N_BEAMS + b for ang in sel for b in range(C.N_BEAMS)]  # noqa: E731
                ps, pb = S.Problem(a[:, cols(small)], ph.tumor, ph.organ), S.Problem(a[:, cols(big)], ph.tumor, ph.organ)
                fs, fb = ps.solve_free(), pb.solve_free()
                if fs.feasible:
                    assert fb.feasible and fb.f <= fs.f + 1e-9
                    n_tested += 1
    assert n_tested >= 12


def test_order_of_the_bounds_on_phantom_rows():
    rows = [r for r in D.rows(cap=C.ORGAN_CAP) if r.get("mip_status") == 0]
    n = 0
    for r in rows:
        tol = 1e-5 * max(1.0, r["f_mip"])
        assert r["f_free"] <= r["f_lpr"] + tol and r["f_lpr"] <= r["f_mip"] + tol and r["f_free"] <= r["f_mip"] + tol
        if r["f_cv1"] is not None:
            assert r["f_mip"] <= r["f_cv1"] + tol
            if r["f_cvk"] is not None:
                assert r["f_cv1"] <= r["f_cvk"] + tol
        elif r["f_cvk"] is not None:
            pytest.fail("CVaR(k) zulässig, CVaR(k+1) nicht: widerspricht der Reihenfolge")
        n += 1
    assert n == 268


def test_frozen_curves_do_not_increase_in_k():
    cs = D.curves()["curves"]
    assert len(cs) == 3 * C.N_CURVE_PHANTOMS
    for c in cs:
        for key in ("f_mip", "f_lpr", "f_cv1", "f_cvk"):
            v = [x for x in c[key] if x is not None]
            assert all(b <= a + 1e-5 * max(1.0, a) for a, b in zip(v, v[1:])), (c["gap"], c["i"], key)


def test_same_seed_same_phantom():
    a = M.make_phantom("mittel", 5)
    b = M.make_phantom("mittel", 5)
    assert a == b and a != M.make_phantom("mittel", 6)
    assert np.allclose(M.standard_matrix(), M.make_matrix(C.GRID, C.ANGLES_DEG, C.N_BEAMS), rtol=0, atol=0)
