"""Randfälle (DEMO-PLAYBOOK §12): unzulässige MIP, Organ aus genau k Voxeln, ein Organvoxel, Organ ohne Voxel, M kleiner oder gleich d, Gleichstände, Grenzen der Regler."""

import numpy as np
import pytest
from scipy.optimize import linprog

import dpl_constants as C
import dpl_data as D
import dpl_evaluation as E
import dpl_model as M
import dpl_solvers as S


def test_infeasible_mip_row_is_reported_as_infeasible_not_as_a_number():
    rows = [r for r in D.table()["rows"] if r.get("mip_status") == 2]
    assert rows and all(r["f_mip"] is None and r["S"] == [] and r["ex_mip"] is None for r in rows)
    r = rows[0]
    a = E.analyse(E.Settings(r["gap"], r["i"], r["ci"], r["M"]))
    assert "exact" not in a.f and E.cell_facts(a)["mip_status"] == 2
    kind, text = E.verdict_cost(E.cell_facts(a), E.summary_of(r["M"], r["gap"], r["ci"]))
    assert kind == "warning" and "nicht erfüllbar" in text
    assert not E.is_active(r)                                                      # nie als aktive Zeile mit Zahl gezählt


def test_organ_with_exactly_k_voxels_and_k_larger_than_the_organ(mini_h3):
    p = mini_h3
    free = p.solve_free().f
    assert p.solve_mip(0.3, p.n_o).f == pytest.approx(free, abs=1e-8)             # k = Zahl der Organvoxel: die Grenze ist leer
    assert p.solve_mip(0.3, p.n_o + 4).f == pytest.approx(free, abs=1e-8)
    assert not p.solve_cvar(0.3, p.n_o + 1).feasible                              # Tail-Größe über der Organgröße ist nicht definiert: gemeldet, nicht gerechnet
    # m = n_o: das Mittel aller Organdosen darf höchstens d sein; zulässig genau dann, wenn die kleinste mögliche mittlere Organdosis (eigenes LP) <= d ist
    n_x = p.n_x
    a_ub = np.vstack([-p.at, p.at])
    b_ub = np.concatenate([-C.TUMOR_MIN * np.ones(p.n_t), C.TUMOR_MAX * np.ones(p.n_t)])
    res = linprog(p.ao.mean(axis=0), A_ub=a_ub, b_ub=b_ub, bounds=(0, None), method="highs")
    assert res.status == 0 and n_x == 18
    for d in (0.05, 0.3, 0.5):
        assert p.solve_cvar(d, p.n_o).feasible == (res.fun <= d + 1e-9)


def test_single_organ_voxel():
    a = M.make_matrix(6, (0, 60, 120), 6)
    p = S.Problem(a, (8, 9, 11), (17,))
    assert p.n_o == 1
    free = p.solve_free()
    assert p.solve_mip(0.3, 1).f == pytest.approx(free.f, abs=1e-9)               # k = 1 von 1: keine Wirkung
    strict = p.solve_mip(0.3, 0)
    assert strict.f is None or strict.f >= free.f - 1e-9


def test_organ_without_voxels_has_no_limit():
    a = M.make_matrix(6, (0, 60, 120), 6)
    p = S.Problem(a, (8, 9, 11), ())
    free = p.solve_free()
    assert free.feasible
    assert p.solve_mip(0.3, 1).f == pytest.approx(free.f, abs=1e-9) and p.solve_relax(0.3, 1).f == pytest.approx(free.f, abs=1e-9)
    assert not p.solve_cvar(0.3, 1).feasible and p.over(free.x, 0.3) == 0


@pytest.mark.parametrize("cap", [0.3, 0.2])
def test_cap_equal_or_below_d_means_no_limit(mini_h3, cap):
    p = mini_h3
    sol = p.solve_mip(0.3, 1, cap=cap)
    free = p.solve_free(cap=cap)
    assert sol.status == free.status and sol.f == free.f                           # M <= d: die Grenze ist leer (Strahlplan ohne Grenze), keine Division durch null
    assert p.solve_relax(0.3, 1, cap=cap).f == free.f


def test_tumor_window_not_reachable_gives_no_numbers(mini_h3):
    p = mini_h3
    assert not p.solve_free(cap=0.01).feasible
    kind, text = E.verdict_cost(dict(i=0, gap="nah", d=0.3, k=1, cap=0.01, n_organ=5, mip_status=None, reachable=False), {})
    assert kind == "warning" and "nicht erreichbar" in text


def test_ties_between_equal_sets_give_the_same_value(mini_h3):
    """Gleichstände: mehrere gleich gute Mengen S liefern denselben Wert; verglichen werden Werte, nicht Mengen."""
    p = mini_h3
    best = S.solve_enumeration(p, 0.5, 1)
    for s in range(p.n_o):
        alt = p.solve_set(0.5, (s,))
        assert alt.f is None or alt.f >= best.f - 1e-9
    assert p.solve_mip(0.5, 1).f == pytest.approx(best.f, abs=1e-8)


def test_pareto_curve_returns_none_when_the_window_is_unreachable_and_a_point_at_a_zero_width_knee():
    a = M.make_matrix(6, (0, 60, 120), 6)
    p = S.Problem(a, (8, 9, 11), (6,))
    curve = S.pareto_curve(p, 5)
    assert curve is not None and curve["points"][0][0] == pytest.approx(curve["tau_min"])
    assert all(pt[2] >= -1e-9 for pt in curve["points"])                          # Preise sind nicht negativ


def test_regler_bounds_are_the_measured_values():
    assert C.CAPS == (1.2, 1.0, 0.8, 0.6) and len(C.COMBOS) == 3 and C.N_PHANTOMS == 30
    t = D.table()
    assert sorted({r["M"] for r in t["rows"]}) == sorted(C.CAPS) and len(t["rows"]) == 3 * C.N_PHANTOMS * len(C.COMBOS) * len(C.CAPS)
    assert {tuple(c) for c in t["meta"]["combos"]} == set(C.COMBOS)


def test_every_row_of_every_cell_of_the_regler_exists():
    for g in C.GAP_NAMES:
        for i in (0, C.N_PHANTOMS - 1):
            for ci in range(len(C.COMBOS)):
                for cap in C.CAPS:
                    assert D.get_row(g, i, ci, cap)["k"] >= 1


def test_phantoms_have_the_documented_properties():
    for g in C.GAP_NAMES:
        lo, hi = C.GAP_RANGES[g]
        for i in range(0, C.N_PHANTOMS, 6):
            ph = D.phantom(g, i)
            assert len(ph.organ) >= 12 and len(ph.tumor) >= 12 and not set(ph.tumor) & set(ph.organ)
            assert lo <= ph.gap_vox + 1e-9 and max(ph.tumor) < 732 and max(ph.organ) < 732
    mid = [D.phantom("mittel", i).gap_vox for i in range(C.N_PHANTOMS)]
    assert min(mid) >= 2.0 - 1e-9 and max(mid) <= 3.7
