"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier; dazu je Preset die erwartete Verdikt-Zeile."""

import pytest

import dpl_constants as C
import dpl_data as D
import dpl_evaluation as E


def _dist(n_active=12, median=1.5, n_proven=30, n_infeasible=0, share=None, cvk_inf=2, cv1_inf=1):
    return dict(n_active=n_active, n_proven=n_proven, n_mip_infeasible=n_infeasible, cost=dict(median=median),
                share=share or dict(n=n_active, mean=0.1, median=0.0, n_lt001=9), cvk_infeasible=cvk_inf, cv1_infeasible=cv1_inf)


def _facts(**kw):
    base = dict(i=3, gap="nah", d=0.3, k=4, cap=1.2, n_organ=20, mip_status=0, reachable=True, f_free=0.3, ex_free=11, has_exact=True, active=True, cost=2.5, ex_exact=4,
                share=0.25, gap_lpr=1.2, gap_cvk=3.4, gap_cv1=1.7, ex_cvk=2, ex_cv1=3)
    base.update(kw)
    return base


# ---------------------------------------------------------------- Kernfrage 1
def test_cost_verdict_active_names_every_number():
    kind, text = E.verdict_cost(_facts(), _dist())
    assert kind == "success"
    assert "Phantom 3, Organ nah, d = 0,3, höchstens k = 4 von 20 Organvoxeln" in text
    assert "11 Organvoxel über d, im exakten Strahlplan 4" in text and "kostet 2,50 %" in text and "ein Phantom, kein Befund" in text
    assert "aktiv in 12 von 30" in text and "Median der Kosten 1,50 %" in text


def test_cost_verdict_inactive_is_an_info_and_says_so():
    kind, text = E.verdict_cost(_facts(active=False, cost=0.0, ex_free=3), _dist(n_active=0))
    assert kind == "info" and "nicht aktiv" in text and "3 Organvoxel über d (erlaubt 4)" in text and "aktiv in 0 von 30" in text


def test_cost_verdict_infeasible_unproven_and_unreachable():
    kind, text = E.verdict_cost(_facts(mip_status=2), _dist(n_infeasible=3))
    assert kind == "warning" and "nicht erfüllbar" in text and "3 unzulässig" in text
    kind, text = E.verdict_cost(_facts(mip_status=1), _dist())
    assert kind == "warning" and "nicht bewiesen" in text
    kind, text = E.verdict_cost(_facts(reachable=False), _dist())
    assert kind == "warning" and "nicht erreichbar" in text


# ---------------------------------------------------------------- Kernfrage 2 und 3
def test_relax_verdict_share_and_gap():
    kind, text = E.verdict_relax(_facts(share=0.25, gap_lpr=1.2), _dist())
    assert kind == "info" and "25 % der Kosten" in text and "1,20 %" in text and "unter 0,01 in 9 von 12" in text
    assert "nichts, was die LP-Relaxation sehen" in E.verdict_relax(_facts(active=False), _dist())[1]
    assert E.verdict_relax(_facts(has_exact=False), _dist())[0] == "info"


def test_cvar_verdict_infeasible_is_a_warning_with_both_forms_named():
    kind, text = E.verdict_cvar(_facts(gap_cvk=None, ex_cvk=None), _dist())
    assert kind == "warning" and "CVaR(k), Form der Literatur, ist unzulässig, obwohl die Aufgabe lösbar ist" in text
    assert "CVaR(k+1), eigene Variante, zahlt 1,70 % mehr als exakt (3 Organvoxel über d)" in text
    assert "CVaR(k) unzulässig in 2 von 12, CVaR(k+1) in 1" in text
    kind, text = E.verdict_cvar(_facts(), _dist())
    assert kind == "info" and "zahlt 3,40 % mehr als exakt (2 Organvoxel über d)" in text and "erlaubt k = 4" in text


def test_share_is_not_defined_without_costs():
    assert E.lpr_share(1.0, 1.0, 1.0) is None
    assert E.lpr_share(1.0, 1.0 + 1e-7, 1.0) is None
    assert E.lpr_share(1.0, 1.2, 1.1) == pytest.approx(0.5)
    assert E.cost_rel(1.02, 1.0) == pytest.approx(0.02) and E.gap_lower(1.0, 0.98) == pytest.approx(0.02) and E.gap_upper(1.0, 1.03) == pytest.approx(0.03)


def test_is_active_threshold_and_needs_a_proof():
    base = dict(mip_status=0, f_free=1.0)
    assert not E.is_active(dict(base, f_mip=1.0 + 0.9e-5)) and E.is_active(dict(base, f_mip=1.0 + 1.1e-5))
    assert not E.is_active(dict(base, f_mip=1.5, mip_status=1)) and not E.is_active(dict(base, f_mip=None, mip_status=2))


def test_number_formats_use_the_decimal_comma():
    assert E.percent(2.5) == "2,5 %" and E.percent(0.1234, 2) == "0,12 %" and E.num(0.5) == "0,5" and E.num(1.2, 3) == "1,200"


# ---------------------------------------------------------------- je Preset die erwartete Verdikt-Zeile
def _preset_verdicts(name):
    pr = C.PRESETS[name]
    a = E.analyse(E.Settings(pr["gap"], pr["seed"], pr["combo"], pr["cap"]))
    f = E.cell_facts(a)
    dist = E.summary_of(pr["cap"], pr["gap"], pr["combo"])
    return f, E.verdict_cost(f, dist), E.verdict_relax(f, dist), E.verdict_cvar(f, dist)


def test_preset_standard_has_an_active_limit_and_feasible_cvar():
    f, cost, relax, cvar = _preset_verdicts("Organ nah, 25 % (Standard)")
    assert f["active"] and cost[0] == "success" and "kostet" in cost[1] and cvar[0] == "info"


def test_preset_fern_reports_no_effect_and_no_large_gap():
    """Gegenfall: ein Preset ohne Effekt darf keinen Effekt melden (DEMO-PLAYBOOK 11a Punkt 7)."""
    f, cost, relax, cvar = _preset_verdicts("Organ fern (Gegenfall)")
    assert not f["active"] and cost[0] == "info" and "nicht aktiv" in cost[1] and "Lücke 0" in relax[1] and "strenger als nötig" in cvar[1]


def test_preset_cvar_infeasible_is_infeasible_for_the_literature_form_only():
    f, cost, relax, cvar = _preset_verdicts("Strenge Grenze 10 %: CVaR unzulässig")
    assert f["gap_cvk"] is None and f["gap_cv1"] is not None and cvar[0] == "warning"
    assert "CVaR(k), Form der Literatur, ist unzulässig" in cvar[1] and "CVaR(k+1), eigene Variante, zahlt" in cvar[1]


def test_preset_highest_price_is_the_maximum_of_the_table():
    f, cost, _, _ = _preset_verdicts("Höchster Preis der Grenze")
    best = max(E.cost_rel(r["f_mip"], r["f_free"]) for r in D.rows(cap=C.ORGAN_CAP) if r.get("mip_status") == 0)
    assert f["cost"] == pytest.approx(100 * best, rel=1e-6) and cost[0] == "success"


def test_preset_small_cap_shows_a_larger_share_than_the_same_cell_at_1_2():
    f, _, relax, _ = _preset_verdicts("Kleine Höchstdosis M = 0,6")
    f12 = E.cell_facts(E.analyse(E.Settings("nah", 8, 1, 1.2)))
    assert f["share"] is not None and f["share"] > 0.1 and (f12["share"] or 0.0) < f["share"] and "Anteil" in relax[1]
