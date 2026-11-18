"""Claims-Tests: jede Zahl der README („Befunde“, „Befunde und Korrekturen“, „Ehrliche Grenzen“, Hook) wird aus den eingefrorenen Tabellen mit denselben
Auswertungsfunktionen wie in der App neu berechnet und in genau der Schreibweise der README gesucht. Ändert sich eine Zahl oder ein Satz der README, schlägt der Test an.

Zählungen und Prozentzahlen sind Funktionen der Tabellen (JSON, auf jeder Plattform gleich): exakt verglichen. Die Tabellen selbst entstehen im Erzeugungsskript; ihre
Stichproben-Neuberechnung (Band, relativ 1e-5) liegt in `tests/test_table.py` und in `python tools/dpl_make_table.py --check`.
Die Zahlen der ersten Rechnung (Vergleich in „Befunde und Korrekturen“) stammen aus deren Ausgaben und sind hier nicht neu gerechnet; die Zahlen dieser Demo schon.
Die Zeit der Live-Rechnung steht als „Messung dieser README, nicht im Test“ in der README."""

from pathlib import Path

import pytest

import dpl_constants as C
import dpl_data as D
import dpl_evaluation as E

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def de(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def has(text):
    assert text in README, f"nicht in der README: {text!r}"


@pytest.fixture(scope="module")
def s1():
    return E.summary_of(C.ORGAN_CAP)


@pytest.fixture(scope="module")
def s2():
    return E.summary_of(C.ORGAN_CAP, second=True)


def _g(name, second=False):
    return E.summary_of(C.ORGAN_CAP, name, None, second)


def test_claim_rows_and_status(s1, s2):
    assert s1["n"] == 270 == s2["n"]
    has(f"**{s1['n_proven']} bewiesen optimal, {s1['n_unproven']} ohne Beweis, {s1['n_mip_infeasible']} unzulässig**")
    has(f"zweite Menge: {s2['n_proven']} bewiesen, {s2['n_mip_infeasible']} unzulässig")
    inf = [r for r in D.table()["rows"] if r["M"] == C.ORGAN_CAP and r["mip_status"] == 2]
    assert len(inf) == 2 and {(r["gap"], r["ci"]) for r in inf} == {("nah", 2)}         # beide Organ nah, d = 0,3, 10 %
    assert C.COMBOS[2] == (0.3, 0.10)


def test_claim_active_rows(s1, s2):
    has(f"**{s1['n_active']} von {s1['n_proven']}**")
    has(", ".join(f"{g} {_g(g)['n_active']} von {_g(g)['n_proven']}" for g in C.GAP_NAMES))
    has(f"zweite Menge {s2['n_active']} von {s2['n_proven']}")
    lo, hi = round(100 * s1["n_active"] / s1["n_proven"]), round(100 * s2["n_active"] / s2["n_proven"])
    has(f"Der Anteil liegt bei **{lo} %** (erste Menge) und **{hi} %** (zweite Menge); mit nur zwei Mengen ist das keine Schwankungsbreite")
    has(f"(fern {_g('fern')['n_active']} gegen {_g('fern', True)['n_active']})")


def test_claim_inactive_rows(s1):
    has(f"{s1['n_inactive']} von {s1['n_proven']}; in {s1['n_inactive_free_ok']} dieser {s1['n_inactive']} Zeilen")
    f, f2 = _g("fern"), _g("fern", True)
    has(f"Organ fern: {f['n_inactive']} von {f['n']} Zeilen nicht aktiv (zweite Menge {f2['n_inactive']} von {f2['n']})")
    has(f"die Grenze ist in {f['n_inactive']} von {f['n']} Zeilen nicht aktiv, in der zweiten Menge in {f2['n_inactive']} von {f2['n']})")


def test_claim_cost_of_the_limit(s1, s2):
    c, c2 = s1["cost"], s2["cost"]
    has(f"Mittel **{de(c['mean'])} %** [{de(c['lo'])}; {de(c['hi'])}], Median {de(c['median'])} %, Maximum {de(c['max'])} %")
    has(f"zweite Menge Mittel {de(c2['mean'])} % [{de(c2['lo'])}; {de(c2['hi'])}], Median {de(c2['median'])} %, Maximum {de(c2['max'])} %")


def test_claim_gap_of_the_relaxation(s1, s2):
    g, g2 = s1["gap_lpr"], s2["gap_lpr"]
    has(f"Mittel **{de(g['mean'])} %** [{de(g['lo'])}; {de(g['hi'])}], Median {de(g['median'])} %; über 1 % in {s1['n_gap_lpr_gt1']} von {g['n']}")
    has(f"zweite Menge Mittel {de(g2['mean'])} % [{de(g2['lo'])}; {de(g2['hi'])}], Median {de(g2['median'])} %, über 1 % in {s2['n_gap_lpr_gt1']} von {g2['n']}")


def test_claim_share_of_the_relaxation(s1, s2):
    a, b = s1["share"], s2["share"]
    has(f"**Median {de(a['median'], 3)}**, Mittel {de(a['mean'], 3)}, Maximum {de(a['max'], 3)}; unter 0,01 in **{a['n_lt001']} von {a['n']}**, über 0,5 in {a['n_gt05']}")
    has(f"zweite Menge Median {de(b['median'], 3)}, unter 0,01 in {b['n_lt001']} von {b['n']}, über 0,5 in {b['n_gt05']}")
    assert a["median"] == 0.0 and b["median"] == 0.0


def test_claim_relaxation_equals_free_plan(s1, s2):
    assert s1["n_lpr_eq_free"] == s1["n_free_excess_within"]
    has(f"in {s1['n_free_excess_within']} von {s1['n_active']} aktiven Zeilen ist der Gesamtüberschuss des Strahlplans ohne Grenze höchstens k (M − d)")
    has(f"zweite Menge {s2['n_lpr_eq_free']} von {s2['n_active']} (Wert) und {s2['n_free_excess_within']} von {s2['n_active']} (Überschuss)")
    has(f"Relaxation hat in {s1['n_lpr_eq_free']} von {s1['n_active']} aktiven Zeilen denselben Zielwert wie der Strahlplan ohne Grenze")
    has(f"Zielwert der Relaxation = Wert ohne Grenze in {s1['n_lpr_eq_free']} von {s1['n_active']} aktiven Zeilen")


def test_claim_cvar_k(s1, s2):
    g, g2 = s1["cvk_gap"], s2["cvk_gap"]
    assert s1["cvk_infeasible_nah"] == s1["cvk_infeasible"]                              # alle Organ nah
    has(f"**unzulässig in {s1['cvk_infeasible']} von {s1['n_active']}** aktiven Zeilen (alle Organ nah; zweite Menge {s2['cvk_infeasible']} von {s2['n_active']})")
    has(f"Lücke in den {g['n']} zulässigen Zeilen Mittel **{de(g['mean'])} %** [{de(g['lo'])}; {de(g['hi'])}], Median {de(g['median'])} %, Maximum {de(g['max'])} %, über 1 % in {s1['cvk_n_gt1']}")
    has(f"zweite Menge Mittel {de(g2['mean'])} % [{de(g2['lo'])}; {de(g2['hi'])}]")


def test_claim_cvar_k_plus_1(s1, s2):
    g, g2, both = s1["cv1_gap"], s2["cv1_gap"], s1["cv1_gap_on_cvk_rows"]
    assert s1["cv1_infeasible_nah"] == s1["cv1_infeasible"] and s2["cv1_infeasible_nah"] == s2["cv1_infeasible"]
    has(f"unzulässig in {s1['cv1_infeasible']} von {s1['n_active']} (alle Organ nah; zweite Menge {s2['cv1_infeasible']} von {s2['n_active']})")
    has(f"Lücke in den {g['n']} zulässigen Zeilen Mittel {de(g['mean'])} % [{de(g['lo'])}; {de(g['hi'])}], Median {de(g['median'])} %, Maximum {de(g['max'])} %, über 1 % in {s1['cv1_n_gt1']}")
    has(f"auf den {s1['n_both']} Zeilen, in denen beide zulässig sind: CVaR(k+1) Mittel {de(both['mean'])} % [{de(both['lo'])}; {de(both['hi'])}], Median {de(both['median'])} %")
    has(f"zweite Menge Mittel {de(g2['mean'])} %")


def test_claim_order_and_count(s1):
    assert s1["order_violations"] == 0 and s1["only_cv1_infeasible"] == 0
    has(f"{s1['order_violations']} Verletzungen (Toleranz 1e-5 relativ) in {s1['n_both']} Zeilen mit beiden CVaR; nur CVaR(k) unzulässig in {s1['only_cvk_infeasible']} Zeilen, nur CVaR(k+1) nie")
    has(f"höchstens k − 1 Organvoxel über d in {s1['n_cvk_le_km1']} von {s1['n_cvk_plans']}, CVaR(k+1)-Strahlplan höchstens k in {s1['n_cv1_le_k']} von {s1['n_cv1_plans']}")
    assert s1["n_cvk_le_km1"] == s1["n_cvk_plans"] and s1["n_cv1_le_k"] == s1["n_cv1_plans"]


def test_claim_organ_voxels_over_d(s1):
    e = s1["ex"]
    has(f"ohne Grenze {de(e['free'], 1)}; exakt {de(e['mip'], 1)}; CVaR(k+1) {de(e['cv1'], 1)}; CVaR(k) {de(e['cvk'], 1)}; erlaubt k im Mittel {de(e['k'], 1)} "
        f"(über alle {s1['n_active']} aktiven Zeilen: ohne Grenze {de(s1['ex_free_mean_act'], 1)}, exakt {de(s1['ex_mip_mean_act'], 1)})")
    has(f"senkt die Zahl der Organvoxel über d im Mittel von {de(s1['ex_free_mean_act'], 1)} auf {de(s1['ex_mip_mean_act'], 1)}")


def test_claim_by_organ_distance():
    parts = [f"{g} {de(_g(g)['gap_lpr']['mean'])} / {de(_g(g)['cv1_gap']['mean'])} / {de(_g(g)['cvk_gap']['mean'])}" for g in C.GAP_NAMES]
    has("; ".join(parts))
    n, m, f = (_g(g)["gap_lpr"]["mean"] for g in C.GAP_NAMES)
    assert n > f and m > f                                                              # Organ nah und mittel größer als fern
    has(f"Lücken sind dort {de(f)} % (Relaxation) und {de(_g('fern')['cv1_gap']['mean'])} % (CVaR(k+1)) in der ersten, "
        f"{de(_g('fern', True)['gap_lpr']['mean'])} % und {de(_g('fern', True)['cv1_gap']['mean'])} % in der zweiten Menge")
    # Reihenfolge nah > mittel > fern in beiden Mengen und in allen drei Spalten; nah und mittel bei der Relaxation nicht klar getrennt
    for second in (False, True):
        for key in ("gap_lpr", "cv1_gap", "cvk_gap"):
            v = [_g(g, second)[key]["mean"] for g in C.GAP_NAMES]
            assert v[0] > v[1] > v[2]
    nah, mit, fern = (_g(g)["gap_lpr"] for g in C.GAP_NAMES)
    assert nah["lo"] < mit["hi"] and mit["lo"] < nah["hi"]                              # Intervalle überlappen
    has(f"Intervalle [{de(nah['lo'])}; {de(nah['hi'])}] und [{de(mit['lo'])}; {de(mit['hi'])}] überlappen), fern liegt klar darunter [{de(fern['lo'])}; {de(fern['hi'])}]")
    assert fern["hi"] < mit["lo"]
    has("Die Reihenfolge nah vor mittel vor fern gilt in beiden Mengen")


def test_claim_gap_classes_are_named_by_the_drawn_minimum_not_the_actual_distance():
    t = D.table()["phantoms"]
    gv = {g: [p["gv"] for p in t[g]] for g in C.GAP_NAMES}
    lo, hi = {g: min(v) for g, v in gv.items()}, {g: max(v) for g, v in gv.items()}
    n_over = sum(1 for x in gv["nah"] if x > 1.5)
    has(f"in der ersten Menge liegt der kleinste Abstand bei nah zwischen {de(lo['nah'])} und {de(hi['nah'])} Voxelbreiten ({n_over} von 30 Phantomen über 1,5), "
        f"bei mittel zwischen {de(lo['mittel'])} und {de(hi['mittel'])}, bei fern zwischen {de(lo['fern'])} und {de(hi['fern'])}")
    assert hi["nah"] > C.GAP_RANGES["nah"][1] and lo["mittel"] > C.GAP_RANGES["nah"][1] and hi["nah"] >= lo["mittel"]       # die Klassen überlappen
    for g in C.GAP_NAMES:
        assert lo[g] >= C.GAP_RANGES[g][0] - 1e-9
    assert all("Mindestmaß" in C.GAP_LABELS[g] for g in C.GAP_NAMES)


def test_claim_cvar_k_infeasible_rows_and_gap_spread(s1, s2):
    """Hook: alle Unzulässigkeiten bei Organ nah, die meisten bei der strengen Kombination; Lücken über 5 %; Spitze der Kosten."""
    act = [r for r in D.rows(cap=C.ORGAN_CAP) if r.get("mip_status") == 0 and E.cost_rel(r["f_mip"], r["f_free"]) > C.ACTIVE_REL]
    inf = [r for r in act if r.get("f_cvk") is None]
    assert len(inf) == 13 and all(r["gap"] == "nah" for r in inf) and sum(1 for r in inf if r["ci"] == 2) == 11
    has(f"(alle {len(inf)} Fälle, {sum(1 for r in inf if r['ci'] == 2)} davon bei der strengen Grenze mit 10 %)")
    costs = sorted((100 * E.cost_rel(r["f_mip"], r["f_free"]) for r in act), reverse=True)
    assert sum(1 for c in costs if c > 5) == 17
    has(f"{sum(1 for c in costs if c > 5)} aktive Zeilen liegen über 5 %, die nächsten Werte sind {de(costs[1])} %, {de(costs[2])} % und {de(costs[3])} %")
    gl = [100 * E.gap_lower(r["f_mip"], r["f_lpr"]) for r in act]
    has(f"einzelne Zeilen liegen darüber: {sum(1 for g in gl if g > 5)} von {len(act)} über 5 %, {sum(1 for g in gl if g > 10)} über 10 %")
    # Mittel der Kosten über alle bewiesenen Zeilen
    proven = [r for r in D.rows(cap=C.ORGAN_CAP) if r.get("mip_status") == 0]
    all_mean = sum(100 * E.cost_rel(r["f_mip"], r["f_free"]) for r in proven) / len(proven)
    has(f"(über alle {len(proven)} Zeilen {de(all_mean)} %)")


def test_claim_by_combination():
    parts = []
    for ci, label in enumerate(("(0,5; 25 %)", "(0,3; 25 %)", "(0,3; 10 %)")):
        t = E.summary_of(C.ORGAN_CAP, None, ci)
        parts.append(f"{label}: aktiv in {t['n_active']} von {t['n_proven']}, Kosten Mittel {de(t['cost']['mean'])} %" if ci == 0 else f"{label}: {t['n_active']} von {t['n_proven']}, {de(t['cost']['mean'])} %")
    has("; ".join(parts))


def test_claim_cap_table():
    rows = E.cap_table()
    assert [r["cap"] for r in rows] == [1.2, 1.0, 0.8, 0.6]
    has("aktive Zeilen " + " / ".join(str(r["n_active"]) for r in rows))
    has("Median " + " / ".join(de(r["share"]["median"], 3) for r in rows))
    has("unter 0,01 in " + " / ".join(str(r["share"]["n_lt001"]) for r in rows))
    has("Lücke der Relaxation Mittel " + " / ".join(de(r["gap_lpr"]["mean"]) for r in rows) + " %")
    has("im Mittel " + " / ".join(de(r["allowed"]) for r in rows))
    assert {r["cv1_infeasible"] for r in rows} == {9}
    has("CVaR(k+1) unzulässig je 9")
    medians = [r["share"]["median"] for r in rows]
    assert medians == sorted(medians) and medians[0] == 0.0                              # der Anteil steigt mit kleinerem M
    has("nur knapp ein Sechstel")
    assert 0.16 <= medians[-1] < 1 / 6


def test_claim_strict_limit(s1, s2):
    has(f"unzulässig in {s1['n_strict_infeasible']} von {s1['n_active']} aktiven Zeilen (zweite Menge {s2['n_strict_infeasible']} von {s2['n_active']})")


def test_claim_nonconvexity(s1, s2):
    parts = ", ".join(f"{g} {_g(g)['wit_viol']} von {_g(g)['wit_n']}" for g in C.GAP_NAMES)
    has(f"**{s1['wit_viol']} von {s1['wit_n']}** aktiven Zeilen mit gebildetem Paar ({round(100 * s1['wit_viol'] / s1['wit_n'])} %; {parts})")
    has(f"zweite Menge {s2['wit_viol']} von {s2['wit_n']} ({round(100 * s2['wit_viol'] / s2['wit_n'])} %)")
    has(f"die Zahl {s1['wit_viol']} von {s1['wit_n']} ist an **konstruierten** Paaren gemessen")


def test_claim_cost_curve():
    cs = E.curve_summary()
    assert cs["n_curves"] == 30 and all(v == 0 for v in cs["viol"].values())
    has(f"{cs['steps']} Kurvenschritte, **0 Verletzungen** der Monotonie")
    has(f"k* gleich der Zahl der Organvoxel über d im Strahlplan ohne Grenze in {cs['kstar_eq']} von 30 Phantomen (höchstens diese Zahl in {cs['kstar_le']})")
    has(f"strenge Höchstdosis unzulässig in {cs['strict_infeasible']} von 30")


def test_claim_gaps_by_k():
    cs = E.curve_summary()
    b = cs["bins"]
    has(f"{cs['n_active_steps']} Kurvenschritte mit aktiver Grenze")
    has(f"bis 0,10: Lücke der Relaxation Mittel {de(b[0]['lpr_mean'])} % (Median {de(b[0]['lpr_median'])} %), Anteil der Kosten in der Relaxation Median {de(b[0]['share_median'], 3)}, "
        f"CVaR(k+1) unzulässig in {b[0]['cv1_inf']} von {b[0]['n']}, CVaR(k) in {b[0]['cvk_inf']}")
    for lab, x in (("über 0,10 bis 0,25", b[1]), ("über 0,25 bis 0,50", b[2]), ("über 0,50", b[3])):
        has(f"{lab}: {de(x['lpr_mean'])} % / {de(x['share_median'], 3)} / " + (f"{x['cv1_inf']} von {x['n']} / {x['cvk_inf']}" if lab.endswith("0,25") and "bis 0,25" in lab else f"{x['cv1_inf']} / {x['cvk_inf']}"))
    has(f"CVaR(k) steht in der Kurve ({cs['cvk_inf_total']} von {cs['n_active_steps']} unzulässig")
    lpr, cv1 = cs["biggest"]["lpr"], cs["biggest"]["cv1"]
    assert lpr[1][0] == "nah" and cv1[1][0] == "nah"
    has(f"Relaxation {de(lpr[0])} % (Organ nah, Phantom {lpr[1][1]}, k = {lpr[1][2]} von {lpr[1][3]} Organvoxeln), CVaR(k+1) {de(cv1[0])} % (Organ nah, Phantom {cv1[1][1]}, k = {cv1[1][2]} von {cv1[1][3]} Organvoxeln)")
    has(f"bei k = ¼ der Organvoxel im Median {de(cs['remaining'][0.25])}, bei k = ½ der Organvoxel {de(cs['remaining'][0.5], 3)}")
    assert b[0]["lpr_mean"] > b[3]["lpr_mean"]                                          # die Lücke ist bei strenger Grenze größer als bei großzügiger
    assert b[2]["cv1_inf"] == b[3]["cv1_inf"] == 0 and b[2]["cvk_inf"] == b[3]["cvk_inf"] == 0


def test_claim_hook_and_limits(s1, s2):
    has(f"in den aktiven Zeilen im Mittel {de(s1['cost']['mean'])} % (über alle Zeilen")
    has(f"in den zulässigen Zeilen im Mittel {de(s1['cvk_gap']['mean'])} % mehr")
    means = [s1["gap_lpr"]["mean"], s1["cv1_gap"]["mean"], s1["cvk_gap"]["mean"]]
    has(f"({de(min(means), 1)} bis {de(max(means), 1)} % im Mittel, Median")
    means_all = means + [s2["gap_lpr"]["mean"], s2["cv1_gap"]["mean"], s2["cvk_gap"]["mean"]]
    has(f"bei {de(min(means_all), 1)} bis {de(max(means_all), 1)} % (zwei Mengen von Zufalls-Seeds)")
    has(f"und kostet in den übrigen {s1['cvk_gap']['n']} im Mittel {de(s1['cvk_gap']['mean'])} % mehr als der exakte Wert")
    meds = [s1["gap_lpr"]["median"], s1["cv1_gap"]["median"], s1["cvk_gap"]["median"]]
    has(f"Median {de(min(meds), 1)} bis {de(max(meds), 1)} %")
    has(f"die Relaxation sieht in {s1['share']['n_gt05']} von {s1['n_active']} Zeilen mehr als die Hälfte der Kosten (zweite Menge {s2['share']['n_gt05']} von {s2['n_active']})")
    has(f"(Maximum {de(s1['cvk_gap']['max'])} % und {de(s1['cv1_gap']['max'])} %; in der zweiten Menge {de(s2['cvk_gap']['max'])} % und {de(s2['cv1_gap']['max'])} %")
    has(f"Kosten der Grenze (Mittel {de(s1['cost']['mean'])} %, Median {de(s1['cost']['median'])} %)")


def test_claim_corrections_section(s1, s2):
    pct = lambda s: round(100 * s["n_active"] / s["n_proven"])  # noqa: E731
    has(f"{s1['n_active']} von {s1['n_proven']} ({pct(s1)} %) und in der zweiten Menge {s2['n_active']} von {s2['n_proven']} ({pct(s2)} %)")
    has(f"Lücke der Relaxation liegt im Mittel bei {de(s1['gap_lpr']['mean'])} %")
    has(f"Maximum {de(s1['cost']['max'])} % statt 36,3 %")
    has(f"**CVaR(k) unzulässig** in {s1['cvk_infeasible']} statt 15 Zeilen, **CVaR(k+1)** in {s1['cv1_infeasible']} wie zuvor")
    has(f"{s1['wit_viol']} von {s1['wit_n']} ({round(100 * s1['wit_viol'] / s1['wit_n'])} %) statt 44 von 140 (31 %)")
    has(f"strenge Höchstdosis unzulässig in {s1['n_strict_infeasible']} von {s1['n_active']} statt 32 von 171")
    parts = [f"{g} {de(_g(g)['gap_lpr']['mean'])} %" for g in C.GAP_NAMES]
    has("Lücke der Relaxation " + ", ".join(parts).replace("mittel", "mittel"))
    cs = E.curve_summary()
    has(f"liegt die Lücke der Relaxation im Mittel bei {de(cs['bins'][0]['lpr_mean'])} % (erste Rechnung 4,32 %), über 0,50 bei {de(cs['bins'][3]['lpr_mean'])} % (0,69 %)")
    has(f"die größte Lücke ist {de(cs['biggest']['lpr'][0])} % statt 29,3 %")
    has(f"({cs['cv1_inf_total']} von {cs['n_active_steps']} Kurvenschritten mit aktiver Grenze)")
    top = max((r for r in D.rows(cap=C.ORGAN_CAP) if r.get("mip_status") == 0), key=lambda r: E.cost_rel(r["f_mip"], r["f_free"]))
    assert (top["gap"], top["i"], top["ci"]) == ("nah", 24, 1) and C.PRESETS["Höchster Preis der Grenze"]["seed"] == top["i"]
    has(f"Organ nah, Zufalls-Seed {top['i']}, d = 0,3, 25 %: {de(100 * E.cost_rel(top['f_mip'], top['f_free']))} %")


def test_claim_organ_sizes():
    sizes = [r["nO"] for t in (D.table(), D.seeds2()) for r in t["rows"]]
    has(f"die Organe der Tabellen haben {min(sizes)} bis {max(sizes)} Voxel, also {min(sizes)} bis {max(sizes)} Binärvariablen je MIP")
    assert 732 == int(E.M.body_mask().sum()) and 192 == C.N_VARS
