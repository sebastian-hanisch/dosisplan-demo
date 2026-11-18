"""Dosisplanung: Dosis-Volumen-Grenze - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anwendungsbeispiel, keine Beratung: erfundenes 2D-Phantom, vereinfachte Strahlenphysik, einheitenlose Dosis, kein klinischer Anspruch.
Eine Dosis-Volumen-Grenze ("höchstens k Organvoxel über der Grenzdosis d") macht die Strahlplanung zu einer ganzzahligen Aufgabe. Die Demo vergleicht
die exakte Lösung (vorgerechnet) mit der LP-Relaxation und der CVaR-Restriktion in zwei Formen.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import dpl_constants as C
import dpl_data as D
import dpl_hand as H
import dpl_solvers as S
import dpl_texts as T
from dpl_evaluation import (
    Settings,
    analyse,
    cap_table,
    cell_facts,
    curve_summary,
    dist_points,
    gap_table,
    h3_results,
    num,
    percent,
    problem_of,
    summary_of,
    verdict_cost,
    verdict_cvar,
    verdict_relax,
)
from dpl_layout import dunkles_design, kopf_css
from dpl_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from dpl_visualization import (
    PANEL_TITLES,
    PLAN_NAMES,
    build_cost_scatter,
    build_curve_plot,
    build_dose_panels,
    build_dvh,
    build_excess_bars,
    build_gap_bars,
    build_pareto,
)

st.set_page_config(page_title="Dosisplanung – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Zielkonflikt: 22 lineare Programme …")
def _pareto(gap, seed):
    return S.pareto_curve(problem_of(gap, seed))


@st.cache_data(show_spinner=False)
def _probe_plans(settings):
    """Strahlpläne a und b (je ein LP zur vorgerechneten Menge) und ihr Mittelpunkt für die Nichtkonvexitäts-Ansicht."""
    row = D.get_row(settings.gap, settings.seed, settings.combo, settings.cap)
    if row.get("Sb") is None:
        return None
    prob = problem_of(settings.gap, settings.seed)
    pa = prob.solve_set(row["d"], row["S"], row["M"])
    pb = prob.solve_set(row["d"], row["Sb"], row["M"])
    mid = 0.5 * (pa.x + pb.x)
    return dict(sorted={"a": sorted(prob.organ_dose(pa.x), reverse=True), "b": sorted(prob.organ_dose(pb.x), reverse=True), "mid": sorted(prob.organ_dose(mid), reverse=True)},
                over={"a": prob.over(pa.x, row["d"]), "b": prob.over(pb.x, row["d"]), "mid": prob.over(mid, row["d"])})


@st.cache_data(show_spinner=False)
def _h2():
    return H.solve_all()


@st.cache_data(show_spinner=False)
def _h3(d, k):
    r = h3_results(d, k)
    prob = r.pop("prob")
    return {n: (sol.status, sol.f, prob.over(sol.x, d) if sol.x is not None else None) for n, sol in r.items() if n not in ("d", "k")}


def _show(verdict):
    kind, text = verdict
    {"success": st.success, "warning": st.warning, "info": st.info}[kind](text)


def _fmt_f(x):
    return "unzulässig" if x is None else f"{x:.5f}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in dpl_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🩻 Dosisplanung: Dosis-Volumen-Grenze")
st.markdown(T.INTRO)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/dosisplan-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Mehrere Strahlen aus verschiedenen Richtungen treffen eine erfundene Scheibe. In ihr liegen ein Tumor, der eine bestimmte Dosis bekommen muss, und daneben ein Organ, das möglichst wenig abbekommen soll. Die Dosis-Volumen-Grenze erlaubt, dass nur eine begrenzte Zahl von Organstellen über einer Schwelle liegt. Diese Zählregel verändert die Aufgabe: Welche Stellen die Ausnahmen sind, muss man erst auswählen.

Die App vergleicht das exakte Ergebnis, das vorab gerechnet wurde, mit der Relaxation, die statt zu zählen einen Gesamtüberschuss erlaubt, und mit der CVaR-Restriktion, die mehr verlangt als nötig. Alles ist erfunden und stark vereinfacht: Anwendungsbeispiel, keine Beratung, kein klinischer Anspruch.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/dosisplan-erklaert.html)
        """
    )
# erklaerseite:einfach:end
st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Phantom**")
    gap = st.selectbox("Organabstand", list(C.GAP_NAMES), key="gap_select", format_func=lambda g: C.GAP_LABELS[g],
                       help="Kleinster Abstand der Voxelmitten von Tumor und Organ, in Voxelbreiten. Die Klassen sind nach dem gezogenen Mindestmaß benannt (nah 1,0 bis 1,5, mittel 2 bis 3, fern 4 bis 6); der tatsächliche Abstand liegt in der ersten Menge bei nah zwischen 1,41 und 2,24, bei mittel zwischen 2,24 und 3,61, bei fern zwischen 4,12 und 6,40.")
    seed = st.slider("Zufalls-Seed (Phantom 0 bis 29)", *bounds("seed_input"), key="seed_input", step=1,
                           help="Welches der 30 Phantome dieses Abstands gezeigt wird. Ein Phantom ist kein Befund: in den Kernfragen und in Experiment 2 stehen die Zahlen über alle 30 daneben.")
    st.button("🎲 Neues Phantom würfeln", width="stretch", on_click=randomize_seed)
    st.markdown("**Grenze und Höchstdosis**")
    combo = st.selectbox("Strenge der Grenze (d; Anteil)", list(range(len(C.COMBOS))), key="combo_select", format_func=lambda c: C.COMBO_LABELS[c],
                         help="Grenzdosis d und erlaubter Anteil der Organvoxel über d; k = Anteil mal Organvoxel, abgerundet, mindestens 1. Nur diese drei Kombinationen sind vorgerechnet.")
    cap = st.select_slider("Organ-Höchstdosis M", options=list(C.CAPS), key="cap_select", format_func=lambda m: f"{m:.1f}".replace(".", ","),
                           help="Größte Dosis, die ein Organvoxel bekommen darf (zugleich die Big-M-Konstante der Grenze). Nur diese vier Werte sind vorgerechnet.")

sync_query_params({"gap_select": gap, "seed_input": int(seed), "combo_select": int(combo), "cap_select": cap, "view_select": st.session_state.get("view_select", C.DEFAULT_VIEW)})

settings = Settings(gap, int(seed), int(combo), float(cap))
a = _analysis(settings)
facts = cell_facts(a)
dist = summary_of(settings.cap, settings.gap, settings.combo)

# --- Ergebnis: eine Grenze, drei Wege ----------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Eine Grenze, drei Wege")
st.caption(
    f"Phantom {settings.seed}, Organ {settings.gap}, {C.COMBO_LABELS[settings.combo]} (k = {a.k} von {a.n_organ} Organvoxeln), Höchstdosis M = {num(settings.cap)}. "
    "Zielgröße: mittlere Dosis außerhalb des Tumors (einheitenlos)."
)
if not facts["reachable"]:
    st.warning("Bei dieser Organ-Höchstdosis ist das Tumorfenster 1,0 bis 1,2 für dieses Phantom nicht erreichbar: es gibt keine Lösung.")
else:
    names = ["free", "exact", "lpr", "cvk", "cv1"]
    cols = st.columns(5)
    labels = {"free": "Ohne Grenze", "exact": "Exakt (vorgerechnet)", "lpr": "LP-Relaxation", "cvk": "CVaR(k), Literatur", "cv1": "CVaR(k+1), eigen"}
    for col, n in zip(cols, names):
        f = a.f.get(n)
        # Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse".
        if f is None:
            col.metric(labels[n], "unzulässig" if n != "exact" else ("unzulässig" if a.row.get("mip_status") == 2 else "nicht bewiesen"))
        elif n == "free":
            col.metric(labels[n], f"{f:.4f}".replace(".", ","), delta=f"{a.over['free']} Organvoxel über d", delta_color="off")
        else:
            over = a.over.get(n)
            col.metric(labels[n], f"{f:.4f}".replace(".", ","), delta=f"{(f - a.f['free']) / a.f['free'] * 100:+.2f} % gegen ohne Grenze".replace(".", ",") + (f", {over} über d" if over is not None else ""),
                       delta_color="inverse")
    st.plotly_chart(build_dose_panels(D.phantom(settings.gap, settings.seed), a.dose, PANEL_TITLES, ["free", "exact", "cvk", "cv1"]), width="stretch", key="dose_panels")
    st.caption("Dosis je Körpervoxel (gemeinsame Farbskala), roter Rand = Tumor, weißer Rand = Organ. Die Relaxation hat in etwas mehr als der Hälfte der aktiven Zeilen (M = 1,2) denselben Zielwert wie ohne Grenze; warum, zeigt Ansicht 3.")
    st.plotly_chart(build_dvh(a.organ_sorted, a.d, a.k, ["free", "exact", "cvk", "cv1"]), width="stretch", key="dvh_main")
    st.caption("Je Strahlplan die Organdosen von der höchsten abwärts: Punkte über der gestrichelten Marke liegen über d, links der senkrechten Marke steht die erlaubte Zahl k.")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Was kostet die Grenze, und was ändert sie am Strahlplan?**")
_show(verdict_cost(facts, dist))
st.markdown("**2. Was sieht die LP-Schranke davon?**")
_show(verdict_relax(facts, dist))
st.markdown("**3. Wie weit reicht die konvexe Näherung (CVaR-Restriktion)?**")
_show(verdict_cvar(facts, dist))

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Experimente")
st.caption("Der Umschalter wählt ein Experiment; nur das gewählte rechnet. Alle Verteilungen kommen aus der vorgerechneten Tabelle (90 Phantome × 3 Kombinationen × 4 Höchstdosen).")
view = st.segmented_control("Experiment", list(C.VIEWS), key="view_select", label_visibility="collapsed")
view = view or C.DEFAULT_VIEW

if view == C.VIEW_D1:
    st.subheader("🔬 0 Vorstufe: eine harte Höchstdosis für alle Organvoxel")
    st.markdown(
        "Die einfachste Organbedingung ist eine **harte Höchstdosis τ für jedes Organvoxel**. Das ist ein lineares Programm; der **Preis** von τ (die Abnahme der Zielgröße je Einheit τ) ist ein Schattenpreis. "
        "Er gilt nur auf einem Abschnitt der Kurve und fällt zum Knie hin auf null. Das ist LP-Lehrsatz und steht ausführlich in [lp-dualitaet-demo](https://github.com/sebastian-hanisch/lp-dualitaet-demo) und "
        "`zwei-kriterien-demo`; hier ist es der Anlass für die Dosis-Volumen-Grenze: Eine harte Höchstdosis für alle Organvoxel kann unzulässig sein."
    )
    curve = _pareto(settings.gap, settings.seed)
    if curve is None:
        st.warning("Für dieses Phantom ist das Tumorfenster nicht erreichbar.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Kleinste mögliche Höchstdosis τ_min", num(curve["tau_min"], 3))
        c2.metric("Knie (größte Organdosis ohne Organgrenze)", num(curve["knee"], 3))
        mid_price = curve["points"][len(curve["points"]) // 2][2]
        c3.metric("Preis in der Mitte der Kurve", num(mid_price, 4))
        st.plotly_chart(build_pareto(curve), width="stretch", key="pareto")
        st.caption(f"{len(curve['points'])} Punkte τ von τ_min bis zum Knie; Preis = −(Summe der Duale der Organzeilen). Ein Phantom.")
    s = summary_of(C.ORGAN_CAP)
    st.info(
        f"Zähleinheit Zeilen (Phantom × Kombination, M = 1,2): eine strenge Höchstdosis d für **alle** Organvoxel (k = 0) ist in {s['n_strict_infeasible']} von {s['n_active']} Zeilen mit aktiver Grenze unzulässig; "
        "die Dosis-Volumen-Grenze erlaubt dort wenigstens k Ausnahmen."
    )

elif view == C.VIEW_COST:
    st.subheader("🔬 2 Wie viel kostet die Grenze, und wann ist es fast nichts?")
    pts = dist_points(settings.gap, settings.combo, settings.cap)
    st.plotly_chart(build_cost_scatter(pts, settings.seed, f"30 Phantome, Organ {settings.gap}, {C.COMBO_LABELS[settings.combo]}, M = {num(settings.cap)}"), width="stretch", key="scatter")
    st.caption("Je Phantom ein Punkt (nur Zeilen mit bewiesenem exaktem Wert). Punkte bei x = 0: die Grenze kostet nichts, der Strahlplan ohne Grenze erfüllt sie schon.")
    rows_t = []
    for g in gap_table(settings.cap):
        rows_t.append({"Organabstand": g["gap"], "Zeilen (bewiesen)": f"{g['n_proven']} von {g['n']}", "Grenze aktiv": g["n_active"], "nicht aktiv, Strahlplan ohne Grenze erfüllt sie": g["n_inactive_free_ok"],
                       "Kosten Mittel (%)": round(g["cost"]["mean"], 2) if g["cost"]["mean"] is not None else None, "Kosten Median (%)": round(g["cost"]["median"], 2) if g["cost"]["median"] is not None else None,
                       "Lücke Relaxation Mittel (%)": round(g["gap_lpr"]["mean"], 2) if g["gap_lpr"]["mean"] is not None else None})
    st.dataframe(pd.DataFrame(rows_t), hide_index=True, width="stretch")
    st.caption(f"Über alle drei Kombinationen, Höchstdosis M = {num(settings.cap)}; Mittelwerte über aktive Zeilen. Zähleinheit Zeilen.")
    st.markdown("**Preis der Grenze und Lücke der Ersatzformen nie vermischen:** Der Preis ist (exakt − ohne Grenze) / ohne Grenze, die Lücke einer Ersatzform ist (Ersatz − exakt) / exakt.")

elif view == C.VIEW_RELAX:
    st.subheader("🔬 3 Warum sieht die LP-Schranke nichts?")
    st.markdown(
        "Die Relaxation erlaubt einen **Gesamtüberschuss** über d von k (M − d), nicht k Voxel. Liegt der Gesamtüberschuss des Strahlplans ohne Grenze darunter, ist dieser Strahlplan für die Relaxation zulässig, und ihr Wert ist der Wert ohne Grenze."
    )
    pts = [p for p in dist_points(settings.gap, settings.combo, settings.cap) if p["active"]]
    if pts:
        st.plotly_chart(build_excess_bars(pts, pts[0]["allowed"], settings.seed), width="stretch", key="excess")
    else:
        st.info("In dieser Auswahl ist keine Zeile aktiv; es gibt keinen Überschuss zu zeigen.")
    rows_t = []
    for r in cap_table():
        rows_t.append({"Höchstdosis M": num(r["cap"]), "aktive Zeilen": r["n_active"], "Anteil der Kosten in der Relaxation, Median": round(r["share"]["median"], 3) if r["share"]["median"] is not None else None,
                       "unter 0,01": f"{r['share']['n_lt001']} von {r['share']['n']}", "Relaxation = Wert ohne Grenze": r["n_lpr_eq_free"], "erlaubter Gesamtüberschuss k (M − d), Mittel": round(r["allowed"], 2),
                       "Lücke Relaxation Mittel (%)": round(r["gap_lpr"]["mean"], 2)})
    st.dataframe(pd.DataFrame(rows_t), hide_index=True, width="stretch")
    st.caption("Alle Abstände und Kombinationen, erste Menge von Zufalls-Seeds. Mit kleinerer Höchstdosis sieht die Relaxation mehr; die Zahlen stehen in der Tabelle, nicht in einer Regel.")

elif view == C.VIEW_CVAR:
    st.subheader("🔬 4 CVaR: Restriktion, kein Ersatz")
    pts = [p for p in dist_points(settings.gap, settings.combo, settings.cap) if p["active"]]
    if pts:
        st.plotly_chart(build_gap_bars(pts, settings.seed), width="stretch", key="cvar_bars")
        st.caption("Lücke der beiden CVaR-Formen zum exakten Wert je Phantom (Zeilen mit aktiver Grenze); ein Kreuz oben bedeutet: unzulässig, obwohl die Aufgabe lösbar ist.")
    else:
        st.info("In dieser Auswahl ist keine Zeile aktiv.")
    s_all = summary_of(settings.cap)
    s_two = summary_of(settings.cap, second=True)
    rows_t = [
        {"Form": "CVaR(k), Form der Literatur", "unzulässig (erste Menge)": f"{s_all['cvk_infeasible']} von {s_all['n_active']}", "unzulässig (zweite Menge)": f"{s_two['cvk_infeasible']} von {s_two['n_active']}",
         "Lücke Mittel (%)": round(s_all["cvk_gap"]["mean"], 2), "Lücke Median (%)": round(s_all["cvk_gap"]["median"], 2), "Organvoxel über d im Mittel": round(s_all["ex"]["cvk"], 2)},
        {"Form": "CVaR(k+1), eigene Variante", "unzulässig (erste Menge)": f"{s_all['cv1_infeasible']} von {s_all['n_active']}", "unzulässig (zweite Menge)": f"{s_two['cv1_infeasible']} von {s_two['n_active']}",
         "Lücke Mittel (%)": round(s_all["cv1_gap"]["mean"], 2), "Lücke Median (%)": round(s_all["cv1_gap"]["median"], 2), "Organvoxel über d im Mittel": round(s_all["ex"]["cv1"], 2)},
    ]
    st.dataframe(pd.DataFrame(rows_t), hide_index=True, width="stretch")
    st.caption(
        f"Aktive Zeilen, alle Abstände und Kombinationen, M = {num(settings.cap)}; Lücke über die Zeilen, in denen die Form zulässig ist. Erlaubt sind im Mittel k = {num(s_all['ex']['k'], 2)} Organvoxel über d (Zeilen, in denen beide Formen zulässig sind)."
    )
    st.markdown(
        "CVaR(k) garantiert höchstens k − 1 Organvoxel über d und verlangt damit mehr als nötig; CVaR(k+1) ist schwächer und hält k ein. Beide sind **obere Schranken**, keine Näherung von beiden Seiten."
    )

elif view == C.VIEW_NONCONVEX:
    st.subheader("🔬 5 Zwei zulässige Strahlpläne, ein unzulässiger Mittelpunkt")
    st.markdown(
        "Ein Strahlplan a (exakt) und ein Strahlplan b, der die Voxel nicht nutzen darf, die a über d bringt, sind beide zulässig. Ihr **Mittelpunkt** liegt in der konvexen Hülle, kann die Grenze aber verletzen: "
        "die zulässige Menge ist die Vereinigung von Teilgebieten, nicht konvex."
    )
    probe = _probe_plans(settings)
    if probe is None:
        st.info("Für diese Zeile gibt es kein Paar (kein exakter Wert, die Grenze ist nicht aktiv oder es gibt keinen zweiten Strahlplan).")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Strahlplan a: Organvoxel über d", probe["over"]["a"])
        m2.metric("Strahlplan b: Organvoxel über d", probe["over"]["b"])
        m3.metric("Mittelpunkt: Organvoxel über d", probe["over"]["mid"], delta=("Grenze verletzt" if probe["over"]["mid"] > a.k else "Grenze eingehalten"), delta_color="off")
        st.plotly_chart(build_dvh(probe["sorted"], a.d, a.k, ["a", "b", "mid"]), width="stretch", key="dvh_probe")
    rows_t = []
    for g in C.GAP_NAMES:
        s = summary_of(settings.cap, g)
        rows_t.append({"Organabstand": g, "Paare gebildet": s["wit_n"], "Mittelpunkt verletzt die Grenze": s["wit_viol"]})
    s = summary_of(settings.cap)
    rows_t.append({"Organabstand": "alle", "Paare gebildet": s["wit_n"], "Mittelpunkt verletzt die Grenze": s["wit_viol"]})
    st.dataframe(pd.DataFrame(rows_t), hide_index=True, width="stretch")
    st.warning(
        "**Die Paare sind konstruiert** (der zweite Strahlplan darf die Überschreitungsvoxel des ersten nicht nutzen). Die Zahl ist keine Häufigkeit bei typischen Strahlplänen; "
        "Craft (2013) argumentiert über typische Strahlpläne, und diese Probe stützt oder widerlegt das nicht."
    )

elif view == C.VIEW_CURVE:
    st.subheader("🔬 6 Wie viele Voxel darf es treffen? Die Kostenkurve über k")
    cur = D.curve(settings.gap, settings.seed)
    st.caption(f"Grenzdosis d = {num(C.CURVE_D)}, Höchstdosis M = 1,2, k von 1 bis zur Zahl der Organvoxel; vorgerechnet für die Phantome 0 bis {C.N_CURVE_PHANTOMS - 1} je Abstandsklasse.")
    if cur is None:
        st.info(f"Für den Zufalls-Seed {settings.seed} ist keine Kurve vorgerechnet (nur Seeds 0 bis {C.N_CURVE_PHANTOMS - 1}); die Tabelle unten gilt für alle {3 * C.N_CURVE_PHANTOMS} Kurven.")
    else:
        st.plotly_chart(build_curve_plot(cur, C.CURVE_D), width="stretch", key="curve")
        st.caption("Die Grenze kostet nur, solange k unter der Zahl der Organvoxel über d des Strahlplans ohne Grenze liegt; danach fallen der exakte Wert und die Relaxation auf null; die CVaR-Kurven bleiben in manchen Schritten darüber, weil sie strenger sind als nötig.")
    cs = curve_summary()
    rows_t = [{"k / Organvoxel": f"{b['lo']:.2f} bis {min(b['hi'], 1.0):.2f}".replace(".", ","), "Zeilen mit aktiver Grenze": b["n"], "Lücke Relaxation Mittel (%)": round(b["lpr_mean"], 2) if b["lpr_mean"] is not None else None,
               "Anteil der Kosten in der Relaxation, Median": round(b["share_median"], 3) if b["share_median"] is not None else None,
               "CVaR(k+1) unzulässig": b["cv1_inf"], "CVaR(k) unzulässig": b["cvk_inf"]} for b in cs["bins"]]
    st.dataframe(pd.DataFrame(rows_t), hide_index=True, width="stretch")
    st.caption(f"Über {cs['n_curves']} Kurven (d = 0,3): {cs['steps']} Kurvenschritte, Verletzungen der Monotonie in k (Wert fällt oder bleibt gleich, Toleranz 1e-5 relativ): "
               f"exakt {cs['viol']['f_mip']}, Relaxation {cs['viol']['f_lpr']}, CVaR(k+1) {cs['viol']['f_cv1']}.")

elif view == C.VIEW_HAND:
    st.subheader("🔬 7 Handinstanz in Brüchen")
    st.markdown(
        "Drei Strahlen, drei Organvoxel (O1 bis O3), ein Tumorvoxel T mit Dosis x₁ + x₂ + x₃ zwischen 1 und 6/5, Grenzdosis d = 1/2, Höchstdosis M = 1, höchstens k = 1 Organvoxel über d. "
        "Alle Werte sind exakte Brüche (Eckenaufzählung, kein Löser)."
    )
    h2 = _h2()
    f0, x0 = h2["free"]
    fe, xe, se = h2["exact"]
    fr, xr = h2["relax"]
    fc1, xc1 = h2["cvar"][2]
    fck, _ = h2["cvar"][1]
    xa, xb, mid = h2["plans"]
    rows_t = [
        {"Weg": "Ohne Grenze", "Wert": H.fmt(f0), "Strahlgewichte x": ", ".join(H.fmt(v) for v in x0), "Organvoxel über d": H.over_count(x0)},
        {"Weg": "Exakt (Aufzählung der Mengen S)", "Wert": H.fmt(fe), "Strahlgewichte x": ", ".join(H.fmt(v) for v in xe), "Organvoxel über d": H.over_count(xe)},
        {"Weg": "LP-Relaxation (Überschuss-Form)", "Wert": H.fmt(fr), "Strahlgewichte x": ", ".join(H.fmt(v) for v in xr), "Organvoxel über d": H.over_count(xr)},
        {"Weg": "CVaR(k+1) = CVaR(2), eigene Variante", "Wert": H.fmt(fc1), "Strahlgewichte x": ", ".join(H.fmt(v) for v in xc1), "Organvoxel über d": H.over_count(xc1)},
        {"Weg": "CVaR(k) = CVaR(1), Form der Literatur", "Wert": "unzulässig" if fck is None else H.fmt(fck), "Strahlgewichte x": "–", "Organvoxel über d": None},
    ]
    st.dataframe(pd.DataFrame(rows_t), hide_index=True, width="stretch")
    st.markdown(
        f"Der Strahlplan ohne Grenze hat den Gesamtüberschuss {H.fmt(h2['excess_free'])} = k (M − d), liegt also genau auf der Grenze der Relaxation: sie sieht nichts. Die Mittelpunkte: a = ({', '.join(H.fmt(v) for v in xa)}), "
        f"b = ({', '.join(H.fmt(v) for v in xb)}), Mitte ({', '.join(H.fmt(v) for v in mid)}) mit {H.over_count(mid)} Organvoxeln über d."
    )
    st.markdown("**Dasselbe auf einem 6 × 6-Gitter** (drei Winkel, sechs Strahlen je Winkel, 24 Körpervoxel, Organ aus fünf Voxeln); LP-Teile live, der exakte Wert durch Aufzählung aller Mengen (6 LPs bei k = 1, 16 bei k = 2):")
    hd, hk = st.columns(2)
    d3 = hd.selectbox("Grenzdosis d", [0.3, 0.5], key="h3_d", format_func=lambda v: num(v))
    k3 = hk.selectbox("Erlaubte Zahl k", [1, 2], key="h3_k")
    r3 = _h3(d3, k3)
    rows3 = [{"Weg": PLAN_NAMES[n] if n in PLAN_NAMES else n, "Wert": _fmt_f(r3[n][1]), "Organvoxel über d": r3[n][2]} for n in ("free", "exact", "lpr", "cvk", "cv1")]
    st.dataframe(pd.DataFrame(rows3), hide_index=True, width="stretch")

st.markdown("---")

# --- Expander ------------------------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
    t_exact, t_relax, t_cvk, t_cv1, t_hand = st.tabs(["Exakt (MIP)", "LP-Relaxation", "CVaR(k) (Literaturform)", "CVaR(k+1) (eigene Variante)", "Von Hand (Handinstanz)"])
    with t_exact:
        st.markdown(T.METHOD_EXACT)
        status = {0: "bewiesen optimal", 1: "nicht bewiesen (Zeitlimit)", 2: "unzulässig", 3: "Fehler"}.get(a.row.get("mip_status"), "–")
        st.markdown(f"Gewählte Zeile: Status {status}, Menge S mit {len(a.row.get('S') or [])} Voxeln, Zielwert {_fmt_f(a.f.get('exact'))}.")
    with t_relax:
        st.markdown(T.METHOD_RELAX)
        st.markdown(f"Gewählte Zeile: Zielwert {_fmt_f(a.f.get('lpr'))}; Gesamtüberschuss des Strahlplans ohne Grenze {num(a.excess_free, 3)}, erlaubt k (M − d) = {num(a.allowed_excess, 3)}.")
    with t_cvk:
        st.markdown(T.METHOD_CVAR_K)
        st.markdown(f"Gewählte Zeile: Zielwert {_fmt_f(a.f.get('cvk'))}, Organvoxel über d: {a.over.get('cvk', '–')}.")
    with t_cv1:
        st.markdown(T.METHOD_CVAR_K1)
        st.markdown(f"Gewählte Zeile: Zielwert {_fmt_f(a.f.get('cv1'))}, Organvoxel über d: {a.over.get('cv1', '–')}.")
    with t_hand:
        st.markdown(T.METHOD_HAND)

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/dosisplan-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind die Dosismatrix $A$ (Voxel × Strahlen), die Tumor- und Organvoxel, das Tumorfenster, die Organ-Höchstdosis $M$, die Grenzdosis $d$ und die erlaubte Zahl $k$. Gesucht sind Strahlgewichte $x \ge 0$ und die Menge der Organvoxel, die über $d$ liegen dürfen. Die drei Zeilen sind das exakte Modell: Zielgröße, Tumorfenster und Dosis-Volumen-Grenze.

| Symbol | Bedeutung |
|---|---|
| $x_j \ge 0$ | Gewicht des Strahls $j$ (192 Strahlen) |
| $D = A x$ | Dosis je Voxel (einheitenlos, relativ zur Tumor-Solldosis 1) |
| $T$, $O$, $N$ | Menge der Tumor-, Organ- und gesunden Voxel; „Nicht-Tumor“ ist $O \cup N$ |
| $c_j$ | mittlere Dosis, die der Strahl $j$ je Einheit Gewicht in allen Nicht-Tumor-Voxeln erzeugt |
| $M$ | Organ-Höchstdosis (1,2, 1,0, 0,8 oder 0,6), zugleich Big-M |
| $d$, $k$ | Grenzdosis und erlaubte Zahl von Organvoxeln über $d$ |
| $z_v$ | 1, wenn das Organvoxel $v$ über $d$ liegen darf |
| $\zeta$, $w_v$ | Hilfsgrößen der CVaR-Restriktion (Rockafellar und Uryasev) |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/dosisplan-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    for line in T.MATH_MODEL:
        st.latex(line)
    st.markdown(T.MATH_NOTE)
    st.markdown("**Annahmen im Überblick**")
    st.markdown(T.ASSUMPTIONS_MARKDOWN)

with st.expander("🚧 Wo die Annahmen enden"):
    st.markdown(T.BOUNDARIES)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
