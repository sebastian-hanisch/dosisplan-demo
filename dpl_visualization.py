"""Plotly-Abbildungen (alle Achsen fest, kein Zoomen: Touch-Scrollen soll nicht am Diagramm hängen bleiben)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dpl_constants as C
import dpl_model as M

PLAN_COLORS = {"free": C.COLOR_FREE, "exact": C.COLOR_EXACT, "lpr": C.COLOR_LPR, "cvk": C.COLOR_CVK, "cv1": C.COLOR_CV1,
               "a": C.COLOR_EXACT, "b": C.COLOR_FREE, "mid": C.COLOR_CVK}
PANEL_TITLES = {"free": "Ohne Grenze", "exact": "Exakt", "lpr": "Relaxation", "cvk": "CVaR(k)", "cv1": "CVaR(k+1)"}
PLAN_NAMES = {"free": "Strahlplan ohne Grenze", "exact": "Exakt (MIP)", "lpr": "LP-Relaxation", "cvk": "CVaR(k), Form der Literatur", "cv1": "CVaR(k+1), eigene Variante",
              "a": "Strahlplan a (exakt)", "b": "Strahlplan b (ohne die Überschreitungen von a)", "mid": "Mittelpunkt von a und b"}


def _fixed(fig, height=360, **layout):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_dose_panels(phantom: M.Phantom, doses: dict, titles: dict, key_order, vmax: float = C.TUMOR_MAX) -> go.Figure:
    """Dosisbilder nebeneinander (gemeinsame Farbskala 0 bis 1,2); roter Rand = Tumor, weißer Rand = Organ."""
    names = [n for n in key_order if n in doses]
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=[titles[n] for n in names], horizontal_spacing=0.03)
    tum = np.nan_to_num(M.body_to_grid(np.isin(np.arange(len(next(iter(doses.values())))), list(phantom.tumor)).astype(float)))
    org = np.nan_to_num(M.body_to_grid(np.isin(np.arange(len(next(iter(doses.values())))), list(phantom.organ)).astype(float)))
    for j, n in enumerate(names, start=1):
        fig.add_trace(go.Heatmap(z=np.round(M.body_to_grid(doses[n]), 4), zmin=0.0, zmax=vmax, colorscale="Viridis", showscale=(j == len(names)),
                                 colorbar=dict(title="Dosis", thickness=10, len=0.8), hovertemplate="Dosis %{z:.3f}<extra></extra>"), row=1, col=j)
        for mask, color in ((tum, C.COLOR_TUMOR), (org, "#ffffff")):
            fig.add_trace(go.Contour(z=mask, contours=dict(start=0.5, end=0.5, size=1, coloring="lines"), line=dict(color=color, width=2),
                                     showscale=False, hoverinfo="skip"), row=1, col=j)
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{'' if j == 1 else j}", row=1, col=j, showticklabels=False)
        fig.update_xaxes(showticklabels=False, row=1, col=j)
    return _fixed(fig, height=300)


def build_dvh(sorted_doses: dict, d: float, k: int, order, title: str = "Organdosen, absteigend sortiert") -> go.Figure:
    """Je Strahlplan die Organdosen der Voxel von der höchsten bis zur niedrigsten; Marke bei d, senkrechte Marke bei k."""
    fig = go.Figure()
    for n in order:
        if n not in sorted_doses:
            continue
        y = sorted_doses[n]
        fig.add_trace(go.Scatter(x=np.arange(1, len(y) + 1), y=np.round(y, 4), mode="lines+markers", name=PLAN_NAMES[n],
                                 line=dict(color=PLAN_COLORS[n], width=2), marker=dict(size=4)))
    n_max = max((len(v) for v in sorted_doses.values()), default=1)
    fig.add_hline(y=d, line_dash="dash", line_color="gray", annotation_text=f"Grenzdosis d = {d:.1f}".replace(".", ","), annotation_position="top right")
    fig.add_vline(x=k + 0.5, line_dash="dot", line_color="gray", annotation_text=f"erlaubt: {k}", annotation_position="top")
    fig.update_xaxes(title="Organvoxel (Rang nach Dosis)", range=[0.5, n_max + 0.5])
    fig.update_yaxes(title="Dosis", range=[0, C.ORGAN_CAP + 0.05])
    return _fixed(fig, title=title, legend=dict(orientation="h", y=-0.25))


def build_cost_scatter(points: list[dict], selected: int | None, title: str) -> go.Figure:
    """Je Phantom ein Punkt: x = Kosten der Grenze (% der Zielgröße), y = Lücke der LP-Relaxation (%); Zeilen ohne aktive Grenze liegen bei x = 0."""
    fig = go.Figure()
    for active, color, name in ((True, C.COLOR_EXACT, "Grenze aktiv"), (False, C.COLOR_GREY, "Grenze nicht aktiv")):
        pts = [p for p in points if p["active"] == active]
        fig.add_trace(go.Scatter(x=[round(p["cost"], 4) for p in pts], y=[round(p["gap_lpr"], 4) for p in pts], mode="markers", name=name,
                                 marker=dict(color=color, size=9), text=[f"Phantom {p['i']}" for p in pts], hovertemplate="%{text}<br>Kosten %{x:.2f} %<br>Lücke %{y:.2f} %<extra></extra>"))
    sel = [p for p in points if p["i"] == selected]
    if sel:
        fig.add_trace(go.Scatter(x=[sel[0]["cost"]], y=[sel[0]["gap_lpr"]], mode="markers", name="gewähltes Phantom",
                                 marker=dict(color="rgba(0,0,0,0)", size=17, line=dict(color=C.COLOR_TUMOR, width=3))))
    fig.update_xaxes(title="Kosten der Grenze (% der Zielgröße)", rangemode="tozero")
    fig.update_yaxes(title="Lücke der LP-Relaxation (%)", rangemode="tozero")
    return _fixed(fig, title=title, legend=dict(orientation="h", y=-0.3))


def build_excess_bars(points: list[dict], allowed: float, selected: int | None) -> go.Figure:
    """Je aktiver Zeile der Gesamtüberschuss des Strahlplans ohne Grenze gegen den von der Relaxation erlaubten Überschuss k (M - d)."""
    fig = go.Figure()
    xs = [f"{p['i']}" for p in points]
    colors = [C.COLOR_TUMOR if p["i"] == selected else C.COLOR_FREE for p in points]
    fig.add_trace(go.Bar(x=xs, y=[round(p["exc"], 4) for p in points], marker_color=colors, name="Gesamtüberschuss des Strahlplans ohne Grenze"))
    fig.add_trace(go.Scatter(x=xs, y=[round(p["allowed"], 4) for p in points], mode="markers", marker=dict(symbol="line-ew", size=14, line=dict(width=3, color="black")),
                             name="von der Relaxation erlaubt: k (M − d)"))
    fig.update_xaxes(title="Phantom (nur Zeilen mit aktiver Grenze)", type="category")
    fig.update_yaxes(title="Summe der Organdosen über d")
    return _fixed(fig, legend=dict(orientation="h", y=-0.35))


def build_gap_bars(points: list[dict], selected: int | None) -> go.Figure:
    """Lücke der beiden CVaR-Formen gegen den exakten Wert je Phantom; unzulässige Zeilen als Kreuz am oberen Rand."""
    fig = go.Figure()
    for name, color, key in (("CVaR(k), Form der Literatur", C.COLOR_CVK, "cvk"), ("CVaR(k+1), eigene Variante", C.COLOR_CV1, "cv1")):
        feas = [p for p in points if p[key] is not None]
        fig.add_trace(go.Bar(x=[str(p["i"]) for p in feas], y=[round(p[key], 4) for p in feas], name=name, marker_color=color))
    top = max([p[k] for p in points for k in ("cvk", "cv1") if p[k] is not None] + [1.0]) * 1.1
    for name, color, key in (("CVaR(k) unzulässig", C.COLOR_CVK, "cvk"), ("CVaR(k+1) unzulässig", C.COLOR_CV1, "cv1")):
        inf = [p for p in points if p[key] is None]
        if inf:
            fig.add_trace(go.Scatter(x=[str(p["i"]) for p in inf], y=[top if key == "cvk" else top * 0.9] * len(inf), mode="markers", name=name,
                                     marker=dict(symbol="x", size=11, color=color, line=dict(width=2))))
    fig.update_xaxes(title="Phantom (nur Zeilen mit aktiver Grenze)", type="category")
    fig.update_yaxes(title="Lücke zum exakten Wert (%)")
    return _fixed(fig, legend=dict(orientation="h", y=-0.35), barmode="group")


def build_curve_plot(curve: dict, d: float) -> go.Figure:
    """Kosten der Grenze (% der Zielgröße des Strahlplans ohne Grenze) über k für den exakten Wert, die Relaxation und beide CVaR-Formen eines Phantoms."""
    fig = go.Figure()
    ff = curve["f_free"]
    for name, key, color in (("Exakt (MIP)", "f_mip", C.COLOR_EXACT), ("LP-Relaxation", "f_lpr", C.COLOR_LPR), ("CVaR(k+1), eigene Variante", "f_cv1", C.COLOR_CV1),
                             ("CVaR(k), Form der Literatur", "f_cvk", C.COLOR_CVK)):
        ks = [k for k, v in zip(curve["k"], curve[key]) if v is not None]
        fig.add_trace(go.Scatter(x=ks, y=[round(100 * (v - ff) / ff, 4) for v in curve[key] if v is not None], mode="lines+markers", name=name, line=dict(color=color, width=2),
                                 marker=dict(size=4)))
    fig.add_vline(x=curve["ex_free"] + 0.5, line_dash="dot", line_color="gray", annotation_text=f"Strahlplan ohne Grenze: {curve['ex_free']} über d", annotation_position="top right")
    fig.update_xaxes(title=f"erlaubte Zahl k von Organvoxeln über d = {d:.1f}".replace(".", ","), range=[0.5, curve["nO"] + 0.5])
    fig.update_yaxes(title="Mehrkosten gegen den Strahlplan ohne Grenze (%)", rangemode="tozero")
    return _fixed(fig, legend=dict(orientation="h", y=-0.3))


def build_pareto(curve: dict) -> go.Figure:
    """Zielgröße f(tau) und Preis (Abnahme je Einheit tau) über der Organ-Höchstdosis tau."""
    pts = curve["points"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=[round(p[0], 4) for p in pts], y=[round(p[1], 5) for p in pts], mode="lines+markers", name="Zielgröße f(τ)", line=dict(color=C.COLOR_EXACT)), secondary_y=False)
    fig.add_trace(go.Scatter(x=[round(p[0], 4) for p in pts], y=[round(p[2], 5) for p in pts], mode="lines+markers", name="Preis (Abnahme je Einheit τ)", line=dict(color=C.COLOR_CVK, dash="dot")),
                  secondary_y=True)
    fig.update_xaxes(title="Organ-Höchstdosis τ (alle Organvoxel)")
    fig.update_yaxes(title="mittlere Dosis außerhalb des Tumors", secondary_y=False)
    fig.update_yaxes(title="Preis", secondary_y=True, rangemode="tozero")
    return _fixed(fig, legend=dict(orientation="h", y=-0.3))
