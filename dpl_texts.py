"""Lange Texte der App (Annahmen, Methoden-Tabs, Formelblock, Grenzen). Dieselben Absätze stehen in der README; tests/test_readme.py vergleicht sie."""

HINT = "**Anwendungsbeispiel, keine Beratung.**"

HINT_LONG = (
    "Erfundenes Phantom, vereinfachte Strahlenphysik (Lehrbuchannahmen), Dosis einheitenlos (relativ zur Tumor-Solldosis 1; die Literatur rechnet in Gray und Volumenanteilen). "
    "Kein klinischer Anspruch: nichts hier ist eine Behandlungs-, Diagnose- oder Planungsempfehlung."
)

INTRO = f"""
{HINT} {HINT_LONG}

In einem erfundenen 2D-Phantom (Gitter 32 × 32, ein Tumor, daneben ein Organ, 12 Einstrahlwinkel × 16 Strahlen) wird die **mittlere Dosis außerhalb des Tumors** minimiert, während der Tumor eine Dosis zwischen 1,0 und 1,2 bekommt.
Eine **Dosis-Volumen-Grenze** („höchstens k Organvoxel über der Grenzdosis d“) macht daraus eine ganzzahlige Aufgabe. Die Demo vergleicht den **exakten Strahlplan** (vorgerechnet) mit der **LP-Relaxation** und der **CVaR-Restriktion** in zwei Formen:
Was kostet die Grenze, was ändert sie am Strahlplan, und was sehen die Ersatzformen davon?

Weiter unten: der **Strahlplan** der gewählten Zelle, drei **Kernfragen**, sieben **Experimente** (Umschalter) und die Expander **🔧 Wie wir das erreichen – vollständiger Methodenvergleich**, **📐 Mathematische Formulierung** und **🚧 Wo die Annahmen enden**. Wer es ohne Fachwörter lesen will, öffnet **💬 In einfachen Worten**.
Gerechnet wird live nur mit linearen Programmen; die ganzzahlige Aufgabe ist **vorgerechnet** und eingefroren.
"""

ASSUMPTIONS_MARKDOWN = """
| Annahme | Wer setzt an |
|---|---|
| Körper = Scheibe auf einem 32 × 32-Gitter (732 Körpervoxel), ein Tumor, ein Organ, Rest gesundes Gewebe; Phantome zufällig erzeugt (Zufalls-Seed), Organabstand als Klasse nah, mittel, fern | Modell, Regler |
| 12 Einstrahlwinkel (30°-Schritte) × 16 parallele Strahlen = 192 Strahlgewichte x ≥ 0; Dosis D = A x mit Dämpfung exp(−μ · Tiefe), μ = 1, und Gauß-Streukern (0,6 Strahlbreiten, bei 3 σ abgeschnitten) | Modell (Lehrbuchannahme, fest) |
| Dosis einheitenlos, relativ zur Tumor-Solldosis 1; Tumorfenster 1,0 bis 1,2 | Modell (Vereinfachung der Demo) |
| Zielgröße: mittlere Dosis aller Nicht-Tumor-Voxel (Organ und gesundes Gewebe) | Modell (fest; andere Zielgrößen ungemessen) |
| Organ-Höchstdosis M (zugleich Big-M der Grenze): 1,2, 1,0, 0,8 oder 0,6 | Regler (nur diese vier gemessen) |
| Dosis-Volumen-Grenze: höchstens k Organvoxel über der Grenzdosis d, k = ⌊Anteil · Organvoxel⌋ (mindestens 1); drei Kombinationen (d; Anteil) | Regler (nur diese drei gemessen) |
| Der exakte Strahlplan ist vorgerechnet (MIP, Zeitlimit 30 s je Zeile); die App rechnet nur lineare Programme live | Modell |
| Ein Tumor, ein Organ, keine Bewegung, keine Unsicherheit der Dosisberechnung, keine Strahlenphysik jenseits des Spielzeug-Kerns | Nicht umgesetzt |
"""

METHOD_EXACT = """
**Exakt (MIP).** Je Organvoxel $v$ eine Binärvariable $z_v$ (1 = darf über $d$ liegen):

$$D_v \\le d + (M - d)\\, z_v,\\qquad z_v \\in \\{0,1\\},\\qquad \\sum_v z_v \\le k$$

Die Menge $S$ der Voxel mit $z_v = 1$ ist die ganze ganzzahlige Entscheidung; zu fester Menge $S$ ist der Strahlplan ein lineares Programm. Die Demo friert für jede Zeile die Menge $S$ und den Wert des zugehörigen LP ein
(**vorgerechnet**, `tools/dpl_make_table.py`, HiGHS, Zeitlimit 30 s, relative Lücke 1e-6; Zeilen ohne Beweis stünden als „nicht bewiesen“ in der Tabelle) und löst in der App nur dieses LP. Der exakte Wert ist ein Wert der Tabelle, kein Live-Ergebnis.
"""

METHOD_RELAX = """
**LP-Relaxation.** Dieselbe Big-M-Form mit $z_v \\in [0, 1]$ statt $\\{0,1\\}$; sie liefert eine **untere Schranke** des exakten Werts. Eliminiert man $z$, ist sie zusammen mit $D_v \\le M$ (für jedes Organvoxel $v$) gleichwertig zu

$$\\sum_v \\max(0,\\; D_v - d) \\le k\\,(M - d)$$

Die Relaxation erlaubt also nicht „$k$ Voxel über $d$“, sondern einen **Gesamtüberschuss** von $k\\,(M-d)$, und viele kleine Überschreitungen passen darunter. Die Gleichwertigkeit gilt nur mit $D_v \\le M$ (die Big-M-Form erzwingt es über $z_v \\le 1$); ohne diese Nebenbedingung war der Wert bei $M = 0{,}6$ in 5 von 19 geprüften Zeilen niedriger, bis zu 0,39 % (eine Handmessung beim Prüfen, nicht im Test). Mit ihr ist es eine Identität (im Test an 20 Zeilen und auf dem Mini-Gitter geprüft), keine Messung.
Dass sie die Kosten der Grenze meist kaum sieht, ist dagegen ein Befund über dieses Phantom (Experiment 3).
"""

METHOD_CVAR_K = """
**CVaR(k), Form der Literatur.** Das Mittel der $k$ größten Organdosen darf höchstens $d$ sein (Rockafellar und Uryasev; so beschrieben bei Kishimoto und Yamashita 2018 und bei Engberg u. a. 2017):

$$\\zeta + \\frac{1}{k}\\sum_v w_v \\le d,\\qquad w_v \\ge D_v - \\zeta,\\quad w_v \\ge 0$$

Jede zulässige Lösung erfüllt die Dosis-Volumen-Grenze, die Umkehrung gilt nicht. Der Strahlplan hat höchstens $k - 1$ Organvoxel über $d$ (im Test geprüft), die Restriktion verlangt also mehr als nötig und ist eine **obere Schranke**; sie kann unzulässig sein, obwohl die Aufgabe lösbar ist.
"""

METHOD_CVAR_K1 = """
**CVaR(k+1), eigene Variante dieser Demo.** Dieselbe Restriktion mit dem Mittel der $k + 1$ größten Organdosen (also $\\frac{1}{k+1}$ statt $\\frac{1}{k}$). Auch sie garantiert höchstens $k$ Organvoxel über $d$ und ist **schwächer** als CVaR(k):
jede Lösung der Form mit $k$ erfüllt sie. Diese Variante ist in keiner gelesenen Quelle zu finden; sie ist eine eigene Ableitung und wird hier neben der Form der Literatur gezeigt, damit der Unterschied sichtbar bleibt.
"""

METHOD_HAND = """
**Von Hand.** Die Handinstanz (Ansicht 7) hat drei Strahlen und drei Organvoxel und rechnet alles in Brüchen: Strahlplan ohne Grenze, die Aufzählung der Mengen $S$, die Relaxation in der Überschuss-Form und beide CVaR-Formen.
Dort hat die Relaxation denselben Wert wie der Strahlplan ohne Grenze (sie sieht nichts), CVaR(k) ist unzulässig, CVaR(k+1) kostet mehr als der exakte Strahlplan, und der Mittelpunkt zweier zulässiger Strahlpläne verletzt die Grenze.
"""

MATH_MODEL = [
    r"\min_{x \ge 0,\, z}\; c^{\top} x \qquad c_j = \frac{1}{|O \cup N|}\sum_{v \in O \cup N} A_{vj}",
    r"1 \le (A x)_v \le 1{,}2 \quad (v \in T)",
    r"(A x)_v \le d + (M - d)\, z_v \quad (v \in O), \qquad z_v \in \{0, 1\}, \qquad \sum_{v \in O} z_v \le k",
]

MATH_NOTE = (
    "Die Modellzeilen sind vollständig: Variablenbereiche, Tumorfenster, die Organ-Höchstdosis $M$ steckt in der Big-M-Konstante ($z_v = 1$ erlaubt bis $M$, wegen $z_v \\le 1$), und die Zielgröße ist die mittlere Dosis aller Nicht-Tumor-Voxel. "
    "Relaxation und CVaR-Restriktionen stehen im Expander oben; für $M \\le d$ gibt es keine Grenze (die Demo wählt $d \\le 0{,}5 < M$)."
)

BOUNDARIES = """
- **Erfundenes Modell.** Phantom, Strahlenkern und Dosis sind Lehrbuchannahmen; alle Zahlen gelten für diese Phantomfamilie (32 × 32, ein Tumor, ein Organ), diese Zielgröße (mittlere Dosis außerhalb des Tumors), die Höchstdosis 1,2 (Experiment 3 variiert sie), drei Kombinationen (d; Anteil) und die gemessenen Zufalls-Seeds. Keine Aussage über echte Strahlpläne.
- **Zähleinheit** der Verteilungen: Zeilen = Phantom × Kombination (d; Anteil), Mittelwerte mit Bootstrap über Phantome (Zeilen desselben Phantoms sind abhängig). „Aktiv“ heißt: die Grenze kostet mehr als 1e-5 relativ.
- **Nichtkonvexität:** Die Zahl der Paare, deren Mittelpunkt die Grenze verletzt, ist an **konstruierten** Paaren gemessen (der zweite Strahlplan darf die Überschreitungsvoxel des ersten nicht nutzen). Sie ist keine Häufigkeit bei typischen Strahlplänen. Craft (2013) argumentiert über typische Strahlpläne, ohne dass hier etwas dafür oder dagegen gemessen wäre.
- **Ungemessen:** andere Zielgrößen; Gitter über 32 × 32 und mehr als 12 Winkel; größere Organe (die Organe der Tabellen haben 14 bis 40 Voxel, also 14 bis 40 Binärvariablen je MIP); schärfere Relaxationen (starke gültige Ungleichungen, Tuncel u. a. 2012, nur Abstract gesehen: die Relaxation der einfachen Form kann dort beliebig schwach sein);
  Verfeinerungen der CVaR-Restriktion (Ausreißer-Ausschluss bei Kishimoto und Yamashita, Zwei-Pass bei Fu u. a.); andere Schwanzgrößen als k und k + 1; mehr als drei Kombinationen (d; Anteil); ein Tumor, ein Organ; μ und Streukern fest. Eine „bessere Schranke“ ist damit **nicht ausgeschlossen**.
- **Auswahl der Einstrahlwinkel** ist eine eigene Aufgabe (Auswahl aus Kandidaten) und steht nicht in dieser Demo: [aufloesung-messorte-demo](https://github.com/sebastian-hanisch/aufloesung-messorte-demo) zeigt sie an anderem Vehikel. Ein zusätzlicher Winkel verschlechtert den Wert nie (im Test an 24 Proben geprüft).
- **Zielkonflikt und Schattenpreis** (Ansicht 0) sind Lehrsatz der LP-Theorie und stehen ausführlich in [lp-dualitaet-demo](https://github.com/sebastian-hanisch/lp-dualitaet-demo) und `zwei-kriterien-demo`; `hart-gegen-weich-demo` zeigt harte gegen weiche Grenzen.
- **Verwandt, aber anders:** [scenario-approach-demo](https://github.com/sebastian-hanisch/scenario-approach-demo) setzt das Verwerfen von Szenarien, ein ganzzahliges Problem, bewusst nicht um; die Dosis-Volumen-Grenze entspricht dieser Aufgabe mit Voxeln als Szenarien (Deutung dieser Demo). [cvar-demo](https://github.com/sebastian-hanisch/cvar-demo) nutzt CVaR als Ziel, hier ist es eine Restriktion.
- **Literatur.** Die Näherung der Dosis-Volumen-Grenze durch das Mittel der heißesten Voxel (CVaR, Rockafellar und Uryasev 2000, 2002) wird seit Romeijn u. a. (2003, 2006) für die Fluenzoptimierung genannt (nach Kishimoto und Yamashita 2018; Engberg u. a. 2017; die Texte von Romeijn u. a. sind hier nicht gelesen); die Variante mit k + 1 heißesten Voxeln ist hier eine eigene Ableitung. Für die Größe der Lücken gibt es in den gelesenen Quellen keine Vergleichszahlen; die Zahlen sind eigene Messungen.
"""
