# Dosisplanung: Dosis-Volumen-Grenze – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-dosisplan-demo.streamlit.app/)** -->

**Anwendungsbeispiel, keine Beratung.** Erfundenes Phantom, vereinfachte Strahlenphysik (Lehrbuchannahmen), Dosis einheitenlos (relativ zur Tumor-Solldosis 1; die Literatur rechnet in Gray und Volumenanteilen). Kein klinischer Anspruch: nichts hier ist eine Behandlungs-, Diagnose- oder Planungsempfehlung.

Fall-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (`demos.html`). In einem erfundenen 2D-Phantom (Gitter 32 × 32, ein Tumor, daneben ein Organ, 12 Einstrahlwinkel × 16 Strahlen) wird die **mittlere Dosis außerhalb des Tumors** minimiert, während der Tumor eine Dosis zwischen 1,0 und 1,2 bekommt. Eine **Dosis-Volumen-Grenze** („höchstens k Organvoxel über der Grenzdosis d“) macht daraus eine ganzzahlige Aufgabe. Die Demo vergleicht den **exakten Strahlplan** (ganzzahliges Programm, **vorgerechnet** und eingefroren) mit der **LP-Relaxation** und der **CVaR-Restriktion** in zwei Formen, der Form der Literatur und einer eigenen, schwächeren Variante. Live rechnet die App nur lineare Programme.

## Warum dieses Problem

Eine Zählgrenze für das Organ kostet in den Zeilen, in denen sie wirkt (133 von 268), im Mittel 1,95 % der Zielgröße (über alle 268 Zeilen 0,97 %) und senkt die Zahl der Organvoxel über d im Mittel von 15,3 auf 4,5 (die Grenze erlaubt höchstens k; dass die Zahl darunter liegt, ist durch die Grenze gebaut, kein Messergebnis). Bei M = 1,2 sieht die LP-Relaxation davon im Median nichts (bei M = 0,6 knapp ein Sechstel). Die konvexe CVaR-Näherung in der Form der Literatur ist in 13 von 133 aktiven Zeilen unzulässig, obwohl die Aufgabe lösbar ist, und kostet in den übrigen 120 im Mittel 1,68 % mehr als der exakte Wert. Die Demo zeigt, **wo** das gilt (Organ nah), wo nicht (Organ fern: die Grenze ist in 72 von 90 Zeilen nicht aktiv, in der zweiten Menge in 66 von 90) und warum die LP-Relaxation so wenig sieht (Überschuss-Form, Experiment 3). Die Kernfragen der App: (1) Was kostet die Grenze, und was ändert sie am Strahlplan? (2) Was sieht die LP-Schranke davon? (3) Wie weit reicht die konvexe Näherung?

**Hook (nach der Messung, nicht nach der alten Hypothese):** Eine Zählgrenze für das Organ kostet im Zielwert in den aktiven Zeilen im Mittel 1,95 % (über alle Zeilen 0,97 %) und senkt die Zahl der Organvoxel über d (im Mittel von 15,3 auf 4,5; durch die Grenze gebaut, kein Messergebnis); bei M = 1,2 sieht die LP-Relaxation davon im Median nichts, und die CVaR-Näherung in der Form der Literatur zahlt in den zulässigen Zeilen im Mittel 1,68 % mehr und ist bei Organ nah unzulässig (alle 13 Fälle, 11 davon bei der strengen Grenze mit 10 %). Im Mittel liegen die Lücken der Ersatzformen bei 1,4 bis 1,9 % (zwei Mengen von Zufalls-Seeds), bei Organ nah bei 2 bis 4 %, einzelne Zeilen deutlich darüber. Der **Preis der Grenze** (1,95 %: exakt gegen Strahlplan ohne Grenze) und die **Lücke einer Ersatzform** ((Ersatz − exakt) / exakt) sind zwei verschiedene Größen und werden nie vermischt; „Kosten der Nichtkonvexität“ ist keine Größe dieser Demo.

## Modell

Körper = Scheibe (Radius 0,95 in [−1, 1]²) auf einem 32 × 32-Gitter, 732 Körpervoxel; Tumorscheibe (Radius 0,16 bis 0,28), daneben eine Organellipse (mindestens 12 Voxel). Organabstand als Klasse, benannt nach dem gezogenen Mindestmaß des kleinsten Abstands der Voxelmitten: nah (1,0 bis 1,5 Voxelbreiten), mittel (2 bis 3), fern (4 bis 6); 30 Phantome je Klasse. Die Erzeugung vergrößert den Abstand, bis er das gezogene Maß erreicht, und er springt auf dem Gitter; deshalb liegt er oft darüber: in der ersten Menge liegt der kleinste Abstand bei nah zwischen 1,41 und 2,24 Voxelbreiten (15 von 30 Phantomen über 1,5), bei mittel zwischen 2,24 und 3,61, bei fern zwischen 4,12 und 6,40. Die Klassen überlappen sich also an den Rändern (nah bis 2,24, mittel ab 2,24). Dosis D = A x mit Strahlgewichten x ≥ 0 (12 Winkel in 30°-Schritten × 16 parallele Strahlen = 192), A[Voxel, Strahl] = exp(−μ · Tiefe) · Gauß(Querabstand), μ = 1, Streukern 0,6 Strahlbreiten, bei 3 σ abgeschnitten: **Lehrbuchannahme**, keine Dosisberechnung. Zielgröße f = mittlere Dosis aller Nicht-Tumor-Voxel (Organ und gesundes Gewebe); Tumorfenster 1,0 ≤ D ≤ 1,2; Organ-Höchstdosis M (1,2, 1,0, 0,8 oder 0,6), zugleich die Big-M-Konstante.

**Dosis-Volumen-Grenze:** höchstens k Organvoxel mit D_v > d. Gemessene Kombinationen (d; Anteil p) ∈ {(0,5; 25 %), (0,3; 25 %), (0,3; 10 %)}, k = ⌊p · Organvoxel⌋, mindestens 1. Drei Wege zur selben Grenze:

| Weg | Form | Art |
|---|---|---|
| Exakt (MIP) | D_v ≤ d + (M − d) z_v, z_v ∈ {0, 1}, Σ z_v ≤ k | Optimum, vorgerechnet |
| LP-Relaxation | z_v ∈ [0, 1]; gleichwertig (zusammen mit D_v ≤ M) Σ max(0, D_v − d) ≤ k (M − d) (Überschuss-Form) | untere Schranke |
| CVaR(k), Form der Literatur | Mittel der k größten Organdosen ≤ d (Rockafellar und Uryasev; so bei Kishimoto/Yamashita 2018 und Engberg u. a. 2017) | obere Schranke (Restriktion) |
| CVaR(k+1), eigene Variante | Mittel der k + 1 größten Organdosen ≤ d | obere Schranke, schwächer als CVaR(k) |

Die **Überschuss-Form** ist zusammen mit D_v ≤ M eine Identität (im Test an 20 Zeilen der Tabelle und auf kleinen Gittern geprüft): die Relaxation erlaubt nicht „k Voxel über d“, sondern einen Gesamtüberschuss von k (M − d), und viele kleine Überschreitungen passen darunter. Ohne D_v ≤ M (die Big-M-Form erzwingt es über z_v ≤ 1) sind beide Formen nicht gleichwertig: bei M = 0,6 war der Wert der Überschuss-Form in 5 von 19 geprüften Zeilen niedriger, bis zu 0,39 % des Relaxationswertes, bei M = 1,2 an 34 Zeilen gleich (Handmessung beim Prüfen, nicht im Test). Dass sie deshalb die Kosten der Grenze meist kaum sieht, ist ein Befund über dieses Phantom (Experiment 3), keine Regel. CVaR(k) erzwingt höchstens k − 1 Organvoxel über d, CVaR(k+1) höchstens k (Zählung im Test geprüft); beide sind gültige Restriktionen, die Umkehrung gilt nicht. **CVaR(k+1) ist eine eigene Ableitung**, in keiner gelesenen Quelle gefunden.

Genaues Modell, Konventionstabelle und Annahmenliste stehen im 📐-Expander der App; der Annahmentext der App steht hier wörtlich:

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

## Methodik

* **Löser:** HiGHS über `scipy.optimize.linprog` (LP) und `scipy.optimize.milp` (MIP). Die MIP läuft **nur** im Erzeugungsskript `tools/dpl_make_table.py` (Zeitlimit 30 s, relative Lücke 1e-6), nie in der App. Die Tabelle `data/dpl_table.json` hält je Phantom, Höchstdosis und Kombination die Menge S der Organvoxel, die über d liegen dürfen, und den Wert des LP zu dieser Menge; die App löst dieses LP und zeigt den exakten Strahlplan. Keine Zeile der Tabellen hat den Status „nicht bewiesen“ (Zeitlimit); jede Zeile ist bewiesen optimal oder unzulässig (Test `test_no_row_is_unproven_and_infeasible_rows_carry_no_numbers`). Eine Zeile ohne Beweis würde in der Tabelle und in der App so gekennzeichnet.
* **Zufall nur über den ganzzahligen SplitMix64** (`dpl_rng.py`, Geometrie mit `exact_cos`: ausgelegt auf dieselben Phantome auf jeder Plattform, unter Windows geprüft), nie über Python-`hash()`; für Phantome, Messung und Bootstrap kommt der Zufall nur daher, nur der Würfel-Knopf der Oberfläche nutzt `random` (für die Wahl des Zufalls-Seeds, nicht für Messwerte). Zufalls-Seed eines Phantoms: Basis · 10⁶ + Abstandsklasse · 10³ + Nummer; **zwei Mengen von Zufalls-Seeds** (erste Menge: Basis 20261118, die Hauptzahlen und die Anzeige der App; zweite Menge: Basis 20261119, die zweite Zahlenreihe neben der ersten). Mittelwerte mit **Bootstrap über Phantome** (2000 Ziehungen, SplitMix64), weil Zeilen desselben Phantoms abhängig sind.
* **Zähleinheit** der Verteilungen: Zeilen = Phantom × Kombination (d; Anteil) bei fester Höchstdosis M. „Aktiv“: die Grenze kostet mehr als 1e-5 relativ. Voxel zählen als „über d“, wenn D_v > d + 1e-6.
* **Kosten der Grenze** = (f_exakt − f_ohne Grenze) / f_ohne Grenze; **Lücke der Relaxation** = (f_exakt − f_Relaxation) / f_exakt; **Lücke einer CVaR-Form** = (f_CVaR − f_exakt) / f_exakt; **Anteil der Kosten in der Relaxation** = (f_Relaxation − f_ohne Grenze) / (f_exakt − f_ohne Grenze).
* **Sieben Experimente** (Umschalter in der App, nur das gewählte rechnet): 0 harte Höchstdosis (Vorstufe, Zielkonflikt), 2 Preis der Grenze und Gegenfall, 3 Warum sieht die LP-Schranke nichts, 4 CVaR als Restriktion, 5 Nichtkonvexität (zwei Strahlpläne, ein Mittelpunkt), 6 Kostenkurve über k, 7 Handinstanz in Brüchen. Die Kernfragen 1 bis 3 mit Verdikt stehen über dem Umschalter.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten für **dieses** Phantom (32 × 32, ein Tumor, ein Organ), die Zielgröße „mittlere Dosis außerhalb des Tumors“, die Höchstdosis M = 1,2 (außer in der Höchstdosis-Zeile), drei Kombinationen (d; Anteil) und die genannten Mengen von Zufalls-Seeds; Zähleinheit Zeilen (Phantom × Kombination), Mittelwerte mit 95-%-Bootstrap-Intervall über Phantome in eckigen Klammern. Keine Aussage über echte Strahlpläne.

| Frage | Befund | Test |
|---|---|---|
| Zeilen und Status des exakten Werts | 270 Zeilen (3 Abstände × 30 Phantome × 3 Kombinationen); **268 bewiesen optimal, 0 ohne Beweis, 2 unzulässig** (beide Organ nah, d = 0,3, 10 %); zweite Menge: 266 bewiesen, 4 unzulässig | `test_claim_rows_and_status` |
| Grenze aktiv | **133 von 268** (nah 61 von 88, mittel 54 von 90, fern 18 von 90); zweite Menge 149 von 266. Der Anteil liegt bei **50 %** (erste Menge) und **56 %** (zweite Menge); mit nur zwei Mengen ist das keine Schwankungsbreite, und die Zahl je Abstandsklasse springt stärker (fern 18 gegen 24) | `test_claim_active_rows` |
| Grenze nicht aktiv | 135 von 268; in 128 dieser 135 Zeilen erfüllt der Strahlplan ohne Grenze die Grenze ohnehin; Organ fern: 72 von 90 Zeilen nicht aktiv (zweite Menge 66 von 90) | `test_claim_inactive_rows` |
| Kosten der Grenze (aktive Zeilen) | Mittel **1,95 %** [1,44; 2,50], Median 0,85 %, Maximum 13,41 %; zweite Menge Mittel 1,75 % [1,21; 2,34], Median 0,67 %, Maximum 25,76 % | `test_claim_cost_of_the_limit` |
| Lücke MIP gegen LP-Relaxation | Mittel **1,50 %** [1,13; 1,87], Median 0,78 %; über 1 % in 56 von 133; zweite Menge Mittel 1,37 % [0,97; 1,81], Median 0,61 %, über 1 % in 57 von 149 | `test_claim_gap_of_the_relaxation` |
| Anteil der Kosten, den die Relaxation sieht | **Median 0,000**, Mittel 0,090, Maximum 0,571; unter 0,01 in **77 von 133**, über 0,5 in 2; zweite Menge Median 0,000, unter 0,01 in 89 von 149, über 0,5 in 8 | `test_claim_share_of_the_relaxation` |
| Relaxation hat in 72 von 133 aktiven Zeilen denselben Zielwert wie der Strahlplan ohne Grenze | in 72 von 133 aktiven Zeilen ist der Gesamtüberschuss des Strahlplans ohne Grenze höchstens k (M − d), und der Wert der Relaxation ist der Wert ohne Grenze (gemessen ist der Zielwert; ob der Löser denselben Strahlplan wählt, ist bei Mehrdeutigkeit nicht gesagt); zweite Menge 85 von 149 (Wert) und 83 von 149 (Überschuss); „denselben Zielwert“ heißt gleich mit relativer Toleranz 1e-9, bei 1e-5 wären es 74 statt 72 in der ersten Menge | `test_claim_relaxation_equals_free_plan` |
| CVaR(k), Form der Literatur | **unzulässig in 13 von 133** aktiven Zeilen (alle Organ nah; zweite Menge 15 von 149), obwohl der exakte Wert existiert; Lücke in den 120 zulässigen Zeilen Mittel **1,68 %** [1,20; 2,19], Median 0,81 %, Maximum 11,13 %, über 1 % in 49; zweite Menge Mittel 1,89 % [1,32; 2,62] | `test_claim_cvar_k` |
| CVaR(k+1), eigene Variante | unzulässig in 9 von 133 (alle Organ nah; zweite Menge 9 von 149); Lücke in den 124 zulässigen Zeilen Mittel 1,63 % [1,11; 2,21], Median 0,74 %, Maximum 14,63 %, über 1 % in 51; auf den 120 Zeilen, in denen beide zulässig sind: CVaR(k+1) Mittel 1,33 % [0,95; 1,74], Median 0,71 %; zweite Menge Mittel 1,88 % | `test_claim_cvar_k_plus_1` |
| Reihenfolge und Zählung | f_Relaxation ≤ f_exakt ≤ f_CVaR(k+1) ≤ f_CVaR(k): 0 Verletzungen (Toleranz 1e-5 relativ) in 120 Zeilen mit beiden CVaR; nur CVaR(k) unzulässig in 4 Zeilen, nur CVaR(k+1) nie; CVaR(k)-Strahlplan hat höchstens k − 1 Organvoxel über d in 120 von 120, CVaR(k+1)-Strahlplan höchstens k in 124 von 124 | `test_claim_order_and_count` |
| Organvoxel über d (Mittel, 120 Zeilen mit beiden CVaR zulässig) | ohne Grenze 15,3; exakt 4,7; CVaR(k+1) 2,5; CVaR(k) 2,1; erlaubt k im Mittel 4,7 (über alle 133 aktiven Zeilen: ohne Grenze 15,3, exakt 4,5) | `test_claim_organ_voxels_over_d` |
| Nach Organabstand (Lücke Relaxation / CVaR(k+1) / CVaR(k), Mittel in %) | nah 2,02 / 3,01 / 3,03; mittel 1,34 / 0,82 / 1,00; fern 0,25 / 0,08 / 0,11. Die Reihenfolge nah vor mittel vor fern gilt in beiden Mengen; nah und mittel sind bei der Lücke der Relaxation nicht klar getrennt (Intervalle [1,43; 2,62] und [0,86; 1,85] überlappen), fern liegt klar darunter [0,11; 0,42]. Die CVaR-Mittel enthalten nur die zulässigen Zeilen und unterschätzen bei nah. Die Klassen sind nach dem gezogenen Mindestmaß benannt (siehe Modell); Organgröße und Abstand sind nicht getrennt | `test_claim_by_organ_distance` |
| Nach Kombination (d; Anteil) | (0,5; 25 %): aktiv in 21 von 90, Kosten Mittel 0,92 %; (0,3; 25 %): 52 von 90, 1,66 %; (0,3; 10 %): 60 von 88, 2,57 % | `test_claim_by_combination` |
| Organ-Höchstdosis M = 1,2 / 1,0 / 0,8 / 0,6 | aktive Zeilen 133 / 134 / 133 / 111; Anteil der Kosten in der Relaxation, Median 0,000 / 0,028 / 0,085 / 0,167; unter 0,01 in 77 / 64 / 54 / 37; Lücke der Relaxation Mittel 1,50 / 1,35 / 1,14 / 0,80 %; erlaubter Gesamtüberschuss k (M − d) im Mittel 3,85 / 2,94 / 2,04 / 1,13; CVaR(k+1) unzulässig je 9 | `test_claim_cap_table` |
| Strenge Höchstdosis (k = 0) | für **alle** Organvoxel unzulässig in 25 von 133 aktiven Zeilen (zweite Menge 25 von 149) | `test_claim_strict_limit` |
| **Nichtkonvexität (konstruiert!)** | Mittelpunkt zweier zulässiger Strahlpläne mit disjunkten Überschreitungsmengen verletzt die Grenze in **37 von 108** aktiven Zeilen mit gebildetem Paar (34 %; nah 13 von 36, mittel 21 von 54, fern 3 von 18); zweite Menge 38 von 125 (30 %) | `test_claim_nonconvexity` |
| Kostenkurve über k (d = 0,3, M = 1,2; 30 Phantome, je 10 je Abstand) | 741 Kurvenschritte, **0 Verletzungen** der Monotonie in k für exakt, Relaxation, CVaR(k+1) und CVaR(k) (Toleranz 1e-5 relativ); die Grenze kostet nur bis k*, k* gleich der Zahl der Organvoxel über d im Strahlplan ohne Grenze in 20 von 30 Phantomen (höchstens diese Zahl in 24); strenge Höchstdosis unzulässig in 4 von 30 | `test_claim_cost_curve` |
| Lücken nach k / Organvoxel (256 Kurvenschritte mit aktiver Grenze) | bis 0,10: Lücke der Relaxation Mittel 1,43 % (Median 1,05 %), Anteil der Kosten in der Relaxation Median 0,031, CVaR(k+1) unzulässig in 4 von 42, CVaR(k) in 6; über 0,10 bis 0,25: 0,75 % / 0,000 / 2 von 76 / 4; über 0,25 bis 0,50: 0,45 % / 0,000 / 0 / 0; über 0,50: 0,36 % / 0,000 / 0 / 0. Größte Lücken: Relaxation 6,27 % (Organ nah, Phantom 1, k = 2 von 25 Organvoxeln), CVaR(k+1) 14,60 % (Organ nah, Phantom 3, k = 2 von 28 Organvoxeln). Von den Kosten bei k = 1 bleiben bei k = ¼ der Organvoxel im Median 0,30, bei k = ½ der Organvoxel 0,001 | `test_claim_gaps_by_k` |
| Handinstanz H2 (Brüche) | ohne Grenze 3/8, exakt 19/48, Relaxation 3/8, CVaR(k+1) 5/12, CVaR(k) unzulässig; beide Lücken 1/19 = 5,26 %; Mittelpunkt (1/3, 1/6, 1/2) hat zwei Organvoxel über d | `test_h2_*` in `tests/test_hand_examples.py` |
| Handinstanz H3 (6 × 6-Gitter, d = 0,3, k = 1) | ohne Grenze 0,46274, Relaxation 0,49042, exakt 0,50795, CVaR(k+1) 0,78814, CVaR(k) unzulässig; die Grenze kostet 9,8 %; bei d = 0,5, k = 1 ist die Relaxation exakt (0,48040) | `test_h3_*` in `tests/test_hand_examples.py` |

**Messung dieser README, nicht im Test:** Die Live-Rechnung der gewählten Zelle (Strahlplan ohne Grenze, Relaxation, beide CVaR-Formen, exakter Strahlplan zur vorgerechneten Menge) dauert über alle 270 Zellen mit M = 1,2 im Median 0,03 bis 0,04 s, einzelne Zellen bis etwa 0,1 s (zwei Messläufe, einer davon unter Last; Maschine des Baus, Python 3.14, HiGHS in scipy); die Pareto-Kurve der Vorstufe (22 LPs) rund 0,14 s.

Die Zahlen dieser Tabelle stehen im Test `tests/test_claims.py`, jede in der hier genannten Schreibweise als Text der README gegen die aus den eingefrorenen Tabellen berechneten Werte geprüft (die Tabellen selbst entstehen im Erzeugungsskript; `tests/test_table.py` und `python tools/dpl_make_table.py --check` rechnen Stichproben neu und vergleichen sie als Band, relativ 1e-5).

## Befunde und Korrekturen gegenüber der ersten Rechnung

Die erste Rechnung (eine Messreihe außerhalb dieses Repos) nutzte einen anderen Zufallsgenerator; dieselben Fragen mit dem Zufall der Demo (SplitMix64) und anderen Phantomen gaben:

* **Abbruchkriterium nicht erfüllt, Hook bleibt:** Die Lücke der Relaxation liegt im Mittel bei 1,50 % (erste Rechnung 1,42 %), also im Mittel weit unter 5 % (einzelne Zeilen liegen darüber: 10 von 133 über 5 %, 1 über 10 %); der Anteil der Kosten in der Relaxation hat den Median 0,000 (erste Rechnung ebenso). Der Hook „kleiner Preis, großer Unterschied im Strahlplan, leere Schranke“ hält.
* **Anzahl aktiver Zeilen:** 133 von 268 (50 %) und in der zweiten Menge 149 von 266 (56 %), statt 171 von 267 (64 %) und 128 von 268 (48 %). Das sind 50 % und 56 % statt 64 % und 48 %; mit nur zwei Mengen ist das keine Schwankungsbreite, die Zahl springt mit den Phantomen.
* **Schwelle „aktiv“:** 1e-5 relativ (die erste Rechnung nutzte 1e-7 bei MIP-Lücke 1e-6, darunter liegen Rundungsreste). Der exakte Wert einer Zeile ist der Wert des LP zur gefundenen Menge S.
* **Kosten der Grenze** Mittel 1,95 % statt 1,92 %, Median 0,85 % statt 0,89 %, Maximum 13,41 % statt 36,3 % (das Maximum ist die Spitze einer Verteilung: 17 aktive Zeilen liegen über 5 %, die nächsten Werte sind 11,54 %, 10,19 % und 9,08 %; in der zweiten Menge ist das Maximum 25,76 %); **Lücke der Relaxation** 1,50 % statt 1,42 %; **CVaR(k) unzulässig** in 13 statt 15 Zeilen, **CVaR(k+1)** in 9 wie zuvor; **Nichtkonvexitäts-Probe** 37 von 108 (34 %) statt 44 von 140 (31 %); strenge Höchstdosis unzulässig in 25 von 133 statt 32 von 171.
* **Nach Organabstand:** Lücke der Relaxation nah 2,02 %, mittel 1,34 %, fern 0,25 % (erste Rechnung 2,34 / 1,01 / 0,38 %): die Reihenfolge nah größer als fern bleibt, die Mitte liegt nun höher.
* **Kostenkurve über k:** „Die Lücken sitzen bei strengen Grenzen“ gilt schwächer. Bei k / Organvoxel bis 0,10 liegt die Lücke der Relaxation im Mittel bei 1,43 % (erste Rechnung 4,32 %), über 0,50 bei 0,36 % (0,69 %); die größte Lücke ist 6,27 % statt 29,3 %. CVaR(k+1) ist nur bei k / Organvoxel bis 0,25 unzulässig (6 von 256 Kurvenschritten mit aktiver Grenze), wie zuvor, nur in anderen Zahlen. Neu: die Literaturform CVaR(k) steht in der Kurve (10 von 256 unzulässig, ebenfalls nur bei k / Organvoxel bis 0,25).
* **Höchstdosis M:** der Anteil der Kosten in der Relaxation steigt mit kleinerem M (Median 0,000 / 0,028 / 0,085 / 0,167), aber auch bei M = 0,6 sieht die Relaxation im Median nur knapp ein Sechstel der Kosten. Das erste Bild (Median bis 0,22) war eine andere Messreihe (anderer Zufallsgenerator, andere Phantome).
* **Preset „Höchster Preis der Grenze“:** das Phantom mit den größten Kosten der Tabelle (Organ nah, Zufalls-Seed 24, d = 0,3, 25 %: 13,41 %); die Wahl des Zufalls-Seeds ist Teil der Messung, nicht der Hook.
* **Gestrichen gegenüber der ersten Rechnung:** die Verteilungen der Vorstufe (Pareto-Kurve konvex 180 von 180, Gültigkeitsstück des Preises, Schutzkosten) und der Winkelauswahl; sie stehen in [lp-dualitaet-demo](https://github.com/sebastian-hanisch/lp-dualitaet-demo), [zwei-kriterien-demo](https://github.com/sebastian-hanisch/zwei-kriterien-demo) und [aufloesung-messorte-demo](https://github.com/sebastian-hanisch/aufloesung-messorte-demo). Die Vorstufe zeigt nur die Kurve des gewählten Phantoms und im Test die Monotonie von Zielgröße und Preis; ein Zahlenbefund über viele Phantome wird hier nicht behauptet. <!-- links-ok: Repo nicht als Ordner lokal vorhanden -->

## Ehrliche Grenzen

* **Erfundenes Modell:** Phantom, Strahlenkern und Dosis sind Lehrbuchannahmen; die Zahlen gelten für diese Phantomfamilie, diese Zielgröße, M = 1,2 (Experiment 3 variiert M), drei Kombinationen (d; Anteil) und die gemessenen Zufalls-Seeds, nicht für echte Strahlpläne. Alle Instanzen sind erfunden, keine echten Daten, keine klinische Plausibilitätsprüfung.
* **Der Effekt ist im Mittel klein, einzelne Zeilen weichen stark ab, und er hängt an den Parametern:** die Kosten der Grenze (Mittel 1,95 %, Median 0,85 %) und die Lücken der Ersatzformen (1,5 bis 1,7 % im Mittel, Median 0,7 bis 0,8 %; erste Menge) hängen an Zielgröße, Höchstdosis, d, p und Organabstand. Bei Organ fern ist die Grenze in 72 von 90 Zeilen nicht aktiv, und die Lücken sind dort 0,25 % (Relaxation) und 0,08 % (CVaR(k+1)) in der ersten, 0,45 % und 0,27 % in der zweiten Menge (kleine Ausgangswerte, die zwischen den Mengen um den Faktor 1,8 bis 3,2 springen).
* **„Leer“ gilt als Median, nicht als Regel:** die Relaxation sieht in 2 von 133 Zeilen mehr als die Hälfte der Kosten (zweite Menge 8 von 149), und im Handbeispiel H3 ist sie bei d = 0,5 exakt. Die CVaR-Restriktionen sind im Mittel nah am exakten Wert (Median 0,7 bis 0,8 %) und in einem Teil der Zeilen weit weg (Maximum 11,13 % und 14,63 %; in der zweiten Menge 23,15 % und 24,36 %; bei beiden Formen liegt in etwa einer von zehn zulässigen Zeilen die Lücke über 5 %) oder unzulässig.
* **Nichtkonvexität nur als Mechanismus:** die Zahl 37 von 108 ist an **konstruierten** Paaren gemessen (der zweite Strahlplan darf die Überschreitungsvoxel des ersten nicht nutzen) und keine Häufigkeit bei typischen Strahlplänen. Craft (2013) argumentiert ohne Messung, die Nichtkonvexität verschwinde bei typischen Plänen oder sei vernachlässigbar; diese Demo prüft das nicht und stützt oder widerlegt es nicht.
* **Ursachen nur als Deutung:** dass die Überschuss-Form die Ursache der leeren Schranke in diesem Phantom ist, folgt aus ihrem Zusammenspiel mit der Zahl kleiner Überschreitungen; gemessen ist die Zählung „Zielwert der Relaxation = Wert ohne Grenze in 72 von 133 aktiven Zeilen“. Ein schärferes Big-M ändert bei M = 1,2 nichts (die größte Dosis, die ein Organvoxel unter Tumorfenster und Höchstdosis erreichen kann, ist die Höchstdosis selbst: Test an drei Phantomen); bei kleineren M gilt diese Begründung nicht allgemein (einzelne Organvoxel erreichen weniger als M), und dort ist es nicht getestet. Aber **schärfere Relaxationen** (starke gültige Ungleichungen, Tuncel u. a. 2012, nur Abstract gesehen: die Relaxation der einfachen Form kann dort beliebig schwach sein) sind nicht gemessen; eine bessere Schranke ist nicht ausgeschlossen.
* **Ungemessen:** andere Zielgrößen; Gitter über 32 × 32 und mehr als 12 Winkel; größere Organe (die Organe der Tabellen haben 14 bis 40 Voxel, also 14 bis 40 Binärvariablen je MIP); Verfeinerungen der CVaR-Restriktion aus der Literatur (Ausreißer-Ausschluss bei Kishimoto und Yamashita, Zwei-Pass bei Fu u. a.); andere Tail-Größen als k und k + 1; mehr als drei Kombinationen (d; Anteil); ein Tumor, ein Organ; μ und Streukern fest; die Nichtkonvexitäts-Probe mit zufällig gezogenen statt konstruierten Paaren. Für die Größe der Lücken gibt es in den gelesenen Quellen keine Vergleichszahlen.
* **Die Einstrahlwinkel sind fest** (12). Die Auswahl weniger Winkel aus vielen ist eine eigene Aufgabe (Auswahl aus Kandidaten) und steht in [aufloesung-messorte-demo](https://github.com/sebastian-hanisch/aufloesung-messorte-demo); ein zusätzlicher Winkel verschlechtert den Wert nie (Test an 24 Proben, `tests/test_eigenschaften_dosis.py`).

## Verifikation

* **Orakel-Tests** (`tests/test_oracle_dosis.py`, anderer Rechenweg): (a) LP-Dualität: Primal = Dual in einem unabhängig gebauten Dual-LP auf sechs Phantomen, Schattenpreis der Pareto-Kurve zwischen linker und rechter Steigung; (b) exakte MIP gegen die Aufzählung aller Mengen S auf 6 × 6-Gittern (32 Instanzen, davon 28 zulässig; die Kette Relaxation ≤ exakt ≤ CVaR(k+1) ≤ CVaR(k) wird nur an den zulässigen geprüft, in den übrigen vier sind MIP und Aufzählung beide unzulässig); (d) Strahlenmatrix: Skalar-Schleife gegen Matrix, Summe des Streukerns gegen die Poisson-Summe; (e) Überschuss-Form = Relaxation an 20 Zeilen; (f) CVaR in Paar-Form gegen Rockafellar-Uryasev, Zählung k − 1 und k; (h) Big-M bei M = 1,2 nicht schärfbar; (i) eingefrorene Tabelle gegen frisch gerechnete LP-Teile.
* **Handbeispiele von Hand** (`tests/test_hand_examples.py`): Strahlenmatrix auf einem 5 × 5-Gitter, Zielkonflikt mit zwei Strahlen (Preis −1), H2 in exakten Brüchen mit dem Solver als zweitem Weg, H3 auf dem 6 × 6-Gitter.
* **Eigenschaftstests** (`tests/test_eigenschaften_dosis.py`): Zielfunktion mal c, Umnummerieren der Strahlen, Werte fallen mit wachsendem k und wachsendem d, k ≥ Organgröße und d ≥ M machen die Grenze leer, k = 0 ist die strenge Höchstdosis, ein zusätzlicher Winkel verschlechtert nie, Reihenfolge der Schranken auf allen 268 bewiesenen Zeilen.
* **Randfälle** (`tests/test_edge_cases.py`): unzulässige MIP wird als „unzulässig“ gemeldet, nie als Zahl; Organ aus genau k Voxeln, ein Organvoxel, Organ ohne Voxel; M kleiner oder gleich d; nicht erreichbares Tumorfenster.
* **Claims, Verdikte, Tabelle, Presets, Layout, Oberfläche:** `tests/test_claims.py`, `tests/test_verdicts.py` (jede Zahl im Text, je Preset die erwartete Verdikt-Zeile; ein Preset ohne Effekt meldet keinen Effekt), `tests/test_table.py` (Phantome aus dem Zufalls-Seed neu erzeugt = eingefroren, `--check` im Kleinen), `tests/test_presets.py`, `tests/test_layout.py` (Schnellstart oben, fester Kopf, Hervorhebung), `tests/test_app.py` (AppTest: jeder Button, jede Ansicht, die Ränder aller Regler), `tests/test_readme.py`, Footer und Laufzeit-Importe.
* **Linux-Check** (Docker, Python 3.12) vor dem ersten Push: Ergebnis steht in der Übergabe, nicht in dieser README. Zählungen und Verhältnisse sind Bänder, Ganzzahlen und Brüche exakt.
* 161 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `dpl_constants.py` | feste Annahmen, Regler-Grenzen, Farben, Presets |
| `dpl_model.py` | Phantom, Strahlenmatrix (Vektor- und Skalarform) |
| `dpl_solvers.py` | Löser: Strahlplan ohne Grenze, MIP, Relaxation (zwei Formen), CVaR (zwei Wege), Aufzählung, Pareto-Kurve, Nichtkonvexitäts-Probe |
| `dpl_rows.py` | Zeilen der vorgerechneten Tabellen (Erzeugung und Tests) |
| `dpl_data.py` | Laden und Nachschlagen der Tabellen |
| `dpl_evaluation.py` | Kennzahlen, Zusammenfassungen, Verdikte |
| `dpl_stats.py` | Bootstrap über Phantome (SplitMix64, vektorisiert) |
| `dpl_hand.py` | Handinstanz in Brüchen (Eckenaufzählung) |
| `dpl_rng.py` | SplitMix64 (aus der geprüften Vorlage kopiert) |
| `dpl_texts.py` | lange Texte der App |
| `dpl_layout.py` | fester Kopf (Stylesheet) |
| `dpl_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Button |
| `dpl_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `data/dpl_table.json`, `data/dpl_seeds2.json`, `data/dpl_curve.json` | vorgerechnete Tabellen (erste und zweite Menge von Zufalls-Seeds, Kostenkurven) |
| `tools/dpl_make_table.py` | erzeugt die Tabellen, `--check` rechnet Stichproben neu |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit-Bibliotheken (ungepinnt) und pytest |
| `tests/` | Orakel, Handbeispiele, Eigenschaften, Randfälle, Claims, Verdikte, Tabelle, Presets, Layout, Oberfläche, README, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

* **MIP live in der App:** die exakte Aufgabe ist vorgerechnet; Live-Rechnung nur für lineare Programme.
* **PDF-Export:** nicht umgesetzt (nichts, was die Aussage trüge).
* **Regler für Strahlbreite und Winkelzahl:** ungemessen.
* **Vergleich mit quadratischer Zielfunktion (L1 gegen L2):** gehört in [nutrition-demo](https://github.com/sebastian-hanisch/nutrition-demo).
* **Szenario-Garantien** (Verwerfen von Szenarien, Konfidenz): stehen in [scenario-approach-demo](https://github.com/sebastian-hanisch/scenario-approach-demo), das das Verwerfen von Szenarien, ein ganzzahliges Problem, bewusst nicht umsetzt; die Dosis-Volumen-Grenze entspricht dieser Aufgabe mit Voxeln als Szenarien (Deutung dieser Demo). CVaR als Ziel steht in [cvar-demo](https://github.com/sebastian-hanisch/cvar-demo), hier ist es eine Restriktion; die Schranke gegen ganzzahlig im Packen in [lp-schranke-demo](https://github.com/sebastian-hanisch/lp-schranke-demo). <!-- links-ok: Repo nicht als Ordner lokal vorhanden -->
* **Zielkonflikt und Schattenpreis** ausführlich: [lp-dualitaet-demo](https://github.com/sebastian-hanisch/lp-dualitaet-demo), [zwei-kriterien-demo](https://github.com/sebastian-hanisch/zwei-kriterien-demo), [hart-gegen-weich-demo](https://github.com/sebastian-hanisch/hart-gegen-weich-demo). <!-- links-ok: Repo nicht als Ordner lokal vorhanden -->

## Quellen

Gelesen (Wortlaut): Craft (2013); Rockafellar und Uryasev (2000 und 2002, jeweils Autoren-Preprint); Kishimoto und Yamashita (2018); Maass, Kim und Aravkin (2022, arXiv-Fassung); Fu, Ungun, Xing und Boyd (2019, ConRad); Engberg u. a. (2017). Nur Abstract oder Metadaten gesehen: Tuncel u. a. (2012), Deasy (1997), Ehrgott u. a., Wu und Mohan (2002), Craft u. a. (2006). **Romeijn u. a. (2003, 2006) sind nicht gelesen** und dienen nur als „genannt in …“: Die Näherung der Dosis-Volumen-Grenze durch das Mittel der heißesten Voxel (CVaR, Rockafellar und Uryasev) wird seit Romeijn u. a. (2003, 2006) für die Fluenzoptimierung genannt (nach Kishimoto und Yamashita 2018; Engberg u. a. 2017); die Variante mit k + 1 heißesten Voxeln ist hier eine eigene Ableitung. Gleichung und Tail-Größe k beruhen auf Kishimoto/Yamashita und Engberg u. a., die Hilfsvariablen auf Rockafellar/Uryasev. Eine Quelle für die Überschuss-Form der Relaxation wurde nicht gefunden (verwandt: eine Summe von Hinge-Termen bei Fu u. a.). **Keine Neuheitsaussage gegenüber der Literatur.** Kishimoto und Yamashita fanden die Methode von Romeijn u. a. auf ihren vier Testfällen unzulässig; das passt qualitativ zu den Unzulässigkeiten hier, ohne Vergleich der Zahlen (andere Instanzen).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
python tools/dpl_make_table.py --check     # Stichproben der Tabellen neu rechnen (einige Minuten)
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
