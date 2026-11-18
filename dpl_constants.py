"""Konstanten: feste Annahmen des Modells, Regler-Grenzen, Farben und Presets.

Das Phantom, die Dämpfung und der Streukern sind Lehrbuchannahmen; die Dosis ist einheitenlos (relativ zur Tumor-Solldosis 1).
Anwendungsbeispiel, keine Beratung: kein klinischer Anspruch."""

# ---- Modell (fest, nicht als Regler: andere Werte sind nicht gemessen)
GRID = 32                      # Gitter 32 x 32
BODY_RADIUS = 0.95             # Körperscheibe im Bereich [-1, 1]^2
N_ANGLES = 12
ANGLES_DEG = tuple(30 * i for i in range(N_ANGLES))
N_BEAMS = 16                   # parallele Strahlen je Winkel
N_VARS = N_ANGLES * N_BEAMS    # 192 Strahlgewichte
MU = 1.0                       # Dämpfung je Längeneinheit
SIGMA_REL = 0.6                # Breite des seitlichen Streukerns in Strahlbreiten
CUT_SIGMA = 3.0                # Streukern bei 3 Sigma abgeschnitten
TUMOR_MIN, TUMOR_MAX = 1.0, 1.2
ORGAN_CAP = 1.2                # Standard-Höchstdosis im Organ (zugleich Big-M der Dosis-Volumen-Grenze)

GAP_NAMES = ("nah", "mittel", "fern")
GAP_RANGES = {"nah": (1.0, 1.5), "mittel": (2.0, 3.0), "fern": (4.0, 6.0)}   # gezogenes Mindestmaß des Abstands Tumor-Organ in Voxelbreiten (kleinster Abstand der Mitten); der tatsächliche Abstand liegt darüber (erste Menge: nah 1,41 bis 2,24, mittel 2,24 bis 3,61, fern 4,12 bis 6,40)
N_PHANTOMS = 30                # Phantome je Abstandsklasse (Zufalls-Seeds 0 bis 29)
SEED_BASIS_1 = 20261118        # erste Menge von Zufalls-Seeds (die App zeigt diese Phantome)
SEED_BASIS_2 = 20261119        # zweite Menge, nur für die Bänder der README
N_CURVE_PHANTOMS = 10          # Phantome je Abstandsklasse mit vorgerechneter Kostenkurve über k
CURVE_D = 0.3

# (Grenzdosis d, Anteil p der erlaubten Organvoxel); k = floor(p * Organvoxel), mindestens 1
COMBOS = ((0.5, 0.25), (0.3, 0.25), (0.3, 0.10))
CAPS = (1.2, 1.0, 0.8, 0.6)    # gemessene Organ-Höchstdosen M (zugleich Big-M)
D1_POINTS = 21                 # Punkte der Pareto-Kurve im Einstieg

# ---- Toleranzen
TOL_COUNT = 1e-6               # Voxel gelten als "über d", wenn D_v > d + TOL_COUNT (HiGHS rechnet auf 1e-7 genau)
ACTIVE_REL = 1e-5              # "aktiv": die Grenze kostet mehr als 1e-5 relativ (unter der MIP-Toleranz 1e-6 liegen nur Rundungsreste)
MIP_TIME_LIMIT = 30.0
MIP_REL_GAP = 1e-6

# ---- Ansichten
VIEW_D1 = "0  Vorstufe: harte Höchstdosis"
VIEW_COST = "2  Preis der Grenze, Gegenfall"
VIEW_RELAX = "3  Warum sieht die LP-Schranke nichts?"
VIEW_CVAR = "4  CVaR: Restriktion, kein Ersatz"
VIEW_NONCONVEX = "5  Zwei Strahlpläne, ein Mittelpunkt"
VIEW_CURVE = "6  Kostenkurve über k"
VIEW_HAND = "7  Handinstanz (Brüche)"
VIEWS = (VIEW_D1, VIEW_COST, VIEW_RELAX, VIEW_CVAR, VIEW_NONCONVEX, VIEW_CURVE, VIEW_HAND)
DEFAULT_VIEW = VIEW_COST

# ---- Farben
COLOR_TUMOR = "#C0392B"
COLOR_ORGAN = "#2F6B65"
COLOR_FREE = "#4c78a8"
COLOR_EXACT = "#2F6B65"
COLOR_LPR = "#8e6bbf"
COLOR_CVK = "#d98f2b"
COLOR_CV1 = "#b5651d"
COLOR_GREY = "rgba(150,150,150,0.6)"

GAP_LABELS = {"nah": "nah (Mindestmaß 1,0 bis 1,5 Voxelbreiten)", "mittel": "mittel (Mindestmaß 2 bis 3)", "fern": "fern (Mindestmaß 4 bis 6)"}
COMBO_LABELS = {0: "d = 0,5, höchstens 25 % der Organvoxel", 1: "d = 0,3, höchstens 25 % der Organvoxel", 2: "d = 0,3, höchstens 10 % der Organvoxel"}

DEFAULTS = {"gap_select": "nah", "seed_input": 8, "combo_select": 1, "cap_select": 1.2, "view_select": DEFAULT_VIEW}

# Presets: Schlüssel gap, seed, combo, cap, view. Die Zufalls-Seeds der beiden Befundpresets stehen nach der Messung fest (tools/dpl_make_table.py,
# tests/test_presets.py prüft je Preset die erwartete Verdikt-Zeile).
PRESETS = {
    "Organ nah, 25 % (Standard)": {"gap": "nah", "seed": 8, "combo": 1, "cap": 1.2, "view": VIEW_COST},
    "Organ fern (Gegenfall)": {"gap": "fern", "seed": 0, "combo": 1, "cap": 1.2, "view": VIEW_COST},
    "Strenge Grenze 10 %: CVaR unzulässig": {"gap": "nah", "seed": 1, "combo": 2, "cap": 1.2, "view": VIEW_CVAR},
    "Höchster Preis der Grenze": {"gap": "nah", "seed": 24, "combo": 1, "cap": 1.2, "view": VIEW_COST},
    "Kleine Höchstdosis M = 0,6": {"gap": "nah", "seed": 8, "combo": 1, "cap": 0.6, "view": VIEW_RELAX},
    "Handinstanz (3 Strahlen, Brüche)": {"gap": "nah", "seed": 8, "combo": 1, "cap": 1.2, "view": VIEW_HAND},
}
PRESET_HELP = {
    "Organ nah, 25 % (Standard)": "Organ dicht am Tumor, Grenzdosis d = 0,3 und höchstens 25 % der Organvoxel darüber. Ein Phantom mit aktiver Grenze; ein Phantom ist kein Befund, in den Kernfragen stehen die Zahlen über alle 30 daneben.",
    "Organ fern (Gegenfall)": "Organ weit vom Tumor: der Strahlplan ohne Grenze erfüllt die Grenze schon, sie kostet nichts. Der Gegenfall zum Hauptbefund.",
    "Strenge Grenze 10 %: CVaR unzulässig": "Nur 10 % der Organvoxel dürfen über d = 0,3 liegen. Die CVaR-Restriktion in der Form der Literatur ist hier unzulässig, obwohl die Aufgabe lösbar ist; die eigene Variante CVaR(k+1) ist es nicht.",
    "Höchster Preis der Grenze": "Das Phantom mit den größten Kosten der Grenze in der gemessenen Tabelle (Organ nah, d = 0,3, 25 %): ein Extremfall, kein typischer.",
    "Kleine Höchstdosis M = 0,6": "Organ-Höchstdosis 0,6 statt 1,2: die Relaxation sieht mehr von den Kosten der Grenze als bei M = 1,2 (Experiment 3).",
    "Handinstanz (3 Strahlen, Brüche)": "Die Handinstanz in Brüchen (Ansicht 7): drei Strahlen, drei Organvoxel, alle Wege exakt nachrechenbar.",
}
