"""Erzeugt die vorgerechneten Tabellen (exakte MIP, LP-Relaxation, CVaR, Strahlpläne als Mengen, Kostenkurve) und prüft sie.

    python tools/dpl_make_table.py main      data/dpl_table.json   erste Menge von Zufalls-Seeds, 90 Phantome x 3 Kombinationen x 4 Höchstdosen (Masken, Zeilen)
    python tools/dpl_make_table.py seeds2    data/dpl_seeds2.json  zweite Menge von Zufalls-Seeds, 90 Phantome x 3 Kombinationen, M = 1,2
    python tools/dpl_make_table.py curve     data/dpl_curve.json   Kostenkurve über k, d = 0,3, je 10 Phantome je Abstandsklasse
    python tools/dpl_make_table.py all                              alle drei Teile nacheinander
    python tools/dpl_make_table.py --check                          rechnet 12 Stichprobenzellen je Tabelle neu und vergleicht (relative Toleranz 1e-5)

Die MIP läuft mit Zeitlimit 30 s und relativer Lücke 1e-6; Zeilen ohne Beweis (Status 1) stehen mit "nicht bewiesen" in der Tabelle und
werden in der App und der README so gekennzeichnet. Die Rechnung dauert einige Minuten bis eine halbe Stunde; die App rechnet sie nie neu."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dpl_constants as C  # noqa: E402
import dpl_model as M  # noqa: E402
import dpl_rows as R  # noqa: E402
import dpl_solvers as S  # noqa: E402

DATA = ROOT / "data"
FILES = {"main": DATA / "dpl_table.json", "seeds2": DATA / "dpl_seeds2.json", "curve": DATA / "dpl_curve.json"}
CHECK_REL = 1e-5


def save(path: Path, data: dict):
    path.parent.mkdir(exist_ok=True)
    path.write_bytes((json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))   # LF auch unter Windows


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def log(text: str):
    print(text, flush=True)


def build_table(base: int, caps, keep_masks: bool) -> dict:
    a = M.standard_matrix()
    rows, phantoms = [], {g: [] for g in C.GAP_NAMES}
    t0 = time.perf_counter()
    for gap in C.GAP_NAMES:
        for i in range(C.N_PHANTOMS):
            ph = M.make_phantom(gap, i, base)
            prob = S.Problem(a, ph.tumor, ph.organ)
            if keep_masks:
                phantoms[gap].append(dict(T=list(ph.tumor), O=list(ph.organ), gv=round(ph.gap_vox, 6)))
            for cap in caps:
                free = prob.solve_free(cap)
                for ci in range(len(C.COMBOS)):
                    rows.append(R.build_row(prob, gap, i, ci, cap, free))
        log("  %s fertig, %d Zeilen, %.0f s" % (gap, len(rows), time.perf_counter() - t0))
    out = dict(meta=dict(base=base, caps=list(caps), combos=[list(c) for c in C.COMBOS], mip_time_limit=C.MIP_TIME_LIMIT, mip_rel_gap=C.MIP_REL_GAP,
                         n_phantoms=C.N_PHANTOMS, tol_count=C.TOL_COUNT), rows=rows)
    if keep_masks:
        out["phantoms"] = phantoms
    return out


def build_curves() -> dict:
    a = M.standard_matrix()
    curves = []
    t0 = time.perf_counter()
    for gap in C.GAP_NAMES:
        for i in range(C.N_CURVE_PHANTOMS):
            ph = M.make_phantom(gap, i, C.SEED_BASIS_1)
            curves.append(R.build_curve(S.Problem(a, ph.tumor, ph.organ), gap, i))
        log("  %s fertig, %.0f s" % (gap, time.perf_counter() - t0))
    return dict(meta=dict(base=C.SEED_BASIS_1, d=C.CURVE_D, cap=C.ORGAN_CAP, mip_time_limit=60.0, mip_rel_gap=C.MIP_REL_GAP), curves=curves)


# ---------------------------------------------------------------- --check

def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= CHECK_REL * max(1.0, abs(a))


def _sample(items, n=12):
    step = max(1, len(items) // n)
    return items[::step][:n]


def check_rows(table: dict, base: int, n: int = 12) -> list[str]:
    """Rechnet n (Standard 12) Stichprobenzeilen neu und vergleicht Werte (relativ 1e-5) und Zählungen (exakt)."""
    a = M.standard_matrix()
    problems = []
    for row in _sample(table["rows"], n):
        ph = M.make_phantom(row["gap"], row["i"], base)
        prob = S.Problem(a, ph.tumor, ph.organ)
        if "phantoms" in table:
            stored = table["phantoms"][row["gap"]][row["i"]]
            if list(ph.tumor) != stored["T"] or list(ph.organ) != stored["O"]:
                problems.append("Maske weicht ab: %s %d" % (row["gap"], row["i"]))
        new = R.build_row(prob, row["gap"], row["i"], row["ci"], row["M"], prob.solve_free(row["M"]))
        for key in ("f_free", "f_mip", "f_lpr", "f_cvk", "f_cv1", "f_maxd"):
            if not _close(row.get(key), new.get(key)):
                problems.append("%s %d ci=%d M=%.1f: %s alt %s neu %s" % (row["gap"], row["i"], row["ci"], row["M"], key, row.get(key), new.get(key)))
        for key in ("k", "nO", "ex_free", "ex_mip", "ex_cvk", "ex_cv1", "mip_status"):
            if row.get(key) != new.get(key):
                problems.append("%s %d ci=%d M=%.1f: %s alt %s neu %s" % (row["gap"], row["i"], row["ci"], row["M"], key, row.get(key), new.get(key)))
    return problems


def check_curves(curves: dict, n: int = 12) -> list[str]:
    a = M.standard_matrix()
    problems = []
    for cur in _sample(curves["curves"], n):
        ph = M.make_phantom(cur["gap"], cur["i"], C.SEED_BASIS_1)
        new = R.build_curve(S.Problem(a, ph.tumor, ph.organ), cur["gap"], cur["i"])
        for key in ("f_mip", "f_lpr", "f_cv1", "f_cvk"):
            for k, (x, y) in enumerate(zip(cur[key], new[key]), start=1):
                if not _close(x, y):
                    problems.append("Kurve %s %d k=%d: %s alt %s neu %s" % (cur["gap"], cur["i"], k, key, x, y))
    return problems


def main(argv):
    if "--check" in argv:
        problems = []
        for name, base in (("main", C.SEED_BASIS_1), ("seeds2", C.SEED_BASIS_2)):
            problems += check_rows(load(FILES[name]), base)
        problems += check_curves(load(FILES["curve"]))
        for p in problems:
            print("ABWEICHUNG:", p)
        print("check: %d Abweichungen" % len(problems))
        return 1 if problems else 0
    parts = [a for a in argv if a in ("main", "seeds2", "curve", "all")]
    if not parts:
        print(__doc__)
        return 2
    if "all" in parts:
        parts = ["main", "seeds2", "curve"]
    for part in parts:
        t0 = time.perf_counter()
        log("Teil %s" % part)
        if part == "main":
            data = build_table(C.SEED_BASIS_1, C.CAPS, True)
        elif part == "seeds2":
            data = build_table(C.SEED_BASIS_2, (C.ORGAN_CAP,), False)
        else:
            data = build_curves()
        save(FILES[part], data)
        log("Teil %s geschrieben (%s), %.0f s" % (part, FILES[part].name, time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
