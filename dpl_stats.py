"""Kleine Statistik: Mittel mit Bootstrap-Intervall über Phantome (Zeilen desselben Phantoms sind abhängig), Median, Zählungen.

Der Bootstrap zieht die Phantome mit Zurücklegen; der Zufall kommt aus dem ganzzahligen SplitMix64 (vektorisiert über einen Zähler,
dieselben Zahlen auf jeder Plattform), nie aus Python-`hash()` oder einer C-Bibliothek."""

import numpy as np

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def splitmix_stream(seed: int, count: int) -> np.ndarray:
    """Die ersten `count` Ausgaben von SplitMix64(seed) als uint64-Feld."""
    with np.errstate(over="ignore"):
        state = np.uint64(seed & ((1 << 64) - 1)) + _GAMMA * np.arange(1, count + 1, dtype=np.uint64)
        z = state
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))


def bootstrap_indices(n_groups: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_groups)-Feld von Gruppenindizes, gezogen mit Zurücklegen."""
    raw = splitmix_stream(seed, n_boot * n_groups)
    return (raw % np.uint64(n_groups)).astype(np.int64).reshape(n_boot, n_groups)


def mean_ci(values, groups, seed: int = 1001, n_boot: int = 2000, lo: float = 2.5, hi: float = 97.5):
    """Mittel und 95-%-Bootstrap-Intervall der Werte, Ziehen über `groups` (je Wert die Phantom-Kennung). Rückgabe (Mittel, unten, oben).
    Ohne Werte (None, None, None)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return None, None, None
    keys = {}
    for val, g in zip(v, groups):
        keys.setdefault(g, []).append(val)
    sums = np.array([sum(x) for x in keys.values()])
    cnts = np.array([len(x) for x in keys.values()], dtype=float)
    idx = bootstrap_indices(len(sums), n_boot, seed)
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return float(v.mean()), float(np.percentile(means, lo)), float(np.percentile(means, hi))


def median(values):
    return None if len(values) == 0 else float(np.median(np.asarray(values, dtype=float)))
