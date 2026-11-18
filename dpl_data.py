"""Vorgerechnete Tabellen laden (data/*.json, erzeugt von tools/dpl_make_table.py) und Phantome, Zeilen und Kurven nachschlagen.

Die MIP wird nie in der App gerechnet: die Tabelle hält je Phantom, Höchstdosis und Kombination den Wert des exakten Strahlplans und die Menge S der Organvoxel,
die über d liegen dürfen. Die App löst zu S ein LP (der Strahlplan), alles Übrige rechnet sie live."""

import json
from functools import lru_cache
from pathlib import Path

import dpl_constants as C
from dpl_model import Phantom

DATA_DIR = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def table() -> dict:
    return _load("dpl_table.json")


def seeds2() -> dict:
    return _load("dpl_seeds2.json")


def curves() -> dict:
    return _load("dpl_curve.json")


@lru_cache(maxsize=None)
def _index() -> dict:
    return {(r["gap"], r["i"], r["ci"], r["M"]): r for r in table()["rows"]}


def get_row(gap: str, i: int, ci: int, cap: float) -> dict:
    return _index()[(gap, i, ci, cap)]


def rows(cap: float | None = None, gap: str | None = None, ci: int | None = None, data: dict | None = None) -> list[dict]:
    """Zeilen einer Tabelle (Standard: erste Menge von Zufalls-Seeds), gefiltert; Zeilen ohne erreichbares Tumorfenster (f_free leer) bleiben draußen."""
    src = (data if data is not None else table())["rows"]
    return [r for r in src if r.get("f_free") is not None and (cap is None or r["M"] == cap) and (gap is None or r["gap"] == gap) and (ci is None or r["ci"] == ci)]


def phantom(gap: str, i: int) -> Phantom:
    p = table()["phantoms"][gap][i]
    return Phantom(gap, i, tuple(p["T"]), tuple(p["O"]), p["gv"])


def curve(gap: str, i: int) -> dict | None:
    for c in curves()["curves"]:
        if c["gap"] == gap and c["i"] == i:
            return c
    return None
