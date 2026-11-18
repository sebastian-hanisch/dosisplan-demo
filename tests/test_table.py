"""Die vorgerechneten Tabellen: Aufbau, Zufall (Phantome aus dem SplitMix64 neu erzeugt = eingefroren), Stichprobenprüfung `--check` und Zeilenenden.

Die MIP läuft nur im Erzeugungsskript (`tools/dpl_make_table.py`); hier wird nachgesehen, ob die Tabelle zum Code passt. Werte als Band (relativ 1e-5),
Ganzzahlen und Zählungen exakt (Lehre aus der Linux-CI 2026-11-15)."""

import importlib.util
from pathlib import Path

import pytest

import dpl_constants as C
import dpl_data as D
import dpl_model as M
import dpl_rng as R
import dpl_stats as ST

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("dpl_make_table", ROOT / "tools" / "dpl_make_table.py")
MK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MK)


def test_table_structure():
    t = D.table()
    assert len(t["rows"]) == 3 * C.N_PHANTOMS * 3 * 4 and t["meta"]["base"] == C.SEED_BASIS_1 and t["meta"]["mip_time_limit"] == 30.0
    assert set(t["phantoms"]) == set(C.GAP_NAMES) and all(len(v) == C.N_PHANTOMS for v in t["phantoms"].values())
    s2 = D.seeds2()
    assert len(s2["rows"]) == 3 * C.N_PHANTOMS * 3 and s2["meta"]["base"] == C.SEED_BASIS_2 and "phantoms" not in s2
    assert len(D.curves()["curves"]) == 3 * C.N_CURVE_PHANTOMS


def test_no_row_is_unproven_and_infeasible_rows_carry_no_numbers():
    for t in (D.table()["rows"], D.seeds2()["rows"]):
        for r in t:
            assert r["mip_status"] in (0, 2), "Zeile mit Zeitlimit: müsste als nicht bewiesen gekennzeichnet werden"
            if r["mip_status"] == 2:
                assert r["f_mip"] is None and r["S"] == [] and r["wit"] is None


def test_k_is_floor_of_the_share_times_organ_voxels_and_at_least_one():
    for r in D.table()["rows"]:
        assert r["k"] == max(1, int(r["p"] * r["nO"] + 1e-9)) and r["nO"] == len(D.phantom(r["gap"], r["i"]).organ)


def test_phantoms_regenerate_from_the_seed_identically():
    """Zufall nur über SplitMix64 und plattformfeste Geometrie: die eingefrorenen Masken sind das, was der Code erzeugt (auch unter Linux)."""
    for g in C.GAP_NAMES:
        for i in (0, 7, 19, 29):
            ph = M.make_phantom(g, i)
            stored = D.phantom(g, i)
            assert ph.tumor == stored.tumor and ph.organ == stored.organ and ph.gap_vox == pytest.approx(stored.gap_vox, abs=1e-6)


def test_splitmix64_reference_values():
    """Referenz: SplitMix64 mit Zufalls-Seed 0 gibt als erste Ausgabe 0xE220A8397B1DCDAF (veröffentlichter Testwert der Verfahrensbeschreibung)."""
    assert R.SplitMix64(0).next_u64() == 0xE220A8397B1DCDAF


def test_vectorised_stream_equals_the_class():
    rng = R.SplitMix64(1234)
    scalar = [rng.next_u64() for _ in range(50)]
    vec = ST.splitmix_stream(1234, 50)
    assert [int(v) for v in vec] == scalar


def test_bootstrap_is_reproducible_and_covers_the_mean():
    vals = [1.0, 2.0, 3.0, 4.0, 10.0, 2.5, 3.5, 1.5]
    groups = [i % 4 for i in range(8)]
    a = ST.mean_ci(vals, groups, seed=5)
    assert a == ST.mean_ci(vals, groups, seed=5)
    assert a[1] <= a[0] <= a[2] and a[0] == pytest.approx(sum(vals) / 8)
    assert ST.mean_ci([], [], seed=1) == (None, None, None)


def test_check_recomputes_a_sample_and_finds_no_deviation():
    """`--check` im Kleinen (3 Zeilen je Tabelle, 1 Kurve): Werte relativ 1e-5, Zählungen exakt."""
    assert MK.check_rows(D.table(), C.SEED_BASIS_1, 3) == []
    assert MK.check_rows(D.seeds2(), C.SEED_BASIS_2, 3) == []
    assert MK.check_curves(D.curves(), 1) == []


def test_check_detects_a_changed_value():
    t = D.table()
    rows = [dict(r) for r in t["rows"]]
    rows[0]["f_lpr"] = rows[0]["f_lpr"] * 1.01
    fake = dict(t, rows=rows[:1])
    assert any("f_lpr" in p for p in MK.check_rows(fake, C.SEED_BASIS_1, 1))


def test_json_files_use_lf_line_endings():
    for name in ("dpl_table.json", "dpl_seeds2.json", "dpl_curve.json"):
        raw = (ROOT / "data" / name).read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")
