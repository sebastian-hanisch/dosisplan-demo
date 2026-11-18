"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt auf gültige Werte."""

import pytest

import dpl_constants as C
import dpl_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid():
    for name, p in C.PRESETS.items():
        assert p["gap"] in C.GAP_NAMES and 0 <= p["seed"] < C.N_PHANTOMS and 0 <= p["combo"] < len(C.COMBOS) and p["cap"] in C.CAPS and p["view"] in C.VIEWS, name


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    assert {P.PRESET_KEYS[k]: v for k, v in first.items()} == C.DEFAULTS


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_preset_names_are_distinct_and_no_preset_is_a_duplicate_of_another():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


@pytest.mark.parametrize("key,raw,expected", [
    ("seed_input", "12", 12), ("seed_input", "99", 29), ("seed_input", "-4", 0), ("seed_input", "abc", None),
    ("gap_select", "fern", "fern"), ("gap_select", "weit", None),
    ("cap_select", "0.8", 0.8), ("cap_select", "0.7", None), ("combo_select", "7", 2),
    ("view_select", C.VIEW_CVAR, C.VIEW_CVAR), ("view_select", "9 unbekannt", None),
])
def test_permalink_values_are_limited_to_valid_ones(key, raw, expected):
    assert P._coerce(P.SETTING_SPECS[key], raw) == expected


def test_combos_give_the_documented_k():
    import dpl_rows as R
    assert [R.k_of(0.25, n) for n in (3, 4, 8, 28)] == [1, 1, 2, 7]               # floor(p n), mindestens 1
    assert [R.k_of(0.10, n) for n in (3, 10, 19, 30, 36)] == [1, 1, 1, 3, 3]
