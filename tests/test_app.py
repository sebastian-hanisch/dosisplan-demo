"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# ---------------------------------------------------------------- Zusatztests dieser Demo (der Anfang der Datei ist die Vorlage)

import pytest  # noqa: E402

import dpl_constants as C  # noqa: E402


def _texts(at) -> str:
    return " ".join(m.value for m in at.markdown) + " " + " ".join(e.value for e in list(at.success) + list(at.info) + list(at.warning))


def test_app_loads_without_exception_and_shows_the_notice():
    at = _fresh_app()
    text = _texts(at)
    assert "Anwendungsbeispiel, keine Beratung" in text and "kein klinischer anspruch" in text.lower()
    assert "Kernfragen" in text and len(at.success) + len(at.info) + len(at.warning) >= 3



@pytest.mark.parametrize("view", list(C.VIEWS))
def test_every_experiment_view_renders_for_three_cells(view):
    at = _fresh_app()
    for gap, seed, combo in (("nah", 8, 1), ("fern", 0, 0), ("mittel", 29, 2)):
        at.selectbox(key="gap_select").set_value(gap)
        at.slider(key="seed_input").set_value(seed)
        at.selectbox(key="combo_select").set_value(combo)
        at.segmented_control(key="view_select").set_value(view).run(timeout=TIMEOUT)
        assert not at.exception, (view, gap, seed, combo, [str(e) for e in at.exception])


def test_every_regler_value_at_its_edges_does_not_raise():
    at = _fresh_app()
    for gap in C.GAP_NAMES:
        for cap in C.CAPS:
            for seed in (0, C.N_PHANTOMS - 1):
                at.selectbox(key="gap_select").set_value(gap)
                at.select_slider(key="cap_select").set_value(cap)
                at.slider(key="seed_input").set_value(seed)
                at.run(timeout=TIMEOUT)
                assert not at.exception, (gap, cap, seed, [str(e) for e in at.exception])


def test_every_preset_renders_in_its_view():
    for name, p in C.PRESETS.items():
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.segmented_control(key="view_select").value == p["view"]
        assert at.selectbox(key="gap_select").value == p["gap"] and at.slider(key="seed_input").value == p["seed"]


def test_the_hint_is_visible_in_every_view():
    at = _fresh_app()
    for view in C.VIEWS:
        at.segmented_control(key="view_select").set_value(view).run(timeout=TIMEOUT)
        assert "Anwendungsbeispiel, keine Beratung" in _texts(at)
