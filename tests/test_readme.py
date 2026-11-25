"""README und Texte: Hinweis „Anwendungsbeispiel, keine Beratung“, keine Planungswörter, kein klinischer Anspruch, App-Text und README gleich, Quellen nur wie gelesen."""

import re
from pathlib import Path

import dpl_constants as C
import dpl_texts as T

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
FOOTER_PREFIX = "Diese Demo ist Teil des Portfolios von"          # der Portfolio-Footer wird von set_demo_footer.py erzeugt und gehört nicht zum Text der Demo

PLANNING_WORDS = [r"\bPlan\b", r"\bPlans\b", "Einzelplan", "Vormessung", "Phase 0", r"\bStück", "ZUSCHNITT", r"\bE\d{1,2}\b", r"PLAN-", "geplant", r"\bLinie\b", "Saatwert", r"\bträgt\b", r"\btragen\b",
                  "läuft lokal"]


def _text_files():
    files = [ROOT / "README.md", ROOT / "app.py"] + sorted(ROOT.glob("dpl_*.py")) + sorted((ROOT / "tools").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    return [f for f in files if f.name != "test_readme.py"]


def _lines_without_footer(path):
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if FOOTER_PREFIX not in ln]


def test_no_planning_words_in_readme_app_texts_docstrings_and_tests():
    hits = []
    for f in _text_files():
        for n, ln in enumerate(_lines_without_footer(f), start=1):
            for pat in PLANNING_WORDS:
                if re.search(pat, ln):
                    hits.append(f"{f.name}:{n}: {pat}: {ln.strip()[:100]}")
    assert not hits, "\n".join(hits)


def test_hint_in_readme_and_app_texts():
    assert README.count("**Anwendungsbeispiel, keine Beratung.**") == 1
    assert "Kein klinischer Anspruch" in README and "Behandlungs-, Diagnose- oder Planungsempfehlung" in README
    assert T.HINT in T.INTRO and T.HINT_LONG in T.INTRO
    assert T.HINT_LONG in README


def test_no_clinical_vocabulary_or_recommendations():
    """Kein „Patient“, kein „Behandlungsplan“, keine Empfehlung außer der ausdrücklichen Verneinung im Hinweis."""
    for f in _text_files():
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"\bPatient", text), f.name
        assert "Behandlungsplan" not in text and "Bestrahlungsplan" not in text, f.name
    for pat in (r"[Ee]mpfehl", r"\bsollte\b", r"\bman sollte\b"):
        for m in re.finditer(pat, README):
            assert "Behandlungs-, Diagnose- oder Planungsempfehlung" in README[max(0, m.start() - 40): m.end() + 60]


def test_app_assumption_table_is_the_same_text_as_in_readme():
    assert T.ASSUMPTIONS_MARKDOWN.strip() in README


def test_readme_does_not_call_the_website_integration_open():
    assert not re.search(r"Einbindung[^.\n]*offen", README) and "geplant" not in README


def test_readme_has_the_live_link_and_no_todo():
    assert README.splitlines().count("**[→ Demo live ausprobieren](https://sebastianhanisch-dosisplan-demo.streamlit.app/)**") == 1
    assert "TODO" not in README


def test_every_named_test_file_and_data_file_exists():
    for name in re.findall(r"`(tests/test_[a-z_0-9]+\.py)`", README) + re.findall(r"`(data/[a-z_0-9]+\.json)`", README) + re.findall(r"`(tools/[a-z_0-9]+\.py)`", README):
        assert (ROOT / name).exists(), name
    for name in re.findall(r"`(test_claim_[a-z_0-9]+)`", README):
        assert any(f"def {name}(" in p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("test_claims.py")), name


def test_sources_are_cited_only_as_read():
    assert "Romeijn u. a. (2003, 2006) sind nicht gelesen" in README and "genannt in" in README
    assert "Keine Neuheitsaussage gegenüber der Literatur" in README.replace("**Keine Neuheitsaussage gegenüber der Literatur.**", "Keine Neuheitsaussage gegenüber der Literatur")
    assert "Kosten der Nichtkonvexität" in README and "keine Größe dieser Demo" in README        # nur als ausdrücklich ausgeschlossene Größe
    assert "passt zu Craft" not in README and "passt zu Craft" not in T.BOUNDARIES
    assert "stützt oder widerlegt es nicht" in README


def test_neighbour_demo_links_are_github_links_to_the_documented_repos():
    for repo in ("lp-dualitaet-demo", "zwei-kriterien-demo", "hart-gegen-weich-demo", "aufloesung-messorte-demo", "scenario-approach-demo", "cvar-demo", "nutrition-demo", "lp-schranke-demo"):
        assert f"https://github.com/sebastian-hanisch/{repo})" in README, repo


def test_decimal_commas_and_german_quotes_in_the_readme_numbers():
    assert not re.search(r"\b\d+\.\d+ %", README)                         # Dezimalpunkt vor Prozent
    assert "Zufalls-Seed" in README and "Zufallsseed" not in README


def test_regler_values_in_the_texts_are_the_constants():
    assert "1,2, 1,0, 0,8 oder 0,6" in README and C.CAPS == (1.2, 1.0, 0.8, 0.6)
    assert "(0,5; 25 %), (0,3; 25 %), (0,3; 10 %)" in README and C.COMBOS == ((0.5, 0.25), (0.3, 0.25), (0.3, 0.10))
