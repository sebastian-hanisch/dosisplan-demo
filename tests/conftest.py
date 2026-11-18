"""Gemeinsame Fixtures: Pfade und die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import dpl_model as M  # noqa: E402
import dpl_solvers as S  # noqa: E402


@pytest.fixture(scope="session")
def matrix():
    return M.standard_matrix()


@pytest.fixture(scope="session")
def mini_h3():
    """6 x 6-Gitter, drei Winkel (0, 60, 120 Grad), sechs Strahlen je Winkel, 24 Körpervoxel; Tumor 8, 9, 11; Organ 6, 14, 17, 22, 23."""
    a = M.make_matrix(6, (0, 60, 120), 6)
    return S.Problem(a, (8, 9, 11), (6, 14, 17, 22, 23))


@pytest.fixture(scope="session")
def mini_h2():
    """Handinstanz H2 als Matrix: Zeile 0 = Tumorvoxel, Zeilen 1 bis 3 = Organvoxel, Zeile 4 = gesundes Gewebe; drei Strahlen."""
    rows = [[1, 1, 1], [0.75, 0.75, 0.5], [0.75, 0, 1], [0, 0.25, 0.5], [0, 0.75, 1]]
    return S.Problem(np.array(rows, dtype=float), (0,), (1, 2, 3))
