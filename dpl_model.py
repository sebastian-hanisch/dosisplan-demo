"""Phantom und Dosismatrix (erfundenes 2D-Modell, vereinfachte Strahlenphysik, kein klinischer Anspruch).

Körper = Scheibe auf einem 32 x 32-Gitter, 12 Einstrahlwinkel x 16 parallele Strahlen = 192 Strahlgewichte x >= 0, Dosis D = A x
(einheitenlos, relativ zur Tumor-Solldosis 1). A[Voxel, Strahl] = exp(-mu * Tiefe) * Gauß(Querabstand), Streukern bei 3 Sigma abgeschnitten.
Das ist eine Lehrbuchannahme, keine Dosisberechnung.

Zufall nur über den ganzzahligen SplitMix64 (`dpl_rng`). Die Geometrie der Phantome (Winkel, Ellipsen) rechnet mit `exact_cos`, damit die
Masken auf jeder Plattform dieselben sind; trotzdem liegen die Masken der 90 Phantome eingefroren in `data/dpl_table.json`, die App liest sie dort.
Herkunft: Modell und Regeln sind für diese Demo neu geschrieben; die Vorlage war eine Messreihe außerhalb des Repos (nicht importiert).
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import dpl_constants as C
from dpl_rng import SplitMix64, exact_cos


def cos_sin(angle: float) -> tuple[float, float]:
    """Kosinus und Sinus für |angle| bis etwa 6,3, auf jeder Plattform bitgleich (sin x = cos(pi/2 - x))."""
    return exact_cos(angle), exact_cos(math.pi / 2.0 - angle)


def voxel_centers(g: int = C.GRID):
    xs = -1.0 + (np.arange(g) + 0.5) * 2.0 / g
    return np.meshgrid(xs, xs, indexing="xy")


def body_mask(g: int = C.GRID) -> np.ndarray:
    x, y = voxel_centers(g)
    return (x ** 2 + y ** 2) <= C.BODY_RADIUS ** 2


def make_matrix(g: int, angles_deg, n_beams: int, mu: float = C.MU, sigma_rel: float = C.SIGMA_REL) -> np.ndarray:
    """Dosismatrix (Körpervoxel x Strahlen); Spalte = Winkelindex * n_beams + Strahlindex."""
    xg, yg = voxel_centers(g)
    body = body_mask(g)
    x, y = xg[body], yg[body]
    omega = 2.0 * C.BODY_RADIUS / n_beams
    sig = sigma_rel * omega
    centers = (np.arange(n_beams) + 0.5) * omega - C.BODY_RADIUS
    blocks = []
    for ang in angles_deg:
        c, s = cos_sin(math.radians(ang))
        u = x * c + y * s
        lat = -x * s + y * c
        u0 = -np.sqrt(np.maximum(C.BODY_RADIUS ** 2 - lat ** 2, 0.0))
        depth = np.maximum(u - u0, 0.0)
        dd = lat[:, None] - centers[None, :]
        w = np.exp(-(dd ** 2) / (2 * sig ** 2))
        w[np.abs(dd) > C.CUT_SIGMA * sig] = 0.0
        blocks.append(w * np.exp(-mu * depth)[:, None])
    return np.hstack(blocks)


@lru_cache(maxsize=1)
def standard_matrix() -> np.ndarray:
    """192 Strahlen auf dem 32 x 32-Gitter (732 Körpervoxel); einmal je Prozess gebaut."""
    a = make_matrix(C.GRID, C.ANGLES_DEG, C.N_BEAMS)
    a.setflags(write=False)
    return a


def beam_scalar(x: float, y: float, angle_deg: float, beam: int, n_beams: int) -> float:
    """Ein Matrixeintrag per Skalarformel (Gegenprobe der Tests und Handwerte): derselbe Ausdruck ohne Vektoren."""
    c, s = cos_sin(math.radians(angle_deg))
    u = x * c + y * s
    lat = -x * s + y * c
    u0 = 0.0 if abs(lat) >= C.BODY_RADIUS else -math.sqrt(C.BODY_RADIUS ** 2 - lat ** 2)
    depth = max(u - u0, 0.0)
    omega = 2.0 * C.BODY_RADIUS / n_beams
    mid = -C.BODY_RADIUS + (beam + 0.5) * omega
    sig = C.SIGMA_REL * omega
    dd = lat - mid
    w = 0.0 if abs(dd) > C.CUT_SIGMA * sig else math.exp(-dd * dd / (2 * sig * sig))
    return w * math.exp(-C.MU * depth)


# ---------------------------------------------------------------- Phantome

@dataclass(frozen=True)
class Phantom:
    """Indizes (in der Reihenfolge der Körpervoxel) von Tumor und Organ; der Rest ist gesundes Gewebe."""
    gap: str
    i: int
    tumor: tuple
    organ: tuple
    gap_vox: float

    @property
    def n_organ(self) -> int:
        return len(self.organ)


def phantom_seed(base: int, gap_index: int, i: int) -> int:
    """Zufalls-Seed des Phantoms i der Abstandsklasse gap_index (1 = nah, 2 = mittel, 3 = fern) in der Menge mit Basis `base`."""
    return base * 1_000_000 + gap_index * 1_000 + i


def _uniform(rng: SplitMix64, lo: float, hi: float) -> float:
    return lo + (hi - lo) * rng.random()


def _ellipse(xg, yg, cx, cy, a, b, phi):
    c, s = cos_sin(phi)
    dx, dy = xg - cx, yg - cy
    u = dx * c + dy * s
    v = -dx * s + dy * c
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def make_phantom(gap: str, i: int, base: int = C.SEED_BASIS_1, g: int = C.GRID, min_vox: int = 12) -> Phantom:
    """Tumorscheibe (Radius 0,16 bis 0,28) mit einer Organellipse daneben (mindestens 12 Voxel). Der kleinste Abstand der Voxelmitten von
    Tumor und Organ wird mit einem Mindestmaß gezogen (nah 1,0 bis 1,5 Voxelbreiten, mittel 2 bis 3, fern 4 bis 6); der tatsächliche Abstand liegt darüber, weil er auf dem Gitter springt (erste Menge: nah 1,41 bis 2,24, mittel 2,24 bis 3,61, fern 4,12 bis 6,40)."""
    gi = C.GAP_NAMES.index(gap) + 1
    lo_gap, hi_gap = C.GAP_RANGES[gap]
    rng = SplitMix64(phantom_seed(base, gi, i))
    xg, yg = voxel_centers(g)
    body = body_mask(g)
    rb = C.BODY_RADIUS
    for _ in range(4000):
        r = _uniform(rng, 0.16, 0.28)
        rad = _uniform(rng, 0.0, rb - r - 0.30)
        ang = _uniform(rng, 0.0, 2 * math.pi)
        ca, sa = cos_sin(ang)
        cx, cy = rad * ca, rad * sa
        a = _uniform(rng, 0.20, 0.32)
        b = _uniform(rng, 0.09, 0.16)
        phi = _uniform(rng, 0.0, math.pi)
        gt = _uniform(rng, lo_gap, hi_gap)
        psi = _uniform(rng, 0.0, 2 * math.pi)
        cp, sp = cos_sin(psi)
        tum = ((xg - cx) ** 2 + (yg - cy) ** 2) <= r ** 2
        if (tum & ~body).any() or tum.sum() < min_vox:
            continue
        ty, tx = np.nonzero(tum)
        found = None
        for dist in np.arange(0.05, 1.2, 0.01):
            org = _ellipse(xg, yg, cx + dist * cp, cy + dist * sp, a, b, phi) & ~tum
            if org.sum() < min_vox:
                continue
            oy, ox = np.nonzero(org)
            dmin = float(np.min(np.sqrt((tx[:, None] - ox[None, :]) ** 2 + (ty[:, None] - oy[None, :]) ** 2)))
            if dmin >= gt:
                found = (org, dmin)
                break
        if found is None:
            continue
        org, dmin = found
        if (org & ~body).any():
            continue
        tb, ob = tum[body], org[body]
        return Phantom(gap, i, tuple(int(v) for v in np.nonzero(tb)[0]), tuple(int(v) for v in np.nonzero(ob)[0]), dmin)
    raise RuntimeError("kein Phantom gefunden")


def body_to_grid(values, g: int = C.GRID) -> np.ndarray:
    """Werte je Körpervoxel auf das Gitter legen (außerhalb des Körpers NaN), für die Dosisbilder."""
    out = np.full((g, g), np.nan)
    out[body_mask(g)] = values
    return out


def label_grid(phantom: Phantom, g: int = C.GRID) -> np.ndarray:
    """0 = gesundes Gewebe, 1 = Tumor, 2 = Organ; außerhalb des Körpers NaN."""
    lab = np.zeros(int(body_mask(g).sum()))
    lab[list(phantom.tumor)] = 1
    lab[list(phantom.organ)] = 2
    return body_to_grid(lab, g)
