"""Shared paths and helpers for the Al 6061-T651 material-card pipeline."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT.parent / (
    "stress-strain-curves-of-aluminum-6061-t651-from-9-lots-at-6-temperatures-"
    "under-uniaxial-and-plain-strain-tension")
OUT = ROOT / "output"
CLEAN_DIR = OUT / "cleaned"
CURVE_DIR = OUT / "curves"
PLOT_DIR = OUT / "plots"
for d in (OUT, CLEAN_DIR, CURVE_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

PREPROCESS_LOG = OUT / "preprocess_log.csv"
PROPERTIES = OUT / "properties.csv"
PROPERTY_STATS = OUT / "property_stats.csv"
PLASTIC_CURVES = CURVE_DIR / "true_plastic_stats.csv"
ENG_CURVES = CURVE_DIR / "engineering_stats.csv"

LEVELS = ("mean", "mean-3sigma", "min")

# Nominal elastic constants used in the card (measured E is compliance-affected)
E_NOMINAL = 68900.0  # MPa
NU = 0.33
DENSITY = 2.70e-9    # t/mm^3


def raw_files():
    return sorted(RAW_DIR.glob("T_*.csv"))


def specimen_id(path):
    """'T_020_A_1_001_022_03.csv' -> ('A_1', 'A')."""
    parts = Path(path).stem.split("_")
    return f"{parts[2]}_{parts[3]}", parts[2]


def load_raw(path):
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    return d[:, 0], d[:, 1]


def offset_stress(e, s, E, offset, start=0):
    """Stress where the curve crosses the line s = E*(e - offset).

    Searches from index `start` (past the elastic fit window) so noise about the
    elastic line cannot produce a spurious crossing. Returns (strain, stress).
    """
    g = s - E * (e - offset)
    idx = np.where(g[start:] <= 0)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    j = start + idx[0]
    if j == 0:
        return e[0], s[0]
    t = g[j - 1] / (g[j - 1] - g[j])
    return e[j - 1] + t * (e[j] - e[j - 1]), s[j - 1] + t * (s[j] - s[j - 1])
