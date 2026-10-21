"""Step 4 - write Abaqus *MATERIAL include files (mean, mean-3sigma, min)."""
from datetime import date

import numpy as np
import pandas as pd

from common import (DENSITY, E_NOMINAL, LEVELS, NU, OUT, PLASTIC_CURVES, PROPERTY_STATS)

THIN_TOL = 0.5  # MPa, max deviation allowed when dropping table rows


def thin(x, y, tol=THIN_TOL):
    """Greedily drop interior points while linear interpolation stays within tol of all points."""
    keep = list(range(len(x)))
    while len(keep) > 2:
        best, best_err = None, np.inf
        for k in range(1, len(keep) - 1):
            trial = keep[:k] + keep[k + 1:]
            err = np.abs(np.interp(x, x[trial], y[trial]) - y).max()
            if err < best_err:
                best, best_err = k, err
        if best_err > tol:
            break
        keep.pop(best)
    return np.array(keep)


def card_name(level):
    return "AL6061T651_" + level.upper().replace("-", "_M")


def main():
    curves = pd.read_csv(PLASTIC_CURVES)
    ps = pd.read_csv(PROPERTY_STATS).set_index("property")
    ep = curves.eps_plastic.to_numpy()

    for level in LEVELS:
        sig = curves[f"true_stress_{level}"].to_numpy()
        idx = thin(ep, sig)
        af = ps.loc["Af_plastic", level]
        lines = [
            f"** Al 6061-T651, 20 C, uniaxial tension, pooled n={int(ps.loc['UTS_MPa', 'n'])}, "
            f"level = {level}",
            f"** Generated {date.today()} by material_card/04_abaqus_card.py; units mm-N-s-MPa (t/mm^3)",
            "** Source: stress-strain-curves-of-aluminum-6061-t651-... (T_020_*.csv, lots A-I)",
            "** Plastic curve: true stress vs true plastic strain, Rp0.01 -> UTS, point-wise statistics",
            f"**   on a common plastic-strain range 0..{ep[-1]:.4f}; truncated at UTS (stress held "
            "constant beyond last row).",
            "** Engineering properties (pooled, this level):",
        ]
        for p in ("Rp0.01_MPa", "Rp0.2_MPa", "UTS_MPa", "Ag_uniform", "Af_total", "E_GPa"):
            lines.append(f"**   {p:<11s} = {ps.loc[p, level]:.4g}   "
                         f"(mean {ps.loc[p, 'mean']:.4g}, std {ps.loc[p, 'std']:.3g})")
        lines += [
            f"** E: nominal {E_NOMINAL:.0f} MPa used; measured E (mean {ps.loc['E_GPa', 'mean']:.1f} GPa)"
            " is affected by machine compliance.",
            f"*MATERIAL, NAME={card_name(level)}",
            "*DENSITY",
            f" {DENSITY:.3e}",
            "*ELASTIC",
            f" {E_NOMINAL:.1f}, {NU}",
            "*PLASTIC",
        ]
        lines += [f" {sig[i]:10.3f}, {ep[i]:.6e}" for i in idx]
        lines += [
            f"** Failure strain (engineering, plastic) at this level: {af:.4f}",
            "** For an optional ductile-damage criterion, convert to equivalent plastic strain at",
            "** fracture (needs triaxiality / necking correction) - not included in this card.",
        ]
        path = OUT / f"{card_name(level)}.inp"
        path.write_text("\n".join(lines) + "\n")
        print(f"{path.name}: {len(idx)} plastic rows, sigma_y = {sig[0]:.1f} MPa, "
              f"last = {sig[idx[-1]]:.1f} MPa @ eps_p {ep[idx[-1]]:.4f}")


if __name__ == "__main__":
    main()
