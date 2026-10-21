"""Step 3 - point-wise statistics: mean, mean-3sigma and min curves.

True stress vs plastic strain (Rp0.01 -> UTS, for the card) and engineering
stress vs strain (0 -> smallest failure strain, for reporting).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from common import (CLEAN_DIR, ENG_CURVES, PLASTIC_CURVES, PLOT_DIR, PREPROCESS_LOG,
                    offset_stress)

N_GRID = 60
EPS_P_FIRST = 1e-4  # first non-zero grid point (Rp0.01 offset)


def true_plastic(row):
    """True stress vs plastic strain from Rp0.01 up to UTS; first point at eps_p = 0."""
    c = pd.read_csv(CLEAN_DIR / f"{row.specimen}.csv")
    e, s = c.eng_strain.to_numpy(), c.eng_stress_MPa.to_numpy()
    E = row.E_fit_MPa
    i_uts = int(s.argmax())
    e, s = e[: i_uts + 1], s[: i_uts + 1]
    start = int(np.argmax(s >= row.fit_stress_hi))
    e_y, s_y = offset_stress(e, s, E, 1e-4, start)

    st = s * (1 + e)
    ep = np.log1p(e) - st / E
    past = e > e_y
    ep = np.r_[0.0, ep[past]]
    st = np.r_[s_y * (1 + e_y), st[past]]
    keep = ep > np.maximum.accumulate(np.r_[-np.inf, ep[:-1]])  # strictly increasing eps_p
    return ep[keep], st[keep]


def stats_on_grid(curves, grid):
    Y = np.array([c(grid) for c in curves])
    m, sd = Y.mean(0), Y.std(0, ddof=1)
    return Y, {"mean": m, "std": sd, "mean-3sigma": m - 3 * sd, "min": Y.min(0), "max": Y.max(0)}


def enforce_hardening(name, y):
    mono = np.maximum.accumulate(y)
    dev = np.abs(mono - y).max()
    if dev > 0:
        print(f"  {name}: non-decreasing fix changed up to {dev:.3f} MPa")
    return mono


def main():
    log = pd.read_csv(PREPROCESS_LOG)

    # --- true stress vs plastic strain (card domain) ---
    tp = {r.specimen: true_plastic(r) for r in log.itertuples()}
    ep_uts_min = min(ep[-1] for ep, _ in tp.values())
    grid = np.r_[0.0, np.geomspace(EPS_P_FIRST, ep_uts_min, N_GRID - 1)]
    Y, st = stats_on_grid([PchipInterpolator(ep, s) for ep, s in tp.values()], grid)
    print(f"common plastic-strain range: 0 .. {ep_uts_min:.4f} "
          f"(limited by {min(tp, key=lambda k: tp[k][0][-1])})")
    for lvl in ("mean", "mean-3sigma", "min"):
        st[lvl] = enforce_hardening(lvl, st[lvl])
    out = pd.DataFrame({"eps_plastic": grid, **{f"true_stress_{k}": v for k, v in st.items()}})
    out.to_csv(PLASTIC_CURVES, index=False, float_format="%.6g")
    pd.DataFrame(Y.T, columns=list(tp), index=grid).rename_axis("eps_plastic").to_csv(
        PLASTIC_CURVES.with_name("true_plastic_specimens.csv"), float_format="%.6g")

    # --- engineering domain (reporting) ---
    eng = {}
    for r in log.itertuples():
        c = pd.read_csv(CLEAN_DIR / f"{r.specimen}.csv")
        eng[r.specimen] = (c.eng_strain.to_numpy(), c.eng_stress_MPa.to_numpy())
    af_min = min(e[-1] for e, _ in eng.values())
    eg = np.linspace(0.0, af_min, 400)
    Ye, se = stats_on_grid([lambda x, e=e, s=s: np.interp(x, e, s) for e, s in eng.values()], eg)
    pd.DataFrame({"eng_strain": eg, **{f"eng_stress_{k}": v for k, v in se.items()}}).to_csv(
        ENG_CURVES, index=False, float_format="%.6g")

    print(out.iloc[[0, 1, 10, 20, 30, 40, 50, -1]].to_string(index=False,
                                                            float_format=lambda v: f"{v:.4g}"))
    below = (st["min"] < st["mean-3sigma"]).sum()
    print(f"grid points where min < mean-3sigma: {below}/{len(grid)}")

    # --- plots ---
    styles = {"mean": ("k", "-"), "mean-3sigma": ("C3", "--"), "min": ("C0", ":")}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for e, s in eng.values():
        ax[0].plot(e, s, color="0.75", lw=0.8)
    for k, (c, ls) in styles.items():
        ax[0].plot(eg, se[k], color=c, ls=ls, lw=2, label=k)
    ax[0].set(xlabel="Engineering strain [-]", ylabel="Engineering stress [MPa]",
              title="Engineering stress-strain (to min failure strain)")
    for ep, s in tp.values():
        ax[1].plot(ep, s, color="0.75", lw=0.8)
    for k, (c, ls) in styles.items():
        ax[1].plot(grid, st[k], color=c, ls=ls, lw=2, label=k)
    ax[1].axvline(ep_uts_min, color="0.4", lw=0.8, ls="-.")
    ax[1].set(xlabel="True plastic strain [-]", ylabel="True stress [MPa]",
              title="True stress vs plastic strain (Rp0.01 to UTS)")
    for a in ax:
        a.grid(alpha=0.3); a.legend()
    fig.suptitle("Al 6061-T651, 20 C - pooled statistics, n=%d" % len(log))
    fig.tight_layout(); fig.savefig(PLOT_DIR / "statistical_curves.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
