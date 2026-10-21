"""Verification - checks on cleaned data, curves and a 1-D J2 return-mapping replay of each card.

Abaqus is not available on this machine, so each card's *ELASTIC/*PLASTIC data is replayed
through a uniaxial radial-return integrator (isotropic hardening, piecewise-linear table,
flat beyond the last row) and compared with the statistical curve it came from.
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CLEAN_DIR, LEVELS, OUT, PLASTIC_CURVES, PLOT_DIR, PREPROCESS_LOG, load_raw,
                    raw_files, specimen_id)

card_name = __import__("04_abaqus_card").card_name


def read_card(path):
    txt = path.read_text()
    E, nu = map(float, re.search(r"\*ELASTIC\n\s*([^\n]+)", txt).group(1).split(","))
    block = txt.split("*PLASTIC\n")[1]
    rows = [list(map(float, l.split(","))) for l in block.splitlines() if l and not l.startswith("**")]
    t = np.array(rows)
    return E, t[:, 0], t[:, 1]


def replay(E, sig_tab, ep_tab, eps_max, n=4000):
    """Strain-driven 1-D radial return with piecewise-linear isotropic hardening."""
    hard = lambda p: np.interp(p, ep_tab, sig_tab)
    slope = lambda p: np.interp(p, ep_tab[1:], np.diff(sig_tab) / np.diff(ep_tab), right=0.0)
    eps, sig, ep = np.linspace(0, eps_max, n), np.zeros(n), 0.0
    eps_p = np.zeros(n)
    for k in range(1, n):
        trial = E * (eps[k] - ep)
        f = abs(trial) - hard(ep)
        if f > 0:
            dp = 0.0
            for _ in range(50):  # Newton on the consistency condition
                r = abs(trial) - E * dp - hard(ep + dp)
                if abs(r) < 1e-9:
                    break
                dp += r / (E + slope(ep + dp))
            ep += dp
            trial = np.sign(trial) * hard(ep)
        sig[k], eps_p[k] = trial, ep
    return eps, sig, eps_p


def main():
    ok = True
    log = pd.read_csv(PREPROCESS_LOG)

    print("1) cleaned data checks")
    for p in raw_files():
        sid, _ = specimen_id(p)
        c = pd.read_csv(CLEAN_DIR / f"{sid}.csv")
        e, s = c.eng_strain.to_numpy(), c.eng_stress_MPa.to_numpy()
        e_raw, s_raw = load_raw(p)
        uts_err = abs(s.max() - np.nanmax(s_raw)) / np.nanmax(s_raw)
        checks = {"starts at (0,0)": e[0] == 0 and s[0] == 0,
                  "strain strictly increasing": np.all(np.diff(e) > 0),
                  "UTS within 1%": uts_err < 0.01}
        bad = [k for k, v in checks.items() if not v]
        ok &= not bad
        print(f"   {sid}: {'OK' if not bad else 'FAIL ' + ', '.join(bad)}  (UTS diff {100*uts_err:.3f} %)")

    print("2) statistical curve checks")
    cv = pd.read_csv(PLASTIC_CURVES)
    for lvl in LEVELS:
        mono = np.all(np.diff(cv[f"true_stress_{lvl}"]) >= 0)
        ok &= mono
        print(f"   {lvl}: non-decreasing {'OK' if mono else 'FAIL'}")
    order = np.all(cv["true_stress_mean"] >= cv["true_stress_mean-3sigma"])
    ok &= order
    print(f"   mean >= mean-3sigma everywhere: {'OK' if order else 'FAIL'}")
    print(f"   min >= mean-3sigma at {(cv['true_stress_min'] >= cv['true_stress_mean-3sigma']).sum()}"
          f"/{len(cv)} points")

    print("3) card replay (1-D J2 radial return)")
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, lvl in zip(("k", "C3", "C0"), LEVELS):
        E, st, ep = read_card(OUT / f"{card_name(lvl)}.inp")
        eps, sig, eps_p = replay(E, st, ep, eps_max=ep[-1] + st[-1] / E + 0.01)
        ref = cv[f"true_stress_{lvl}"].to_numpy()
        got = np.interp(cv.eps_plastic, eps_p[eps_p > 0], sig[eps_p > 0])
        mask = cv.eps_plastic > 2e-4  # replay resolution near first yield
        err = np.abs(got - ref)[mask].max()
        ok &= err < 1.0
        print(f"   {lvl}: {len(st)} rows, max |replay - statistical curve| = {err:.3f} MPa "
              f"{'OK' if err < 1.0 else 'FAIL'}")
        # back to engineering for comparison with test data
        ax.plot(np.expm1(eps), sig / np.exp(eps), color=c, lw=2, label=f"card replay: {lvl}")
    for r in log.itertuples():
        cc = pd.read_csv(CLEAN_DIR / f"{r.specimen}.csv")
        e, s = cc.eng_strain.to_numpy(), cc.eng_stress_MPa.to_numpy()
        e_corr = e - s / r.E_fit_MPa + s / E  # remove compliance: re-base elastic strain on nominal E
        ax.plot(e_corr, s, color="0.75", lw=0.8)
    ax.set(xlabel="Engineering strain (elastic part on nominal E) [-]",
           ylabel="Engineering stress [MPa]", title="Abaqus card replay vs test curves", xlim=(0, 0.13))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(PLOT_DIR / "card_replay.png", dpi=120); plt.close(fig)

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
