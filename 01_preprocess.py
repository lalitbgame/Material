"""Step 1 - clean each raw engineering stress-strain curve.

sanitise -> cut fracture tail -> remove spikes -> toe compensation (start at 0,0)
-> resample on a uniform grid -> Savitzky-Golay smoothing of the plastic zone.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import medfilt, savgol_filter

from common import (CLEAN_DIR, PLOT_DIR, PREPROCESS_LOG, load_raw, offset_stress,
                    raw_files, specimen_id)

SPIKE_FACTOR = 5.0     # x robust scatter of the median-filter residual
SPIKE_FLOOR = 1.0      # MPa, never flag residuals below this
FRACTURE_DROP = 0.10   # fraction of UTS, single-step drop that marks fracture
FRACTURE_LEVEL = 0.50  # fraction of UTS, stress level after UTS that marks fracture
DE_ELASTIC = 2e-4
DE_PLASTIC = 5e-4
SG_WINDOW, SG_ORDER = 11, 3
KNEE_TANGENT = 0.02   # smoothing starts once d(sigma)/d(eps) < this fraction of E


def sanitise(e, s):
    ok = np.isfinite(e) & np.isfinite(s)
    e, s = e[ok], s[ok]
    # keep only points that advance in strain (strictly increasing)
    prev_max = np.maximum.accumulate(np.r_[-np.inf, e[:-1]])
    keep = e > prev_max
    return e[keep], s[keep], int((~ok).sum()), int((~keep).sum())


def cut_fracture(e, s):
    uts = s.max()
    i_uts = int(s.argmax())
    for j in range(i_uts + 1, len(s)):
        if s[j] < FRACTURE_LEVEL * uts or s[j] - s[j - 1] < -FRACTURE_DROP * uts:
            return e[:j], s[:j]
    return e, s


def remove_spikes(e, s):
    r = s - medfilt(s, 11)
    r[:5] = r[-5:] = 0.0  # medfilt zero-pads the ends
    scatter = 1.4826 * np.median(np.abs(r[r != 0])) if np.any(r != 0) else 0.0
    bad = np.abs(r) > max(SPIKE_FACTOR * scatter, SPIKE_FLOOR)
    return e[~bad], s[~bad], int(bad.sum())


def elastic_fit(e, s):
    """Steepest well-correlated linear window between 20 % and 60 % of a provisional Rp0.2."""
    m = (s > 0.1 * s.max()) & (s < 0.4 * s.max())
    E0, b0 = np.polyfit(e[m], s[m], 1)
    _, rp02 = offset_stress(e, s, E0, 0.002 - b0 / E0, start=int(np.argmax(m)))
    zone = np.where((s >= 0.2 * rp02) & (s <= 0.6 * rp02) & (np.arange(len(s)) < s.argmax()))[0]
    width = max(5, len(zone) // 3)
    best = None
    for k in range(zone[0], zone[-1] - width + 2):
        sl = slice(k, k + width)
        E, b = np.polyfit(e[sl], s[sl], 1)
        r2 = np.corrcoef(e[sl], s[sl])[0, 1] ** 2
        cand = (r2 >= 0.999, E if r2 >= 0.999 else r2, E, b, k, k + width - 1, r2)
        if best is None or cand[:2] > best[:2]:
            best = cand
    _, _, E, b, i0, i1, r2 = best
    return E, b, i0, i1, r2


def resample_and_smooth(e, s, e_knee):
    f = PchipInterpolator(e, s)
    g_el = np.arange(0.0, e_knee, DE_ELASTIC)
    g_pl = np.arange(e_knee, e[-1], DE_PLASTIC)
    grid = np.r_[g_el, g_pl, e[-1]]
    grid = np.unique(grid)
    sg = f(grid)
    pl = grid >= e_knee
    if pl.sum() > SG_WINDOW:
        smooth = savgol_filter(sg[pl], SG_WINDOW, SG_ORDER, mode="interp")
        w = np.clip(np.arange(pl.sum()) / SG_WINDOW, 0.0, 1.0)  # blend in, no step at the knee
        sg[pl] = (1 - w) * sg[pl] + w * smooth
    sg[0] = 0.0
    i_uts = int(sg.argmax())
    sg[: i_uts + 1] = np.maximum.accumulate(sg[: i_uts + 1])
    return grid, sg


def process(path):
    sid, lot = specimen_id(path)
    e_raw, s_raw = load_raw(path)
    e, s, n_nan, n_backward = sanitise(e_raw, s_raw)
    raw_uts = s.max()
    e, s = cut_fracture(e, s)
    e, s, n_spike = remove_spikes(e, s)

    E, b, i0, i1, r2 = elastic_fit(e, s)
    e0 = -b / E                       # toe-corrected strain origin
    e_c = np.r_[0.0, e[i0:] - e0]
    s_c = np.r_[0.0, s[i0:]]
    keep = np.r_[True, np.diff(e_c) > 0]
    e_c, s_c = e_c[keep], s_c[keep]

    _, rp02 = offset_stress(e_c, s_c, E, 0.002, start=i1 - i0 + 1)
    fine = np.arange(rp02 / E, e_c[-1], 1e-5)
    flat = fine[PchipInterpolator(e_c, s_c).derivative()(fine) < KNEE_TANGENT * E]
    e_knee = max(flat[0] if len(flat) else 0.0, rp02 / E + 0.004)
    grid, sg = resample_and_smooth(e_c, s_c, e_knee)

    pd.DataFrame({"eng_strain": grid, "eng_stress_MPa": sg}).to_csv(
        CLEAN_DIR / f"{sid}.csv", index=False, float_format="%.7g")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, lim in zip(ax, (None, (0, 0.012))):
        a.plot(e_raw, s_raw, ".", ms=2.5, color="0.6", label="raw")
        a.plot(grid, sg, "-", lw=1.2, color="C0", label="cleaned")
        a.plot(grid[E * grid < 1.1 * raw_uts], E * grid[E * grid < 1.1 * raw_uts], ":", lw=0.8, color="C3", label=f"E fit = {E/1000:.1f} GPa")
        a.set_xlabel("Engineering strain [-]"); a.set_ylabel("Engineering stress [MPa]")
        if lim:
            a.set_xlim(*lim); a.set_ylim(0, 1.15 * raw_uts)
        a.grid(alpha=0.3)
    ax[0].legend(); ax[0].set_title(f"{sid}  raw vs cleaned"); ax[1].set_title("elastic / toe region")
    fig.tight_layout(); fig.savefig(PLOT_DIR / f"clean_{sid}.png", dpi=110); plt.close(fig)

    return dict(specimen=sid, lot=lot, file=path.name, E_fit_MPa=E, fit_R2=r2,
                fit_stress_lo=s[i0], fit_stress_hi=s[i1],
                toe_shift=e0, nan_rows=n_nan, backward_pts=n_backward, spikes=n_spike,
                raw_UTS=raw_uts, clean_UTS=sg.max(), raw_pts=len(e_raw), clean_pts=len(grid),
                fracture_strain=grid[-1])


def main():
    log = [process(p) for p in raw_files()]
    df = pd.DataFrame(log)
    df.to_csv(PREPROCESS_LOG, index=False, float_format="%.6g")
    print(df[["specimen", "E_fit_MPa", "fit_R2", "toe_shift", "nan_rows", "backward_pts",
              "spikes", "raw_UTS", "clean_UTS", "fracture_strain"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for sid in df.specimen:
        c = pd.read_csv(CLEAN_DIR / f"{sid}.csv")
        ax.plot(c.eng_strain, c.eng_stress_MPa, lw=1, label=sid)
    ax.set_xlabel("Engineering strain [-]"); ax.set_ylabel("Engineering stress [MPa]")
    ax.set_title("Al 6061-T651, 20 C - cleaned curves (n=%d)" % len(df))
    ax.grid(alpha=0.3); ax.legend(ncol=4, fontsize=7)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "cleaned_all.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
