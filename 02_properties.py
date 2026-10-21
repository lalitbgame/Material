"""Step 2 - E, Rp0.01 (yield), Rp0.2 (proof), UTS, uniform elongation, failure strain.

Per-specimen values from the cleaned curves + pooled statistics and Grubbs outlier check.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import (CLEAN_DIR, PLOT_DIR, PREPROCESS_LOG, PROPERTIES, PROPERTY_STATS,
                    offset_stress)

PROPS = ["E_GPa", "Rp0.01_MPa", "Rp0.2_MPa", "UTS_MPa", "Ag_uniform", "Af_total", "Af_plastic"]


def grubbs_outlier(x, alpha=0.05):
    """Two-sided Grubbs test; returns index of the outlier or None."""
    n = len(x)
    z = np.abs(x - x.mean()) / x.std(ddof=1)
    t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    g_crit = (n - 1) / np.sqrt(n) * np.sqrt(t**2 / (n - 2 + t**2))
    i = int(z.argmax())
    return i if z[i] > g_crit else None


def specimen_props(row):
    c = pd.read_csv(CLEAN_DIR / f"{row.specimen}.csv")
    e, s = c.eng_strain.to_numpy(), c.eng_stress_MPa.to_numpy()
    E = row.E_fit_MPa
    start = int(np.argmax(s >= row.fit_stress_hi))  # past the elastic fit window
    _, rp001 = offset_stress(e, s, E, 1e-4, start)
    _, rp02 = offset_stress(e, s, E, 2e-3, start)
    i_uts = int(s.argmax())
    return {"specimen": row.specimen, "lot": row.lot, "E_GPa": E / 1000,
            "Rp0.01_MPa": rp001, "Rp0.2_MPa": rp02, "UTS_MPa": s[i_uts],
            "Ag_uniform": e[i_uts], "Af_total": e[-1], "Af_plastic": e[-1] - s[-1] / E}


def main():
    log = pd.read_csv(PREPROCESS_LOG)
    df = pd.DataFrame([specimen_props(r) for r in log.itertuples()])
    df.to_csv(PROPERTIES, index=False, float_format="%.6g")

    rows = []
    for p in PROPS:
        x = df[p].to_numpy()
        m, sd = x.mean(), x.std(ddof=1)
        out = grubbs_outlier(x)
        rows.append({"property": p, "n": len(x), "mean": m, "std": sd, "CoV_%": 100 * sd / m,
                     "min": x.min(), "max": x.max(), "mean-3sigma": m - 3 * sd,
                     "grubbs_outlier": "" if out is None else df.specimen[out]})
    st = pd.DataFrame(rows)
    st.to_csv(PROPERTY_STATS, index=False, float_format="%.6g")

    pd.set_option("display.width", 160)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print()
    print(st.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    lots = sorted(df.lot.unique())
    for ax, p in zip(axes.flat, PROPS):
        for k, lot in enumerate(lots):
            y = df[df.lot == lot][p]
            ax.plot([k] * len(y), y, "o", ms=6, color=f"C{k % 10}")
        r = st[st.property == p].iloc[0]
        ax.axhline(r["mean"], color="k", lw=1, label="mean")
        ax.axhline(r["mean-3sigma"], color="C3", lw=1, ls="--", label="mean-3σ")
        ax.axhline(r["min"], color="0.5", lw=1, ls=":", label="min")
        ax.set_xticks(range(len(lots)), lots); ax.set_title(p); ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("Al 6061-T651, 20 C - properties by lot (n=%d)" % len(df))
    fig.tight_layout(); fig.savefig(PLOT_DIR / "properties_by_lot.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
