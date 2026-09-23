"""Figures describing the data and the cohorts (supplementary material)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "ISIC2017": "#4C72B0", "ISIC2018": "#55A868", "ISIC2019": "#C44E52",
    "ISIC2020": "#8172B2", "Fitzpatrick17k": "#CCB974", "DDI": "#64B5CD",
    "PAD-UFES-20": "#937860",
}
TONE_COLORS = {"I-II": "#F2D3B6", "III-IV": "#C68642", "V-VI": "#5C3317"}


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def _save(fig, out_dir, name):
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / f"{name}.png")
        fig.savefig(p / f"{name}.pdf")
    return fig


# --------------------------------------------------------------------------
def fig_attrition(log: pd.DataFrame, out_dir=None, name="cohort_attrition"):
    """Attrition at each eligibility step."""
    set_style()
    steps = log.dropna(subset=["n_after"]).reset_index(drop=True)
    k = len(steps)
    fig, ax = plt.subplots(figsize=(9.6, 1.5 * k + 0.8))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.5 * k)

    box_w, box_h, x0 = 4.4, 0.78, 0.25
    for i, r in steps.iterrows():
        yc = 1.5 * k - 0.75 - 1.5 * i
        ax.add_patch(plt.Rectangle((x0, yc - box_h / 2), box_w, box_h,
                                   fc="#EDF2F8", ec="#3E5468", lw=1.2, zorder=2))
        ax.text(x0 + box_w / 2, yc + 0.15, str(r["step"]), ha="center",
                va="center", fontsize=9, weight="bold", zorder=3)
        ax.text(x0 + box_w / 2, yc - 0.18, f"n = {int(r['n_after']):,}",
                ha="center", va="center", fontsize=10.5, zorder=3)

        if i < k - 1:
            ax.annotate("", xy=(x0 + box_w / 2, yc - box_h / 2 - 0.70),
                        xytext=(x0 + box_w / 2, yc - box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#3E5468", lw=1.3))

        nxt = steps.iloc[i + 1] if i < k - 1 else None
        if nxt is not None and pd.notna(nxt["n_removed"]) and nxt["n_removed"] > 0:
            ymid = yc - box_h / 2 - 0.35
            ax.annotate("", xy=(x0 + box_w + 1.05, ymid),
                        xytext=(x0 + box_w / 2, ymid),
                        arrowprops=dict(arrowstyle="-|>", color="#A6423F", lw=1.1))
            txt = f"excluded  n = {int(nxt['n_removed']):,}\n{nxt['note']}"
            ax.text(x0 + box_w + 1.15, ymid, txt, fontsize=7.6, va="center",
                    ha="left", color="#7A2E2E",
                    bbox=dict(boxstyle="round,pad=0.35", fc="#FBEDEC",
                              ec="#D9B3B0", lw=0.7))
    ax.set_title("Cohort construction", loc="left", pad=12)
    return _save(fig, out_dir, name)


def fig_class_by_source(df: pd.DataFrame, out_dir=None, name="class_by_source"):
    set_style()
    ct = pd.crosstab(df["source"], df["dx"])
    order = [c for c in ["MEL", "BCC", "SCC", "AK", "NV", "BKL", "DF", "VASC", "UNK", "OTHER"]
             if c in ct.columns]
    ct = ct[order].sort_values(order[0] if order else ct.columns[0], ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ct.plot(kind="barh", stacked=True, ax=axes[0], colormap="tab20", width=0.72)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("images (log scale)")
    axes[0].set_ylabel("")
    axes[0].set_title("Class counts per release",
                      loc="left", fontsize=9.5)
    axes[0].legend(ncol=5, fontsize=7, loc="lower right")

    frac = ct.div(ct.sum(axis=1), axis=0)
    frac.plot(kind="barh", stacked=True, ax=axes[1], colormap="tab20",
              width=0.72, legend=False)
    axes[1].set_xlabel("proportion of release")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    axes[1].set_title("Class proportions per release", loc="left", fontsize=9.5)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig_verification_bias(vb: pd.DataFrame, ps: pd.DataFrame,
                          out_dir=None, name="verification_bias"):
    """Histopathology confirmation by outcome, and its effect on prevalence."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    v = vb.dropna(subset=["verification_gap_pp"]).sort_values("verification_gap_pp")
    x = np.arange(len(v))
    axes[0].barh(x - 0.19, v["malignant"], height=0.36, color="#C44E52", label="malignant")
    axes[0].barh(x + 0.19, v["benign"], height=0.36, color="#4C72B0", label="benign")
    axes[0].set_yticks(x); axes[0].set_yticklabels(v["source"])
    axes[0].set_xlabel("% of class with histopathology confirmation")
    axes[0].set_xlim(0, 105)
    for i, r in enumerate(v.itertuples(index=False)):
        axes[0].text(102, i, f"gap {r.verification_gap_pp:+.0f} pp",
                     va="center", fontsize=8, color="#7A2E2E")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Histopathology confirmation by outcome", loc="left")

    p = ps.dropna(subset=["prevalence_histo_only"]).sort_values("prevalence_all")
    x2 = np.arange(len(p))
    axes[1].barh(x2 - 0.19, p["prevalence_all"] * 100, height=0.36,
                 color="#999", label="all confirmation methods")
    axes[1].barh(x2 + 0.19, p["prevalence_histo_only"] * 100, height=0.36,
                 color="#C44E52", label="histopathology only")
    axes[1].set_yticks(x2); axes[1].set_yticklabels(p["source"])
    axes[1].set_xlabel("apparent malignancy prevalence (%)")
    axes[1].legend(loc="lower right")
    axes[1].set_title("Prevalence before and after the histopathology restriction", loc="left")
    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig_overlap(matrix: pd.DataFrame, out_dir=None, name="release_overlap"):
    set_style()
    m = matrix.copy()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(np.log10(m.values + 1), cmap="rocket_r" if "rocket_r" in
                   plt.colormaps() else "magma_r")
    ax.set_xticks(range(len(m.columns))); ax.set_xticklabels(m.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index)
    for i in range(len(m.index)):
        for j in range(len(m.columns)):
            v = m.values[i, j]
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=7.5,
                    color="white" if np.log10(v + 1) > np.log10(m.values.max() + 1) * 0.55 else "black")
    ax.set_title("Shared image identifiers between sources", loc="left")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="log10(shared ids + 1)")
    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig_tone_coverage(tc: pd.DataFrame, out_dir=None, name="skin_tone_coverage"):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    d = tc.sort_values("n", ascending=True)
    axes[0].barh(d["source"], d["fitz_labelled_%"], color="#55A868")
    axes[0].set_xlabel("% of images with a Fitzpatrick label")
    axes[0].set_xlim(0, 105)
    for i, v in enumerate(d["fitz_labelled_%"]):
        axes[0].text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=8)
    axes[0].set_title("Images with a Fitzpatrick label", loc="left")

    grps = ["I-II", "III-IV", "V-VI"]
    bottom = np.zeros(len(d))
    for g in grps:
        vals = d[f"{g}_n"].fillna(0).values
        axes[1].barh(d["source"], vals, left=bottom, color=TONE_COLORS[g],
                     edgecolor="white", label=f"FST {g}")
        bottom += vals
    axes[1].set_xlabel("images with skin-tone stratum assigned")
    axes[1].legend(title="stratum")
    axes[1].set_title("Images per skin-tone group", loc="left")
    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig_power(feas: pd.DataFrame, out_dir=None, name="precision_by_stratum"):
    """Approximate AUC interval half-width and calibration slope SE per stratum."""
    set_style()
    d = feas.dropna(subset=["AUC_95CI_halfwidth"]).copy()
    d["label"] = d.apply(lambda r: f"{r['source']} · {r.get('fst_group', 'all')}", axis=1)
    d = d.sort_values("AUC_95CI_halfwidth")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = ["#C44E52" if v > 0.10 else "#DD8452" if v > 0.05 else "#55A868"
              for v in d["AUC_95CI_halfwidth"]]
    axes[0].barh(d["label"], d["AUC_95CI_halfwidth"], color=colors)
    axes[0].axvline(0.05, ls="--", c="grey", lw=0.9)
    axes[0].set_xlabel("half-width of 95% CI for AUC (assumed AUC 0.85)")
    for i, (v, n) in enumerate(zip(d["AUC_95CI_halfwidth"], d["malignant"])):
        axes[0].text(v + 0.002, i, f"{v:.3f}  ({int(n)} events)", va="center", fontsize=7.5)
    axes[0].set_title("AUC precision per stratum", loc="left")

    ev = np.arange(5, 400)
    axes[1].plot(ev, [1.0 / np.sqrt(e * 1.5) for e in ev], color="#4C72B0", lw=1.8)
    axes[1].axhline(0.15, ls="--", c="grey", lw=0.9)
    axes[1].text(300, 0.16, "SE 0.15", fontsize=8, color="grey")
    for _, r in d.iterrows():
        if r["malignant"] >= 2:
            axes[1].scatter(r["malignant"], 1.0 / np.sqrt(r["malignant"] * 1.5),
                            s=34, zorder=3, color="#C44E52")
            axes[1].annotate(r["label"], (r["malignant"],
                                          1.0 / np.sqrt(r["malignant"] * 1.5)),
                             textcoords="offset points", xytext=(6, 4), fontsize=7)
    axes[1].set_xlabel("malignant events in stratum")
    axes[1].set_ylabel("approx. SE of calibration slope")
    axes[1].set_title("Calibration slope precision by number of events", loc="left")
    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig_missingness(df: pd.DataFrame, out_dir=None, name="metadata_missingness"):
    set_style()
    cols = ["dx", "label_malignant", "confirm_histo", "patient_id", "lesion_id",
            "fitzpatrick", "age_approx", "sex", "anatom_site"]
    m = (df.groupby("source")[cols].apply(lambda g: g.isna().mean() * 100))
    fig, ax = plt.subplots(figsize=(8.4, 0.55 * len(m) + 2.2))
    im = ax.imshow(m.values, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha="right")
    ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 55 else "black")
    ax.grid(False)
    ax.set_title("Metadata missingness by source (%)", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="% missing")
    fig.tight_layout()
    return _save(fig, out_dir, name)
