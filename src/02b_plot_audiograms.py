"""
02b_plot_audiograms.py

Plot audiograms for all subjects in master_clean.csv using audiometric data
from the TIDE-Compiled-Version2 Excel file.

Run order: after 02_plot_demographics.py, before 03_convert_to_bids.py.
Run from src/: python 02b_plot_audiograms.py

Column format in Excel:
  ARE_<freq>  Air conduction Right Ear
  ALE_<freq>  Air conduction Left Ear

Saves (to tinnorm_dir/plots/audiograms/):
  audiogram_group.pdf         — standard freqs, both ears, TIN vs CTRL
  audiogram_by_site.pdf       — standard freqs, 2 rows × 7 cols (ear × site)
  audiogram_hf.pdf            — extended HF freqs, both ears, TIN vs CTRL
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Constants & paths ─────────────────────────────────────────────────────────
CTRL_COLOR = "#1f77b4"
TIN_COLOR  = "#d62728"
PALETTE    = {"Control": CTRL_COLOR, "Tinnitus": TIN_COLOR}

tinnorm_dir  = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
saving_dir   = tinnorm_dir / "plots" / "audiograms"
AUDIOGRAM_XLS = Path("../material/TIDE-Compiled-Version2_3-26-2026.xlsx")
os.makedirs(saving_dir, exist_ok=True)

# Standard clinical audiogram and extended HF frequencies (Hz)
STANDARD_FREQS = [125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000]
HF_FREQS       = [10000, 11200, 12500, 14000, 16000]
ALL_AIR_FREQS  = STANDARD_FREQS + HF_FREQS

# ── Load & merge ──────────────────────────────────────────────────────────────
df_master = pd.read_csv("../material/master_clean.csv")
df_master["subject_id"] = df_master["subject_id"].astype(str)

df_excel = pd.read_excel(AUDIOGRAM_XLS, sheet_name="Sheet1")
df_excel["Subject ID"] = df_excel["Subject ID"].astype(str).str.strip()

df_merged = df_master[["subject_id", "site", "group"]].merge(
    df_excel, left_on="subject_id", right_on="Subject ID", how="inner"
)
print(f"Subjects matched for audiogram analysis: {len(df_merged)}")
print(f"  Controls: {(df_merged['group']==0).sum()}, "
      f"Tinnitus: {(df_merged['group']==1).sum()}")

# ── Frequency string parser ───────────────────────────────────────────────────
def _parse_freq(freq_str: str) -> int:
    s = str(freq_str).strip()
    return int(float(s[:-1]) * 1000) if s.endswith("k") else int(float(s))


# ── Build long-format dataframe ───────────────────────────────────────────────
def _to_long(df, col_prefix_re, ear_label, keep_freqs):
    """Melt one ear's columns to long format, keeping only keep_freqs."""
    cols = [c for c in df.columns if c.startswith(col_prefix_re)]
    id_cols = ["subject_id", "site", "group"]
    dfs = []
    for col in cols:
        freq_str = col.split("_", 1)[1]
        freq_hz  = _parse_freq(freq_str)
        if freq_hz not in keep_freqs:
            continue
        tmp = df[id_cols + [col]].copy()
        tmp.rename(columns={col: "threshold"}, inplace=True)
        tmp["ear"]  = ear_label
        tmp["freq"] = freq_hz
        dfs.append(tmp)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# Air conduction (standard + HF); ARE_=Right Ear, ALE_=Left Ear
df_air = pd.concat([
    _to_long(df_merged, "ARE_", "Right", ALL_AIR_FREQS),
    _to_long(df_merged, "ALE_", "Left",  ALL_AIR_FREQS),
], ignore_index=True)
df_air["threshold"]   = pd.to_numeric(df_air["threshold"], errors="coerce")
df_air["group_label"] = df_air["group"].map({0: "Control", 1: "Tinnitus"})
df_air = df_air.dropna(subset=["threshold"])

print(f"  Air-conduction rows: {len(df_air)}")

# ── Audiogram plotting helper ─────────────────────────────────────────────────
def _freq_labels(freqs):
    return [f"{f//1000:g}k" if f >= 1000 else str(f) for f in sorted(freqs)]


def plot_audiometry(data, hue="group_label", row=None, col="ear",
                    row_order=None, col_order=None, palette=PALETTE,
                    ylim=(90, -10), markersize=5, err_lw=2.5,
                    height=3.0, aspect=1.5, filename="plot.pdf",
                    xticks_first_col_only=False):
    """
    Audiogram FacetGrid following the convention in the Tinception reference script.
    X-axis (frequency) is placed at top; Y-axis (threshold) is inverted (0 dB at top).
    Left ear is displayed in the left panel (col_order default: ["Left", "Right"]).
    """
    if col == "ear" and col_order is None:
        col_order = ["Left", "Right"]

    facet_kws = dict(height=height, aspect=aspect, legend_out=False, despine=False)
    if row is not None:
        facet_kws["row"] = row
        if row_order is not None:
            facet_kws["row_order"] = row_order
    if col is not None:
        facet_kws["col"] = col
        if col_order is not None:
            facet_kws["col_order"] = col_order

    # Compute the global frequency order BEFORE creating the grid so it can be
    # passed as `order` to pointplot. Without this, each facet builds its own
    # categorical x-axis from only the data it sees — sites missing a frequency
    # (e.g. Zuerich has no 3 kHz) get shifted positions and wrong tick labels.
    freqs = sorted(data["freq"].unique())
    tick_labs = _freq_labels(freqs)

    g = sns.FacetGrid(data=data, **facet_kws)
    g.map_dataframe(
        sns.pointplot, x="freq", y="threshold", hue=hue,
        palette=palette, errorbar="se",
        order=freqs,
        markersize=markersize, linewidth=2,
        capsize=0.25, err_kws={"linewidth": err_lw},
    )

    for ax in g.axes.flat:
        if not ax.has_data():
            continue
        ax.invert_yaxis()
        ax.set_ylim(ylim)
        ax.grid(True, axis="both", linewidth=0.6, alpha=0.3)

        # Move x-axis to top (audiogram convention)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", rotation=45)
        ax.set_xticks(range(len(freqs)))
        ax.set_xticklabels(tick_labs, ha="left", fontsize=8)

        ax.spines[["right", "bottom"]].set_visible(False)
        for s in ["top", "left"]:
            ax.spines[s].set_linewidth(1.5)

        # Remove per-axis legends added by pointplot
        if ax.get_legend():
            ax.get_legend().remove()

    # Hide x-tick labels on all columns except the first
    if xticks_first_col_only and g.axes.ndim == 2:
        for r in range(g.axes.shape[0]):
            for c in range(0, g.axes.shape[1]):
                g.axes[r, c].set_xticklabels([])
                g.axes[r, c].tick_params(axis="x", which="both", length=0)
                g.axes[r, c].set_xlabel("")

    g.set_ylabels("Hearing threshold (dB HL)")
    g.tight_layout()
    g.figure.subplots_adjust(top=0.88)

    fpath = saving_dir / filename
    g.figure.savefig(fpath, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(g.figure)
    print(f"  Saved → {fpath}")


# ── Plot 1: Standard audiogram — group comparison, both ears ──────────────────
print("\n── Plot 1: Group audiogram (standard frequencies, both ears) ──")
df_std = df_air[df_air["freq"].isin(STANDARD_FREQS)]
plot_audiometry(df_std, hue="group_label", col="ear",
                filename="audiogram_group.pdf")

# ── Plot 2: Audiogram per site — 2 rows × 7 cols (ear × site) ────────────────
print("\n── Plot 2: Audiogram by site (standard frequencies, 2 rows × 7 cols) ──")
site_order = sorted(df_std["site"].unique())
plot_audiometry(
    df_std, hue="group_label", row="ear", col="site",
    row_order=["Left", "Right"], col_order=site_order,
    ylim=(50, -10), xticks_first_col_only=True,
    markersize=3, err_lw=2.0, height=2.2, aspect=1.4,
    filename="audiogram_by_site.pdf",
)

# ── Plot 3: Extended HF audiogram — group comparison, both ears ───────────────
print("\n── Plot 3: Extended HF audiogram (125 Hz – 16 kHz, both ears) ──")
plot_audiometry(df_air, hue="group_label", col="ear",
                ylim=(100, -10), filename="audiogram_hf.pdf")

print(f"\nDone. Audiograms saved to: {saving_dir}")
