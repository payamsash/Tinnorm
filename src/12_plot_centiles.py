"""
12_plot_centiles.py

Centile plots for normative model predictions.

Shows the normative trajectory (5th / 50th / 95th centile bands) as a function
of a continuous covariate (age or PTA4_mean), with individual subject Z-scores
overlaid as scatter points.

  data_mode="train" → controls as scatter (sanity check: should lie within bands)
  data_mode="test"  → tinnitus as scatter (key comparison)

Supported modalities: power, aperiodic, regional, global, graph.
Connectivity modalities require conn_mode ("coh", "pli", "plv").

Run from src/:  python 12_plot_centiles.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pcntoolkit import NormativeModel, NormData, plot_centiles_advanced


# ── Paths & constants ─────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
HM_DIR      = TINNORM_DIR / "harmonized"
MODELS_DIR  = TINNORM_DIR / "models"
SAVING_DIR  = TINNORM_DIR / "plots" / "centiles"

DEMOGRAPHIC_COLS = ["subject_id", "SITE", "group", "age", "sex", "PTA4_mean"]
COVARIATES       = ["age", "sex", "PTA4_mean"]   # model covariates (not x-axis)
BATCH_EFFECTS    = ["SITE"]


# ── Path helpers ──────────────────────────────────────────────────────────────

def _suffix(mode: str, conn_mode: str | None) -> str:
    return f"{mode}_{conn_mode}" if conn_mode else mode


def _hm_path(preproc_level: int, space: str,
             mode: str, conn_mode: str | None) -> Path:
    return (HM_DIR / f"preproc_{preproc_level}" / space
            / f"{_suffix(mode, conn_mode)}_hm.csv")


def _model_path(preproc_level: int, space: str,
                mode: str, conn_mode: str | None) -> Path:
    return (MODELS_DIR / f"preproc_{preproc_level}" / space
            / _suffix(mode, conn_mode) / "full_model")


# ── Column-name resolution ────────────────────────────────────────────────────

def _resolve_brain_labels(
        feature_cols: list,
        brain_labels: list,
        frequency: str | None,
        mode: str,
        conn_mode: str | None,
) -> list:
    """
    Return the actual column names in the HM file for the requested ROIs and band.

    Discovers suffix conventions automatically so all modalities are handled:
      power          : {roi}_{band}
      aperiodic      : {roi}_exponent  /  {roi}_offset          (frequency=None)
      regional/global: {roi}_{band}_{conn_mode}
      graph          : {roi}_{band}_clustering / {roi}_{band}_local_eff
    """
    result = []
    for bl in brain_labels:
        candidates = [c for c in feature_cols if c.startswith(f"{bl}_")]

        if mode == "aperiodic":
            candidates = [c for c in candidates
                          if c in (f"{bl}_exponent", f"{bl}_offset")]
        elif frequency:
            candidates = [c for c in candidates if f"_{frequency}" in c]
            if conn_mode and mode != "graph":
                candidates = [c for c in candidates
                              if c.endswith(f"_{conn_mode}")]

        result.extend(candidates)

    return sorted(set(result))


# ── Main plotting function ────────────────────────────────────────────────────

def run_centile_plot(
        brain_labels: list,
        frequency: str | None,
        covar: str,
        data_mode: str,
        mode: str,
        space: str,
        preproc_level: int,
        conn_mode: str | None = None,
) -> None:
    """
    Load NormativeModel and HM data; call plot_centiles_advanced; save PDFs.

    Parameters
    ----------
    brain_labels : list[str]
        ROI names without frequency suffix, e.g. ["transversetemporal-rh"].
    frequency : str or None
        Freq band string ("alpha_1", "theta", …). None for aperiodic.
    covar : str
        Covariate for the x-axis: "age" or "PTA4_mean".
    data_mode : "train" | "test"
        Which group appears as scatter: controls (train) or tinnitus (test).
    mode : str
        Modality: "power", "aperiodic", "regional", "global", "graph".
    space : str
        "source" or "sensor".
    preproc_level : int
    conn_mode : str or None
        "coh", "pli", "plv" for connectivity modalities; None otherwise.
    """
    hm_path    = _hm_path(preproc_level, space, mode, conn_mode)
    model_path = _model_path(preproc_level, space, mode, conn_mode)

    if not hm_path.exists():
        print(f"  HM file not found: {hm_path}")
        return
    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        return

    # ── Load & align data ─────────────────────────────────────────────────────
    df = pd.read_csv(hm_path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

    df_master = pd.read_csv("../material/master_clean.csv")
    df["subject_id"]        = df["subject_id"].astype(str)
    df_master["subject_id"] = df_master["subject_id"].astype(str)
    df = df.merge(df_master[["subject_id"]], on="subject_id", how="inner")

    # ── Separate features from demographics ───────────────────────────────────
    present_demo  = [c for c in DEMOGRAPHIC_COLS if c in df.columns]
    response_cols = [c for c in df.columns if c not in present_demo]

    # ── Resolve which columns correspond to the requested brain labels + band ─
    bls = _resolve_brain_labels(response_cols, brain_labels, frequency, mode, conn_mode)
    if not bls:
        band_str = frequency if frequency else "all"
        print(f"  No matching columns for brain_labels={brain_labels}, "
              f"frequency={band_str} — skipping.")
        return

    # ── Build NormData objects ────────────────────────────────────────────────
    norm_kw = dict(
        covariates    = COVARIATES,
        batch_effects = BATCH_EFFECTS,
        response_vars = response_cols,
        subject_ids   = "subject_id",
    )
    norm_ctrl = NormData.from_dataframe(name="train",
                                        dataframe=df.query("group == 0"),
                                        **norm_kw)
    norm_tin  = NormData.from_dataframe(name="test",
                                        dataframe=df.query("group == 1"),
                                        **norm_kw)

    scatter_data = norm_ctrl if data_mode == "train" else norm_tin

    # ── Load model and plot ───────────────────────────────────────────────────
    model = NormativeModel.load(str(model_path))

    figs, re_vars = plot_centiles_advanced(
        model,
        centiles        = [0.05, 0.5, 0.95],
        covariate       = covar,
        batch_effects   = {"SITE": df["SITE"].unique().tolist()},
        scatter_data    = scatter_data,
        show_other_data = False,
        harmonize_data  = True,
        show_yhat       = True,
        brain_labels    = bls,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    suf     = _suffix(mode, conn_mode)
    out_dir = SAVING_DIR / f"preproc{preproc_level}_{space}_{suf}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fig, re_var in zip(figs, re_vars):
        fname = out_dir / f"{re_var}_{covar}_{data_mode}.pdf"
        fig.savefig(fname, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fname}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    preproc_level = 2
    space         = "source"

    # Covariates to use as x-axis (one run per covariate)
    covariates = ["age", "PTA4_mean"]

    # Scatter both groups in separate runs
    data_modes = ["train", "test"]   # train = controls, test = tinnitus

    # Representative ROIs: primary auditory, auditory association, DMN, motor
    brain_labels = [
        "transversetemporal-lh",  "transversetemporal-rh",
        "superiortemporal-lh",    "superiortemporal-rh",
        "supramarginal-lh",       "supramarginal-rh",
        "precuneus-lh",           "precuneus-rh",
        "precentral-lh",          "precentral-rh",
    ]

    # (mode, conn_mode, freq_bands_to_plot)
    # Use None for frequency when the modality has no band structure (aperiodic)
    configs = [
        ("power",     None,  ["alpha_1", "alpha_2", "beta_1", "theta"]),
        ("aperiodic", None,  [None]),
        ("regional",  "coh", ["alpha_1", "alpha_2", "beta_1"]),
        ("global",    "coh", ["alpha_1", "alpha_2", "beta_1"]),
    ]

    for mode, conn_mode, frequencies in configs:
        for frequency in frequencies:
            for covar in covariates:
                for data_mode in data_modes:
                    suf      = _suffix(mode, conn_mode)
                    band_str = frequency if frequency else "all"
                    print(f"\n── {suf} | {band_str} | covar={covar} | {data_mode} ──")
                    try:
                        run_centile_plot(
                            brain_labels  = brain_labels,
                            frequency     = frequency,
                            covar         = covar,
                            data_mode     = data_mode,
                            mode          = mode,
                            space         = space,
                            preproc_level = preproc_level,
                            conn_mode     = conn_mode,
                        )
                    except Exception as e:
                        print(f"  Failed: {e}")


## NOTE: this will only work if we save the models, which we usually dont, but maybe at the very end!