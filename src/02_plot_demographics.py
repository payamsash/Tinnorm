import os
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency, spearmanr

# -------------------------------------------------------------------------
# CONSTANTS & PATHS
# -------------------------------------------------------------------------
TIN_COLOR  = "#C99700"
CTRL_COLOR = "#1f77b4"
SEX_COLORS = {"Male": "#2e86ab", "Female": "#e84855"}

tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
features_dir = tinnorm_dir / "features"
saving_dir = tinnorm_dir / "plots" / "demographics"
os.makedirs(saving_dir, exist_ok=True)

# -------------------------------------------------------------------------
# LOAD & PREPARE DATA
# -------------------------------------------------------------------------
df = pd.read_csv("../material/master_new.csv")
cols_to_keep = ["subject_id", "site", "sex", "age", "group", "PTA4_mean", "PTA4_HF", "THI", "TFI"]
df_plot = df[cols_to_keep].copy()
df_plot["sex"]   = df_plot["sex"].map({1: "Male", 2: "Female"})
df_plot["group"] = df_plot["group"].map({0: "Control", 1: "Tinnitus"})

# -------------------------------------------------------------------------
# SUBJECT MATCHING & DATA INTEGRITY CHECKS
# -------------------------------------------------------------------------
check_text = "power_sensor_preproc_1.zip"
ids_list = set(
    fname.stem[4:9] for fname in features_dir.iterdir()
    if str(fname).endswith(check_text)
)
ids_df = set(df_plot["subject_id"].astype(str))

only_in_list = sorted(ids_list - ids_df)
only_in_df   = sorted(ids_df - ids_list)
common_ids   = ids_list & ids_df

# -------------------------------------------------------------------------
# STATISTICAL TESTS
# -------------------------------------------------------------------------

def _pval_badge(p):
    if p < 0.001:
        return p, "***", "#dc2626"
    if p < 0.01:
        return p, "**", "#ea580c"
    if p < 0.05:
        return p, "*", "#d97706"
    return p, "ns", "#16a34a"


def _fmt_p(p):
    return f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001"


# Between-group tests (Tinnitus vs Control)
continuous_cols = ["age", "PTA4_mean", "PTA4_HF"]
group_stats = {}
for col in continuous_cols:
    tin  = df_plot.loc[df_plot["group"] == "Tinnitus", col].dropna()
    ctrl = df_plot.loc[df_plot["group"] == "Control",  col].dropna()
    stat, p = mannwhitneyu(tin, ctrl, alternative="two-sided")
    group_stats[col] = {"U": stat, "p": p, **dict(zip(("sig", "color"), _pval_badge(p)[1:]))}

sex_ct = pd.crosstab(df_plot["group"], df_plot["sex"])
chi2_sex_group, p_sex_group, _, _ = chi2_contingency(sex_ct)
group_stats["sex"] = {"chi2": chi2_sex_group, "p": p_sex_group,
                      **dict(zip(("sig", "color"), _pval_badge(p_sex_group)[1:]))}

# Between-site tests (Kruskal-Wallis)
site_stats = {}
for col in continuous_cols:
    groups = [df_plot.loc[df_plot["site"] == s, col].dropna() for s in df_plot["site"].unique()]
    groups = [g for g in groups if len(g) >= 3]
    stat, p = kruskal(*groups)
    site_stats[col] = {"H": stat, "p": p, **dict(zip(("sig", "color"), _pval_badge(p)[1:]))}

sex_ct_site = pd.crosstab(df_plot["site"], df_plot["sex"])
chi2_sex_site, p_sex_site, _, _ = chi2_contingency(sex_ct_site)
site_stats["sex"] = {"chi2": chi2_sex_site, "p": p_sex_site,
                     **dict(zip(("sig", "color"), _pval_badge(p_sex_site)[1:]))}

# THI–TFI agreement (tinnitus only)
tin_df = df_plot[df_plot["group"] == "Tinnitus"].dropna(subset=["THI", "TFI"])
rho_thi_tfi, p_thi_tfi = spearmanr(tin_df["THI"], tin_df["TFI"])

# Per-variable descriptive stats
def _desc(df_in, col):
    return f"{df_in[col].mean():.1f} ± {df_in[col].std():.1f}"

# -------------------------------------------------------------------------
# CONSOLE OUTPUT
# -------------------------------------------------------------------------
counts = pd.crosstab(df_plot["site"], df_plot["group"])
general_summary  = df_plot.groupby(["site", "group"])[continuous_cols].agg(["mean", "std"]).round(2)
tinnitus_summary = df_plot[df_plot["group"] == "Tinnitus"].groupby("site")[["THI", "TFI"]].agg(["mean", "std"]).round(2)

print(f"Subjects only in features ({len(only_in_list)}): {only_in_list or 'None'}")
print(f"Subjects only in CSV     ({len(only_in_df)}):   {only_in_df or 'None'}")
print("\n" + "="*20 + " SITE × GROUP COUNTS " + "="*20)
print(counts)
print("\n" + "="*20 + " DEMOGRAPHICS SUMMARY " + "="*20)
print(general_summary)
print("\n" + "="*15 + " TINNITUS SEVERITY SUMMARY " + "="*15)
print(tinnitus_summary)
print()
for col, s in group_stats.items():
    print(f"  Group diff [{col}]: {_fmt_p(s['p'])} {s['sig']}")
print(f"\n  THI–TFI Spearman rho = {rho_thi_tfi:.3f}, {_fmt_p(p_thi_tfi)}")

# -------------------------------------------------------------------------
# FILTER TO MATCHED SUBJECTS
# -------------------------------------------------------------------------
df_plot = df_plot[df_plot["subject_id"].astype(str).isin(common_ids)]

# -------------------------------------------------------------------------
# HTML HELPER FUNCTIONS
# -------------------------------------------------------------------------

def _badge_html(p, sig, color):
    return (f'<span style="display:inline-block;padding:2px 8px;border-radius:12px;'
            f'background:{color}22;color:{color};font-weight:700;font-size:11px;'
            f'border:1px solid {color}66;">{sig}</span>')


def _stat_card(label, value, sub=""):
    return (f'<div class="stat-card"><div class="stat-lbl">{label}</div>'
            f'<div class="stat-val">{value}</div>'
            f'{"<div class=stat-sub>" + sub + "</div>" if sub else ""}</div>')


def _row_bg(i):
    return "#f8fafc" if i % 2 == 0 else "#ffffff"


def _build_group_test_table():
    rows = ""
    labels = {"age": "Age (years)", "PTA4_mean": "PTA4 Mean (dB HL)",
               "PTA4_HF": "PTA4 HF (dB HL)", "sex": "Sex (M/F)"}
    for i, (col, s) in enumerate(group_stats.items()):
        lbl = labels.get(col, col)
        if col == "sex":
            test_str = f"χ²({sex_ct.shape[0]-1}) = {s['chi2']:.2f}"
            tin_val  = f"{(df_plot['sex']=='Male').sum()} M / {(df_plot['sex']=='Female').sum()} F"
            ctrl_val = "—"
        else:
            test_str = f"U = {s['U']:.0f}"
            tin_val  = _desc(df_plot[df_plot["group"]=="Tinnitus"], col)
            ctrl_val = _desc(df_plot[df_plot["group"]=="Control"],  col)
        badge = _badge_html(s["p"], s["sig"], s["color"])
        rows += (f'<tr style="background:{_row_bg(i)}"><td>{lbl}</td>'
                 f'<td>{tin_val}</td><td>{ctrl_val}</td>'
                 f'<td>{test_str}</td><td>{_fmt_p(s["p"])}</td><td>{badge}</td></tr>')
    return rows


def _build_site_test_table():
    labels = {"age": "Age (years)", "PTA4_mean": "PTA4 Mean (dB HL)",
               "PTA4_HF": "PTA4 HF (dB HL)", "sex": "Sex"}
    rows = ""
    for i, (col, s) in enumerate(site_stats.items()):
        lbl = labels.get(col, col)
        if col == "sex":
            test_str = f"χ²({sex_ct_site.shape[0]-1}) = {s['chi2']:.2f}"
        else:
            test_str = f"H = {s['H']:.2f}"
        badge = _badge_html(s["p"], s["sig"], s["color"])
        rows += (f'<tr style="background:{_row_bg(i)}"><td>{lbl}</td>'
                 f'<td>{test_str}</td><td>{_fmt_p(s["p"])}</td><td>{badge}</td></tr>')
    return rows


def _build_counts_table():
    header = "<tr><th>Site</th>"
    for g in counts.columns:
        header += f"<th>{g}</th>"
    header += "<th>Total</th></tr>"
    body = ""
    for i, (site, row) in enumerate(counts.iterrows()):
        body += f'<tr style="background:{_row_bg(i)}"><td><strong>{site}</strong></td>'
        for v in row:
            body += f"<td>{v}</td>"
        body += f"<td><strong>{row.sum()}</strong></td></tr>"
    body += f'<tr style="background:#f1f5f9;font-weight:700"><td>Total</td>'
    for v in counts.sum():
        body += f"<td>{v}</td>"
    body += f"<td>{counts.sum().sum()}</td></tr>"
    return header + body


def _build_tinnitus_severity_table():
    cols_sev = ["THI", "TFI"]
    tdf = df_plot[df_plot["group"] == "Tinnitus"]
    header = "<tr><th>Site</th>"
    for c in cols_sev:
        header += f"<th>{c} Mean ± SD</th><th>{c} Median</th><th>{c} Range</th>"
    header += "<th>N</th></tr>"
    body = ""
    for i, site in enumerate(sorted(tdf["site"].unique())):
        sg = tdf[tdf["site"] == site]
        body += f'<tr style="background:{_row_bg(i)}"><td><strong>{site}</strong></td>'
        for c in cols_sev:
            vals = sg[c].dropna()
            if len(vals):
                body += (f"<td>{vals.mean():.1f} ± {vals.std():.1f}</td>"
                         f"<td>{vals.median():.1f}</td>"
                         f"<td>{vals.min():.0f}–{vals.max():.0f}</td>")
            else:
                body += "<td>—</td><td>—</td><td>—</td>"
        body += f"<td>{len(sg)}</td></tr>"
    # Total row
    body += f'<tr style="background:#f1f5f9;font-weight:700"><td>All sites</td>'
    for c in cols_sev:
        vals = tdf[c].dropna()
        body += (f"<td>{vals.mean():.1f} ± {vals.std():.1f}</td>"
                 f"<td>{vals.median():.1f}</td>"
                 f"<td>{vals.min():.0f}–{vals.max():.0f}</td>")
    body += f"<td>{len(tdf)}</td></tr>"
    return header + body


# -------------------------------------------------------------------------
# GENERATE HTML
# -------------------------------------------------------------------------
n_tin  = (df_plot["group"] == "Tinnitus").sum()
n_ctrl = (df_plot["group"] == "Control").sum()

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tinnorm — Demographics & Data Quality Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f1f5f9; color: #1e293b; line-height: 1.6;
    padding: 32px 24px;
  }}
  .page-header {{
    max-width: 1100px; margin: 0 auto 28px;
    border-left: 5px solid {TIN_COLOR}; padding-left: 16px;
  }}
  .page-header h1 {{ font-size: 26px; font-weight: 700; color: #0f172a; }}
  .page-header p  {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
             max-width: 1100px; margin: 0 auto 20px; }}
  .grid-1 {{ max-width: 1100px; margin: 0 auto 20px; }}
  .card {{
    background: #fff; border-radius: 10px; padding: 22px 26px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07); border: 1px solid #e2e8f0;
  }}
  .card h2 {{ font-size: 15px; font-weight: 700; color: #0f172a;
              margin-bottom: 14px; padding-bottom: 8px;
              border-bottom: 2px solid #f1f5f9; }}
  .card h3 {{ font-size: 12px; font-weight: 600; color: #64748b;
              text-transform: uppercase; letter-spacing: .6px;
              margin: 14px 0 6px; }}
  .kpi-row {{ display: flex; gap: 14px; flex-wrap: wrap; }}
  .kpi {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 14px 18px; flex: 1; min-width: 110px; }}
  .kpi-val {{ font-size: 26px; font-weight: 700; color: #0f172a; }}
  .kpi-lbl {{ font-size: 11px; color: #64748b; text-transform: uppercase;
              letter-spacing: .5px; margin-top: 2px; }}
  .kpi.tin  {{ border-left: 4px solid {TIN_COLOR}; }}
  .kpi.ctrl {{ border-left: 4px solid {CTRL_COLOR}; }}
  .kpi.warn {{ border-left: 4px solid #dc2626; }}
  .kpi.ok   {{ border-left: 4px solid #16a34a; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 6px; }}
  th {{ background: #f1f5f9; color: #475569; font-weight: 600;
        padding: 9px 12px; text-align: left; border: 1px solid #e2e8f0; }}
  td {{ padding: 8px 12px; border: 1px solid #e2e8f0; vertical-align: middle; }}
  .mono {{ font-family: "SFMono-Regular", Consolas, monospace;
           font-size: 12px; background: #f8fafc; padding: 10px 14px;
           border-radius: 6px; border-left: 4px solid #cbd5e1;
           color: #475569; word-break: break-all; margin-top: 6px; }}
  .legend-dot {{ display:inline-block; width:10px; height:10px;
                 border-radius:50%; margin-right:5px; vertical-align:middle; }}
  .note {{ font-size: 11px; color: #94a3b8; margin-top: 8px; }}
  .rho-box {{ display:inline-flex; align-items:center; gap:8px;
              background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
              padding:10px 16px; margin-top:8px; font-size:13px; }}
  .rho-val {{ font-size:20px; font-weight:700; color:#7c3aed; }}
</style>
</head>
<body>

<div class="page-header">
  <h1>Tinnorm — Demographics &amp; Data Quality Report</h1>
  <p>Generated from master_new.csv · Matched against features directory</p>
</div>

<!-- KPIs -->
<div class="grid-1">
  <div class="card">
    <h2>Dataset Overview</h2>
    <div class="kpi-row">
      {_stat_card("Matched Subjects", len(common_ids), "")}
      {_stat_card("Tinnitus", n_tin, "")}
      {_stat_card("Controls", n_ctrl, "")}
      {_stat_card("Sites", df_plot["site"].nunique(), "")}
      {_stat_card("Only in Features", len(only_in_list), "")}
      {_stat_card("Only in CSV", len(only_in_df), "")}
    </div>
  </div>
</div>

<!-- Replace _stat_card with proper KPI HTML -->
<style>
  .stat-card {{ background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                padding:14px 18px;flex:1;min-width:110px; }}
  .stat-val   {{ font-size:26px;font-weight:700;color:#0f172a; }}
  .stat-lbl   {{ font-size:11px;color:#64748b;text-transform:uppercase;
                 letter-spacing:.5px;margin-top:2px; }}
  .stat-sub   {{ font-size:11px;color:#94a3b8;margin-top:2px; }}
</style>

<!-- Data Integrity -->
<div class="grid-2">
  <div class="card">
    <h2>Data Integrity — Subject Matching</h2>
    <h3>Missing from CSV (in features folder)</h3>
    <div class="mono">{", ".join(only_in_list) if only_in_list else "✓ None"}</div>
    <h3>Missing from features (in CSV)</h3>
    <div class="mono">{", ".join(only_in_df) if only_in_df else "✓ None"}</div>
  </div>

  <div class="card">
    <h2>THI ↔ TFI Agreement (Tinnitus only)</h2>
    <p style="font-size:13px;color:#475569;">Spearman correlation between Tinnitus Handicap Inventory and
    Tinnitus Functional Index scores across all tinnitus subjects (N = {len(tin_df)}).</p>
    <div class="rho-box">
      <div>
        <span class="rho-val">ρ = {rho_thi_tfi:.3f}</span><br>
        <span style="font-size:12px;color:#64748b;">{_fmt_p(p_thi_tfi)}&nbsp;
          {_badge_html(p_thi_tfi, *_pval_badge(p_thi_tfi)[1:])}
        </span>
      </div>
    </div>
    <p class="note">N = {len(tin_df)} tinnitus subjects with both THI and TFI available.</p>
  </div>
</div>

<!-- Site × Group counts -->
<div class="grid-1">
  <div class="card">
    <h2>Site × Group Cohort Counts</h2>
    <table>
      {_build_counts_table()}
    </table>
  </div>
</div>

<!-- Between-group statistical tests -->
<div class="grid-1">
  <div class="card">
    <h2>Between-Group Comparisons (Tinnitus vs Control)</h2>
    <p class="note">Continuous: Mann-Whitney U (two-sided). Categorical: χ² test.</p>
    <table>
      <tr><th>Variable</th><th>Tinnitus (mean ± SD)</th><th>Control (mean ± SD)</th>
          <th>Test statistic</th><th>p-value</th><th>Significance</th></tr>
      {_build_group_test_table()}
    </table>
  </div>
</div>

<!-- Between-site statistical tests -->
<div class="grid-1">
  <div class="card">
    <h2>Between-Site Comparisons (Kruskal-Wallis / χ²)</h2>
    <p class="note">Continuous: Kruskal-Wallis H-test across all sites. Categorical: χ² test. Site effects here would indicate potential confounders.</p>
    <table>
      <tr><th>Variable</th><th>Test statistic</th><th>p-value</th><th>Significance</th></tr>
      {_build_site_test_table()}
    </table>
  </div>
</div>

<!-- Tinnitus severity -->
<div class="grid-1">
  <div class="card">
    <h2>Tinnitus Severity by Site (THI &amp; TFI)</h2>
    <p class="note">THI: 0–100 (≤16 Slight, ≤36 Mild, ≤56 Moderate, ≤76 Severe, &gt;76 Catastrophic).
    TFI: 0–100 (tinnitus group only).</p>
    <table>
      {_build_tinnitus_severity_table()}
    </table>
  </div>
</div>

</body>
</html>
"""

html_file = saving_dir / "demographics_summary.html"
with open(html_file, "w") as f:
    f.write(html_content)

# -------------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------------
site_names = sorted(df_plot["site"].unique())
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

palette_map = {
    "group": {"Tinnitus": TIN_COLOR, "Control": CTRL_COLOR},
    "sex":   SEX_COLORS,
}

plot_configs = {
    "age":      {"xlim": [5,  85],  "label": "Age (years)",                   "tinnitus_only": False},
    "PTA4_mean":{"xlim": [-10, 80], "label": "PTA4 Mean (dB HL)",             "tinnitus_only": False},
    "PTA4_HF":  {"xlim": [-10, 100],"label": "High-Frequency PTA4 (dB HL)",  "tinnitus_only": False},
    "THI":      {"xlim": [0, 100],  "label": "THI Score",                     "tinnitus_only": True},
    "TFI":      {"xlim": [0, 100],  "label": "TFI Score",                     "tinnitus_only": True},
}

for feature, config in plot_configs.items():
    for hue in ["group", "sex"]:
        if config["tinnitus_only"] and hue == "group":
            continue

        data_to_plot = (df_plot[df_plot["group"] == "Tinnitus"]
                        if config["tinnitus_only"] else df_plot)

        if data_to_plot.groupby(hue)[feature].count().min() == 0:
            continue

        xlim = config["xlim"]
        pal  = palette_map[hue]

        # Build ridge-style facet grid
        g = sns.FacetGrid(
            data_to_plot, row="site", hue=hue, aspect=4.0, height=1.5,
            palette=pal, row_order=site_names, xlim=xlim
        )

        g.map(sns.kdeplot, feature, bw_adjust=0.5, clip_on=False, clip=xlim,
              fill=True, alpha=0.65, linewidth=1.2)
        g.map(sns.kdeplot, feature, clip_on=False, clip=xlim,
              linewidth=1.8, bw_adjust=0.5)
        g.refline(y=0, linewidth=1.5, linestyle="-", color="#cccccc", clip_on=False)

        g.figure.subplots_adjust(hspace=0.2, top=0.85)
        g.set_titles(row_template="{row_name}")
        g.set(yticks=[], ylabel="", xlabel=config["label"])
        g.despine(bottom=False, left=True)

        # Build legend manually so colors always match
        hue_vals = sorted(data_to_plot[hue].dropna().unique())
        patches = [mpatches.Patch(facecolor=pal[v], label=str(v)) for v in hue_vals]
        g.figure.legend(
            handles=patches, title=hue.capitalize(),
            loc="upper right", frameon=True,
            fontsize=9, title_fontsize=9,
            bbox_to_anchor=(0.98, 0.97),
        )

        output_filename = f"{hue}_{feature}_distribution.pdf"
        g.figure.savefig(saving_dir / output_filename, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(g.figure)

print(f"Done. Plots and demographics_summary.html saved to:\n  {saving_dir}")