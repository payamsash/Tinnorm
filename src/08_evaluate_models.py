from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## read the necessary files (fix paths)
model_dir = Path("/Users/payamsadeghishabestari/Tinnorm/material/test_model_results")
ev_fname = model_dir / "results" / "statistics_test.csv"
df = pd.read_csv(ev_fname)


freq_band = "alpha_1"
metrics = ["EXPV", "MSLL", "SMSE"]

## plot averaged histograms
df = df.query('statistic == @metrics').T
df = df.rename(columns=df.iloc[0]).iloc[1:].reset_index(drop=False)
df = df[df["index"].str.endswith(freq_band)]
df = df.rename(columns={"index": "bl"})
df = df.melt(
            id_vars="bl",         
            value_vars=metrics, 
            var_name="metric",    
            value_name="value"
        )

pal = [
        sns.cubehelix_palette(start=2)[1],
        sns.cubehelix_palette(start=2)[-1],
        sns.cubehelix_palette(rot=-.2)[2]
        ]
xlims = {
        "EXPV": [-0.2, 0.5],
        "MSLL": [-10, 5],
        "SMSE":  [0.5, 2]
        }

def label(x, color, label):
        ax = plt.gca()
        ax.text(0.8, .1, f"metric = {label}", fontstyle='italic', color=color,
                ha="left", va="center", transform=ax.transAxes)

g = sns.FacetGrid(
                data=df,
                row="metric",
                hue="metric",
                aspect=2.8,
                height=1.6,
                sharey=False,
                sharex=False,
                row_order=metrics,
                xlim=None,
                palette=pal
                )

g.map(sns.kdeplot,
        "value",
        fill=True,
        alpha=0.5,
        linewidth=1.5
        )

g.map(sns.kdeplot, "value", color="k", lw=1)
g.refline(y=0, linewidth=0.5, linestyle="-", color="k", clip_on=False)
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

for idx, (metric, xlim) in zip(range(len(metrics)), xlims.items()):
        g.axes[idx][0].set_xlim(xlim)

g.axes[idx][0].set_xlabel(r"Distribution of evaluation metrics")
g.map(label, "metric")

## plot brain maps


## plot connection maps



