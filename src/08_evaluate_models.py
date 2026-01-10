from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mne.viz import Brain
from mne import read_labels_from_annot


## read the necessary files (fix paths and make it functoin tbc from main)
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
plt.show()


## plot brain maps
def plot_brain(
        metric_vals,
        palette,
        surf="pial_semi_inflated",
        subject="fsaverage",
        subjects_dir=None,
        alpha=0.7,
        ):
        """
        Plot region-wise brain metrics on the fsaverage cortical surface.

        Each cortical label is colored according to the corresponding metric value.
        Four standard views are rendered and combined into a single matplotlib figure:
        left/right hemispheres * lateral/medial views.

        Parameters
        ----------
        metric_vals : array-like, shape (n_labels,)
                Metric values associated with cortical labels from the annotation.
                The order must match the labels returned by
                ``mne.read_labels_from_annot(..., subject=subject)[:-1]``.
        palette : seaborn color palette or None
                Seaborn palette used to map normalized metric values to colors.
                If None, a dark red sequential palette is used.
        surf : str
                Surface to use for visualization (e.g., ``"pial"``,
                ``"inflated"``, ``"pial_semi_inflated"``).
        subject : str
                FreeSurfer subject name (default: ``"fsaverage"``).
        subjects_dir : path-like or None
                FreeSurfer subjects directory. If None, uses MNE default.
        alpha : float
                Alpha transparency for labels (0-1).

        Returns
        -------
        fig : matplotlib.figure.Figure
                Figure containing the four brain views.

        Notes
        -----
        - This function uses ``mne.viz.Brain`` with the ``pyvistaqt`` backend.
        - Metric values are min-max normalized before color mapping.
        - Screenshots are automatically cropped to remove white margins.
        """


        # Prepare metric values
        metric_vals = np.asarray(metric_vals, dtype=float)

        if metric_vals.ndim != 1:
                raise ValueError("metric_vals must be a 1D array.")

        # Min–max normalization (robust to constant values)
        vmin, vmax = metric_vals.min(), metric_vals.max()
        if np.isclose(vmin, vmax):
                metric_norm = np.zeros_like(metric_vals)
        else:
                metric_norm = (metric_vals - vmin) / (vmax - vmin)

        # Load labels
        labels = read_labels_from_annot(subject=subject, subjects_dir=subjects_dir)
        labels = labels[:-1]  # drop 'unknown'

        if len(labels) != len(metric_norm):
                raise ValueError(
                f"Expected {len(labels)} metric values, got {len(metric_norm)}."
                )

        # Color mapping
        colors = np.asarray(palette.as_hex())
        color_idx = np.round(metric_norm * (len(colors) - 1)).astype(int)
        label_colors = colors[color_idx]
        
        # Brain plotting helper
        brain_kwargs = dict(
                subject=subject,
                subjects_dir=subjects_dir,
                surf=surf,
                background="white",
                cortex=["#b8b4ac", "#b8b4ac"],
        )

        def _render_view(hemi, view):
                """Render one brain view and return cropped screenshot."""
                brain = Brain(hemi=hemi, views=view, **brain_kwargs)

                for label, color in zip(labels, label_colors):
                        if label.hemi == hemi:
                                brain.add_label(
                                label,
                                hemi=hemi,
                                color=color,
                                borders=False,
                                alpha=alpha,
                                )

                img = brain.screenshot()
                brain.close()

                # Crop white margins
                nonwhite = (img != 255).any(axis=-1)
                rows = nonwhite.any(axis=1)
                cols = nonwhite.any(axis=0)

                return img[rows][:, cols]

        # Render views
        screenshots = [
                _render_view("lh", "lateral"),
                _render_view("rh", "lateral"),
                _render_view("lh", "medial"),
                _render_view("rh", "medial"),
        ]
        
        # Assemble figure
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        for ax, img in zip(axes.flat, screenshots):
                ax.imshow(img)
                ax.axis("off")

        return fig