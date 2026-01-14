from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mne.viz import Brain
from mne import read_labels_from_annot
from matplotlib.colors import ListedColormap



def plot_dist(ev_fname):

        df = pd.read_csv(ev_fname)
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

        # xlims = {
        #         "EXPV": [-0.2, 0.5],
        #         "MSLL": [-10, 5],
        #         "SMSE":  [0.5, 2]
        #         }
        # for idx, (metric, xlim) in zip(range(len(metrics)), xlims.items()):
        #         g.axes[idx][0].set_xlim(xlim)

        g.axes[-1][0].set_xlabel(r"Distribution of evaluation metrics")
        g.map(label, "metric")
        
        return g.figure


## plot brain maps
def plot_brain(
        ev_fname,
        metric,
        freq_band,
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
        df = pd.read_csv(ev_fname)
        df = df.query(f'statistic == "{metric}"')
        df = df.filter(regex=rf'_{freq_band}$')
        metric_vals = np.squeeze(df.values)

        if metric_vals.ndim != 1:
                raise ValueError("metric_vals must be a 1D array.")

        vmin, vmax = metric_vals.min(), metric_vals.max()
        metrics_norm = (metric_vals - vmin) / (vmax - vmin)

        # Load labels
        labels = read_labels_from_annot(subject=subject, subjects_dir=subjects_dir, verbose=False)[:-1]

        if len(labels) != len(metric_vals):
                raise ValueError(
                f"Expected {len(labels)} metric values, got {len(metric_vals)}."
                )

        # Color mapping
        colors = np.asarray(palette.as_hex())
        color_idx = np.round(metrics_norm * (len(colors) - 1)).astype(int) ## most important line
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
                brain.add_annotation("aparc", borders=True, color="white")                           
                img = brain.screenshot()
                brain.close()

                # Crop white margins
                nonwhite = (img != 255).any(axis=-1)
                rows = nonwhite.any(axis=1)
                cols = nonwhite.any(axis=0)

                return img[rows][:, cols]

        # Render views
        screenshots = [
                _render_view("rh", "lateral"),
                _render_view("lh", "lateral"),
                # _render_view("lh", "medial"),
                # _render_view("rh", "medial"),
        ]
        
        # Assemble figure
        fig, axes = plt.subplots(1, 2, figsize=(9, 7))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        im = None
        for ax, img in zip(axes.flat, screenshots):
                im = ax.imshow(img, 
                                cmap=ListedColormap(palette),
                                vmin=vmin,
                                vmax=vmax
                                )
                ax.axis("off")

        cbar = fig.colorbar(
                        im,
                        ax=axes,
                        orientation="horizontal",
                        fraction=0.08,
                        pad=0.1,
                        shrink=0.5
                        )
        cbar.set_label(f"{metric} score")
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"Min: {vmin:.2f}", f"Max: {vmax:.2f}"])
        
        return fig

if __name__ == "__main__":

        mode = "power"
        space = "source"
        preproc_level = 2
        data_mode = "test"
        freq_band = "alpha_1"
        metric = "MSLL"

        tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
        models_dir = tinnorm_dir / "models"
        model_dir = models_dir / f"{mode}_{space}_preproc_{preproc_level}" / "for_eval"

        saving_dir = tinnorm_dir / "plots"
        ev_fname = model_dir / "results" / f"statistics_train_{data_mode}.csv"

        pal = sns.color_palette("Purples", n_colors=200)

        fig_dist = plot_dist(ev_fname)
        fig_brain = plot_brain(
                        ev_fname,
                        metric,
                        freq_band,
                        palette=pal,
                        surf="inflated",
                        subject="fsaverage",
                        subjects_dir=None,
                        alpha=0.7,
                        )

        kwargs = {
                "format": "pdf",
                "dpi": 300,
                "bbox_inches": "tight"
                }
        for fig, title in zip([fig_dist, fig_brain], ["dist", "brain"]):
                fig.savefig(
                        saving_dir / f"{title}_{mode}_{metric}_{data_mode}_{freq_band}_preproc_level_{preproc_level}.pdf",
                        **kwargs
                        )