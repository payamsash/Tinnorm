from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from seaborn import cubehelix_palette, color_palette
from mne import read_labels_from_annot, vertex_to_mni
from mne.viz import Brain
from nilearn.plotting import plot_connectome
from nichord.chord import plot_chord


models_dir = Path("../material")
mode = "source"
modality = "power"
preproc_level = 2 
conn_mode = None
train_or_test = "test"
freq_band = "alpha_1"

## read the results of the model
fname = models_dir / f"test_model_results" / "results" / f"Z_{train_or_test}.csv" # fix the path
df = pd.read_csv(fname)
df = df.filter(regex="_alpha_1")

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

        vmin, vmax = metric_vals.min(), metric_vals.max()

        # Load labels
        labels = read_labels_from_annot(subject=subject, subjects_dir=subjects_dir)[:-1]

        if len(labels) != len(metric_vals):
                raise ValueError(
                f"Expected {len(labels)} metric values, got {len(metric_vals)}."
                )

        # Color mapping
        colors = np.asarray(palette.as_hex())
        color_idx = np.round(metric_vals * (len(colors) - 1)).astype(int) ## most important line
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
        cbar.set_label("W score")
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"Min: {vmin:.2f}", f"Max: {vmax:.2f}"])

        return fig


def plot_brain_connectome(
                        metric_vals,
                        palette,
                        subject="fsaverage",
                        subjects_dir=None,
                        edge_threshold="98%",
                        ):
        """
        Plot brain connectivity using both a 3D connectome view and a circular (chord) diagram.

        This function:
        1. Converts a 1D array of metric values (representing the lower triangle of a
        symmetric connectivity matrix) into a full adjacency matrix.
        2. Computes node coordinates in MNI space based on FreeSurfer labels.
        3. Plots the 3D brain connectome using nilearn's plot_connectome.
        4. Plots a circular connectome (chord diagram) showing connectivity between
        brain regions and their associated functional networks.

        Parameters
        ----------
        metric_vals : np.ndarray
                1D array of connectivity or metric values corresponding to the lower triangle
                (excluding diagonal) of a symmetric ROI × ROI matrix.
                Length must be n_labels*(n_labels-1)/2, where n_labels is the number of ROIs.
        palette : list or np.ndarray
                A list of RGB tuples or colormap array used to color the edges in the plots.
        subject : str, default "fsaverage"
                Name of the FreeSurfer subject (used to read labels and compute MNI coordinates).
        subjects_dir : str or None, default None
                Path to FreeSurfer SUBJECTS_DIR. If None, uses the environment variable SUBJECTS_DIR.
        edge_threshold : str or float, default "98%"
                Threshold for edges to be displayed. Can be a percentile string like "98%" or a numeric value.

        Returns
        -------
        None
                The function plots figures but does not return any values.

        Notes
        -----
        - The function expects FreeSurfer annotations for the aparc atlas.
        - Circular connectome plotting uses `map_roi_to_yeo7()` to assign ROIs to Yeo 7 networks.
        - Edge weights for the chord diagram are computed as the mean of `df.mean().values`.
        - Node coordinates are converted from label centers to MNI space.
        - Both 3D brain connectome and circular plots are displayed in the same call.

        """

        labels = read_labels_from_annot(subject=subject, subjects_dir=subjects_dir)[:-1]

        ## creating node coords
        node_coords = []
        for label in labels:

                if label.hemi == 'lh':
                        hemi = 0
                if label.hemi == 'rh':
                        hemi = 1
        
        center_vertex = label.center_of_mass(
                                        subject=subject, 
                                        restrict_vertices=False, 
                                        subjects_dir=subjects_dir
                                        )
        mni_pos = vertex_to_mni(
                                center_vertex,
                                hemis=hemi,
                                subject=subject,
                                subjects_dir=subjects_dir
                                )
        node_coords.append(mni_pos)

        node_coords = np.array(node_coords)
        
        ## creating the graph matrix
        graph = np.zeros((len(labels), len(labels)))
        r, c = np.tril_indices(len(labels), k=-1)
        graph[r, c] = metric_vals
        graph[c, r] = metric_vals
        
        ## plotting the brain connectome
        fig, ax = plt.subplots(1, 1, figsize=(11, 3))
        edge_kwargs = {"lw": 3}
        plot_connectome(
                        adjacency_matrix=graph,
                        node_coords=node_coords,
                        display_mode="lzry",
                        edge_vmin=None,
                        edge_vmax=None,
                        edge_cmap=ListedColormap(palette),
                        node_color='k',
                        node_size=10,
                        axes=ax,
                        colorbar=True,
                        edge_threshold=edge_threshold,
                        edge_kwargs=edge_kwargs
                        )
        fig.tight_layout()

        ## plotting the circular connectome
        edge_idxs = [(i, j) for i, j in zip(r, c)]
        network_order = list(set(map_roi_to_yeo7().values()))
        network_color = dict(zip(network_order, color_palette("Set2")))

        plot_chord(
                idx_to_label=map_roi_to_yeo7(),
                edges=edge_idxs,
                edge_weights=df.mean().values,
                network_order=network_order,
                network_colors=network_color,
                linewidths=1,
                cmap=ListedColormap(palette),
                black_BG=True,
                label_fontsize=18,
                edge_threshold=edge_threshold
                )

## w scores on brain power or node strength or regional metric
plot_brain(
            metric_vals=df.mean().values,
            palette=cubehelix_palette(rot=-.2, n_colors=200),
            surf="inflated"
            )



def map_roi_to_yeo7():
        roi_to_yeo7 = {
                0: "VAN",   # bankssts-lh
                1: "VAN",   # bankssts-rh

                2: "DMN",   # caudalanteriorcingulate-lh
                3: "DMN",   # caudalanteriorcingulate-rh

                4: "DAN",   # caudalmiddlefrontal-lh
                5: "DAN",   # caudalmiddlefrontal-rh

                6: "VIS",   # cuneus-lh
                7: "VIS",   # cuneus-rh

                8: "DMN",   # entorhinal-lh
                9: "DMN",   # entorhinal-rh

                10: "DMN",   # frontalpole-lh
                11: "DMN",   # frontalpole-rh

                12: "VAN",   # fusiform-lh
                13: "VAN",   # fusiform-rh

                14: "VAN",   # inferiorparietal-lh
                15: "VAN",   # inferiorparietal-rh

                16: "VIS",   # inferiortemporal-lh
                17: "VIS",   # inferiortemporal-rh

                18: "VAN",   # insula-lh
                19: "VAN",   # insula-rh

                20: "DMN",   # isthmuscingulate-lh
                21: "DMN",   # isthmuscingulate-rh

                22: "VIS",   # lateraloccipital-lh
                23: "VIS",   # lateraloccipital-rh

                24: "FPN",   # lateralorbitofrontal-lh
                25: "FPN",   # lateralorbitofrontal-rh

                26: "VIS",   # lingual-lh
                27: "VIS",   # lingual-rh

                28: "DMN",   # medialorbitofrontal-lh
                29: "DMN",   # medialorbitofrontal-rh

                30: "VAN",   # middletemporal-lh
                31: "VAN",   # middletemporal-rh

                32: "SMN",   # paracentral-lh
                33: "SMN",   # paracentral-rh

                34: "DMN",   # parahippocampal-lh
                35: "DMN",   # parahippocampal-rh

                36: "FPN",   # parsopercularis-lh
                37: "FPN",   # parsopercularis-rh

                38: "FPN",   # parsorbitalis-lh
                39: "FPN",   # parsorbitalis-rh

                40: "FPN",   # parstriangularis-lh
                41: "FPN",   # parstriangularis-rh

                42: "VIS",   # pericalcarine-lh
                43: "VIS",   # pericalcarine-rh

                44: "SMN",   # postcentral-lh
                45: "SMN",   # postcentral-rh

                46: "DMN",   # posteriorcingulate-lh
                47: "DMN",   # posteriorcingulate-rh

                48: "SMN",   # precentral-lh
                49: "SMN",   # precentral-rh

                50: "DMN",   # precuneus-lh
                51: "DMN",   # precuneus-rh

                52: "DMN",   # rostralanteriorcingulate-lh
                53: "DMN",   # rostralanteriorcingulate-rh

                54: "FPN",   # rostralmiddlefrontal-lh
                55: "FPN",   # rostralmiddlefrontal-rh

                56: "FPN",   # superiorfrontal-lh
                57: "FPN",   # superiorfrontal-rh

                58: "DAN",   # superiorparietal-lh
                59: "DAN",   # superiorparietal-rh

                60: "SMN",   # superiortemporal-lh
                61: "SMN",   # superiortemporal-rh

                62: "VAN",   # supramarginal-lh
                63: "VAN",   # supramarginal-rh

                64: "DMN",   # temporalpole-lh
                65: "DMN",   # temporalpole-rh

                66: "SMN",   # transversetemporal-lh
                67: "SMN"    # transversetemporal-rh
        }
        return roi_to_yeo7
