#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
import numpy as np
import sebastians_matplotlib_addons as seb
import json_utils


paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)
seb.matplotlib.rcParams.update(res.analysis["plot"]["matplotlib"])

weights_thrown2expected = json_utils.tree.read(
    os.path.join(
        paths["analysis_dir"],
        "0040_weights_from_thrown_to_expected_energy_spectrum",
    )
)
passing_trigger = json_utils.tree.read(
    os.path.join(paths["analysis_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    os.path.join(paths["analysis_dir"], "0056_passing_basic_quality")
)

particle_colors = res.analysis["plot"]["particle_colors"]

PARTICLES = res.PARTICLES

# Read features
# =============

tables = {}

for pk in PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        _table = arc.read_table(
            levels_and_columns={
                "primary": [snt.IDX, "energy_GeV"],
                "features": "__all__",
            }
        )

    idx_common = snt.intersection(
        [
            passing_trigger[pk]["idx"],
            passing_quality[pk]["idx"],
        ]
    )

    tables[pk] = snt.cut_and_sort_table_on_indices(
        table=_table,
        common_indices=idx_common,
    )

# guess bin edges
lims = {}
Sfeatures = irf.event_table.structure.init_features_level_structure()

for fk in Sfeatures:
    lims[fk] = {}
    for pk in PARTICLES:
        lims[fk][pk] = {}
        features = tables[pk]["features"]
        num_bins = int(np.sqrt(features.shape[0]))
        num_bin_edges = num_bins + 1
        lims[fk][pk]["bin_edges"] = {}
        lims[fk][pk]["bin_edges"]["num"] = num_bin_edges

        start, stop = irf.features.find_values_quantile_range(
            values=features[fk], quantile_range=[0.01, 0.99]
        )
        if "log(x)" in Sfeatures[fk]["transformation"]["function"]:
            start = 10 ** np.floor(np.log10(start))
            stop = 10 ** np.ceil(np.log10(stop))
        else:
            if start >= 0.0:
                start = 0.9 * start
            else:
                start = 1.1 * start
            if stop >= 0.0:
                stop = 1.1 * stop
            else:
                stop = 0.9 * stop

        lims[fk][pk]["bin_edges"]["start"] = start
        lims[fk][pk]["bin_edges"]["stop"] = stop

# find same bin-edges for all particles
for fk in Sfeatures:
    starts = [lims[fk][pk]["bin_edges"]["start"] for pk in PARTICLES]
    stops = [lims[fk][pk]["bin_edges"]["stop"] for pk in PARTICLES]
    nums = [lims[fk][pk]["bin_edges"]["num"] for pk in PARTICLES]
    start = np.min(starts)
    stop = np.max(stops)
    num = np.max(nums)
    for pk in PARTICLES:
        lims[fk][pk]["bin_edges"]["stop"] = stop
        lims[fk][pk]["bin_edges"]["start"] = start
        lims[fk][pk]["bin_edges"]["num"] = num

for fk in Sfeatures:
    fig = seb.figure(style=seb.FIGURE_1_1)
    ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])

    for pk in PARTICLES:
        reweight_spectrum = np.interp(
            x=tables[pk]["primary"]["energy_GeV"],
            xp=weights_thrown2expected[pk]["weights_vs_energy"]["energy_GeV"],
            fp=weights_thrown2expected[pk]["weights_vs_energy"]["mean"],
        )

        if "log(x)" in Sfeatures[fk]["transformation"]["function"]:
            myspace = np.geomspace
        else:
            myspace = np.linspace

        bin_edges_fk = myspace(
            lims[fk][pk]["bin_edges"]["start"],
            lims[fk][pk]["bin_edges"]["stop"],
            lims[fk][pk]["bin_edges"]["num"],
        )
        bin_counts_fk = np.histogram(
            tables[pk]["features"][fk], bins=bin_edges_fk
        )[0]
        bin_counts_weight_fk = np.histogram(
            tables[pk]["features"][fk],
            weights=reweight_spectrum,
            bins=bin_edges_fk,
        )[0]

        bin_counts_unc_fk = irf.utils._divide_silent(
            numerator=np.sqrt(bin_counts_fk),
            denominator=bin_counts_fk,
            default=np.nan,
        )
        bin_counts_weight_norm_fk = irf.utils._divide_silent(
            numerator=bin_counts_weight_fk,
            denominator=np.sum(bin_counts_weight_fk),
            default=0,
        )

        seb.ax_add_histogram(
            ax=ax,
            bin_edges=bin_edges_fk,
            bincounts=bin_counts_weight_norm_fk,
            linestyle="-",
            linecolor=particle_colors[pk],
            linealpha=1.0,
            bincounts_upper=bin_counts_weight_norm_fk
            * (1 + bin_counts_unc_fk),
            bincounts_lower=bin_counts_weight_norm_fk
            * (1 - bin_counts_unc_fk),
            face_color=particle_colors[pk],
            face_alpha=0.3,
        )

    if "log(x)" in Sfeatures[fk]["transformation"]["function"]:
        ax.loglog()
    else:
        ax.semilogy()

    irf.summary.figure.mark_ax_airshower_spectrum(ax=ax)
    ax.set_xlabel("{:s} / {:s}".format(fk, Sfeatures[fk]["unit"]))
    ax.set_ylabel("relative intensity / 1")
    seb.ax_add_grid(ax)
    ax.set_xlim(
        [
            lims[fk][pk]["bin_edges"]["start"],
            lims[fk][pk]["bin_edges"]["stop"],
        ]
    )
    ax.set_ylim([1e-5, 1.0])
    fig.savefig(os.path.join(paths["out_dir"], f"{fk:s}.jpg"))
    seb.close(fig)
