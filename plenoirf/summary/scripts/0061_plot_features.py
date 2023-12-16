#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
import numpy as np
import sebastians_matplotlib_addons as seb
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

weights_thrown2expected = json_utils.tree.read(
    os.path.join(
        pa["summary_dir"],
        "0040_weights_from_thrown_to_expected_energy_spectrum",
    )
)
passing_trigger = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)

PARTICLES = irf_config["config"]["particles"]
SITES = irf_config["config"]["sites"]

particle_colors = sum_config["plot"]["particle_colors"]

# Read features
# =============

tables = {}
for sk in SITES:
    tables[sk] = {}
    for pk in PARTICLES:
        _table = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                sk,
                pk,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        idx_common = spt.intersection(
            [
                passing_trigger[sk][pk]["idx"],
                passing_quality[sk][pk]["idx"],
            ]
        )

        tables[sk][pk] = spt.cut_and_sort_table_on_indices(
            table=_table,
            common_indices=idx_common,
            level_keys=["primary", "features"],
        )

# guess bin edges
lims = {}
Sfeatures = irf.table.STRUCTURE["features"]

for fk in Sfeatures:
    lims[fk] = {}
    for sk in SITES:
        lims[fk][sk] = {}
        for pk in PARTICLES:
            lims[fk][sk][pk] = {}
            features = tables[sk][pk]["features"]
            num_bins = int(np.sqrt(features.shape[0]))
            num_bin_edges = num_bins + 1
            lims[fk][sk][pk]["bin_edges"] = {}
            lims[fk][sk][pk]["bin_edges"]["num"] = num_bin_edges

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

            lims[fk][sk][pk]["bin_edges"]["start"] = start
            lims[fk][sk][pk]["bin_edges"]["stop"] = stop

# find same bin-edges for all particles
for fk in Sfeatures:
    for sk in SITES:
        starts = [lims[fk][sk][pk]["bin_edges"]["start"] for pk in PARTICLES]
        stops = [lims[fk][sk][pk]["bin_edges"]["stop"] for pk in PARTICLES]
        nums = [lims[fk][sk][pk]["bin_edges"]["num"] for pk in PARTICLES]
        start = np.min(starts)
        stop = np.max(stops)
        num = np.max(nums)
        for pk in PARTICLES:
            lims[fk][sk][pk]["bin_edges"]["stop"] = stop
            lims[fk][sk][pk]["bin_edges"]["start"] = start
            lims[fk][sk][pk]["bin_edges"]["num"] = num

for fk in Sfeatures:
    for sk in SITES:
        fig = seb.figure(style=seb.FIGURE_1_1)
        ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])

        for pk in PARTICLES:
            reweight_spectrum = np.interp(
                x=tables[sk][pk]["primary"]["energy_GeV"],
                xp=weights_thrown2expected[sk][pk]["weights_vs_energy"][
                    "energy_GeV"
                ],
                fp=weights_thrown2expected[sk][pk]["weights_vs_energy"][
                    "mean"
                ],
            )

            if "log(x)" in Sfeatures[fk]["transformation"]["function"]:
                myspace = np.geomspace
            else:
                myspace = np.linspace

            bin_edges_fk = myspace(
                lims[fk][sk][pk]["bin_edges"]["start"],
                lims[fk][sk][pk]["bin_edges"]["stop"],
                lims[fk][sk][pk]["bin_edges"]["num"],
            )
            bin_counts_fk = np.histogram(
                tables[sk][pk]["features"][fk], bins=bin_edges_fk
            )[0]
            bin_counts_weight_fk = np.histogram(
                tables[sk][pk]["features"][fk],
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
                lims[fk][sk][pk]["bin_edges"]["start"],
                lims[fk][sk][pk]["bin_edges"]["stop"],
            ]
        )
        ax.set_ylim([1e-5, 1.0])
        fig.savefig(
            os.path.join(pa["out_dir"], "{:s}_{:s}.jpg".format(sk, fk))
        )
        seb.close(fig)
