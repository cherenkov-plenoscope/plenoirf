#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import confusion_matrix
import os
from os.path import join as opj
import numpy as np
import sebastians_matplotlib_addons as seb
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]

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

num_bins = 32
min_number_samples = num_bins

max_relative_leakage = sum_config["quality"]["max_relative_leakage"]
min_reconstructed_photons = sum_config["quality"]["min_reconstructed_photons"]

distance_bin_edges = np.geomspace(5e3, 25e3, num_bins + 1)

for sk in SITES:
    for pk in PARTICLES:
        event_table = spt.read(
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

        table = spt.cut_and_sort_table_on_indices(
            table=event_table,
            common_indices=idx_common,
            level_keys=[
                "primary",
                "cherenkovsize",
                "cherenkovpool",
                "core",
                "trigger",
                "features",
            ],
        )

        fk = "image_smallest_ellipse_object_distance"
        true_airshower_maximum_altitude = table["cherenkovpool"][
            "maximum_asl_m"
        ]
        image_smallest_ellipse_object_distance = table["features"][fk]

        event_weights = np.interp(
            x=table["primary"]["energy_GeV"],
            fp=weights_thrown2expected[sk][pk]["weights_vs_energy"]["mean"],
            xp=weights_thrown2expected[sk][pk]["weights_vs_energy"][
                "energy_GeV"
            ],
        )

        cm = confusion_matrix.init(
            ax0_key="true_airshower_maximum_altitude",
            ax0_values=true_airshower_maximum_altitude,
            ax0_bin_edges=distance_bin_edges,
            ax1_key="image_smallest_ellipse_object_distance",
            ax1_values=image_smallest_ellipse_object_distance,
            ax1_bin_edges=distance_bin_edges,
            weights=event_weights,
            min_exposure_ax0=min_number_samples,
            default_low_exposure=0.0,
        )

        fig = seb.figure(style=seb.FIGURE_1_1)
        ax_c = seb.add_axes(fig=fig, span=[0.25, 0.27, 0.55, 0.65])
        ax_h = seb.add_axes(fig=fig, span=[0.25, 0.11, 0.55, 0.1])
        ax_cb = seb.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
        _pcm_confusion = ax_c.pcolormesh(
            cm["ax0_bin_edges"],
            cm["ax1_bin_edges"],
            np.transpose(cm["counts_normalized_on_ax0"]),
            cmap="Greys",
            norm=seb.plt_colors.PowerNorm(gamma=0.5),
        )
        seb.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
        irf.summary.figure.mark_ax_airshower_spectrum(ax=ax_c)
        ax_c.set_aspect("equal")
        ax_c.set_title("normalized for each column")
        ax_c.set_ylabel(
            fk + " / " + irf.table.STRUCTURE["features"][fk]["unit"]
        )
        ax_c.loglog()
        seb.ax_add_grid(ax_c)

        ax_h.semilogx()
        ax_h.set_xlim(
            [np.min(cm["ax0_bin_edges"]), np.max(cm["ax1_bin_edges"])]
        )
        ax_h.set_xlabel("true maximum of airshower / m")
        ax_h.set_ylabel("num. events / 1")
        irf.summary.figure.mark_ax_thrown_spectrum(ax_h)
        ax_h.axhline(min_number_samples, linestyle=":", color="k")
        seb.ax_add_histogram(
            ax=ax_h,
            bin_edges=cm["ax0_bin_edges"],
            bincounts=cm["exposure_ax0"],
            linestyle="-",
            linecolor="k",
        )
        fig.savefig(opj(pa["out_dir"], sk + "_" + pk + "_maximum.jpg"))
        seb.close(fig)
