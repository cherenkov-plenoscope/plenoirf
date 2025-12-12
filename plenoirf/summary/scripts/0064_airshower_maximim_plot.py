#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import confusion_matrix
import os
from os.path import join as opj
import numpy as np
import sebastians_matplotlib_addons as sebplt
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

PARTICLES = res.PARTICLES

weights_thrown2expected = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"],
        "0040_weights_from_thrown_to_expected_energy_spectrum",
    )
)

passing_trigger = res.read_passed_trigger(
    opj(res.paths["analysis_dir"], "0055_passing_trigger"),
    trigger_mode_key="far_accepting_focus_and_near_rejecting_focus",
)
passing_quality = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)

num_bins = 32
min_number_samples = 3 * num_bins

max_relative_leakage = res.analysis["quality"]["max_relative_leakage"]
min_reconstructed_photons = res.analysis["quality"][
    "min_reconstructed_photons"
]

distance_bin_edges = np.linspace(10e3, 22.5e3, num_bins + 1)

STRUCTURE = irf.event_table.structure.init_event_table_structure()

for pk in PARTICLES:

    uid_common = snt.logic.intersection(
        passing_trigger[pk]["uid"],
        passing_quality[pk]["uid"],
    )

    with res.open_event_table(particle_key=pk) as arc:
        table = arc.query(
            levels_and_columns={
                "primary": ["uid", "energy_GeV"],
                "instrument_pointing": ["uid", "zenith_rad"],
                "cherenkovpool": ["uid", "z_emission_p50_m"],
                "features": [
                    "uid",
                    "image_smallest_ellipse_object_distance",
                ],
            },
            indices=uid_common,
            sort=True,
        )

    true_airshower_maximum_altitude = table["cherenkovpool"][
        "z_emission_p50_m"
    ]
    reco_airshower_maximum_altitude = (
        np.cos(table["instrument_pointing"]["zenith_rad"])
        * table["features"]["image_smallest_ellipse_object_distance"]
    )

    event_weights = np.interp(
        x=table["primary"]["energy_GeV"],
        fp=weights_thrown2expected[pk]["weights_vs_energy"]["mean"],
        xp=weights_thrown2expected[pk]["weights_vs_energy"]["energy_GeV"],
    )

    cm = confusion_matrix.init(
        ax0_key="true_airshower_maximum_altitude",
        ax0_values=true_airshower_maximum_altitude,
        ax0_bin_edges=distance_bin_edges,
        ax1_key="reco_airshower_maximum_altitude",
        ax1_values=reco_airshower_maximum_altitude,
        ax1_bin_edges=distance_bin_edges,
        weights=event_weights,
        min_exposure_ax0=min_number_samples,
        default_low_exposure=0.0,
    )

    fig = sebplt.figure(style=sebplt.FIGURE_1_1)
    ax_c = sebplt.add_axes(fig=fig, span=[0.25, 0.27, 0.55, 0.65])
    ax_h = sebplt.add_axes(fig=fig, span=[0.25, 0.11, 0.55, 0.1])
    ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
    _pcm_confusion = ax_c.pcolormesh(
        cm["ax0_bin_edges"],
        cm["ax1_bin_edges"],
        np.transpose(cm["counts_normalized_on_ax0"]),
        cmap=res.PARTICLE_COLORMAPS[pk],
        norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
    )
    sebplt.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
    irf.summary.figure.mark_ax_airshower_spectrum(ax=ax_c)
    ax_c.set_aspect("equal")
    ax_c.set_title("normalized for each column")
    ax_c.set_ylabel("(depth of smallest ellipse) cos(zenith) / m")
    # ax_c.loglog()
    sebplt.ax_add_grid(ax_c)

    ax_h.semilogx()
    ax_h.set_xlim([np.min(cm["ax0_bin_edges"]), np.max(cm["ax0_bin_edges"])])
    ax_h.set_xlabel("true maximum of airshower / m")
    ax_h.set_ylabel("num. events / 1")
    # irf.summary.figure.mark_ax_thrown_spectrum(ax_h)
    ax_h.axhline(min_number_samples, linestyle=":", color="k")
    sebplt.ax_add_histogram(
        ax=ax_h,
        bin_edges=cm["ax0_bin_edges"],
        bincounts=cm["exposure_ax0"],
        linestyle="-",
        linecolor=res.PARTICLE_COLORS[pk],
    )
    fig.savefig(opj(res.paths["out_dir"], f"{pk}_maximum.jpg"))
    sebplt.close(fig)

res.stop()
