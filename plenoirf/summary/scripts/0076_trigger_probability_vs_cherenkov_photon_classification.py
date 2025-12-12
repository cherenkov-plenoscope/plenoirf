#!/usr/bin/python
import sys
import numpy as np
import os
from os.path import join as opj
import plenoirf as irf
import sparse_numeric_table as snt
import sebastians_matplotlib_addons as sebplt
import json_utils
import propagate_uncertainties
import warnings


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

passing_trigger = res.read_passed_trigger(
    opj(res.paths["analysis_dir"], "0055_passing_trigger"),
    trigger_mode_key="far_accepting_focus_and_near_rejecting_focus",
)
passing_quality = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)

energy_bin = res.energy_binning(key="10_bins_per_decade")
zenith_bin = res.zenith_binning(key="3_bins_per_45deg")


def get_uid_in_range(event_table, level, column, start, stop):
    mask = np.logical_and(
        event_table[level][column] >= start,
        event_table[level][column] < stop,
    )
    return event_table[level][event_table.index_key][mask]


counts_cache_path = os.path.join(res.paths["cache_dir"], "counts.json")

if not os.path.exists(counts_cache_path):
    categories = [
        "thrown",
        "loose_trigger",
        "trigger",
        "cherenkovclassification",
    ]
    counts = {}
    for pk in res.PARTICLES:
        counts[pk] = {}
        for cat in categories:
            counts[pk][cat] = np.zeros(
                shape=(zenith_bin["num"], energy_bin["num"]), dtype=int
            )

    for pk in res.PARTICLES:
        for zbin in range(zenith_bin["num"]):
            for ebin in range(energy_bin["num"]):
                event_table = res.event_table(particle_key=pk).query(
                    levels_and_columns={
                        "primary": ["uid"],
                        "trigger": ["uid", "response_pe"],
                        "features": ["uid"],
                    },
                    zenith_bin_indices=[zbin],
                    energy_bin_indices=[ebin],
                )
                uid_thrown = event_table["primary"]["uid"]

                uid_passed_loose_trigger = get_uid_in_range(
                    event_table=event_table,
                    level="trigger",
                    column="response_pe",
                    start=res.config["sum_trigger"]["threshold_pe"],
                    stop=float("inf"),
                )

                uid_passed_trigger = passing_trigger[pk].uid(
                    zenith_bin_indices=[zbin],
                    energy_bin_indices=[ebin],
                )

                uid_cherenkovclassification = event_table["features"]["uid"]

                counts[pk]["thrown"][zbin, ebin] = len(
                    event_table["primary"]["uid"]
                )
                counts[pk]["loose_trigger"][zbin, ebin] = len(
                    uid_passed_loose_trigger
                )
                counts[pk]["trigger"][zbin, ebin] = len(
                    snt.logic.intersection(
                        uid_passed_loose_trigger,
                        uid_passed_trigger,
                    )
                )
                counts[pk]["cherenkovclassification"][zbin, ebin] = len(
                    snt.logic.intersection(
                        uid_passed_loose_trigger,
                        uid_passed_trigger,
                        uid_cherenkovclassification,
                    )
                )
    json_utils.write(counts_cache_path, counts)

counts = json_utils.read(counts_cache_path)

warning_messages_to_be_ignored = [
    "invalid value encountered in multiply",
    "invalid value encountered in divide",
    "divide by zero encountered in reciprocal",
    "divide by zero encountered in power",
]

for pk in res.PARTICLES:
    n_trg = counts[pk]["trigger"].astype(float)
    n_trg_au = np.sqrt(n_trg)

    n_cls = counts[pk]["cherenkovclassification"].astype(float)
    n_cls_au = np.sqrt(n_cls)

    with warnings.catch_warnings():
        for message in warning_messages_to_be_ignored:
            warnings.filterwarnings("ignore", message=message)

        ratio, ratio_au = propagate_uncertainties.divide(
            x=n_cls, x_au=n_cls_au, y=n_trg, y_au=n_trg_au
        )

    fig = sebplt.figure(style=irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(
        fig=fig,
        span=[0.15, 0.15, 0.7, 0.8],
        style={"spines": ["bottom"], "axes": ["x", "y"], "grid": False},
    )
    ax.set_yticks([])
    axs = {}
    for zbin in range(zenith_bin["num"]):
        axs[zbin] = sebplt.add_axes(
            fig=fig,
            span=[0.15, 0.95 - (1 + zbin) * (0.8 / 3), 0.7, 0.23],
            style={"spines": ["left", "bottom"], "axes": ["y"], "grid": True},
        )
        sebplt.ax_add_histogram(
            ax=axs[zbin],
            bin_edges=energy_bin["edges"],
            bincounts=ratio[zbin],
            bincounts_upper=ratio[zbin] + ratio_au[zbin] / 2,
            bincounts_lower=ratio[zbin] - ratio_au[zbin] / 2,
            linestyle="-",
            linecolor=res.PARTICLE_COLORS[pk],
            face_color=res.PARTICLE_COLORS[pk],
            face_alpha=0.2,
        )
        axs[zbin].semilogx()
        axs[zbin].set_xlim(energy_bin["limits"])
        axs[zbin].set_ylim([-0.05, 1.05])
        sebplt.ax_add_grid_with_explicit_ticks(
            ax=axs[zbin],
            xticks=np.geomspace(1e0, 1e3, 4),
            yticks=np.linspace(0, 1, 6),
        )
        sebplt.add_axes_zenith_range_indicator(
            fig=fig,
            zenith_bin_edges_rad=zenith_bin["edges"],
            zenith_bin=zbin,
            span=[0.85, 1 - (1 + zbin) * (0.8 / 3), 0.15, 0.15],
            fontsize=7,
        )

    ax.semilogx()
    ax.set_xlim(energy_bin["limits"])
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel(
        "num. events passing feature extraction /\nnum. events passing trigger\n\n"
    )
    fig.savefig(opj(res.paths["out_dir"], pk + "_ratio.jpg"))
    sebplt.close(fig)

res.stop()
