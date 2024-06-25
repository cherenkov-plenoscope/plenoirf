#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import json_utils

paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)
seb.matplotlib.rcParams.update(res.analysis["plot"]["matplotlib"])

cr = json_utils.tree.read(
    os.path.join(
        paths["analysis_dir"], "0100_trigger_acceptance_for_cosmic_particles"
    )
)

energy_bin = json_utils.read(
    os.path.join(paths["analysis_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance"]

particle_colors = res.analysis["plot"]["particle_colors"]

trigger_thresholds = np.array(
    res.analysis["trigger"][res.site_key]["ratescan_thresholds_pe"]
)
analysis_trigger_threshold = res.analysis["trigger"][res.site_key][
    "threshold_pe"
]

for source_key in irf.summary.figure.SOURCES:
    for tt in range(len(trigger_thresholds)):
        fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

        text_y = 0
        for pk in res.PARTICLES:
            Q = np.array(cr[pk][source_key]["mean"][tt])
            Q_au = np.array(cr[pk][source_key]["absolute_uncertainty"][tt])

            seb.ax_add_histogram(
                ax=ax,
                bin_edges=energy_bin["edges"],
                bincounts=Q,
                linestyle="-",
                linecolor=particle_colors[pk],
                bincounts_upper=Q + Q_au,
                bincounts_lower=Q - Q_au,
                face_color=particle_colors[pk],
                face_alpha=0.25,
            )

            ax.text(
                0.9,
                0.1 + text_y,
                pk,
                color=particle_colors[pk],
                transform=ax.transAxes,
            )
            text_y += 0.06

        ax.set_xlabel("energy / GeV")
        ax.set_ylabel(
            "{:s} / {:s}".format(
                irf.summary.figure.SOURCES[source_key]["label"],
                irf.summary.figure.SOURCES[source_key]["unit"],
            )
        )
        ax.set_ylim(
            irf.summary.figure.SOURCES[source_key]["limits"]["passed_trigger"]
        )
        ax.loglog()
        ax.set_xlim(energy_bin["limits"])

        if trigger_thresholds[tt] == analysis_trigger_threshold:
            fig.savefig(os.path.join(paths["out_dir"], f"{source_key:s}.jpg"))
        ax.set_title(
            "trigger-threshold: {:d} p.e.".format(trigger_thresholds[tt])
        )
        fig.savefig(
            os.path.join(paths["out_dir"], f"{source_key:s}_{tt:06d}.jpg")
        )
        seb.close(fig)
