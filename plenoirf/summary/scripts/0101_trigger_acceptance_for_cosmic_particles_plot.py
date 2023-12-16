#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
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

cr = json_utils.tree.read(
    os.path.join(
        pa["summary_dir"], "0100_trigger_acceptance_for_cosmic_particles"
    )
)

energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance"]

particle_colors = sum_config["plot"]["particle_colors"]

for sk in SITES:
    trigger_thresholds = np.array(
        sum_config["trigger"][sk]["ratescan_thresholds_pe"]
    )
    analysis_trigger_threshold = sum_config["trigger"][sk]["threshold_pe"]

    for source_key in irf.summary.figure.SOURCES:
        for tt in range(len(trigger_thresholds)):
            fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
            ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

            text_y = 0
            for pk in PARTICLES:
                Q = np.array(cr[sk][pk][source_key]["mean"][tt])
                Q_au = np.array(
                    cr[sk][pk][source_key]["absolute_uncertainty"][tt]
                )

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
                irf.summary.figure.SOURCES[source_key]["limits"][
                    "passed_trigger"
                ]
            )
            ax.loglog()
            ax.set_xlim(energy_bin["limits"])

            if trigger_thresholds[tt] == analysis_trigger_threshold:
                fig.savefig(
                    os.path.join(
                        pa["out_dir"],
                        "{:s}_{:s}.jpg".format(
                            sk,
                            source_key,
                        ),
                    )
                )
            ax.set_title(
                "trigger-threshold: {:d} p.e.".format(trigger_thresholds[tt])
            )
            fig.savefig(
                os.path.join(
                    pa["out_dir"],
                    "{:s}_{:s}_{:06d}.jpg".format(
                        sk,
                        source_key,
                        tt,
                    ),
                )
            )
            seb.close(fig)
