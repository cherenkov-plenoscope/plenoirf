#!/usr/bin/python
import sys
import os
from os.path import join as opj
import propagate_uncertainties
import numpy as np
import magnetic_deflection as mdfl
import spherical_coordinates
import sparse_numeric_table as snt
import plenoirf as irf
import copy
import sebastians_matplotlib_addons as sebplt
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(
    run_dir=paths["plenoirf_dir"]
)
sum_config = irf.summary.read_summary_config(summary_dir=paths["analysis_dir"])
sebplt.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(paths["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
PLT = sum_config["plot"]

passing_trigger = json_utils.tree.read(
    opj(paths["analysis_dir"], "0055_passing_trigger")
)

energy_bin = json_utils.read(
    opj(paths["analysis_dir"], "0005_common_binning", "energy.json")
)["point_spread_function"]

MAX_SCATTER_DEG = 20
NUM_POPULATED_SCATTER_BINS = 11
c_bin_edges_deg = {}
for pk in PARTICLES:
    max_scatter_deg = irf_config["config"]["particles"][pk][
        "max_scatter_angle_deg"
    ]
    _c_bin_edges = np.linspace(
        0,
        max_scatter_deg**2,
        NUM_POPULATED_SCATTER_BINS,
    )
    _c_bin_edges = np.sqrt(_c_bin_edges)
    _c_bin_edges = list(_c_bin_edges)
    _c_bin_edges.append(MAX_SCATTER_DEG)
    _c_bin_edges = np.array(_c_bin_edges)
    c_bin_edges_deg[pk] = _c_bin_edges

FIGURE_STYLE = {"rows": 1080, "cols": 1350, "fontsize": 1}

o = {}
for sk in SITES:
    o[sk] = {}
    for pk in PARTICLES:
        o[sk][pk] = {}

        evttab = snt.read(
            path=opj(
                paths["plenoirf_dir"],
                "event_table",
                sk,
                pk,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        passed_trigger = snt.make_mask_of_right_in_left(
            left_indices=evttab["primary"]["uid"],
            right_indices=passing_trigger[sk][pk]["uid"],
        )

        _scatter_rad = spherical_coordinates.angle_between_az_zd(
            azimuth1_rad=evttab["primary"]["azimuth_rad"],
            zenith1_rad=evttab["primary"]["zenith_rad"],
            azimuth2_rad=evttab["primary"]["magnet_azimuth_rad"],
            zenith2_rad=evttab["primary"]["magnet_zenith_rad"],
        )
        scatter_deg = np.rad2deg(_scatter_rad)

        o[sk][pk]["thrown"] = []
        o[sk][pk]["detected"] = []

        for ex in range(energy_bin["num"]):
            print("histogram", sk, pk, "energy", ex)
            emask = np.logical_and(
                evttab["primary"]["energy_GeV"] >= energy_bin["edges"][ex],
                evttab["primary"]["energy_GeV"] < energy_bin["edges"][ex + 1],
            )

            detected = np.histogram(
                scatter_deg[emask],
                weights=passed_trigger[emask],
                bins=c_bin_edges_deg[pk],
            )[0]

            thrown = np.histogram(
                scatter_deg[emask],
                bins=c_bin_edges_deg[pk],
            )[0]

            o[sk][pk]["detected"].append(detected)
            o[sk][pk]["thrown"].append(thrown)

        o[sk][pk]["thrown"] = np.array(o[sk][pk]["thrown"])
        o[sk][pk]["detected"] = np.array(o[sk][pk]["detected"])

        with np.errstate(divide="ignore", invalid="ignore"):
            o[sk][pk]["thrown_au"] = (
                np.sqrt(o[sk][pk]["thrown"]) / o[sk][pk]["thrown"]
            )
            o[sk][pk]["detected_au"] = (
                np.sqrt(o[sk][pk]["detected"]) / o[sk][pk]["detected"]
            )

            ratio, ratio_au = propagate_uncertainties.divide(
                x=o[sk][pk]["detected"].astype(np.float),
                x_au=o[sk][pk]["detected_au"],
                y=o[sk][pk]["thrown"].astype(np.float),
                y_au=o[sk][pk]["thrown_au"],
            )

        o[sk][pk]["ratio"] = ratio
        o[sk][pk]["ratio_au"] = ratio_au


AXSPAN = copy.deepcopy(irf.summary.figure.AX_SPAN)
AXSPAN = [AXSPAN[0], AXSPAN[1], AXSPAN[2], AXSPAN[3]]

for sk in SITES:
    for pk in PARTICLES:
        sk_pk_dir = opj(paths["out_dir"], sk, pk)
        os.makedirs(sk_pk_dir, exist_ok=True)

        for ex in range(energy_bin["num"]):
            print("plot", sk, pk, "energy", ex)

            fig = sebplt.figure(style=irf.summary.figure.FIGURE_STYLE)

            axr = sebplt.add_axes(
                fig=fig,
                span=[AXSPAN[0], 0.6, AXSPAN[2], 0.3],
                style={
                    "spines": ["left", "bottom"],
                    "axes": ["x", "y"],
                    "grid": True,
                },
            )
            axi = sebplt.add_axes(
                fig=fig,
                span=[AXSPAN[0], AXSPAN[1], AXSPAN[2], 0.33],
                style={
                    "spines": ["left", "bottom"],
                    "axes": ["x", "y"],
                    "grid": True,
                },
            )

            axi.set_xlim(
                [np.min(c_bin_edges_deg[pk]), np.max(c_bin_edges_deg[pk])]
            )
            axi.set_ylim([0.1, 1e6])
            axi.semilogy()
            axi.set_xlabel("scatter angle / $1^\\circ$")
            axi.set_ylabel("intensity / 1")

            axr.set_xlim(
                [np.min(c_bin_edges_deg[pk]), np.max(c_bin_edges_deg[pk])]
            )
            axr.set_ylim([1e-4, 1.0])
            axr.semilogy()
            axr.set_ylabel("(detected\n/ thrown) / 1")

            sebplt.ax_add_histogram(
                ax=axi,
                bin_edges=c_bin_edges_deg[pk],
                bincounts=o[sk][pk]["thrown"][ex],
                linestyle=":",
                linecolor=PLT["particle_colors"][pk],
                linealpha=0.5,
                bincounts_upper=None,
                bincounts_lower=None,
                face_color=None,
                face_alpha=None,
                label=None,
                draw_bin_walls=False,
            )

            sebplt.ax_add_histogram(
                ax=axi,
                bin_edges=c_bin_edges_deg[pk],
                bincounts=o[sk][pk]["detected"][ex],
                linestyle="-",
                linecolor=PLT["particle_colors"][pk],
                linealpha=1.0,
                bincounts_upper=o[sk][pk]["detected"][ex]
                + o[sk][pk]["detected_au"][ex],
                bincounts_lower=o[sk][pk]["detected"][ex]
                - o[sk][pk]["detected_au"][ex],
                face_color=PLT["particle_colors"][pk],
                face_alpha=0.33,
                label=None,
                draw_bin_walls=False,
            )

            sebplt.ax_add_histogram(
                ax=axr,
                bin_edges=c_bin_edges_deg[pk],
                bincounts=o[sk][pk]["ratio"][ex],
                linestyle="-",
                linecolor=PLT["particle_colors"][pk],
                linealpha=1.0,
                bincounts_upper=o[sk][pk]["ratio"][ex]
                + o[sk][pk]["ratio_au"][ex],
                bincounts_lower=o[sk][pk]["ratio"][ex]
                - o[sk][pk]["ratio_au"][ex],
                face_color=PLT["particle_colors"][pk],
                face_alpha=0.33,
                label=None,
                draw_bin_walls=False,
            )

            axr.set_title(
                "energy {: 7.1f} - {: 7.1f} GeV".format(
                    energy_bin["edges"][ex],
                    energy_bin["edges"][ex + 1],
                ),
            )

            fig.savefig(
                opj(
                    sk_pk_dir,
                    "{:s}_{:s}_energy{:06d}.jpg".format(
                        sk,
                        pk,
                        ex,
                    ),
                )
            )
            sebplt.close(fig)


for sk in SITES:
    for pk in PARTICLES:
        print("plot 2D", sk, pk)

        fig = sebplt.figure(style=irf.summary.figure.FIGURE_STYLE)
        ax = sebplt.add_axes(fig=fig, span=[AXSPAN[0], AXSPAN[1], 0.6, 0.7])

        ax_cb = sebplt.add_axes(
            fig=fig,
            span=[0.85, AXSPAN[1], 0.02, 0.7],
            # style=sebplt.AXES_BLANK,
        )

        ax.set_xlim(energy_bin["limits"])
        ax.set_ylim([np.min(c_bin_edges_deg[pk]), np.max(c_bin_edges_deg[pk])])
        ax.semilogx()

        ax.set_xlabel("energy / GeV")
        ax.set_ylabel("scatter angle / $1^\\circ$")

        ratio = np.array(o[sk][pk]["ratio"])
        ratio[np.isnan(ratio)] = 0.0
        pcm_ratio = ax.pcolormesh(
            energy_bin["edges"],
            c_bin_edges_deg[pk],
            np.transpose(ratio),
            norm=sebplt.plt_colors.LogNorm(),
            cmap="terrain_r",
            vmin=1e-4,
            vmax=1e-0,
        )

        sebplt.plt.colorbar(
            pcm_ratio,
            cax=ax_cb,
            extend="max",
            label="trigger-probability / 1",
        )

        ratio_ru = o[sk][pk]["ratio_au"] / o[sk][pk]["ratio"]

        num_c_bins = len(c_bin_edges_deg[pk]) - 1
        for iy in range(num_c_bins):
            for ix in range(energy_bin["num"]):
                if ratio_ru[ix][iy] > 0.1 or np.isnan(ratio_ru[ix][iy]):
                    sebplt.ax_add_hatches(
                        ax=ax,
                        ix=ix,
                        iy=iy,
                        x_bin_edges=energy_bin["edges"],
                        y_bin_edges=c_bin_edges_deg[pk],
                    )

        max_scatter_deg = irf_config["config"]["particles"][pk][
            "max_scatter_angle_deg"
        ]
        min_energy_GeV = np.min(
            irf_config["config"]["particles"][pk]["energy_bin_edges_GeV"]
        )
        max_energy_GeV = np.max(
            irf_config["config"]["particles"][pk]["energy_bin_edges_GeV"]
        )

        ax.plot(
            [min_energy_GeV, max_energy_GeV],
            [max_scatter_deg, max_scatter_deg],
            "k:",
            alpha=0.1,
        )
        ax.plot(
            [min_energy_GeV, min_energy_GeV],
            [0, max_scatter_deg],
            "k:",
            alpha=0.1,
        )
        ax.plot(
            [max_energy_GeV, max_energy_GeV],
            [0, max_scatter_deg],
            "k:",
            alpha=0.1,
        )

        fig.savefig(
            opj(
                paths["out_dir"],
                "{:s}_{:s}.jpg".format(
                    sk,
                    pk,
                ),
            )
        )
        sebplt.close(fig)
