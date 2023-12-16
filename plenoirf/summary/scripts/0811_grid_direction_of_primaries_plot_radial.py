#!/usr/bin/python
import sys
import os
import propagate_uncertainties
import numpy as np
import magnetic_deflection as mdfl
import spherical_coordinates
import sparse_numeric_table as spt
import plenoirf as irf
import copy
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
PLT = sum_config["plot"]

passing_trigger = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)

energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
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

        evttab = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                sk,
                pk,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        passed_trigger = spt.make_mask_of_right_in_left(
            left_indices=evttab["primary"][spt.IDX],
            right_indices=passing_trigger[sk][pk]["idx"],
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

        for ex in range(energy_bin["num_bins"]):
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
        sk_pk_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(sk_pk_dir, exist_ok=True)

        for ex in range(energy_bin["num_bins"]):
            print("plot", sk, pk, "energy", ex)

            fig = seb.figure(style=irf.summary.figure.FIGURE_STYLE)

            axr = seb.add_axes(
                fig=fig,
                span=[AXSPAN[0], 0.6, AXSPAN[2], 0.3],
                style={
                    "spines": ["left", "bottom"],
                    "axes": ["x", "y"],
                    "grid": True,
                },
            )
            axi = seb.add_axes(
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

            seb.ax_add_histogram(
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

            seb.ax_add_histogram(
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

            seb.ax_add_histogram(
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
                os.path.join(
                    sk_pk_dir,
                    "{:s}_{:s}_energy{:06d}.jpg".format(
                        sk,
                        pk,
                        ex,
                    ),
                )
            )
            seb.close(fig)


for sk in SITES:
    for pk in PARTICLES:
        print("plot 2D", sk, pk)

        fig = seb.figure(style=irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=[AXSPAN[0], AXSPAN[1], 0.6, 0.7])

        ax_cb = seb.add_axes(
            fig=fig,
            span=[0.85, AXSPAN[1], 0.02, 0.7],
            # style=seb.AXES_BLANK,
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
            norm=seb.plt_colors.LogNorm(),
            cmap="terrain_r",
            vmin=1e-4,
            vmax=1e-0,
        )

        seb.plt.colorbar(
            pcm_ratio,
            cax=ax_cb,
            extend="max",
            label="trigger-probability / 1",
        )

        ratio_ru = o[sk][pk]["ratio_au"] / o[sk][pk]["ratio"]

        num_c_bins = len(c_bin_edges_deg[pk]) - 1
        for iy in range(num_c_bins):
            for ix in range(energy_bin["num_bins"]):
                if ratio_ru[ix][iy] > 0.1 or np.isnan(ratio_ru[ix][iy]):
                    seb.ax_add_hatches(
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
            os.path.join(
                pa["out_dir"],
                "{:s}_{:s}.jpg".format(
                    sk,
                    pk,
                ),
            )
        )
        seb.close(fig)
