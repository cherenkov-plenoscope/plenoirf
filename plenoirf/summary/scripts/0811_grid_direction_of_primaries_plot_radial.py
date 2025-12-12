#!/usr/bin/python
import sys
import os
import pickle
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

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

passing_trigger = res.read_passed_trigger(
    opj(res.paths["analysis_dir"], "0055_passing_trigger"),
    trigger_mode_key="far_accepting_focus_and_near_rejecting_focus",
)
energy_bin = res.energy_binning(key="point_spread_function")


MAX_SCATTER_DEG = 45
NUM_POPULATED_SCATTER_BINS = 11
c_bin_edges_deg = {}
for pk in res.PARTICLES:
    _c_bin_edges = np.linspace(
        0,
        MAX_SCATTER_DEG**2,
        NUM_POPULATED_SCATTER_BINS,
    )
    _c_bin_edges = np.sqrt(_c_bin_edges)
    _c_bin_edges = list(_c_bin_edges)
    _c_bin_edges.append(MAX_SCATTER_DEG)
    _c_bin_edges = np.array(_c_bin_edges)
    c_bin_edges_deg[pk] = _c_bin_edges

FIGURE_STYLE = {"rows": 1080, "cols": 1380, "fontsize": 1}


cache_path = opj(res.paths["out_dir"], "__cache__.pkl")

if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
        o = pickle.loads(f.read())
else:
    o = {}
    for pk in res.PARTICLES:
        o[pk] = {}

        with res.open_event_table(particle_key=pk) as arc:
            evttab = arc.query(
                levels_and_columns={
                    "primary": (
                        "uid",
                        "azimuth_rad",
                        "zenith_rad",
                        "energy_GeV",
                    ),
                    "instrument_pointing": (
                        "uid",
                        "azimuth_rad",
                        "zenith_rad",
                    ),
                }
            )

        evttab = snt.logic.cut_and_sort_table_on_indices(
            table=evttab,
            common_indices=evttab["instrument_pointing"]["uid"],
        )
        scatter_rad = spherical_coordinates.angle_between_az_zd(
            azimuth1_rad=evttab["primary"]["azimuth_rad"],
            zenith1_rad=evttab["primary"]["zenith_rad"],
            azimuth2_rad=evttab["instrument_pointing"]["azimuth_rad"],
            zenith2_rad=evttab["instrument_pointing"]["zenith_rad"],
        )
        scatter_deg = np.rad2deg(scatter_rad)

        mask_trigger = snt.logic.make_mask_of_right_in_left(
            left_indices=evttab["primary"]["uid"],
            right_indices=passing_trigger[pk].uid(),
        )

        o[pk]["thrown"] = []
        o[pk]["detected"] = []

        for ebin in range(energy_bin["num"]):
            print("histogram", pk, "energy", ebin)

            mask_energy = np.logical_and(
                evttab["primary"]["energy_GeV"] >= energy_bin["edges"][ebin],
                evttab["primary"]["energy_GeV"]
                < energy_bin["edges"][ebin + 1],
            )

            mask_energy_trigger = np.logical_and(mask_energy, mask_trigger)

            detected = np.histogram(
                scatter_deg[mask_energy_trigger],
                bins=c_bin_edges_deg[pk],
            )[0]

            thrown = np.histogram(
                scatter_deg[mask_energy],
                bins=c_bin_edges_deg[pk],
            )[0]

            o[pk]["detected"].append(detected)
            o[pk]["thrown"].append(thrown)

        o[pk]["thrown"] = np.array(o[pk]["thrown"])
        o[pk]["detected"] = np.array(o[pk]["detected"])

        with np.errstate(divide="ignore", invalid="ignore"):
            o[pk]["thrown_au"] = np.sqrt(o[pk]["thrown"]) / o[pk]["thrown"]
            o[pk]["detected_au"] = (
                np.sqrt(o[pk]["detected"]) / o[pk]["detected"]
            )

            ratio, ratio_au = propagate_uncertainties.divide(
                x=o[pk]["detected"].astype(float),
                x_au=o[pk]["detected_au"],
                y=o[pk]["thrown"].astype(float),
                y_au=o[pk]["thrown_au"],
            )

        o[pk]["ratio"] = ratio
        o[pk]["ratio_au"] = ratio_au

    with open(cache_path, "wb") as f:
        f.write(pickle.dumps(o))


AXSPAN = copy.deepcopy(irf.summary.figure.AX_SPAN)
AXSPAN = [AXSPAN[0], AXSPAN[1], AXSPAN[2], AXSPAN[3]]


for pk in res.PARTICLES:
    os.makedirs(opj(res.paths["out_dir"], pk), exist_ok=True)

    for ebin in range(energy_bin["num"]):
        print("plot", pk, "energy", ebin)

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
            bincounts=o[pk]["thrown"][ebin],
            linestyle=":",
            linecolor=res.PARTICLE_COLORS[pk],
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
            bincounts=o[pk]["detected"][ebin],
            linestyle="-",
            linecolor=res.PARTICLE_COLORS[pk],
            linealpha=1.0,
            bincounts_upper=o[pk]["detected"][ebin]
            + o[pk]["detected_au"][ebin],
            bincounts_lower=o[pk]["detected"][ebin]
            - o[pk]["detected_au"][ebin],
            face_color=res.PARTICLE_COLORS[pk],
            face_alpha=0.33,
            label=None,
            draw_bin_walls=False,
        )

        sebplt.ax_add_histogram(
            ax=axr,
            bin_edges=c_bin_edges_deg[pk],
            bincounts=o[pk]["ratio"][ebin],
            linestyle="-",
            linecolor=res.PARTICLE_COLORS[pk],
            linealpha=1.0,
            bincounts_upper=o[pk]["ratio"][ebin] + o[pk]["ratio_au"][ebin],
            bincounts_lower=o[pk]["ratio"][ebin] - o[pk]["ratio_au"][ebin],
            face_color=res.PARTICLE_COLORS[pk],
            face_alpha=0.33,
            label=None,
            draw_bin_walls=False,
        )

        axr.set_title(
            "energy {: 7.1f} - {: 7.1f} GeV".format(
                energy_bin["edges"][ebin],
                energy_bin["edges"][ebin + 1],
            ),
        )

        fig.savefig(
            opj(res.paths["out_dir"], pk, f"{pk:s}_energy{ebin:06d}.jpg")
        )
        sebplt.close(fig)


for pk in res.PARTICLES:
    print("plot 2D", pk)

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

    ratio = np.array(o[pk]["ratio"])
    ratio[np.isnan(ratio)] = 0.0
    pcm_ratio = ax.pcolormesh(
        energy_bin["edges"],
        c_bin_edges_deg[pk],
        np.transpose(ratio),
        norm=sebplt.plt_colors.LogNorm(
            vmin=1e-4,
            vmax=1e-0,
        ),
        cmap="terrain_r",
    )

    sebplt.plt.colorbar(
        pcm_ratio,
        cax=ax_cb,
        extend="max",
        label="trigger probability / 1",
    )

    ratio_ru = o[pk]["ratio_au"] / o[pk]["ratio"]

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

    min_energy_GeV = energy_bin["start"]
    max_energy_GeV = energy_bin["stop"]

    ax.plot(
        [min_energy_GeV, max_energy_GeV],
        [MAX_SCATTER_DEG, MAX_SCATTER_DEG],
        "k:",
        alpha=0.1,
    )
    ax.plot(
        [min_energy_GeV, min_energy_GeV],
        [0, MAX_SCATTER_DEG],
        "k:",
        alpha=0.1,
    )
    ax.plot(
        [max_energy_GeV, max_energy_GeV],
        [0, MAX_SCATTER_DEG],
        "k:",
        alpha=0.1,
    )

    fig.savefig(opj(res.paths["out_dir"], f"{pk:s}.jpg"))
    sebplt.close(fig)

res.stop()
