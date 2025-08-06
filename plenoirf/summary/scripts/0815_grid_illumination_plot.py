#!/usr/bin/python
import sys
import os
from os.path import join as opj
import numpy as np
import sparse_numeric_table as snt
import plenoirf as irf
import sebastians_matplotlib_addons as sebplt
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

passing_trigger = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)

energy_bin = res.energy_binning(key="point_spread_function")
num_grid_bins_on_edge = res.config["ground_grid"]["geometry"][
    "num_bins_each_axis"
]

GH = irf.ground_grid.GroundGrid(
    bin_width_m=res.config["ground_grid"]["geometry"]["bin_width_m"],
    num_bins_each_axis=res.config["ground_grid"]["geometry"][
        "num_bins_each_axis"
    ],
    center_x_m=0,
    center_y_m=0,
)

MAX_AIRSHOWER_PER_ENERGY_BIN = 100

MAX_CHERENKOV_INTENSITY = (
    10.0 * res.config["ground_grid"]["threshold_num_photons"]
)

FIGURE_STYLE = {"rows": 1080, "cols": 1350, "fontsize": 1}

for pk in res.PARTICLES:
    ggi_path = opj(
        res.response_path(particle_key=pk), "ground_grid_intensity.zip"
    )

    uid_ggi = []
    ggi = irf.ground_grid.intensity.Reader(ggi_path)
    for uid in ggi:
        uid_ggi.append(uid)

    uid_trigger_ggi = list(
        set.intersection(set(uid_ggi), set(passing_trigger[pk]["uid"]))
    )

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(levels_and_columns={"primary": "__all__"})

    detected_events = snt.logic.cut_table_on_indices(
        table=event_table,
        common_indices=uid_trigger_ggi,
    )

    # summarize
    # ---------
    grid_intensities = []
    num_airshowers = []
    for ebin in range(energy_bin["num"]):
        energy_GeV_start = energy_bin["edges"][ebin]
        energy_GeV_stop = energy_bin["edges"][ebin + 1]

        mask_energy = np.logical_and(
            detected_events["primary"]["energy_GeV"] > energy_GeV_start,
            detected_events["primary"]["energy_GeV"] <= energy_GeV_stop,
        )
        uid_energy_range = detected_events["primary"]["uid"][mask_energy]
        grid_intensity = np.zeros(
            (num_grid_bins_on_edge, num_grid_bins_on_edge)
        )
        num_airshower = 0
        for uid in uid_energy_range:
            ground_grid_cell_intensity = ggi[uid]
            for cell in ground_grid_cell_intensity:
                grid_intensity[cell["x_bin"], cell["y_bin"]] += cell["size"]
            num_airshower += 1
            if num_airshower == MAX_AIRSHOWER_PER_ENERGY_BIN:
                break

        grid_intensities.append(grid_intensity)
        num_airshowers.append(num_airshower)

    grid_intensities = np.array(grid_intensities)
    num_airshowers = np.array(num_airshowers)

    # write
    # -----
    for ebin in range(energy_bin["num"]):
        grid_intensity = grid_intensities[ebin]
        num_airshower = num_airshowers[ebin]

        normalized_grid_intensity = grid_intensity
        if num_airshower > 0:
            normalized_grid_intensity /= num_airshower

        fig = sebplt.figure(style=FIGURE_STYLE)
        ax = sebplt.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
        ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.1, 0.02, 0.8])
        ax.set_aspect("equal")
        _pcm_grid = ax.pcolormesh(
            GH["x_bin"]["edges"] * 1e-3,
            GH["y_bin"]["edges"] * 1e-3,
            np.transpose(normalized_grid_intensity),
            norm=sebplt.plt_colors.PowerNorm(
                gamma=1.0 / 4.0, vmin=0, vmax=MAX_CHERENKOV_INTENSITY
            ),
            cmap="Blues",
        )
        sebplt.plt.colorbar(_pcm_grid, cax=ax_cb, extend="max")
        ax.set_title(
            "num. airshower {: 6d}, energy {: 7.1f} - {: 7.1f} GeV".format(
                num_airshower,
                energy_bin["edges"][ebin],
                energy_bin["edges"][ebin + 1],
            ),
            family="monospace",
        )
        ax.set_xlabel(r"$x$ / km")
        ax.set_ylabel(r"$y$ / km")
        sebplt.ax_add_grid(ax)
        fig.savefig(
            opj(
                res.paths["out_dir"],
                f"{pk:s}_grid_area_pasttrigger_{ebin:06d}.jpg",
            )
        )
        sebplt.close(fig)

res.stop()
