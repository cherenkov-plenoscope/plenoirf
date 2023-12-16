#!/usr/bin/python
import sys
import os
from os.path import join as opj
import numpy as np
import sparse_numeric_table as spt
import plenoirf as irf
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

passing_trigger = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)

energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["point_spread_function"]

num_grid_bins_on_edge = irf_config["grid_geometry"]["num_bins_diameter"]

MAX_AIRSHOWER_PER_ENERGY_BIN = 100

MAX_CHERENKOV_INTENSITY = (
    10.0 * irf_config["config"]["grid"]["threshold_num_photons"]
)

FIGURE_STYLE = {"rows": 1080, "cols": 1350, "fontsize": 1}

for sk in SITES:
    for pk in PARTICLES:
        prefix_str = "{:s}_{:s}".format(sk, pk)

        # read
        # ----
        detected_grid_histograms = irf.grid.read_histograms(
            path=opj(
                pa["run_dir"],
                "event_table",
                sk,
                pk,
                "grid.tar",
            ),
            indices=passing_trigger[sk][pk]["idx"],
        )
        idx_passed_trigger_and_in_debug_output = np.array(
            list(detected_grid_histograms.keys())
        )

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

        detected_events = spt.cut_table_on_indices(
            table=event_table,
            common_indices=idx_passed_trigger_and_in_debug_output,
            level_keys=[
                "primary",
                "grid",
                "core",
                "cherenkovsize",
                "cherenkovpool",
                "cherenkovsizepart",
                "cherenkovpoolpart",
                "trigger",
            ],
        )

        # summarize
        # ---------
        grid_intensities = []
        num_airshowers = []
        for energy_idx in range(energy_bin["num_bins"]):
            energy_GeV_start = energy_bin["edges"][energy_idx]
            energy_GeV_stop = energy_bin["edges"][energy_idx + 1]
            energy_mask = np.logical_and(
                detected_events["primary"]["energy_GeV"] > energy_GeV_start,
                detected_events["primary"]["energy_GeV"] <= energy_GeV_stop,
            )
            idx_energy_range = detected_events["primary"][energy_mask][spt.IDX]
            grid_intensity = np.zeros(
                (num_grid_bins_on_edge, num_grid_bins_on_edge)
            )
            num_airshower = 0
            for idx in idx_energy_range:
                grid_intensity += irf.grid.bytes_to_histogram(
                    detected_grid_histograms[idx]
                )
                num_airshower += 1
                if num_airshower == MAX_AIRSHOWER_PER_ENERGY_BIN:
                    break

            grid_intensities.append(grid_intensity)
            num_airshowers.append(num_airshower)

        grid_intensities = np.array(grid_intensities)
        num_airshowers = np.array(num_airshowers)

        # write
        # -----
        for energy_idx in range(energy_bin["num_bins"]):
            grid_intensity = grid_intensities[energy_idx]
            num_airshower = num_airshowers[energy_idx]

            normalized_grid_intensity = grid_intensity
            if num_airshower > 0:
                normalized_grid_intensity /= num_airshower

            fig = seb.figure(style=FIGURE_STYLE)
            ax = seb.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
            ax_cb = seb.add_axes(fig=fig, span=[0.85, 0.1, 0.02, 0.8])
            ax.set_aspect("equal")
            _pcm_grid = ax.pcolormesh(
                irf_config["grid_geometry"]["xy_bin_edges"] * 1e-3,
                irf_config["grid_geometry"]["xy_bin_edges"] * 1e-3,
                np.transpose(normalized_grid_intensity),
                norm=seb.plt_colors.PowerNorm(gamma=1.0 / 4.0),
                cmap="Blues",
                vmin=0,
                vmax=MAX_CHERENKOV_INTENSITY,
            )
            seb.plt.colorbar(_pcm_grid, cax=ax_cb, extend="max")
            ax.set_title(
                "num. airshower {: 6d}, energy {: 7.1f} - {: 7.1f} GeV".format(
                    num_airshower,
                    energy_bin["edges"][energy_idx],
                    energy_bin["edges"][energy_idx + 1],
                ),
                family="monospace",
            )
            ax.set_xlabel("$x$ / km")
            ax.set_ylabel("$y$ / km")
            seb.ax_add_grid(ax)
            fig.savefig(
                opj(
                    pa["out_dir"],
                    "{:s}_{:s}_{:06d}.jpg".format(
                        prefix_str,
                        "grid_area_pasttrigger",
                        energy_idx,
                    ),
                )
            )
            seb.close(fig)
