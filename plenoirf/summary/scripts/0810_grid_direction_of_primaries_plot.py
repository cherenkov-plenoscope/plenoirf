#!/usr/bin/python
import sys
import os
from os.path import join as opj
import numpy as np
import magnetic_deflection as mdfl
import spherical_coordinates
import sparse_numeric_table as spt
import plenoirf as irf
import sebastians_matplotlib_addons as seb
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

passing_trigger = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)

os.makedirs(pa["out_dir"], exist_ok=True)

energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["point_spread_function"]

c_bin_edges_deg = np.linspace(
    sum_config["direction_binning"]["radial_angle_deg"] * (-1.0),
    sum_config["direction_binning"]["radial_angle_deg"],
    sum_config["direction_binning"]["num_bins"] + 1,
)

FIGURE_STYLE = {"rows": 1080, "cols": 1350, "fontsize": 1}

for site_key in irf_config["config"]["sites"]:
    for particle_key in irf_config["config"]["particles"]:
        prefix_str = "{:s}_{:s}".format(site_key, particle_key)

        # read
        # ----
        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                site_key,
                particle_key,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        # summarize
        # ---------
        mask_triggered = spt.make_mask_of_right_in_left(
            left_indices=event_table["primary"][spt.IDX],
            right_indices=passing_trigger[site_key][particle_key]["idx"],
        )
        (primary_cx, primary_cy) = spherical_coordinates.az_zd_to_cx_cy(
            azimuth_deg=np.rad2deg(event_table["primary"]["azimuth_rad"]),
            zenith_deg=np.rad2deg(event_table["primary"]["zenith_rad"]),
        )

        intensity_cube = []
        exposure_cube = []
        num_events_stack = []
        for ex in range(energy_bin["num_bins"]):
            energy_mask = np.logical_and(
                event_table["primary"]["energy_GeV"]
                >= energy_bin["edges"][ex],
                event_table["primary"]["energy_GeV"]
                < energy_bin["edges"][ex + 1],
            )

            num_events = np.sum(mask_triggered[energy_mask])
            num_events_stack.append(num_events)

            his = np.histogram2d(
                np.rad2deg(primary_cx[energy_mask]),
                np.rad2deg(primary_cy[energy_mask]),
                weights=mask_triggered[energy_mask],
                bins=[c_bin_edges_deg, c_bin_edges_deg],
            )[0]

            exposure = np.histogram2d(
                np.rad2deg(primary_cx[energy_mask]),
                np.rad2deg(primary_cy[energy_mask]),
                bins=[c_bin_edges_deg, c_bin_edges_deg],
            )[0]

            his[exposure > 0] = his[exposure > 0] / exposure[exposure > 0]
            exposure_cube.append(exposure > 0)
            intensity_cube.append(his)
        intensity_cube = np.array(intensity_cube)
        exposure_cube = np.array(exposure_cube)
        num_events_stack = np.array(num_events_stack)

        # write
        # -----
        vmax = np.max(intensity_cube)

        for energy_idx in range(energy_bin["num_bins"]):
            fig = seb.figure(style=FIGURE_STYLE)
            ax = seb.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
            ax_cb = fig.add_axes([0.85, 0.1, 0.02, 0.8])
            ax.set_aspect("equal")
            _pcm_grid = ax.pcolormesh(
                c_bin_edges_deg,
                c_bin_edges_deg,
                np.transpose(intensity_cube[energy_idx, :, :]),
                norm=seb.plt_colors.PowerNorm(gamma=0.5),
                cmap="Blues",
                vmin=0.0,
                vmax=vmax,
            )
            ax.set_xlim([np.min(c_bin_edges_deg), np.max(c_bin_edges_deg)])
            ax.set_ylim([np.min(c_bin_edges_deg), np.max(c_bin_edges_deg)])
            seb.plt.colorbar(_pcm_grid, cax=ax_cb, extend="max")
            ax.set_xlabel("$c_x$ / $1^\\circ$")
            ax.set_ylabel("$c_y$ / $1^\\circ$")
            seb.ax_add_grid(ax)
            ax.set_title(
                "num. airshower {: 6d}, energy {: 7.1f} - {: 7.1f} GeV".format(
                    num_events_stack[energy_idx],
                    energy_bin["edges"][energy_idx],
                    energy_bin["edges"][energy_idx + 1],
                ),
                family="monospace",
            )
            for rr in [10, 20, 30, 40, 50]:
                seb.ax_add_circle(
                    ax=ax,
                    x=0,
                    y=0,
                    r=rr,
                    color="k",
                    linewidth=0.66,
                    linestyle="-",
                    alpha=0.1,
                )

            num_c_bins = len(c_bin_edges_deg) - 1
            for ix in range(num_c_bins):
                for iy in range(num_c_bins):
                    if not exposure_cube[energy_idx][ix][iy]:
                        seb.ax_add_hatches(
                            ax=ax,
                            ix=ix,
                            iy=iy,
                            x_bin_edges=c_bin_edges_deg,
                            y_bin_edges=c_bin_edges_deg,
                        )
            fig.savefig(
                opj(
                    pa["out_dir"],
                    "{:s}_{:s}_{:06d}.{:s}".format(
                        prefix_str,
                        "grid_direction_pasttrigger",
                        energy_idx,
                        "jpg",
                    ),
                )
            )
            seb.close(fig)
