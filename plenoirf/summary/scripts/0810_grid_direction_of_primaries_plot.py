#!/usr/bin/python
import sys
import os
from os.path import join as opj
import numpy as np
import magnetic_deflection as mdfl
import spherical_coordinates
import spherical_histogram
import sparse_numeric_table as snt
import plenoirf as irf
import sebastians_matplotlib_addons as sebplt
import json_utils
import warnings

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

passing_trigger = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)

energy_bin = res.energy_binning(key="point_spread_function")
zenith_bin = res.zenith_binning("once")

cmap = sebplt.plt.colormaps["inferno"].resampled(256)

dome = spherical_histogram.HemisphereHistogram(num_vertices=256)

for pk in res.PARTICLES:

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={
                "primary": "__all__",
            }
        )

    # summarize
    # ---------
    mask_triggered = snt.logic.make_mask_of_right_in_left(
        left_indices=event_table["primary"]["uid"],
        right_indices=passing_trigger[pk]["uid"],
    )

    intensity_cube = []
    exposure_cube = []
    num_events_stack = []
    for ebin in range(energy_bin["num"]):
        print(pk, ebin)
        mask_energy = np.logical_and(
            event_table["primary"]["energy_GeV"] >= energy_bin["edges"][ebin],
            event_table["primary"]["energy_GeV"]
            < energy_bin["edges"][ebin + 1],
        )

        mask_energy_trigger = np.logical_and(mask_energy, mask_triggered)

        num_events = np.sum(mask_triggered[mask_energy])
        num_events_stack.append(num_events)

        dome.reset()
        dome.assign_azimuth_zenith(
            azimuth_rad=event_table["primary"]["azimuth_rad"][
                mask_energy_trigger
            ],
            zenith_rad=event_table["primary"]["zenith_rad"][
                mask_energy_trigger
            ],
        )
        _i_per_sr = dome.bin_counts / dome.bin_geometry.faces_solid_angles

        dome.reset()
        dome.assign_azimuth_zenith(
            azimuth_rad=event_table["primary"]["azimuth_rad"][mask_energy],
            zenith_rad=event_table["primary"]["zenith_rad"][mask_energy],
        )
        _e_per_sr = dome.bin_counts / dome.bin_geometry.faces_solid_angles

        _i_per_sr[_e_per_sr > 0] = (
            _i_per_sr[_e_per_sr > 0] / _e_per_sr[_e_per_sr > 0]
        )
        exposure_cube.append(_e_per_sr > 0)
        intensity_cube.append(_i_per_sr)

    intensity_cube = np.array(intensity_cube)
    exposure_cube = np.array(exposure_cube)
    num_events_stack = np.array(num_events_stack)

    # write
    # -----
    vmax = np.max(intensity_cube)

    for ebin in range(energy_bin["num"]):
        fig = sebplt.figure(
            style={"rows": 1380, "cols": 1280, "fontsize": 1.0}
        )
        ax = sebplt.add_axes(
            fig=fig, span=[0.0, 0.0, 1, 0.9], style=sebplt.AXES_BLANK
        )

        vx, vy, vz = dome.bin_geometry.vertices.T
        vaz, vzd = spherical_coordinates.cx_cy_cz_to_az_zd(vx, vy, vz)

        faces_counts_per_solid_angle_p99 = np.percentile(
            intensity_cube[ebin], q=99
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in divide"
            )
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered in divide"
            )
            faces_colors = cmap(
                intensity_cube[ebin] / faces_counts_per_solid_angle_p99
            )

        sebplt.hemisphere.ax_add_faces(
            ax=ax,
            azimuths_rad=vaz,
            zeniths_rad=vzd,
            faces=dome.bin_geometry.faces,
            faces_colors=faces_colors,
        )

        ax.set_aspect("equal")
        sebplt.hemisphere.ax_add_grid_stellarium_style(
            ax=ax, color="grey", linewidth=0.33
        )
        sebplt.hemisphere.ax_add_ticklabel_text(ax=ax, color="grey")

        ax.set_title(
            "num. airshower {: 6d}, energy {: 7.1f} - {: 7.1f} GeV".format(
                num_events_stack[ebin],
                energy_bin["edges"][ebin],
                energy_bin["edges"][ebin + 1],
            ),
            family="monospace",
        )
        fig.savefig(
            opj(
                res.paths["out_dir"],
                f"{pk:s}_grid_direction_pasttrigger_{ebin:06d}.jpg",
            )
        )
        sebplt.close(fig)

res.stop()
