#!/usr/bin/python
import sys
import os
from os.path import join as opj
import numpy as np
import magnetic_deflection as mdfl
import rename_after_writing as rnw
import spherical_coordinates
import spherical_histogram
import sparse_numeric_table as snt
import plenoirf as irf
import sebastians_matplotlib_addons as sebplt
import json_utils
import warnings

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

passing_trigger = res.read_passed_trigger(
    opj(res.paths["analysis_dir"], "0055_passing_trigger"),
    trigger_mode_key="far_accepting_focus_and_near_rejecting_focus",
)

energy_bin = res.energy_binning(key="5_bins_per_decade")
zenith_bin = res.zenith_binning("3_bins_per_45deg")

cmap = sebplt.plt.colormaps["magma_r"].resampled(256)

dome = spherical_histogram.HemisphereHistogram(num_vertices=256)


def write(path, intensity_cube, exposure_cube, num_events_stack):
    with rnw.Directory(path) as tmp:
        with rnw.open(opj(tmp, "intensity_cube.npy"), "wb") as f:
            np.save(f, intensity_cube)
        with rnw.open(opj(tmp, "exposure_cube.npy"), "wb") as f:
            np.save(f, exposure_cube)
        with rnw.open(opj(tmp, "num_events_stack.npy"), "wb") as f:
            np.save(f, num_events_stack)


def read(path):
    out = {}
    with open(opj(path, "intensity_cube.npy"), "rb") as f:
        out["intensity_cube"] = np.load(f)
    with open(opj(path, "exposure_cube.npy"), "rb") as f:
        out["exposure_cube"] = np.load(f)
    with open(opj(path, "num_events_stack.npy"), "rb") as f:
        out["num_events_stack"] = np.load(f)
    return out


for pk in res.PARTICLES:

    pk_cache_dir = opj(res.paths["cache_dir"], pk)

    if os.path.exists(pk_cache_dir):
        continue

    intensity_cube = []
    exposure_cube = []
    num_events_stack = []
    for enbin in range(energy_bin["num"]):
        print(
            pk,
            f"en: {enbin + 1:d}/{energy_bin['num']:d}",
        )

        event_table = res.event_table(particle_key=pk).query(
            levels_and_columns={
                "primary": ["uid", "azimuth_rad", "zenith_rad"],
            },
            energy_start_GeV=energy_bin["edges"][enbin],
            energy_stop_GeV=energy_bin["edges"][enbin + 1],
        )

        mask_triggered = snt.logic.make_mask_of_right_in_left(
            left_indices=event_table["primary"]["uid"],
            right_indices=passing_trigger[pk].uid(
                energy_start_GeV=energy_bin["edges"][enbin],
                energy_stop_GeV=energy_bin["edges"][enbin + 1],
            ),
        )

        num_events = np.sum(mask_triggered)
        num_events_stack.append(num_events)

        dome.reset()
        dome.assign_azimuth_zenith(
            azimuth_rad=event_table["primary"]["azimuth_rad"][mask_triggered],
            zenith_rad=event_table["primary"]["zenith_rad"][mask_triggered],
        )
        _i_per_sr = dome.bin_counts / dome.bin_geometry.faces_solid_angles

        dome.reset()
        dome.assign_azimuth_zenith(
            azimuth_rad=event_table["primary"]["azimuth_rad"],
            zenith_rad=event_table["primary"]["zenith_rad"],
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

    write(
        path=pk_cache_dir,
        intensity_cube=intensity_cube,
        exposure_cube=exposure_cube,
        num_events_stack=num_events_stack,
    )


for pk in res.PARTICLES:
    A = read(opj(res.paths["cache_dir"], pk))

    vmax = np.max(A["intensity_cube"])

    for enbin in range(energy_bin["num"]):
        fig = sebplt.figure(
            style={"rows": 1380, "cols": 1280, "fontsize": 1.0}
        )
        ax = sebplt.add_axes(
            fig=fig, span=[0.0, 0.0, 1, 0.9], style=sebplt.AXES_BLANK
        )

        vx, vy, vz = dome.bin_geometry.vertices.T
        vaz, vzd = spherical_coordinates.cx_cy_cz_to_az_zd(vx, vy, vz)

        faces_counts_per_solid_angle_p99 = np.percentile(
            A["intensity_cube"][enbin], q=99
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in divide"
            )
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered in divide"
            )
            faces_colors = cmap(
                A["intensity_cube"][enbin] / faces_counts_per_solid_angle_p99
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
                A["num_events_stack"][enbin],
                energy_bin["edges"][enbin],
                energy_bin["edges"][enbin + 1],
            ),
            family="monospace",
        )
        fig.savefig(
            opj(
                res.paths["out_dir"],
                f"{pk:s}_grid_direction_pasttrigger_{enbin:06d}.jpg",
            )
        )
        sebplt.close(fig)

res.stop()
