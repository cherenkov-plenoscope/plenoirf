#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import json_utils
import spherical_coordinates
import solid_angle_utils
import binning_utils
import spherical_histogram
import sebastians_matplotlib_addons as sebplt


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

hh = spherical_histogram.HemisphereHistogram(
    num_vertices=4_000,
    max_zenith_distance_rad=np.deg2rad(90),
)


for pk in res.PARTICLES:

    cmap = res.PARTICLE_COLORMAPS[pk].resampled(256)

    table_bin_by_bin = res.event_table(particle_key=pk).query(
        levels_and_columns={
            "primary": [
                "uid",
                "azimuth_rad",
                "zenith_rad",
            ]
        },
        bin_by_bin=True,
    )

    hh.reset()
    for table in table_bin_by_bin:
        hh.assign_azimuth_zenith(
            azimuth_rad=table["primary"]["azimuth_rad"],
            zenith_rad=table["primary"]["zenith_rad"],
        )

    pointing_range_color = "yellowgreen"
    fstyle, _ = irf.summary.figure.style("4:3")
    fig = sebplt.figure(fstyle)
    ax = sebplt.add_axes(
        fig=fig, span=[0.0, 0.0, 0.8, 1], style=sebplt.AXES_BLANK
    )
    ax_colorbar = sebplt.add_axes(
        fig=fig,
        span=[0.8, 0.1, 0.02, 0.8],
    )

    vx, vy, vz = hh.bin_geometry.vertices.T
    vaz, vzd = spherical_coordinates.cx_cy_cz_to_az_zd(vx, vy, vz)
    faces_counts_per_solid_angle = (
        hh.bin_counts / hh.bin_geometry.faces_solid_angles
    )
    faces_counts_per_solid_angle_p99 = np.percentile(
        faces_counts_per_solid_angle, q=99
    )
    faces_colors = cmap(
        faces_counts_per_solid_angle / faces_counts_per_solid_angle_p99
    )
    sebplt.hemisphere.ax_add_faces(
        ax=ax,
        azimuths_rad=vaz,
        zeniths_rad=vzd,
        faces=hh.bin_geometry.faces,
        faces_colors=faces_colors,
    )
    ax.set_aspect("equal")
    ptg_circ_az = np.linspace(0, np.pi * 2, 360)
    ptg_circ_zd = res.config["pointing"]["range"][
        "max_zenith_distance_rad"
    ] * np.ones(shape=ptg_circ_az.shape)
    sebplt.hemisphere.ax_add_plot(
        ax=ax,
        azimuths_rad=ptg_circ_az,
        zeniths_rad=ptg_circ_zd,
        linewidth=2,
        color=pointing_range_color,
    )
    sebplt.hemisphere.ax_add_grid_stellarium_style(
        ax=ax, color="grey", linewidth=0.33
    )
    sebplt.hemisphere.ax_add_ticklabel_text(ax=ax, color="grey")

    _norm = sebplt.plt_colors.Normalize(
        vmin=0.0, vmax=faces_counts_per_solid_angle_p99, clip=False
    )
    _mappable = sebplt.plt.cm.ScalarMappable(norm=_norm, cmap=cmap)
    sebplt.plt.colorbar(mappable=_mappable, cax=ax_colorbar)
    ax_colorbar.set_ylabel(r"intensity / sr$^{-1}$")
    fig.savefig(opj(res.paths["out_dir"], f"{pk:s}_directions_thrown.jpg"))
    sebplt.close(fig)

res.stop()
