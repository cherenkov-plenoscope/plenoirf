#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
import json_utils
import spherical_coordinates
import solid_angle_utils
import binning_utils
import spherical_histogram
import sebastians_matplotlib_addons as seb


paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)
seb.matplotlib.rcParams.update(res.analysis["plot"]["matplotlib"])

energy_bin = json_utils.read(
    os.path.join(paths["analysis_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance"]

hh = spherical_histogram.HemisphereHistogram(
    num_vertices=12_000,
    max_zenith_distance_rad=np.deg2rad(90),
)

cmap = seb.plt.colormaps["inferno"].resampled(256)

for pk in res.PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        table = arc.query(
            levels_and_columns={
                "primary": [
                    "uid",
                    "energy_GeV",
                    "azimuth_rad",
                    "zenith_rad",
                ],
                "instrument_pointing": "__all__",
                "groundgrid": "__all__",
            }
        )

    table = snt.logic.sort_table_on_common_indices(
        table=table,
        common_indices=table["primary"]["uid"],
        index_key="uid",
    )

    hh.reset()
    hh.assign_azimuth_zenith(
        azimuth_rad=table["primary"]["azimuth_rad"],
        zenith_rad=table["primary"]["zenith_rad"],
    )

    pointing_range_color = "yellowgreen"
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=[0.0, 0.0, 1, 1], style=seb.AXES_BLANK)
    ax_legend = seb.add_axes(
        fig=fig, span=[0.8, 0.05, 0.2, 0.9], style=seb.AXES_BLANK
    )
    ax_legend.set_xlim([0, 1])
    ax_legend.set_ylim([0, 1])
    ax_colorbar = seb.add_axes(
        fig=fig,
        span=[0.8, 0.4, 0.02, 0.5],
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
    seb.hemisphere.ax_add_faces(
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
    seb.hemisphere.ax_add_plot(
        ax=ax,
        azimuths_rad=ptg_circ_az,
        zeniths_rad=ptg_circ_zd,
        linewidth=0.66,
        color=pointing_range_color,
    )
    seb.hemisphere.ax_add_grid_stellarium_style(
        ax=ax, color="grey", linewidth=0.33
    )
    seb.hemisphere.ax_add_ticklabel_text(ax=ax, color="grey")
    ax_legend.plot(
        [0.0, 0.1],
        [0.1, 0.1],
        color=pointing_range_color,
        linewidth=0.66,
    )
    ax_legend.text(
        x=0.15,
        y=0.1,
        s="pointing\nrange",
        fontsize=9,
        verticalalignment="center",
    )
    ax_legend.text(
        x=0.0,
        y=0.2,
        s=f"site: {res.SITE['name']:s}",
        fontsize=9,
        verticalalignment="center",
    )
    ax_legend.text(
        x=0.0,
        y=0.3,
        s=f"particle: {pk:s}",
        fontsize=9,
        verticalalignment="center",
    )
    _norm = seb.plt_colors.Normalize(
        vmin=0.0, vmax=faces_counts_per_solid_angle_p99, clip=False
    )
    _mappable = seb.plt.cm.ScalarMappable(norm=_norm, cmap=cmap)
    seb.plt.colorbar(mappable=_mappable, cax=ax_colorbar)
    ax_colorbar.set_ylabel(r"intensity / sr$^{-1}$")
    fig.savefig(
        os.path.join(paths["out_dir"], f"{pk:s}_directions_thrown.jpg")
    )
    seb.close(fig)
