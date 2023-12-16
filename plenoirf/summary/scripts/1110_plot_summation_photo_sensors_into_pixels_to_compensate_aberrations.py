#!/usr/bin/python
import sys
import plenoirf as irf
import plenopy as pl
import scipy
import os
import json_utils
import numpy as np
import sebastians_matplotlib_addons as seb

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

AXES_STYLE = {"spines": ["left", "bottom"], "axes": ["x", "y"], "grid": False}

os.makedirs(pa["out_dir"], exist_ok=True)

light_field_geometry = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
)

OBJECT_DISTANCE = 999e3

# pick pixels on diagonal of image
# --------------------------------
NUM_PIXEL = 6
pixels = []

c_direction = np.array([1, 1])
c_direction = c_direction / np.linalg.norm(c_direction)

for i, off_axis_angle in enumerate(np.deg2rad(np.linspace(0, 3, NUM_PIXEL))):
    cxy = c_direction * off_axis_angle
    pixel = {
        "name_in_figure": chr(i + ord("A")),
        "cxy": cxy,
        "off_axis_angle": off_axis_angle,
        "id": light_field_geometry.pixel_pos_tree.query(cxy)[1],
        "opening_angle": np.deg2rad(0.06667) * 0.6,
    }
    pixels.append(pixel)

lixel_cx_cy_tree = scipy.spatial.cKDTree(
    np.array([light_field_geometry.cx_mean, light_field_geometry.cy_mean]).T
)

for pixel in pixels:
    proto_mask = lixel_cx_cy_tree.query(pixel["cxy"], k=2000)
    mask = []
    for j, angle_between_pixel_and_lixel in enumerate(proto_mask[0]):
        if angle_between_pixel_and_lixel <= pixel["opening_angle"]:
            mask.append(proto_mask[1][j])
    pixel["photosensor_ids"] = np.array(mask)

    pixel["photosensor_mask"] = np.zeros(
        light_field_geometry.number_lixel, dtype=np.bool
    )
    pixel["photosensor_mask"][pixel["photosensor_ids"]] = True

for pixel in pixels:
    xs = light_field_geometry.lixel_positions_x[pixel["photosensor_ids"]]
    ys = light_field_geometry.lixel_positions_y[pixel["photosensor_ids"]]
    pixel["mean_position_of_photosensors_on_sensor_plane"] = np.array(
        [
            np.mean(xs),
            np.mean(ys),
        ]
    )


pixel_spacing_rad = (
    light_field_geometry.sensor_plane2imaging_system.pixel_FoV_hex_flat2flat
)
eye_outer_radius_m = (
    (1 / np.sqrt(3))
    * pixel_spacing_rad
    * light_field_geometry.sensor_plane2imaging_system.expected_imaging_system_focal_length
)


# plot individual pixels
# ----------------------

ROI_RADIUS = 0.35

for pixel in pixels:
    fig = seb.figure(style={"rows": 360, "cols": 360, "fontsize": 0.7})
    ax = seb.add_axes(fig=fig, span=[0.0, 0.0, 1, 1], style=AXES_STYLE)

    _x, _y = pixel["mean_position_of_photosensors_on_sensor_plane"]
    xlim = [_x - ROI_RADIUS, _x + ROI_RADIUS]
    ylim = [_y - ROI_RADIUS, _y + ROI_RADIUS]

    poseye = irf.summary.figure.positions_of_eyes_in_roi(
        light_field_geometry=light_field_geometry,
        roi={"x": xlim, "y": ylim},
        margin=0.2,
    )

    pl.plot.light_field_geometry.ax_add_polygons_with_colormap(
        polygons=light_field_geometry.lixel_polygons,
        I=pixel["photosensor_mask"],
        ax=ax,
        cmap="binary",
        edgecolors="grey",
        linewidths=0.33,
        xlim=xlim,
        ylim=ylim,
    )

    for peye in poseye:
        (_x, _y) = poseye[peye]
        seb.ax_add_hexagon(
            ax=ax,
            x=_x,
            y=_y,
            r_outer=eye_outer_radius_m,
            orientation_deg=0,
            color="black",
            linestyle="-",
            linewidth=0.5,
        )

    ax.text(
        s=pixel["name_in_figure"],
        x=0.1,
        y=0.7,
        fontsize=48,
        transform=ax.transAxes,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    off_axis_angle_mdeg = int(1000 * np.rad2deg(pixel["off_axis_angle"]))
    fig.savefig(
        os.path.join(
            pa["out_dir"],
            "aberration_pixel_{pixel:0d}_{angle:0d}mdeg.jpg".format(
                pixel=pixel["id"], angle=off_axis_angle_mdeg
            ),
        ),
    )
    seb.close("all")

# plot all pixels overview
# -------------------------
fig = fig = seb.figure(style={"rows": 720, "cols": 720, "fontsize": 0.7})
ax = seb.add_axes(fig=fig, span=[0.16, 0.16, 0.82, 0.82])

overview_photosensor_mask = np.zeros(
    light_field_geometry.number_lixel, dtype=np.bool
)
for pixel in pixels:
    overview_photosensor_mask[pixel["photosensor_ids"]] = True

pl.plot.light_field_geometry.ax_add_polygons_with_colormap(
    polygons=light_field_geometry.lixel_polygons,
    I=overview_photosensor_mask,
    ax=ax,
    cmap="binary",
    edgecolors="grey",
    linewidths=(0.02,),
)

poseye = irf.summary.figure.positions_of_eyes_in_roi(
    light_field_geometry=light_field_geometry,
    roi={"x": [-10, 10], "y": [-10, 10]},
    margin=0.2,
)

for peye in poseye:
    (_x, _y) = poseye[peye]
    seb.ax_add_hexagon(
        ax=ax,
        x=_x,
        y=_y,
        r_outer=eye_outer_radius_m,
        orientation_deg=0,
        color="black",
        linestyle="-",
        alpha=0.5,
        linewidth=0.2,
    )


for pixel in pixels:
    _x, _y = pixel["mean_position_of_photosensors_on_sensor_plane"]

    Ax = _x - ROI_RADIUS
    Ay = _y - ROI_RADIUS
    Bx = _x + ROI_RADIUS
    By = _y - ROI_RADIUS

    Cx = _x + ROI_RADIUS
    Cy = _y + ROI_RADIUS
    Dx = _x - ROI_RADIUS
    Dy = _y + ROI_RADIUS

    ax.plot([Ax, Bx], [Ay, By], "k", linewidth=0.5)
    ax.plot([Bx, Cx], [By, Cy], "k", linewidth=0.5)
    ax.plot([Cx, Dx], [Cy, Dy], "k", linewidth=0.5)
    ax.plot([Dx, Ax], [Dy, Ay], "k", linewidth=0.5)
    ax.text(
        s=pixel["name_in_figure"],
        x=_x - 1.0 * ROI_RADIUS,
        y=_y + 1.25 * ROI_RADIUS,
        fontsize=16,
    )

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("$x\\,/\\,$m")
ax.set_ylabel("$y\\,/\\,$m")

fig.savefig(os.path.join(pa["out_dir"], "aberration_overview.jpg"))
seb.close("all")

# export table
# ------------

with open(
    os.path.join(pa["out_dir"], "aberration_overview.txt"), "wt"
) as fout:
    for pixel in pixels:
        _x, _y = pixel["mean_position_of_photosensors_on_sensor_plane"]
        fout.write(
            "{:s} & {:d} & {:.2f} & {:.2f}\\\\\n".format(
                pixel["name_in_figure"],
                pixel["id"],
                np.rad2deg(pixel["off_axis_angle"]),
                np.hypot(_x, _y),
            )
        )
