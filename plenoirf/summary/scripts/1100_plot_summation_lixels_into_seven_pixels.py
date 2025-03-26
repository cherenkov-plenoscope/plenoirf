#!/usr/bin/python
import sys
import plenoirf as irf
import plenopy as pl
import os
from os.path import join as opj
import numpy as np
import json_utils
import json
import sebastians_matplotlib_addons as sebplt
import matplotlib
from matplotlib.collections import PolyCollection
from plenopy.light_field_geometry.LightFieldGeometry import init_lixel_polygons

DARKMODE = True
FINE_STEPS = 9
rrr = 2

paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)

if DARKMODE:
    sebplt.plt.style.use("dark_background")
    stroke = "white"
    backc = "black"
    EXT = ".dark.png"
    cmap = "binary_r"
    BEAM_ALPHA = 0.4
    colors = ["gray", "g", "b", "r", "c", "m", "orange"]
else:
    EXT = ".jpg"
    stroke = "black"
    backc = "white"
    cmap = "binary"
    BEAM_ALPHA = 0.2
    colors = ["k", "g", "b", "r", "c", "m", "orange"]

sebplt.matplotlib.rcParams.update(res.analysis["plot"]["matplotlib"])

light_field_geometry = pl.LightFieldGeometry(
    opj(
        paths["plenoirf_dir"],
        "plenoptics",
        "instruments",
        res.instrument_key,
        "light_field_geometry",
    )
)

region_of_interest_on_sensor_plane = {"x": [-0.35, 0.35], "y": [-0.35, 0.35]}


if FINE_STEPS is not None:
    object_distances = np.geomspace(10e3, 60e3, FINE_STEPS).tolist() + [999e3]
    object_distances = np.array(object_distances)
else:
    object_distances = [21e3, 29e3, 999e3]

# object_distances = [3e3, 5e3, 9e3, 15e3, 25e3, 999e3]
central_seven_pixel_ids = [4221, 4124, 4222, 4220, 4125, 4317, 4318]


XY_LABELS_ALWAYS = True

linewidths = 0.25

pixel_spacing_rad = (
    light_field_geometry.sensor_plane2imaging_system.pixel_FoV_hex_flat2flat
)
eye_outer_radius_m = (
    (1 / np.sqrt(3))
    * pixel_spacing_rad
    * light_field_geometry.sensor_plane2imaging_system.expected_imaging_system_focal_length
)
image_outer_radius_rad = 0.5 * (
    light_field_geometry.sensor_plane2imaging_system.max_FoV_diameter
    - pixel_spacing_rad
)

image_geometry = pl.trigger.geometry.init_trigger_image_geometry(
    image_outer_radius_rad=image_outer_radius_rad,
    pixel_spacing_rad=pixel_spacing_rad,
    pixel_radius_rad=pixel_spacing_rad / 2,
    max_number_nearest_lixel_in_pixel=7,
)


def lixel_in_region_of_interest(
    light_field_geometry, lixel_id, roi, margin=0.1
):
    return irf.summary.figure.is_in_roi(
        x=light_field_geometry.lixel_positions_x[lixel_id],
        y=light_field_geometry.lixel_positions_y[lixel_id],
        roi=roi,
        margin=margin,
    )


lixel_polygons = init_lixel_polygons(
    lixel_positions_x=light_field_geometry.lixel_positions_x,
    lixel_positions_y=light_field_geometry.lixel_positions_y,
    lixel_outer_radius=light_field_geometry.lixel_outer_radius,
)

poseye = irf.summary.figure.positions_of_eyes_in_roi(
    light_field_geometry=light_field_geometry,
    lixel_polygons=lixel_polygons,
    roi=region_of_interest_on_sensor_plane,
    margin=0.2,
)

AXES_STYLE = {"spines": ["left", "bottom"], "axes": ["x", "y"], "grid": False}

for obj, object_distance in enumerate(object_distances):
    fig = sebplt.figure(
        style={"rows": 960 * rrr, "cols": 1280 * rrr, "fontsize": 1.244 * rrr}
    )
    ax = sebplt.add_axes(
        fig=fig, span=[0.15, 0.15, 0.85 * (3 / 4), 0.85], style=AXES_STYLE
    )
    ax2 = sebplt.add_axes(fig=fig, span=[0.82, 0.15, 0.2 * (3 / 4), 0.85])

    cpath = opj(paths["out_dir"], "lixel_to_pixel_{:06d}.json".format(obj))

    # compute a list of pixels where a lixel contributes to.
    if not os.path.exists(cpath):
        lixel_to_pixel = (
            pl.trigger.geometry.estimate_projection_of_light_field_to_image(
                light_field_geometry=light_field_geometry,
                object_distance=object_distance,
                image_pixel_cx_rad=image_geometry["pixel_cx_rad"],
                image_pixel_cy_rad=image_geometry["pixel_cy_rad"],
                image_pixel_radius_rad=image_geometry["pixel_radius_rad"],
                max_number_nearest_lixel_in_pixel=image_geometry[
                    "max_number_nearest_lixel_in_pixel"
                ],
            )
        )

        json_utils.write(cpath, lixel_to_pixel, indent=None)
    else:
        with open(cpath, "rt") as f:
            lixel_to_pixel = json.loads(f.read())

    colored_lixels = np.zeros(light_field_geometry.number_lixel, dtype=bool)
    for i, pixel_id in enumerate(central_seven_pixel_ids):
        valid_polygons = []
        additional_colored_lixels = np.zeros(
            light_field_geometry.number_lixel, dtype=bool
        )
        for j, poly in enumerate(lixel_polygons):
            if pixel_id in lixel_to_pixel[j]:
                valid_polygons.append(poly)
                additional_colored_lixels[j] = True

        coll = PolyCollection(
            valid_polygons,
            facecolors=[colors[i] for _ in range(len(valid_polygons))],
            edgecolors="none",
            linewidths=None,
        )
        ax.add_collection(coll)

        colored_lixels += additional_colored_lixels

    not_colored = np.invert(colored_lixels)
    not_colored_polygons = []
    for j, poly in enumerate(lixel_polygons):
        if not_colored[j]:
            if lixel_in_region_of_interest(
                light_field_geometry=light_field_geometry,
                lixel_id=j,
                roi=region_of_interest_on_sensor_plane,
                margin=0.1,
            ):
                not_colored_polygons.append(poly)

    coll = PolyCollection(
        not_colored_polygons,
        facecolors=[backc for _ in range(len(not_colored_polygons))],
        edgecolors="gray",
        linewidths=linewidths,
    )
    ax.add_collection(coll)

    for peye in poseye:
        (_x, _y) = poseye[peye]
        sebplt.ax_add_hexagon(
            ax=ax,
            x=_x,
            y=_y,
            r_outer=eye_outer_radius_m,
            orientation_deg=0,
            color=stroke,
            linestyle="-",
            linewidth=linewidths * 2,
        )

    if obj == 0 or XY_LABELS_ALWAYS:
        ax.set_xlabel("$x\\,/\\,$m")
        ax.set_ylabel("$y\\,/\\,$m")

    ax.set_xlim(region_of_interest_on_sensor_plane["x"])
    ax.set_ylim(region_of_interest_on_sensor_plane["y"])

    ax2.set_axis_off()
    ax2.set_xlim([-1.0, 1.0])
    ax2.set_ylim([-0.05, 3.95])
    t = object_distance / 1e3 / 20
    irf.summary.figure.add_rays_to_ax(
        ax=ax2,
        object_distance=t,
        N=9,
        linewidth=4.7,
        color=irf.summary.figure.COLOR_BEAM_RGBA,
        alpha=BEAM_ALPHA,
    )
    ax2.plot(
        [
            -1,
            1,
        ],
        [-0.1, -0.1],
        color=backc,
        linewidth=10,
        alpha=1.0,
    )
    ax2.plot(
        [
            -1.0,
            1.0,
        ],
        [0, 0],
        color=stroke,
        linewidth=0.5 * linewidths,
    )

    ax2.text(
        x=-0.6,
        y=2 * t,
        s="{:0.0f}$\\,$km".format(object_distance / 1e3),
        fontsize=12,
    )
    if obj + 1 == len(object_distances):
        ax2.text(x=-0.6, y=3.7, s="infinity", fontsize=12)

    fig.savefig(
        opj(
            paths["out_dir"],
            "refocus_lixel_summation_7_{obj:d}{ext:s}".format(
                obj=obj, ext=EXT
            ),
        )
    )
    sebplt.close("all")
