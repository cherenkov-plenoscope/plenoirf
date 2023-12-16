#!/usr/bin/python
import sys
import plenoirf as irf
import plenopy as pl
import os
import numpy as np
import json_utils
import sebastians_matplotlib_addons as seb
import matplotlib
from matplotlib.collections import PolyCollection

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

light_field_geometry = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
)

region_of_interest_on_sensor_plane = {"x": [-0.35, 0.35], "y": [-0.35, 0.35]}

object_distances = [21e3, 29e3, 999e3]
# object_distances = [3e3, 5e3, 9e3, 15e3, 25e3, 999e3]
central_seven_pixel_ids = [4221, 4124, 4222, 4220, 4125, 4317, 4318]
colors = ["k", "g", "b", "r", "c", "m", "orange"]

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
    return is_in_roi(
        x=light_field_geometry.lixel_positions_x[lixel_id],
        y=light_field_geometry.lixel_positions_y[lixel_id],
        roi=roi,
        margin=margin,
    )


poseye = irf.summary.figure.positions_of_eyes_in_roi(
    light_field_geometry=light_field_geometry,
    roi=region_of_interest_on_sensor_plane,
    margin=0.2,
)

AXES_STYLE = {"spines": ["left", "bottom"], "axes": ["x", "y"], "grid": False}

for obj, object_distance in enumerate(object_distances):
    fig = seb.figure(style={"rows": 960, "cols": 1280, "fontsize": 1.244})
    ax = seb.add_axes(
        fig=fig, span=[0.15, 0.15, 0.85 * (3 / 4), 0.85], style=AXES_STYLE
    )
    ax2 = seb.add_axes(fig=fig, span=[0.82, 0.15, 0.2 * (3 / 4), 0.85])

    cpath = os.path.join(
        pa["out_dir"], "lixel_to_pixel_{:06d}.json".format(obj)
    )

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

        json_utils.write(path=cpath, out_dict=lixel_to_pixel, indent=None)
    else:
        lixel_to_pixel = json_utils.read(path=cpath)

    colored_lixels = np.zeros(light_field_geometry.number_lixel, dtype=np.bool)
    for i, pixel_id in enumerate(central_seven_pixel_ids):
        valid_polygons = []
        additional_colored_lixels = np.zeros(
            light_field_geometry.number_lixel, dtype=np.bool
        )
        for j, poly in enumerate(light_field_geometry.lixel_polygons):
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
    for j, poly in enumerate(light_field_geometry.lixel_polygons):
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
        facecolors=["w" for _ in range(len(not_colored_polygons))],
        edgecolors="gray",
        linewidths=linewidths,
    )
    ax.add_collection(coll)

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
        alpha=0.2,
    )
    ax2.plot(
        [
            -1,
            1,
        ],
        [-0.1, -0.1],
        color="white",
        linewidth=10,
        alpha=1.0,
    )
    ax2.plot(
        [
            -1.0,
            1.0,
        ],
        [0, 0],
        color="k",
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
        os.path.join(
            pa["out_dir"],
            "refocus_lixel_summation_7_{obj:d}.jpg".format(obj=obj),
        )
    )
    seb.close("all")
