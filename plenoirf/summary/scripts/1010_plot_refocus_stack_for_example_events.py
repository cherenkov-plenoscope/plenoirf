#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import plenopy as pl
import binning_utils
import gamma_ray_reconstruction as gamrec
import sebastians_matplotlib_addons as sebplt
import json_utils
import plenoptics
import solid_angle_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

passing_trigger = res.read_passed_trigger(
    opj(res.paths["analysis_dir"], "0055_passing_trigger"),
    trigger_mode_key="far_accepting_focus_and_near_rejecting_focus",
)
passing_quality = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)

light_field_geometry = pl.LightFieldGeometry(
    opj(
        res.paths["plenoirf_dir"],
        "plenoptics",
        "instruments",
        res.instrument_key,
        "light_field_geometry",
    )
)
max_FoV_diameter_deg = np.rad2deg(
    light_field_geometry.sensor_plane2imaging_system.max_FoV_diameter
)

SAMPLE = {
    "bin_edges_pe": np.geomspace(7.5e1, 7.5e5, 5),
    "count": np.array([10, 5, 5, 5]),
}

FIC_SCALE = 2
FIG_ROWS = 360 * FIC_SCALE
FIG_COLS = 640 * FIC_SCALE

REGION_OF_INTEREST_DEG = 3.25
CMAP_GAMMA = 0.5


# same picture binning as in plenoptics/plot_phantom_source
# ---------------------------------------------------------
image_edge_ticks_deg = np.linspace(-3, 3, 7)
image_edge_bin = binning_utils.Binning(
    bin_edges=np.deg2rad(np.linspace(-3.5, 3.5, int(3 * (7 / 0.067)))),
)
image_bin_solid_angle_sr = np.mean(image_edge_bin["widths"]) ** 2
image_bin_solid_angle_deg2 = solid_angle_utils.sr2squaredeg(
    image_bin_solid_angle_sr
)
image_bins = [image_edge_bin["edges"], image_edge_bin["edges"]]


def table_to_dict(ta):
    out = {}
    for col in ta:
        out[col] = recarray_to_dict(ta[col])
    return out


def recarray_to_dict(ra):
    out = {}
    for key in ra.dtype.names:
        out[key] = ra[key]
    return out


def counter_init(sample):
    return np.zeros(len(sample["count"]), dtype=int)


def counter_not_full(counter, sample):
    for i in range(len(counter)):
        if counter[i] < sample["count"][i]:
            return True
    return False


def counter_can_add(counter, pe, sample):
    b = np.digitize(x=pe, bins=sample["bin_edges_pe"])
    if b == 0:
        return False
    if b == len(sample["bin_edges_pe"]):
        return False
    bix = b - 1

    if counter[bix] < sample["count"][bix]:
        return True
    else:
        return False


def counter_add(counter, pe, sample):
    b = np.digitize(x=pe, bins=sample["bin_edges_pe"])
    assert b > 0
    assert b < len(sample["bin_edges_pe"])
    bix = b - 1
    counter[bix] += 1
    return counter


def loph_to_time_lixel_repr(loph):
    _photon_arrival_times_s = (
        loph["sensor"]["time_slice_duration"]
        * loph["photons"]["arrival_time_slices"]
    )
    _photon_lixel_ids = loph["photons"]["channels"]
    return (_photon_arrival_times_s, _photon_lixel_ids)


def make_image_like_in_plenoptics_phantom(
    light_field_geometry,
    loph_record,
    object_distance_m,
):
    light_field = (
        loph_record["photons"]["arrival_time_slices"],
        loph_record["photons"]["channels"],
    )
    img = plenoptics.analysis.image.compute_image(
        light_field_geometry=light_field_geometry,
        light_field=loph_to_time_lixel_repr(loph=loph_record),
        object_distance=object_distance_m,
        bins=image_bins,
        prng=np.random.Generator(np.random.PCG64(obj_idx)),
    )
    return img


def plot_image_like_in_plenoptics_phantom(
    path,
    image_edge_bin,
    image_edge_ticks_deg,
    img,
    img_vmax,
    cmapkey,
    max_FoV_diameter_deg,
    roi_limits_deg=None,
    NPIX=1280,
):
    fig = sebplt.figure(style={"rows": NPIX, "cols": NPIX, "fontsize": 1.0})
    ax = sebplt.add_axes(fig=fig, span=[0.0, 0.0, 1, 1])
    ax.set_aspect("equal")
    cmap = ax.pcolormesh(
        np.rad2deg(image_edge_bin["edges"]),
        np.rad2deg(image_edge_bin["edges"]),
        np.transpose(img),
        cmap=cmapkey,
        norm=sebplt.plt_colors.PowerNorm(
            gamma=plenoptics.plot.CMAPS[cmapkey]["gamma"],
            vmin=0.0,
            vmax=img_vmax,
        ),
    )
    sebplt.ax_add_circle(
        ax=ax,
        x=0.0,
        y=0.0,
        r=0.5 * max_FoV_diameter_deg,
        linewidth=0.33,
        linestyle="-",
        color=plenoptics.plot.CMAPS[cmapkey]["linecolor"],
        num_steps=360 * 5,
    )
    if roi_limits_deg is None:
        ax.set_xlim(np.rad2deg(image_edge_bin["limits"]))
        ax.set_ylim(np.rad2deg(image_edge_bin["limits"]))
    else:
        ax.set_xlim(roi_limits_deg["cx"])
        ax.set_ylim(roi_limits_deg["cy"])

    sebplt.ax_add_grid_with_explicit_ticks(
        ax=ax,
        xticks=image_edge_ticks_deg,
        yticks=image_edge_ticks_deg,
        linewidth=0.33,
        color=plenoptics.plot.CMAPS[cmapkey]["linecolor"],
    )
    fig.savefig(path)
    sebplt.close(fig)


def plot_image_focus_bar_like_in_plenoptics_phantom(
    path,
    object_distances,
    obj_idx,
    NPIX=1280,
):
    fig = sebplt.figure(
        style={"rows": NPIX, "cols": NPIX // 4, "fontsize": 2.0}
    )
    ax = sebplt.add_axes(
        fig=fig,
        span=[0.75, 0.1, 0.2, 0.8],
        style={"spines": ["left"], "axes": ["y"], "grid": False},
    )
    ax.set_ylim([0, 2.5e1])
    ax.set_xlim([0, 1])
    ax.set_ylabel("depth / km")
    ax.plot(
        [0, 1],
        [
            object_distances[obj_idx] * 1e-3,
            object_distances[obj_idx] * 1e-3,
        ],
        color="black",
        linewidth=2,
    )
    fig.savefig(path)
    sebplt.close(fig)


def plot_image_colorbar_like_in_plenoptics_phantom(
    path,
    cmapkey,
    img_vmax_pe_per_deg2,
    orientation="vertical",
):
    if orientation == "horizontal":
        fig_style = {"rows": 120, "cols": 1280, "fontsize": 1}
        ax_span = [0.1, 0.8, 0.8, 0.15]
        text_pos = [0.5, -4.7]
        text_rotation = 0
    elif orientation == "vertical":
        fig_style = {"rows": 1280, "cols": 240, "fontsize": 1}
        ax_span = [0.1, 0.1, 0.15, 0.8]
        text_pos = [3.5, 0.15]
        text_rotation = 90

    fig = sebplt.figure(style=fig_style)
    ax = sebplt.add_axes(fig, ax_span)

    # get the 'cmap'
    ax_not_shown = sebplt.add_axes(fig, [1.1, 1.1, 0.1, 0.1])
    cmap = ax_not_shown.pcolormesh(
        [0, 1],
        [0, 1],
        [[1]],
        cmap=cmapkey,
        norm=sebplt.plt_colors.PowerNorm(
            gamma=plenoptics.plot.CMAPS[cmapkey]["gamma"],
            vmin=0.0,
            vmax=img_vmax_pe_per_deg2 * 1e-3,
        ),
    )
    ax.text(
        text_pos[0],
        text_pos[1],
        r"reco. Cherenkov$\,/\,$ k$\,$(photo electron) (1$^\circ$)$^{-2}$",
        rotation=text_rotation,
        fontsize=15,
    )
    sebplt.plt.colorbar(cmap, cax=ax, extend="max", orientation=orientation)
    fig.savefig(path)
    sebplt.close(fig)


def make_roi_limits(
    cx_rad, cy_rad, region_of_interest_deg=REGION_OF_INTEREST_DEG
):
    out = {}
    cx_deg = np.rad2deg(cx_rad)
    cy_deg = np.rad2deg(cy_rad)
    cr_deg = region_of_interest_deg / 2.0

    out["cx"] = (cx_deg - cr_deg, cx_deg + cr_deg)
    out["cy"] = (cy_deg - cr_deg, cy_deg + cr_deg)
    return out


cmapkey = "magma_r"  # "inferno"
depths = np.geomspace(2.5e3, 25e3, 8)  # 36 for presentations #
number_depths = len(depths)

image_rays = pl.image.ImageRays(light_field_geometry=light_field_geometry)

for pk in res.PARTICLES:
    pk_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    uid_trigger_and_quality = snt.logic.intersection(
        passing_trigger[pk]["uid"],
        passing_quality[pk]["uid"],
    )

    event_table = res.event_table(particle_key=pk).query(
        levels_and_columns={
            "primary": ["uid", "energy_GeV", "azimuth_rad", "zenith_rad"],
            "groundgrid_choice": ("uid", "core_x_m", "core_y_m"),
            "cherenkovpool": ["uid", "z_emission_p50_m"],
            "features": ["uid", "num_photons"],
        },
        indices=uid_trigger_and_quality,
        sort=True,
    )

    run = pl.photon_stream.loph.LopfTarReader(
        opj(
            res.response_path(particle_key=pk),
            "reconstructed_cherenkov.loph.tar",
        )
    )

    os.makedirs(opj(res.paths["out_dir"], pk), exist_ok=True)

    counter = counter_init(SAMPLE)
    while counter_not_full(counter, SAMPLE):
        try:
            event = next(run)
        except StopIteration:
            break

        airshower_uid, loph_record = event

        # mandatory
        # ---------
        if airshower_uid not in uid_trigger_and_quality:
            continue

        # optional for cherry picking
        # ---------------------------
        num_pe = len(loph_record["photons"]["arrival_time_slices"])
        if not counter_can_add(counter, num_pe, SAMPLE):
            continue

        event_cx = np.median(
            light_field_geometry.cx_mean[loph_record["photons"]["channels"]]
        )
        event_cy = np.median(
            light_field_geometry.cy_mean[loph_record["photons"]["channels"]]
        )

        roi_limits = make_roi_limits(cx_rad=event_cx, cy_rad=event_cy)

        event_off_deg = np.rad2deg(np.hypot(event_cx, event_cy))
        if event_off_deg > 2.5:
            continue

        event_entry = event_table.query(
            indices=np.array([airshower_uid]),
        )

        core_m = np.hypot(
            event_entry["groundgrid_choice"]["core_x_m"][0],
            event_entry["groundgrid_choice"]["core_y_m"][0],
        )
        if core_m > num_pe / 5:
            print(f"nope, core {core_m:f}m, size {num_pe:f}pe")
            continue

        counter = counter_add(counter, num_pe, SAMPLE)

        evt_dir = opj(pk_dir, f"{airshower_uid:012d}")
        os.makedirs(evt_dir, exist_ok=True)

        tabpath = opj(evt_dir, f"{pk:s}_{airshower_uid:012d}_truth.json")
        json_utils.write(
            tabpath,
            table_to_dict(event_entry),
            indent=4,
        )

        # prepare image intensities
        # -------------------------
        image_stack = np.zeros(
            shape=(number_depths, light_field_geometry.number_pixel)
        )

        for dek in range(number_depths):
            depth = depths[dek]
            (
                pixel_indicies,
                inside_fov,
            ) = image_rays.pixel_ids_of_lixels_in_object_distance(depth)

            # populate image:
            for channel_id in loph_record["photons"]["channels"]:
                if inside_fov[channel_id]:
                    pixel_id = pixel_indicies[channel_id]
                    image_stack[dek, pixel_id] += 1

        # prepare image intensities 2
        # ---------------------------
        images_dir = os.path.join(evt_dir, "images.cache")
        os.makedirs(images_dir, exist_ok=True)

        img_vmax = 0.0
        for obj_idx in range(number_depths):
            reco_object_distance = depths[obj_idx]

            image_path = os.path.join(
                images_dir, "{:06d}.float32".format(obj_idx)
            )

            if os.path.exists(image_path):
                img = plenoptics.analysis.image.read_image(path=image_path)
            else:
                light_field = (
                    loph_record["photons"]["arrival_time_slices"],
                    loph_record["photons"]["channels"],
                )
                img = plenoptics.analysis.image.compute_image(
                    light_field_geometry=light_field_geometry,
                    light_field=loph_to_time_lixel_repr(loph=loph_record),
                    object_distance=reco_object_distance,
                    bins=image_bins,
                    prng=np.random.Generator(np.random.PCG64(obj_idx)),
                )
                plenoptics.analysis.image.write_image(
                    path=image_path, image=img
                )

            img_vmax = np.max([img_vmax, np.max(img)])

        # plot images 2
        # -------------
        fig_colorbar_path = opj(
            evt_dir,
            f"plo.colorbar.{pk:s}_{airshower_uid:012d}.jpg",
        )
        plot_image_colorbar_like_in_plenoptics_phantom(
            path=fig_colorbar_path,
            cmapkey=cmapkey,
            img_vmax_pe_per_deg2=img_vmax / image_bin_solid_angle_deg2,
        )

        for dek in range(number_depths):
            image_path = os.path.join(images_dir, "{:06d}.float32".format(dek))

            fig_image_path = opj(
                evt_dir,
                f"plo.image.{pk:s}_{airshower_uid:012d}_focus{dek:03d}.jpg",
            )

            fig_image_roi_path = opj(
                evt_dir,
                f"plo.image.roi.{pk:s}_{airshower_uid:012d}_focus{dek:03d}.jpg",
            )
            fig_depthbar_path = opj(
                evt_dir,
                f"plo.depthbar.{pk:s}_{airshower_uid:012d}_{dek:03d}.jpg",
            )

            if not os.path.exists(fig_image_path):
                img = plenoptics.analysis.image.read_image(path=image_path)
                plot_image_like_in_plenoptics_phantom(
                    path=fig_image_path,
                    image_edge_bin=image_edge_bin,
                    image_edge_ticks_deg=image_edge_ticks_deg,
                    img=img,
                    img_vmax=img_vmax,
                    cmapkey=cmapkey,
                    max_FoV_diameter_deg=max_FoV_diameter_deg,
                )
                plot_image_like_in_plenoptics_phantom(
                    path=fig_image_roi_path,
                    image_edge_bin=image_edge_bin,
                    image_edge_ticks_deg=image_edge_ticks_deg,
                    img=img,
                    img_vmax=img_vmax,
                    cmapkey=cmapkey,
                    max_FoV_diameter_deg=max_FoV_diameter_deg,
                    roi_limits_deg=roi_limits,
                )
                plot_image_focus_bar_like_in_plenoptics_phantom(
                    path=fig_depthbar_path,
                    object_distances=depths,
                    obj_idx=dek,
                    NPIX=1280,
                )

        # plot images
        # -----------
        continue

        for dek in range(number_depths):
            print(pk, airshower_uid, "depth", dek, "counter", counter)
            figpath = opj(
                evt_dir,
                f"{pk:s}_{airshower_uid:012d}_{dek:03d}.jpg",
            )
            if os.path.exists(figpath):
                continue

            depth = depths[dek]

            fig = sebplt.figure(
                style={
                    "rows": FIG_ROWS,
                    "cols": FIG_COLS,
                    "fontsize": 0.5 * FIC_SCALE,
                }
            )
            ax = sebplt.add_axes(fig=fig, span=[0.175, 0.1, 0.7, 0.85])
            axr = sebplt.add_axes(fig=fig, span=[0.1, 0.1, 0.05, 0.85])
            cax = sebplt.add_axes(fig=fig, span=[0.8, 0.15, 0.025, 0.75])

            colbar = pl.plot.image.add2ax(
                ax=ax,
                I=image_stack[dek, :],
                px=np.rad2deg(light_field_geometry.pixel_pos_cx),
                py=np.rad2deg(light_field_geometry.pixel_pos_cy),
                colormap=cmapkey,
                hexrotation=30,
                vmin=0,
                vmax=np.max(image_stack),
                colorbar=False,
                norm=sebplt.plt_colors.PowerNorm(gamma=CMAP_GAMMA),
            )
            ax.set_aspect("equal")
            sebplt.plt.colorbar(colbar, cax=cax)

            ax.set_xlim(roi_limits["cx"])
            ax.set_ylim(roi_limits["cy"])

            ax.set_ylabel(r"$c_y\,/\,1^{\circ}$")
            fig.text(
                x=0.47,
                y=0.15,
                s=r"$c_x\,/\,1^{\circ}$",
                color="grey",
            )

            pl.plot.ruler.add2ax_object_distance_ruler(
                ax=axr,
                object_distance=depth,
                object_distance_min=min(depths) * 0.9,
                object_distance_max=max(depths) * 1.1,
                label=r"depth$\,/\,$km",
                print_value=False,
                color="black",
            )

            fig.savefig(figpath)
            sebplt.close(fig)

res.stop()
