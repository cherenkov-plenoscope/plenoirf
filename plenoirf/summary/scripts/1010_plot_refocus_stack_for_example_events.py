#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import plenopy as pl
import gamma_ray_reconstruction as gamrec
import sebastians_matplotlib_addons as sebplt
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

passing_trigger = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)

lfg = pl.LightFieldGeometry(
    opj(
        res.paths["plenoirf_dir"],
        "plenoptics",
        "instruments",
        res.instrument_key,
        "light_field_geometry",
    )
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


NUM_EVENTS_PER_PARTICLE = 10
MIN_NUM_PHOTONS = 1000
colormap = "inferno"

depths = np.geomspace(2.5e3, 25e3, 12)
number_depths = len(depths)

image_rays = pl.image.ImageRays(light_field_geometry=lfg)

for pk in res.PARTICLES:
    pk_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    uid_trigger_and_quality = snt.logic.intersection(
        [
            passing_trigger[pk]["uid"],
            passing_quality[pk]["uid"],
        ]
    )

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={
                "groundgrid_choice": ("uid", "core_x_m", "core_y_m"),
            }
        )
        event_table = snt.logic.cut_and_sort_table_on_indices(
            event_table,
            common_indices=uid_trigger_and_quality,
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

        event_cx = np.median(lfg.cx_mean[loph_record["photons"]["channels"]])
        event_cy = np.median(lfg.cy_mean[loph_record["photons"]["channels"]])
        event_off_deg = np.rad2deg(np.hypot(event_cx, event_cy))
        if event_off_deg > 2.5:
            continue

        event_entry = snt.logic.cut_and_sort_table_on_indices(
            event_table,
            common_indices=np.array([airshower_uid]),
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
        image_stack = np.zeros(shape=(number_depths, lfg.number_pixel))

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

        # plot images
        # -----------
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
                px=np.rad2deg(lfg.pixel_pos_cx),
                py=np.rad2deg(lfg.pixel_pos_cy),
                colormap=colormap,
                hexrotation=30,
                vmin=0,
                vmax=np.max(image_stack),
                colorbar=False,
                norm=sebplt.plt_colors.PowerNorm(gamma=CMAP_GAMMA),
            )
            ax.set_aspect("equal")
            sebplt.plt.colorbar(colbar, cax=cax)

            # region of interest
            roi_cx_deg = np.rad2deg(event_cx)
            roi_cy_deg = np.rad2deg(event_cy)
            cxstart = roi_cx_deg - REGION_OF_INTEREST_DEG / 2
            cxstop = roi_cx_deg + REGION_OF_INTEREST_DEG / 2
            cystart = roi_cy_deg - REGION_OF_INTEREST_DEG / 2
            cystop = roi_cy_deg + REGION_OF_INTEREST_DEG / 2
            ax.set_xlim([cxstart, cxstop])
            ax.set_ylim([cystart, cystop])

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
