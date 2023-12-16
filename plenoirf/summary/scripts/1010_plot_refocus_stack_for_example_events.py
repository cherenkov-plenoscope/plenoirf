#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import plenopy as pl
import gamma_ray_reconstruction as gamrec
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
passing_quality = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)

lfg = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
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
    return np.zeros(len(sample["count"]), dtype=np.int)


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

# SITES = irf_config["config"]["sites"]
SITES = ["namibia"]

# PARTICLES= irf_config["config"]["particles"]
PARTICLES = ["gamma", "proton", "helium"]

for sk in SITES:
    sk_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(sk_dir, exist_ok=True)
    for pk in PARTICLES:
        pk_dir = os.path.join(sk_dir, pk)
        os.makedirs(pk_dir, exist_ok=True)

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar"
            ),
            structure=irf.table.STRUCTURE,
        )
        common_idx = spt.intersection(
            [passing_trigger[sk][pk]["idx"], passing_quality[sk][pk]["idx"]]
        )
        events_truth = spt.cut_and_sort_table_on_indices(
            event_table,
            common_indices=common_idx,
            level_keys=[
                "primary",
                "cherenkovsize",
                "grid",
                "cherenkovpool",
                "cherenkovsizepart",
                "cherenkovpoolpart",
                "core",
                "trigger",
                "pasttrigger",
                "cherenkovclassification",
            ],
        )

        run = pl.photon_stream.loph.LopfTarReader(
            os.path.join(
                pa["run_dir"], "event_table", sk, pk, "cherenkov.phs.loph.tar"
            )
        )

        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        counter = counter_init(SAMPLE)
        while counter_not_full(counter, SAMPLE):
            try:
                event = next(run)
            except StopIteration:
                break

            airshower_id, loph_record = event

            # mandatory
            # ---------
            if airshower_id not in passing_trigger[sk][pk]["idx"]:
                continue

            if airshower_id not in passing_quality[sk][pk]["idx"]:
                continue

            # optional for cherry picking
            # ---------------------------
            num_pe = len(loph_record["photons"]["arrival_time_slices"])
            if not counter_can_add(counter, num_pe, SAMPLE):
                continue

            event_cx = np.median(
                lfg.cx_mean[loph_record["photons"]["channels"]]
            )
            event_cy = np.median(
                lfg.cy_mean[loph_record["photons"]["channels"]]
            )
            event_off_deg = np.rad2deg(np.hypot(event_cx, event_cy))
            if event_off_deg > 2.5:
                continue

            event_truth = spt.cut_and_sort_table_on_indices(
                events_truth,
                common_indices=np.array([airshower_id]),
            )

            core_m = np.hypot(
                event_truth["core"]["core_x_m"][0],
                event_truth["core"]["core_x_m"][0],
            )
            if core_m > num_pe / 5:
                print("nope", core_m, num_pe)
                continue

            counter = counter_add(counter, num_pe, SAMPLE)

            evt_dir = os.path.join(pk_dir, "{:012d}".format(airshower_id))
            os.makedirs(evt_dir, exist_ok=True)

            tabpath = os.path.join(
                evt_dir, "{:s}_{:012d}_truth.json".format(pk, airshower_id)
            )
            json_utils.write(
                path=tabpath,
                out_dict=table_to_dict(event_truth),
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
                print(sk, pk, airshower_id, dek, counter)
                figpath = os.path.join(
                    evt_dir,
                    "{:s}_{:012d}_{:03d}.jpg".format(pk, airshower_id, dek),
                )
                if os.path.exists(figpath):
                    continue

                depth = depths[dek]

                fig = seb.figure(
                    style={
                        "rows": FIG_ROWS,
                        "cols": FIG_COLS,
                        "fontsize": 0.5 * FIC_SCALE,
                    }
                )
                ax = seb.add_axes(fig=fig, span=[0.175, 0.1, 0.7, 0.85])
                axr = seb.add_axes(fig=fig, span=[0.1, 0.1, 0.05, 0.85])
                cax = seb.add_axes(fig=fig, span=[0.8, 0.15, 0.025, 0.75])

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
                    norm=seb.plt_colors.PowerNorm(gamma=CMAP_GAMMA),
                )
                ax.set_aspect("equal")
                seb.plt.colorbar(colbar, cax=cax)

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
                seb.close(fig)
