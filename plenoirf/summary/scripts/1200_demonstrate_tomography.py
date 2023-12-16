#!/usr/bin/python
import sys
import plenoirf as irf
import plenopy as pl
import scipy
import os
import numpy as np
import json_utils
import sparse_numeric_table as spt
import sebastians_matplotlib_addons as seb
import glob
import tempfile
import tarfile
import skimage
import multiprocessing
from skimage.draw import circle as skimage_draw_circle


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

production_key = "demo_helium_for_tomography"
demo_helium_dir = os.path.join(pa["run_dir"], production_key)

if not os.path.exists(demo_helium_dir):
    job = irf.production.example.make_helium_demo_for_tomography(
        run_dir=pa["run_dir"],
        num_air_showers=50,
        production_key=production_key,
        site_key="namibia",
        max_scatter_radius_m=150,
    )
    irf.instrument_response.run_job(job)
    irf.instrument_response.reduce(
        run_dir=pa["run_dir"],
        production_key=production_key,
        site_key="namibia",
        particle_key="helium",
        LAZY=False,
    )


NUM_EVENTS_PER_PARTICLE = 5
MIN_NUM_CHERENKOV_PHOTONS = 200
MAX_CORE_DISTANCE = 400
TOMO_NUM_ITERATIONS = 550
NUM_THREADS = 6

os.makedirs(pa["out_dir"], exist_ok=True)

light_field_geometry = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
)
FIELD_OF_VIEW_RADIUS_DEG = 0.5 * np.rad2deg(
    light_field_geometry.sensor_plane2imaging_system.max_FoV_diameter
)


def get_airshower_uids_in_directory(directory, wildcard="*.tar"):
    raw_sensor_paths = glob.glob(os.path.join(directory, wildcard))
    filenames = [os.path.split(raw_path)[1] for raw_path in raw_sensor_paths]
    airshower_uid_strs = [os.path.splitext(fn)[0] for fn in filenames]
    airshower_uids = [int(uidstr) for uidstr in airshower_uid_strs]
    return set(airshower_uids)


def read_simulation_truth_for_tomography(
    raw_sensor_response_dir, uid, tomography_binning
):
    with tempfile.TemporaryDirectory() as tmpdir:
        uid_str = irf.unique.UID_FOTMAT_STR.format(uid)
        response_path = os.path.join(tmpdir, uid_str)
        response_tar_path = os.path.join(
            raw_sensor_response_dir, uid_str + ".tar"
        )
        with tarfile.open(response_tar_path, "r") as tf:
            tf.extractall(response_path)
            event = pl.Event(
                path=response_path,
                light_field_geometry=light_field_geometry,
            )
            simulation_truth = (
                pl.Tomography.Image_Domain.Simulation_Truth.init(
                    event=event,
                    binning=tomography_binning,
                )
            )
    return simulation_truth


def make_binning_from_ligh_field_geometry(light_field_geometry):
    lfg = light_field_geometry
    design = lfg.sensor_plane2imaging_system

    NUM_CAMERAS_ON_DIAGONAL = int(
        0.707 * design.max_FoV_diameter / design.pixel_FoV_hex_flat2flat
    )
    NUM_CAMERAS_ON_DIAGONAL = 2 * (NUM_CAMERAS_ON_DIAGONAL // 2)

    binning = pl.Tomography.Image_Domain.Binning.init(
        focal_length=design.expected_imaging_system_focal_length,
        cx_min=-0.5 * design.max_FoV_diameter,
        cx_max=+0.5 * design.max_FoV_diameter,
        number_cx_bins=NUM_CAMERAS_ON_DIAGONAL,
        cy_min=-0.5 * design.max_FoV_diameter,
        cy_max=+0.5 * design.max_FoV_diameter,
        number_cy_bins=NUM_CAMERAS_ON_DIAGONAL,
        obj_min=40 * design.expected_imaging_system_focal_length,
        obj_max=240 * design.expected_imaging_system_focal_length,
        number_obj_bins=NUM_CAMERAS_ON_DIAGONAL // 4,
    )
    return binning


def ax_add_tomography(ax, binning, reconstruction, simulation_truth):
    # as cube
    # -------
    ivolrec = pl.Tomography.Image_Domain.Binning.volume_intensity_as_cube(
        volume_intensity=reconstruction["reconstructed_volume_intensity"],
        binning=binning,
    )
    ivolrec = ivolrec / np.sum(ivolrec)

    ivoltru = pl.Tomography.Image_Domain.Binning.volume_intensity_as_cube(
        volume_intensity=simulation_truth["true_volume_intensity"],
        binning=binning,
    )
    ivoltru = ivoltru / np.sum(ivoltru)

    # validity
    # --------
    if np.any(np.isnan(ivoltru)):
        print(
            uid,
            "Failed to construct volume of true Cherenkov-emission.",
        )
        return

    # normalizing
    # -----------
    imax = np.max([np.max(ivolrec), np.max(ivoltru)])
    imin = np.min([np.min(ivolrec[ivolrec > 0]), np.min(ivoltru[ivoltru > 0])])

    UU = 0.2
    the = np.deg2rad(50)

    for iz in range(binning["number_sen_z_bins"]):
        projection = np.array(
            [
                [np.cos(the), -np.sin(the) * (1.0 / UU), 0],
                [np.sin(the), np.cos(the) * UU, 5.3 - (5.3 * iz)],
                [0, 0, 1],
            ]
        )
        intensity_rgb = np.zeros(
            shape=(
                binning["number_cx_bins"],
                binning["number_cy_bins"],
                4,
            ),
            dtype=np.float,
        )
        intensity_rgb[:, :, 0] = ivolrec[:, :, iz] / imax
        intensity_rgb[:, :, 1] = ivoltru[:, :, iz] / imax
        intensity_rgb[:, :, 2] = (
            intensity_rgb[:, :, 0] * intensity_rgb[:, :, 1]
        ) ** (1 / 3)

        seb.pseudo3d.ax_add_mesh_intensity_to_alpha(
            ax=ax,
            projection=projection,
            x_bin_edges=np.rad2deg(binning["cx_bin_edges"]),
            y_bin_edges=np.rad2deg(binning["cy_bin_edges"]),
            intensity_rgb=intensity_rgb,
            threshold=0.01,
            gamma=0.5,
        )
        seb.pseudo3d.ax_add_grid(
            ax=ax,
            projection=projection,
            x_bin_edges=np.rad2deg(binning["cx_bin_edges"]),
            y_bin_edges=np.rad2deg(binning["cy_bin_edges"]),
            alpha=0.22,
            linewidth=0.1,
            color="k",
            linestyle="-",
        )
        seb.pseudo3d.ax_add_circle(
            ax=ax,
            projection=projection,
            x=0.0,
            y=0.0,
            r=3.25,
            alpha=0.66,
            linewidth=0.1,
            color="k",
            linestyle="-",
        )


def write_figure_tomography(path, binning, reconstruction, simulation_truth):
    RRR = 1280
    axes_style = {"spines": [], "axes": [], "grid": False}
    fig = seb.figure({"rows": 4 * RRR, "cols": RRR, "fontsize": 1})
    ax = seb.add_axes(fig=fig, span=[0.0, 0.0, 1.0, 1.0], style=axes_style)
    ax_add_tomography(
        ax=ax,
        binning=binning,
        reconstruction=reconstruction,
        simulation_truth=simulation_truth,
    )
    ax.set_aspect("equal")
    fig.savefig(path, transparent=True)
    seb.close(fig)


binning = make_binning_from_ligh_field_geometry(
    light_field_geometry=light_field_geometry
)

pl.Tomography.Image_Domain.Binning.write(
    binning=binning,
    path=os.path.join(
        pa["out_dir"], "binning_for_tomography_in_image_domain.json"
    ),
)

system_matrix_path = os.path.join(
    pa["out_dir"], "system_matrix_for_tomography_in_image_domain.bin"
)

pool = multiprocessing.Pool(NUM_THREADS)

if not os.path.exists(system_matrix_path):
    jobs = pl.Tomography.System_Matrix.make_jobs(
        light_field_geometry=light_field_geometry,
        sen_x_bin_edges=binning["sen_x_bin_edges"],
        sen_y_bin_edges=binning["sen_y_bin_edges"],
        sen_z_bin_edges=binning["sen_z_bin_edges"],
        random_seed=sum_config["random_seed"],
        num_lixels_in_job=1000,
        num_samples_per_lixel=10,
    )

    results = pool.map(pl.Tomography.System_Matrix.run_job, jobs)

    sparse_system_matrix = pl.Tomography.System_Matrix.reduce_results(results)
    pl.Tomography.System_Matrix.write(
        sparse_system_matrix=sparse_system_matrix, path=system_matrix_path
    )

sparse_system_matrix = pl.Tomography.System_Matrix.read(
    path=system_matrix_path
)

tomo_psf = pl.Tomography.Image_Domain.Point_Spread_Function.init(
    sparse_system_matrix=sparse_system_matrix
)

for sk in ["namibia"]:
    for pk in ["helium"]:
        sk_pk_dir = os.path.join(pa["run_dir"], production_key, sk, pk)

        print("===", sk, pk, "===")

        print("- read event_table")
        event_table = spt.read(
            path=os.path.join(
                sk_pk_dir,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        print("- find events with full output")
        raw_sensor_response_dir = os.path.join(sk_pk_dir, "past_trigger.map")
        have_raw_sensor_response_uids = get_airshower_uids_in_directory(
            directory=raw_sensor_response_dir
        )

        print("- read list-of-photons")
        run = pl.photon_stream.loph.LopfTarReader(
            os.path.join(sk_pk_dir, "cherenkov.phs.loph.tar")
        )

        event_counter = 0
        while event_counter < NUM_EVENTS_PER_PARTICLE:
            try:
                event = next(run)
            except StopIteration:
                break

            uid, loph_record = event

            num_cherenkov_photons = len(loph_record["photons"]["channels"])
            if num_cherenkov_photons < MIN_NUM_CHERENKOV_PHOTONS:
                print(uid, "not enough photons")
                continue

            if uid not in have_raw_sensor_response_uids:
                print(uid, "no full output avaiable")
                continue

            event_core = spt.cut_and_sort_table_on_indices(
                table=event_table,
                common_indices=np.array([uid]),
                level_keys=["core"],
            )["core"]
            event_core_distance = np.hypot(
                event_core["core_x_m"][0],
                event_core["core_y_m"][0],
            )

            if event_core_distance > MAX_CORE_DISTANCE:
                print(uid, "distance to core too large", event_core_distance)
                continue

            event_counter += 1

            print(sk, pk, uid, event_core_distance)

            simulation_truth = read_simulation_truth_for_tomography(
                raw_sensor_response_dir=raw_sensor_response_dir,
                uid=uid,
                tomography_binning=binning,
            )

            result_dir = os.path.join(
                pa["out_dir"], sk, pk, irf.unique.UID_FOTMAT_STR.format(uid)
            )

            os.makedirs(result_dir, exist_ok=True)
            reconstruction_path = os.path.join(
                result_dir, "reconstruction.json"
            )

            if not os.path.exists(reconstruction_path):
                reconstruction = (
                    pl.Tomography.Image_Domain.Reconstruction.init(
                        light_field_geometry=light_field_geometry,
                        photon_lixel_ids=loph_record["photons"]["channels"],
                        binning=binning,
                    )
                )
                with open(reconstruction_path, "wt") as f:
                    f.write(json_utils.dumps(reconstruction))

            with open(reconstruction_path, "rt") as f:
                reconstruction = json_utils.loads(f.read())

            num_missing_iterations = (
                TOMO_NUM_ITERATIONS - reconstruction["iteration"]
            )
            if num_missing_iterations > 0:
                for i in range(num_missing_iterations):
                    reconstruction = (
                        pl.Tomography.Image_Domain.Reconstruction.iterate(
                            reconstruction=reconstruction,
                            point_spread_function=tomo_psf,
                        )
                    )
                    if reconstruction["iteration"] % 25 == 0:
                        stack_path = os.path.join(
                            result_dir,
                            "stack_{:06d}.jpg".format(
                                reconstruction["iteration"]
                            ),
                        )
                        write_figure_tomography(
                            path=stack_path,
                            binning=binning,
                            reconstruction=reconstruction,
                            simulation_truth=simulation_truth,
                        )
                with open(reconstruction_path, "wt") as f:
                    f.write(json_utils.dumps(reconstruction))
