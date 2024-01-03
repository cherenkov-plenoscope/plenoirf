import os
from os import path as op
from os.path import join as opj
import tarfile
import numpy as np

import corsika_primary as cpw
import json_utils
import sparse_numeric_table as spt
import atmospheric_cherenkov_response as acr
import spherical_coordinates
import dynamicsizerecarray

from .. import bookkeeping
from .. import ground_grid
from .. import event_table
from .. import outer_telescope_array
from .. import tar_append

from . import transform_cherenkov_bunches
from . import job_io


def run_job(job, logger):
    cache_path = os.path.join(job["paths"]["tmp_dir"], "corsika_and_grid")

    if os.path.exists(cache_path) and job["cache"]:
        logger.info("corsika_and_grid, read cache")
        return job_io.read(path=cache_path)
    else:
        logger.info("corsika_and_grid, run corsika")
        job = corsika_and_grid(job=job, logger=logger)

        if job["cache"]:
            logger.info("corsika_and_grid, write cache")
            job_io.write(path=cache_path, job=job)

    return job


def corsika_and_grid(job, logger):
    logger.info("corsika_and_grid, start corsika")

    with cpw.cherenkov.CherenkovEventTapeWriter(
        path=job["paths"]["tmp"]["cherenkov_pools"]
    ) as evttar, tarfile.open(
        job["paths"]["tmp"]["ground_grid_intensity"], "w"
    ) as imgtar, tarfile.open(
        job["paths"]["tmp"]["ground_grid_intensity_roi"], "w"
    ) as imgroitar:
        with cpw.CorsikaPrimary(
            corsika_path=job["config"]["executables"]["corsika_primary_path"],
            steering_dict=job["run"]["corsika_primary_steering"],
            stdout_path=job["paths"]["tmp"]["corsika_stdout"],
            stderr_path=job["paths"]["tmp"]["corsika_stderr"],
            particle_output_path=job["paths"]["tmp"]["particle_pools_dat"],
        ) as corsika_run:
            logger.info("corsika_and_grid, corsika is ready")
            evttar.write_runh(runh=corsika_run.runh)

            for event_idx, corsika_event in enumerate(corsika_run):
                corsika_evth, cherenkov_reader = corsika_event

                cherenkov_bunches = read_all_cherenkov_bunches(
                    cherenkov_reader=cherenkov_reader
                )

                uid = nail_down_event_identity(
                    corsika_evth=corsika_evth,
                    event_idx=event_idx,
                    corsika_primary_steering=job["run"][
                        "corsika_primary_steering"
                    ],
                )
                logger.info(
                    "corsika_and_grid, shower uid {:s}".format(uid["uid_str"])
                )

                primary_rec = make_primary_record(
                    uid=uid,
                    corsika_evth=corsika_evth,
                    corsika_primary_steering=job["run"][
                        "corsika_primary_steering"
                    ],
                    primary_directions=job["run"]["primary_directions"],
                )
                job["event_table"]["primary"].append_record(primary_rec)

                pointing_rec = make_pointing_record(
                    uid=uid, pointings=job["run"]["pointings"]
                )
                job["event_table"]["pointing"].append_record(pointing_rec)
                _ = pointing_rec.pop("idx")
                pointing = pointing_rec

                cherenkovsize_rec = make_cherenkovsize_record(
                    uid=uid,
                    cherenkov_bunches=cherenkov_bunches,
                )
                job["event_table"]["cherenkovsize"].append_record(
                    cherenkovsize_rec
                )

                cherenkov_pool_median_x_m = 0.0
                cherenkov_pool_median_y_m = 0.0
                if cherenkovsize_rec["num_bunches"] > 0:
                    cherenkovpool_rec = make_cherenkovpool_record(
                        uid=uid,
                        cherenkov_bunches=cherenkov_bunches,
                    )
                    job["event_table"]["cherenkovpool"].append_record(
                        cherenkovpool_rec
                    )
                    cherenkov_pool_median_x_m = cherenkovpool_rec["x_median_m"]
                    cherenkov_pool_median_y_m = cherenkovpool_rec["y_median_m"]

                groundgrid_config = ground_grid.make_ground_grid_config(
                    bin_width_m=job["config"]["ground_grid"]["geometry"][
                        "bin_width_m"
                    ],
                    num_bins_each_axis=job["config"]["ground_grid"][
                        "geometry"
                    ]["num_bins_each_axis"],
                    cherenkov_pool_median_x_m=cherenkov_pool_median_x_m,
                    cherenkov_pool_median_y_m=cherenkov_pool_median_y_m,
                    prng=job["prng"],
                )

                groundgrid = ground_grid.GroundGrid(
                    bin_width_m=groundgrid_config["bin_width_m"],
                    num_bins_each_axis=groundgrid_config["num_bins_each_axis"],
                    center_x_m=groundgrid_config["center_x_m"],
                    center_y_m=groundgrid_config["center_y_m"],
                )

                fov_mask = mask_cherenkov_bunches_in_instruments_field_of_view(
                    cherenkov_bunches=cherenkov_bunches,
                    pointing=pointing,
                    field_of_view_half_angle_rad=job["instrument"][
                        "field_of_view_half_angle_rad"
                    ],
                )
                cherenkov_bunches_in_fov = cherenkov_bunches[fov_mask]
                del cherenkov_bunches

                groundgrid_result, groundgrid_debug = ground_grid.assign(
                    groundgrid=groundgrid,
                    cherenkov_bunches=cherenkov_bunches_in_fov,
                    threshold_num_photons=job["config"]["ground_grid"][
                        "threshold_num_photons"
                    ],
                    prng=job["prng"],
                )

                groundgrid_rec = make_groundgrid_record(
                    uid=uid,
                    groundgrid_config=groundgrid_config,
                    groundgrid_result=groundgrid_result,
                    groundgrid=groundgrid,
                )
                job["event_table"]["groundgrid"].append_record(groundgrid_rec)

                if groundgrid_result["choice"]:
                    cherenkov_bunches_in_choice = cherenkov_bunches_in_fov[
                        groundgrid_result["choice"]["cherenkov_bunches_idxs"]
                    ]
                    del cherenkov_bunches_in_fov

                    cherenkov_bunches_in_instrument = transform_cherenkov_bunches.from_obervation_level_to_instrument(
                        cherenkov_bunches=cherenkov_bunches_in_choice,
                        instrument_pointing=pointing,
                        instrument_pointing_model=job["config"]["pointing"][
                            "model"
                        ],
                        instrument_x_m=groundgrid_result["choice"]["core_x_m"],
                        instrument_y_m=groundgrid_result["choice"]["core_y_m"],
                        speed_of_ligth_m_per_s=job["instrument"][
                            "local_speed_of_light_m_per_s"
                        ],
                    )
                    del cherenkov_bunches_in_choice

                    core_rec = make_core_record(
                        uid=uid,
                        groundgrid_result_choice=groundgrid_result["choice"],
                    )
                    job["event_table"]["core"].append_record(core_rec)

                    EventTape_append_event(
                        evttar=evttar,
                        corsika_evth=corsika_evth,
                        cherenkov_bunches=cherenkov_bunches_in_instrument,
                        core_x_m=groundgrid_result["choice"]["core_x_m"],
                        core_y_m=groundgrid_result["choice"]["core_y_m"],
                    )

                    ImgRoiTar_append(
                        imgroitar=imgroitar,
                        uid=uid,
                        groundgrid_result=groundgrid_result,
                        groundgrid_debug=groundgrid_debug,
                    )

                    if uid["uid"] in job["run"]["event_uids_for_debugging"]:
                        ImgTar_append(
                            imgtar=imgtar,
                            uid=uid,
                            groundgrid=groundgrid,
                            groundgrid_debug=groundgrid_debug,
                        )

                    cherenkovsizepart_rec = make_cherenkovsize_record(
                        uid=uid,
                        cherenkov_bunches=cherenkov_bunches_in_instrument,
                    )
                    job["event_table"]["cherenkovsizepart"].append_record(
                        cherenkovsizepart_rec
                    )

                    if cherenkovsizepart_rec["num_bunches"] > 0:
                        cherenkovpoolpart_rec = make_cherenkovpool_record(
                            uid=uid,
                            cherenkov_bunches=cherenkov_bunches_in_instrument,
                        )
                        job["event_table"]["cherenkovpoolpart"].append_record(
                            cherenkovpoolpart_rec
                        )

                with open(
                    opj(
                        job["paths"]["tmp_dir"],
                        uid["uid_str"] + "_ground_grid.json",
                    ),
                    "wt",
                ) as f:
                    f.write(json_utils.dumps(groundgrid_result))

    logger.info("corsika_and_grid, particle output from dat to tar")
    cpw.particles.dat_to_tape(
        dat_path=job["paths"]["tmp"]["particle_pools_dat"],
        tape_path=job["paths"]["tmp"]["particle_pools_tar"],
    )

    return job


def nail_down_event_identity(
    corsika_evth, corsika_primary_steering, event_idx
):
    run_id = int(corsika_evth[cpw.I.EVTH.RUN_NUMBER])
    assert run_id == corsika_primary_steering["run"]["run_id"]
    event_id = event_idx + 1
    assert event_id == corsika_evth[cpw.I.EVTH.EVENT_NUMBER]
    uid = bookkeeping.uid.make_uid(run_id=run_id, event_id=event_id)
    uid_str = bookkeeping.uid.make_uid_str(run_id=run_id, event_id=event_id)

    out = {
        "record": {spt.IDX: uid},
        "uid": uid,
        "uid_str": uid_str,
        "run_id": run_id,
        "event_id": event_id,
        "event_idx": event_idx,
    }
    return out


def make_primary_record(
    uid, corsika_evth, corsika_primary_steering, primary_directions
):
    primary = corsika_primary_steering["primaries"][uid["event_idx"]]
    primary_direction = primary_directions[uid["event_idx"]]

    rec = uid["record"].copy()
    rec["particle_id"] = primary["particle_id"]
    rec["energy_GeV"] = primary["energy_GeV"]
    rec["azimuth_rad"] = primary["azimuth_rad"]
    rec["zenith_rad"] = primary["zenith_rad"]
    rec["depth_g_per_cm2"] = primary["depth_g_per_cm2"]

    rec["momentum_x_GeV_per_c"] = corsika_evth[
        cpw.I.EVTH.PX_MOMENTUM_GEV_PER_C
    ]
    rec["momentum_y_GeV_per_c"] = corsika_evth[
        cpw.I.EVTH.PY_MOMENTUM_GEV_PER_C
    ]
    rec["momentum_z_GeV_per_c"] = (
        -1.0 * corsika_evth[cpw.I.EVTH.PZ_MOMENTUM_GEV_PER_C]
    )

    rec["starting_height_asl_m"] = (
        cpw.CM2M * corsika_evth[cpw.I.EVTH.STARTING_HEIGHT_CM]
    )
    obs_lvl_intersection = acr.utils.ray_plane_x_y_intersection(
        support=[0, 0, rec["starting_height_asl_m"]],
        direction=[
            rec["momentum_x_GeV_per_c"],
            rec["momentum_y_GeV_per_c"],
            rec["momentum_z_GeV_per_c"],
        ],
        plane_z=corsika_primary_steering["run"]["observation_level_asl_m"],
    )

    rec["starting_x_m"] = -1.0 * obs_lvl_intersection[0]
    rec["starting_y_m"] = -1.0 * obs_lvl_intersection[1]

    rec["first_interaction_height_asl_m"] = (
        -1.0 * cpw.CM2M * corsika_evth[cpw.I.EVTH.Z_FIRST_INTERACTION_CM]
    )

    rec["solid_angle_thrown_sr"] = primary_direction["solid_angle_thrown_sr"]
    rec["inner_atmopsheric_magnetic_cutoff"] = primary_direction["cutoff"]

    return rec


def make_pointing_record(uid, pointings):
    rec = uid["record"].copy()
    for key in ["azimuth_rad", "zenith_rad"]:
        rec[key] = pointings[uid["event_idx"]][key]
    return rec


def make_cherenkovsize_record(uid, cherenkov_bunches):
    rec = uid["record"].copy()
    rec["num_bunches"] = cherenkov_bunches.shape[0]
    rec["num_photons"] = np.sum(cherenkov_bunches[:, cpw.I.BUNCH.BUNCH_SIZE_1])
    return rec


def make_cherenkovpool_record(uid, cherenkov_bunches):
    rec = uid["record"].copy()
    cb = cherenkov_bunches
    assert cb.shape[0] > 0
    rec["maximum_asl_m"] = cpw.CM2M * np.median(
        cb[:, cpw.I.BUNCH.EMISSOION_ALTITUDE_ASL_CM]
    )
    rec["wavelength_median_nm"] = np.abs(
        np.median(cb[:, cpw.I.BUNCH.WAVELENGTH_NM])
    )
    rec["cx_median_rad"] = np.median(cb[:, cpw.I.BUNCH.CX_RAD])
    rec["cy_median_rad"] = np.median(cb[:, cpw.I.BUNCH.CY_RAD])
    rec["x_median_m"] = cpw.CM2M * np.median(cb[:, cpw.I.BUNCH.X_CM])
    rec["y_median_m"] = cpw.CM2M * np.median(cb[:, cpw.I.BUNCH.Y_CM])
    rec["bunch_size_median"] = np.median(cb[:, cpw.I.BUNCH.BUNCH_SIZE_1])
    return rec


def make_groundgrid_record(
    uid, groundgrid_config, groundgrid_result, groundgrid
):
    rec = uid["record"].copy()
    rec.update(groundgrid_config)

    rec["num_bins_thrown"] = groundgrid["num_bins_thrown"]
    rec["num_bins_above_threshold"] = groundgrid_result[
        "num_bins_above_threshold"
    ]
    rec["area_thrown_m2"] = groundgrid["area_thrown_m2"]

    rec["num_photons_overflow"] = groundgrid_result["num_photons_overflow"]

    # compare scatter
    # ---------------
    scathist = groundgrid_result["scatter_histogram"]
    num_bins = len(scathist["bin_counts"])
    assert num_bins == 16
    for rbin in range(num_bins):
        rec["scatter_rbin_{:02d}".format(rbin)] = scathist["bin_counts"][rbin]
    return rec


def make_core_record(uid, groundgrid_result_choice):
    rec = uid["record"].copy()
    for key in event_table.structure.init_core_level_structure():
        rec[key] = groundgrid_result_choice[key]
    return rec


def read_all_cherenkov_bunches(cherenkov_reader):
    return np.vstack([b for b in cherenkov_reader])


def mask_cherenkov_bunches_in_instruments_field_of_view(
    cherenkov_bunches,
    pointing,
    field_of_view_half_angle_rad,
):
    OVERHEAD = 2.0
    return mask_cherenkov_bunches_in_cone(
        cherenkov_bunches_cx=cherenkov_bunches[:, cpw.I.BUNCH.CX_RAD],
        cherenkov_bunches_cy=cherenkov_bunches[:, cpw.I.BUNCH.CY_RAD],
        cone_azimuth_rad=pointing["azimuth_rad"],
        cone_zenith_rad=pointing["zenith_rad"],
        cone_half_angle_rad=OVERHEAD * field_of_view_half_angle_rad,
    )


def mask_cherenkov_bunches_in_cone(
    cherenkov_bunches_cx,
    cherenkov_bunches_cy,
    cone_half_angle_rad,
    cone_azimuth_rad,
    cone_zenith_rad,
):
    cone_cx, cone_cy = spherical_coordinates.az_zd_to_cx_cy(
        azimuth_rad=cone_azimuth_rad,
        zenith_rad=cone_zenith_rad,
    )
    delta_rad = spherical_coordinates.angle_between_cx_cy(
        cx1=cherenkov_bunches_cx,
        cy1=cherenkov_bunches_cy,
        cx2=cone_cx,
        cy2=cone_cy,
    )
    return delta_rad < cone_half_angle_rad


def EventTape_append_event(
    evttar,
    corsika_evth,
    cherenkov_bunches,
    core_x_m,
    core_y_m,
):
    evth = corsika_evth.copy()
    evth[cpw.I.EVTH.NUM_REUSES_OF_CHERENKOV_EVENT] = 1.0
    evth[cpw.I.EVTH.X_CORE_CM(reuse=1)] = cpw.M2CM * core_x_m
    evth[cpw.I.EVTH.Y_CORE_CM(reuse=1)] = cpw.M2CM * core_y_m
    evttar.write_evth(evth=evth)
    evttar.write_payload(payload=cherenkov_bunches)


def ImgRoiTar_append(imgroitar, uid, groundgrid_result, groundgrid_debug):
    bb = outer_telescope_array.init_binning()
    roi_array = ground_grid.bin_photon_assignment_to_array_roi(
        bin_photon_assignment=groundgrid_debug["bin_photon_assignment"],
        x_bin=groundgrid_result["choice"]["bin_idx_x"],
        y_bin=groundgrid_result["choice"]["bin_idx_y"],
        r_bin=bb["num_bins_radius"],
        dtype=np.float32,
    )
    tar_append.tar_append(
        tarout=imgroitar,
        filename=uid["uid_str"] + ".f4.gz",
        filebytes=ground_grid.io.histogram_to_bytes(roi_array),
    )


def ImgTar_append(imgtar, uid, groundgrid, groundgrid_debug):
    img = ground_grid.bin_photon_assignment_to_array(
        bin_photon_assignment=groundgrid_debug["bin_photon_assignment"],
        num_bins_each_axis=groundgrid["num_bins_each_axis"],
        dtype=np.float32,
    )
    tar_append.tar_append(
        tarout=imgtar,
        filename=uid["uid_str"] + ".f4.gz",
        filebytes=ground_grid.io.histogram_to_bytes(img),
    )
