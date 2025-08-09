import os
from os.path import join as opj
import tarfile
import numpy as np
import gzip
import hashlib

import corsika_primary as cpw
import json_utils
import json_line_logger
import pickle
import atmospheric_cherenkov_response as acr
import spherical_coordinates
import dynamicsizerecarray
import rename_after_writing as rnw
import sparse_numeric_table as snt

from .. import bookkeeping
from .. import ground_grid
from .. import event_table
from .. import outer_telescope_array
from .. import tar_append
from .. import seeding
from .. import utils
from .. import event_table

from . import transform_cherenkov_bunches
from . import cherenkov_bunch_storage


def run(env, part, seed):
    name = __name__.split(".")[-1]
    module_work_dir = opj(env["work_dir"], part, name)

    if os.path.exists(module_work_dir):
        return

    os.makedirs(module_work_dir)
    logger = json_line_logger.LoggerFile(opj(module_work_dir, "log.jsonl"))
    logger.info(__name__)
    logger.info(f"seed: {seed:d}")

    prng = np.random.Generator(np.random.PCG64(seed))

    logger.info("reading primaries_and_pointings.")
    with gzip.open(
        opj(
            env["work_dir"],
            "prm2cer",
            "draw_primaries_and_pointings",
            "result.pkl.gz",
        ),
        "rb",
    ) as fin:
        dpp = pickle.loads(fin.read())

    logger.info("reading event_uids_for_debugging.")
    event_uids_for_debugging = json_utils.read(
        path=opj(
            env["work_dir"],
            "prm2cer",
            "draw_event_uids_for_debugging",
            "event_uids_for_debugging.json",
        )
    )

    evttab = snt.SparseNumericTable(index_key="uid")
    additional_level_keys = [
        "primary",
        "cherenkovsize",
        "cherenkovpool",
        "groundgrid",
        "groundgrid_choice",
    ]
    for key in additional_level_keys:
        evttab = event_table.add_empty_level(evttab, key)

    logger.info("simulating showers.")
    evttab = corsika_first_run(
        env=env,
        prng=prng,
        evttab=evttab,
        corsika_and_grid_work_dir=module_work_dir,
        corsika_primary_steering=dpp["corsika_primary_steering"],
        primary_directions=dpp["primary_directions"],
        event_uids_for_debugging=event_uids_for_debugging,
        logger=logger,
    )

    event_table.write_certain_levels_to_path(
        evttab=evttab,
        path=opj(module_work_dir, "event_table.snt.zip"),
        level_keys=additional_level_keys,
    )
    logger.info("done.")
    json_line_logger.shutdown(logger=logger)

    # tidy up and compress
    utils.gzip_file(opj(module_work_dir, "log.jsonl"))
    utils.gzip_file(opj(module_work_dir, "cherenkovpools_md5.json"))
    utils.gzip_file(opj(module_work_dir, "corsika.stdout.txt"))
    utils.gzip_file(opj(module_work_dir, "corsika.stderr.txt"))
    os.remove(opj(module_work_dir, "particle_pools.dat"))


def corsika_first_run(
    env,
    prng,
    evttab,
    corsika_and_grid_work_dir,
    corsika_primary_steering,
    primary_directions,
    event_uids_for_debugging,
    logger,
):
    logger.info("Start corsika first run.")
    work_dir = corsika_and_grid_work_dir
    os.makedirs(work_dir, exist_ok=True)

    cherenkovpools_md5 = {}

    with tarfile.open(
        opj(work_dir, "ground_grid_intensity.tar"), "w"
    ) as imgtar, tarfile.open(
        opj(work_dir, "ground_grid_intensity_roi.tar"), "w"
    ) as imgroitar:
        with cpw.CorsikaPrimary(
            steering_dict=corsika_primary_steering,
            stdout_path=opj(work_dir, "corsika.stdout.txt"),
            stderr_path=opj(work_dir, "corsika.stderr.txt"),
            particle_output_path=opj(work_dir, "particle_pools.dat"),
        ) as corsika_run:
            logger.info("corsika is ready.")

            GGH = ground_grid.GGH()

            for event_idx, corsika_event in enumerate(corsika_run):
                corsika_evth, cherenkov_block_reader = corsika_event
                uid = nail_down_event_identity(
                    corsika_evth=corsika_evth,
                    event_idx=event_idx,
                    corsika_primary_steering=corsika_primary_steering,
                )

                logger.debug(
                    json_line_logger.xml(
                        "EventTime", uid=uid["uid_str"], status="start."
                    )
                )

                evttab["primary"].append_record(
                    make_primary_record(
                        uid=uid,
                        corsika_evth=corsika_evth,
                        corsika_primary_steering=corsika_primary_steering,
                        primary_directions=primary_directions,
                    )
                )

                groundgrid_config = ground_grid.make_ground_grid_config(
                    bin_width_m=env["config"]["ground_grid"]["geometry"][
                        "bin_width_m"
                    ],
                    num_bins_each_axis=env["config"]["ground_grid"][
                        "geometry"
                    ]["num_bins_each_axis"],
                    prng=prng,
                )
                groundgrid = ground_grid.GroundGrid(**groundgrid_config)
                groundgrid_record = make_groundgrid_record(
                    uid=uid,
                    groundgrid=groundgrid,
                )

                logger.debug(
                    json_line_logger.xml(
                        "EventTime",
                        uid=uid["uid_str"],
                        status="cherenkov production start ...",
                    )
                )

                cherenkovmd5 = hashlib.md5()
                cherenkovsizestats = (
                    cherenkov_bunch_storage.CherenkovSizeStatistics()
                )
                cherenkovpoolstats = (
                    cherenkov_bunch_storage.CherenkovPoolStatistics()
                )
                GGH.init_groundgrid(groundgrid=groundgrid)

                for cherenkov_block in cherenkov_block_reader:
                    cherenkovmd5.update(cherenkov_block.tobytes())

                    cherenkovsizestats.assign_cherenkov_bunches(
                        cherenkov_bunches=cherenkov_block
                    )
                    cherenkovpoolstats.assign_cherenkov_bunches(
                        cherenkov_bunches=cherenkov_block
                    )
                    GGH.assign_cherenkov_bunches(
                        cherenkov_bunches=cherenkov_block
                    )

                logger.debug(
                    json_line_logger.xml(
                        "EventTime",
                        uid=uid["uid_str"],
                        status="cherenkov production done.",
                    )
                )

                cherenkovpools_md5[uid["uid_str"]] = cherenkovmd5.hexdigest()

                cherenkovsize_rec = cherenkovsizestats.make_record()
                cherenkovsize_rec.update(uid["record"])
                evttab["cherenkovsize"].append_record(cherenkovsize_rec)

                if cherenkovsize_rec["num_bunches"] > 0:
                    logger.debug(
                        json_line_logger.xml(
                            "EventTime",
                            uid=uid["uid_str"],
                            status="has cherenkov light.",
                        )
                    )

                    cherenkovpool_rec = cherenkovpoolstats.make_record()
                    cherenkovpool_rec.update(uid["record"])
                    evttab["cherenkovpool"].append_record(cherenkovpool_rec)

                    groundgrid_histogram = GGH.get_histogram()
                    groundgrid_result = ground_grid.make_result(
                        groundgrid=groundgrid,
                        groundgrid_histogram=groundgrid_histogram,
                        threshold_num_photons=env["config"]["ground_grid"][
                            "threshold_num_photons"
                        ],
                        prng=prng,
                    )

                    groundgrid_record["num_bins_above_threshold"] = (
                        groundgrid_result["num_bins_above_threshold"]
                    )

                    if groundgrid_result["choice"]:
                        logger.debug(
                            json_line_logger.xml(
                                "EventTime",
                                uid=uid["uid_str"],
                                status="has grid bins above threshold.",
                            )
                        )

                        evttab["groundgrid_choice"].append_record(
                            make_groundgrid_choice_record(
                                uid=uid,
                                groundgrid_result=groundgrid_result,
                            )
                        )

                        ImgRoiTar_append(
                            imgroitar=imgroitar,
                            uid=uid,
                            groundgrid_result=groundgrid_result,
                            groundgrid_histogram=groundgrid_histogram,
                        )
                        if uid["uid"] in event_uids_for_debugging:
                            ImgTar_append(
                                imgtar=imgtar,
                                uid=uid,
                                groundgrid_histogram=groundgrid_histogram,
                            )

                evttab["groundgrid"].append_record(groundgrid_record)

                logger.debug(
                    json_line_logger.xml(
                        "EventTime", uid=uid["uid_str"], status="stop."
                    )
                )

            GGH.close()

    logger.info("Dump cherenkovpools_md5 checksums.")
    with rnw.open(opj(work_dir, "cherenkovpools_md5.json"), "wt") as fl:
        fl.write(json_utils.dumps(cherenkovpools_md5))

    logger.info("Convert particle output from .dat to .tar.")
    cpw.particles.dat_to_tape(
        dat_path=opj(work_dir, "particle_pools.dat"),
        tape_path=opj(work_dir, "particle_pools.tar.gz"),
    )

    return evttab


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
        "record": {"uid": uid},
        "uid": uid,
        "uid_str": uid_str,
        "run_id": run_id,
        "event_id": event_id,
        "event_idx": event_idx,
        "uid_path": bookkeeping.uid.make_uid_path(
            run_id=run_id, event_id=event_id
        ),
    }
    return out


def make_primary_record(
    uid, corsika_evth, corsika_primary_steering, primary_directions
):
    primary = corsika_primary_steering["primaries"][uid["event_idx"]]
    primary_direction = primary_directions[uid["uid_str"]]

    rec = uid["record"].copy()
    rec["particle_id"] = primary["particle_id"]
    rec["energy_GeV"] = primary["energy_GeV"]

    # momentum
    rec["phi_rad"] = primary["phi_rad"]
    rec["theta_rad"] = primary["theta_rad"]

    # pointing
    rec["azimuth_rad"] = spherical_coordinates.corsika.phi_to_az(
        phi_rad=primary["phi_rad"]
    )
    rec["zenith_rad"] = spherical_coordinates.corsika.theta_to_zd(
        theta_rad=primary["theta_rad"]
    )

    rec["depth_g_per_cm2"] = primary["depth_g_per_cm2"]

    mom = cpw.I.EVTH.get_momentum_vector_GeV_per_c(evth=corsika_evth)
    rec["momentum_x_GeV_per_c"] = mom[0]
    rec["momentum_y_GeV_per_c"] = mom[1]
    rec["momentum_z_GeV_per_c"] = mom[2]

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
    rec["containment_quantile_in_solid_angle_thrown"] = primary_direction[
        "sky_draw_quantile"
    ]

    pd = primary_direction

    if pd["method"] == "magnetic_deflection_skymap" and not pd["cutoff"]:
        rec["draw_primary_direction_method"] = (
            event_table.structure.METHOD_SKYMAP_VALID
        )
    elif pd["method"] == "magnetic_deflection_skymap" and pd["cutoff"]:
        rec["draw_primary_direction_method"] = (
            event_table.structure.METHOD_SKYMAP_CUTOFF_FALLBACK_FULL_SKY
        )
    elif pd["method"] == "viewcone":
        rec["draw_primary_direction_method"] = (
            event_table.structure.METHOD_VIEWCONE
        )
    else:
        raise AssertionError("Can not assign draw_primary_direction_method")

    return rec


def make_groundgrid_record(uid, groundgrid):
    rec = uid["record"].copy()
    config_keys = [
        "bin_width_m",
        "num_bins_each_axis",
        "center_x_m",
        "center_y_m",
    ]
    for key in config_keys:
        rec[key] = groundgrid[key]
    rec["num_bins_thrown"] = groundgrid["num_bins_thrown"]
    rec["area_thrown_m2"] = groundgrid["area_thrown_m2"]

    # This will be overwritten in case there are any bins above threshold.
    rec["num_bins_above_threshold"] = 0
    return rec


def make_groundgrid_choice_record(uid, groundgrid_result):
    rec = uid["record"].copy()
    choice = groundgrid_result["choice"]
    for key in choice:
        rec[key] = choice[key]

    # compare scatter
    # ---------------
    scathist = groundgrid_result["scatter_histogram"]
    num_bins = len(scathist["bin_counts"])
    assert num_bins == 16
    for rbin in range(num_bins):
        rec["scatter_rbin_{:02d}".format(rbin)] = scathist["bin_counts"][rbin]
    return rec


def ImgRoiTar_append(imgroitar, uid, groundgrid_result, groundgrid_histogram):
    roi = ImgRoiTar_apply_cut(
        groundgrid_histogram=groundgrid_histogram,
        groundgrid_choice_bin_idx_x=groundgrid_result["choice"]["bin_idx_x"],
        groundgrid_choice_bin_idx_y=groundgrid_result["choice"]["bin_idx_y"],
    )

    tar_append.tar_append(
        tarout=imgroitar,
        filename=uid["uid_path"] + ".i4_i4_f8.gz",
        filebytes=gzip.compress(roi.tobytes()),
    )


def ImgRoiTar_apply_cut(
    groundgrid_histogram,
    groundgrid_choice_bin_idx_x,
    groundgrid_choice_bin_idx_y,
):
    bb = outer_telescope_array.init_binning()
    bin_radius = (bb["num_bins_on_edge"] - 1) // 2

    dyn_roi = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=ground_grid.histogram2d.make_histogram2d_dtype()
    )
    for entry in groundgrid_histogram:
        dx = entry["x_bin"] - groundgrid_choice_bin_idx_x
        dy = entry["y_bin"] - groundgrid_choice_bin_idx_y
        if abs(dx) <= bin_radius and abs(dy) <= bin_radius:
            dyn_roi.append_numpy_record_or_numpy_void(entry)
    roi = dyn_roi.to_recarray()
    return roi


def ImgTar_append(imgtar, uid, groundgrid_histogram):
    tar_append.tar_append(
        tarout=imgtar,
        filename=uid["uid_path"] + ".i4_i4_f8.gz",
        filebytes=gzip.compress(groundgrid_histogram.tobytes()),
    )
