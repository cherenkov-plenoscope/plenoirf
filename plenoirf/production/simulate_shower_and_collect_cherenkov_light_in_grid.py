import os
import tarfile
import numpy as np
import gzip
import hashlib

import corsika_primary as cpw
import json_utils
from json_line_logger import xml
import pickle
import sparse_numeric_table as snt
import atmospheric_cherenkov_response as acr
import spherical_coordinates
import dynamicsizerecarray
import rename_after_writing as rnw

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


def run(env, seed, logger):
    opj = os.path.join
    logger.info(__name__ + ": start ...")

    corsika_and_grid_work_dir = opj(env["work_dir"], __name__)

    if os.path.exists(corsika_and_grid_work_dir):
        logger.info(__name__ + ": already done. skip computation.")
        return

    logger.info(__name__ + ": simulating showers ...")
    prng = np.random.Generator(np.random.PCG64(seed))

    with open(
        opj(
            env["work_dir"],
            "plenoirf.production.draw_primaries_and_pointings",
            "result.pkl",
        ),
        "rb",
    ) as fin:
        dpp = pickle.loads(fin.read())

    with open(
        opj(
            env["work_dir"],
            "plenoirf.production.draw_event_uids_for_debugging.json",
        ),
        "rt",
    ) as fin:
        event_uids_for_debugging = json_utils.loads(fin.read())

    evttab = {}
    evttab = event_table.add_empty_level(evttab, "primary")
    evttab = event_table.add_empty_level(evttab, "instrument_pointing")
    evttab = event_table.add_empty_level(evttab, "cherenkovsize")
    evttab = event_table.add_empty_level(evttab, "cherenkovpool")
    evttab = event_table.add_empty_level(evttab, "groundgrid")
    evttab = event_table.add_empty_level(evttab, "groundgrid_result")
    evttab = event_table.add_empty_level(evttab, "cherenkovsizepart")
    evttab = event_table.add_empty_level(evttab, "cherenkovpoolpart")

    evttab = stage_one(
        env=env,
        prng=prng,
        evttab=evttab,
        corsika_and_grid_work_dir=corsika_and_grid_work_dir,
        corsika_primary_steering=dpp["corsika_primary_steering"],
        primary_directions=dpp["primary_directions"],
        instrument_pointings=dpp["instrument_pointings"],
        event_uids_for_debugging=event_uids_for_debugging,
        logger=logger,
    )

    evttab = stage_two(
        env=env,
        prng=prng,
        evttab=evttab,
        corsika_and_grid_work_dir=corsika_and_grid_work_dir,
        corsika_primary_steering=dpp["corsika_primary_steering"],
        primary_directions=dpp["primary_directions"],
        instrument_pointings=dpp["instrument_pointings"],
        event_uids_for_debugging=event_uids_for_debugging,
        logger=logger,
    )

    event_table.write_all_levels_to_path(
        evttab=evttab,
        path=os.path.join(corsika_and_grid_work_dir, "event_table.tar"),
    )

    cherenkov_bunch_storage.filter_by_event_uids(
        inpath=opj(corsika_and_grid_work_dir, "cherenkov_pools.tar"),
        outpath=opj(corsika_and_grid_work_dir, "cherenkov_pools.debug.tar"),
        event_uids=event_uids_for_debugging,
    )

    logger.info(__name__ + ": remove temporary cherenkov_pool_storage files.")
    cherenkov_pool_storage_in_fov_path = opj(
        corsika_and_grid_work_dir,
        "cherenkov_pool_storage_in_field_of_view.tar",
    )
    if os.path.exists(cherenkov_pool_storage_in_fov_path):
        os.remove(cherenkov_pool_storage_in_fov_path)
    cherenkov_pool_storage_path = opj(
        corsika_and_grid_work_dir, "cherenkov_pool_storage.tar"
    )
    if os.path.exists(cherenkov_pool_storage_path):
        os.remove(cherenkov_pool_storage_path)

    logger.info(__name__ + ": ... done.")


def stage_one(
    env,
    prng,
    evttab,
    corsika_and_grid_work_dir,
    corsika_primary_steering,
    primary_directions,
    instrument_pointings,
    event_uids_for_debugging,
    logger,
):
    opj = os.path.join
    logger.info(__name__ + ": start corsika stage one")
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
            stdout_path=opj(work_dir, "corsika.1.stdout.txt"),
            stderr_path=opj(work_dir, "corsika.1.stderr.txt"),
            particle_output_path=opj(work_dir, "particle_pools.dat"),
        ) as corsika_run:
            logger.info(__name__ + ": corsika is ready")

            GGH = ground_grid.GGH()

            for event_idx, corsika_event in enumerate(corsika_run):
                corsika_evth, cherenkov_block_reader = corsika_event
                uid = nail_down_event_identity(
                    corsika_evth=corsika_evth,
                    event_idx=event_idx,
                    corsika_primary_steering=corsika_primary_steering,
                )

                evttab["instrument_pointing"].append_record(
                    make_instrument_pointing_record(
                        uid=uid, instrument_pointings=instrument_pointings
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
                evttab["groundgrid"].append_record(
                    make_groundgrid_record(
                        uid=uid,
                        groundgrid=groundgrid,
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

                cherenkovpools_md5[uid["uid_str"]] = cherenkovmd5.hexdigest()

                cherenkovsize_rec = cherenkovsizestats.make_record()
                cherenkovsize_rec.update(uid["record"])
                evttab["cherenkovsize"].append_record(cherenkovsize_rec)

                if cherenkovsize_rec["num_bunches"] > 0:
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

                    if groundgrid_result["choice"]:
                        evttab["groundgrid_result"].append_record(
                            make_groundgrid_result_record(
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

            GGH.close()

    logger.info(__name__ + ": dump cherenkovpools_md5 checksums.")
    with rnw.open(opj(work_dir, "cherenkovpools_md5.json"), "wt") as fl:
        fl.write(json_utils.dumps(cherenkovpools_md5))

    logger.info(__name__ + ": convert particle output from .dat to .tar")
    cpw.particles.dat_to_tape(
        dat_path=opj(work_dir, "particle_pools.dat"),
        tape_path=opj(work_dir, "particle_pools.tar.gz"),
    )

    return evttab


def stage_two(
    env,
    prng,
    evttab,
    corsika_and_grid_work_dir,
    corsika_primary_steering,
    primary_directions,
    instrument_pointings,
    event_uids_for_debugging,
    logger,
):
    opj = os.path.join
    logger.info(__name__ + ": start corsika stage one")
    work_dir = corsika_and_grid_work_dir

    with open(opj(work_dir, "cherenkovpools_md5.json"), "rt") as fl:
        cherenkovpools_md5 = json_utils.loads(fl.read())

    evttab_groundgrid_by_uid = {}
    for rec in evttab["groundgrid"]:
        evttab_groundgrid_by_uid[rec[snt.IDX]] = rec

    evttab_groundgrid_result_by_uid = {}
    for rec in evttab["groundgrid_result"]:
        evttab_groundgrid_result_by_uid[rec[snt.IDX]] = rec

    evttab_instrument_pointing_by_uid = {}
    for rec in evttab["instrument_pointing"]:
        evttab_instrument_pointing_by_uid[rec[snt.IDX]] = rec

    with cpw.cherenkov.CherenkovEventTapeWriter(
        path=opj(work_dir, "cherenkov_pools.tar")
    ) as evttar, cpw.CorsikaPrimary(
        steering_dict=corsika_primary_steering,
        stdout_path=opj(work_dir, "corsika.2.stdout.txt"),
        stderr_path=opj(work_dir, "corsika.2.stderr.txt"),
        particle_output_path=opj(work_dir, "particle_pools.dat"),
    ) as corsika_run:
        logger.info(__name__ + ": corsika is ready")
        evttar.write_runh(runh=corsika_run.runh)

        for event_idx, corsika_event in enumerate(corsika_run):
            corsika_evth, cherenkov_block_reader = corsika_event
            uid = nail_down_event_identity(
                corsika_evth=corsika_evth,
                event_idx=event_idx,
                corsika_primary_steering=corsika_primary_steering,
            )

            instrument_pointing = evttab_instrument_pointing_by_uid[uid["uid"]]
            groundgrid_config = evttab_groundgrid_by_uid[uid["uid"]]
            groundgrid = ground_grid.GroundGrid(
                bin_width_m=groundgrid_config["bin_width_m"],
                num_bins_each_axis=groundgrid_config["num_bins_each_axis"],
                center_x_m=groundgrid_config["center_x_m"],
                center_y_m=groundgrid_config["center_y_m"],
            )

            cherenkovmd5 = hashlib.md5()
            if uid["uid"] in evttab_groundgrid_result_by_uid:
                groundgrid_result = evttab_groundgrid_result_by_uid[uid["uid"]]
                cherenkovsizepartstats = (
                    cherenkov_bunch_storage.CherenkovSizeStatistics()
                )
                cherenkovpoolpartstats = (
                    cherenkov_bunch_storage.CherenkovPoolStatistics()
                )

                custom_evth = make_evth_with_core(
                    corsika_evth=corsika_evth,
                    core_x_m=groundgrid_result["core_x_m"],
                    core_y_m=groundgrid_result["core_y_m"],
                )
                evttar.write_evth(evth=custom_evth)

                for cherenkov_block in cherenkov_block_reader:
                    cherenkovmd5.update(cherenkov_block.tobytes())
                    cherenkov_in_sphere_block = (
                        cherenkov_bunch_storage.cut_in_sphere(
                            cherenkov_bunches=cherenkov_block,
                            sphere_obs_level_x_m=groundgrid_result["core_x_m"],
                            sphere_obs_level_y_m=groundgrid_result["core_y_m"],
                            sphere_radius_m=groundgrid[
                                "bin_smallest_enclosing_radius_m"
                            ],
                        )
                    )
                    cherenkovsizepartstats.assign_cherenkov_bunches(
                        cherenkov_bunches=cherenkov_in_sphere_block
                    )
                    cherenkovpoolpartstats.assign_cherenkov_bunches(
                        cherenkov_bunches=cherenkov_in_sphere_block
                    )

                    cherenkov_in_instrument_block = transform_cherenkov_bunches.from_obervation_level_to_instrument(
                        cherenkov_bunches=cherenkov_in_sphere_block,
                        instrument_pointing=instrument_pointing,
                        instrument_pointing_model=env["config"]["pointing"][
                            "model"
                        ],
                        instrument_x_m=groundgrid_result["core_x_m"],
                        instrument_y_m=groundgrid_result["core_y_m"],
                        speed_of_ligth_m_per_s=env["instrument"][
                            "local_speed_of_light_m_per_s"
                        ],
                    )
                    evttar.write_payload(cherenkov_in_instrument_block)

                cherenkovsizepart_rec = cherenkovsizepartstats.make_record()
                cherenkovsizepart_rec.update(uid["record"])
                evttab["cherenkovsizepart"].append_record(
                    cherenkovsizepart_rec
                )

                cherenkovpoolpart_rec = cherenkovpoolpartstats.make_record()
                cherenkovpoolpart_rec.update(uid["record"])
                evttab["cherenkovpoolpart"].append_record(
                    cherenkovpoolpart_rec
                )

                assert_expected_num_photons_in_sphere(
                    num_photons_in_sphere=cherenkovsizepartstats.num_photons,
                    num_photons_in_groundgrid_bin=groundgrid_result[
                        "bin_num_photons"
                    ],
                )

            else:
                for cherenkov_block in cherenkov_block_reader:
                    cherenkovmd5.update(cherenkov_block.tobytes())

            assert cherenkovpools_md5[uid["uid"]] == cherenkovmd5.hexdigest()

    return evttab


def assert_expected_num_photons_in_sphere(
    num_photons_in_sphere,
    num_photons_in_groundgrid_bin,
):
    assert num_photons_in_sphere >= num_photons_in_groundgrid_bin


def corsika_and_grid(
    env,
    prng,
    evttab,
    corsika_and_grid_work_dir,
    corsika_primary_steering,
    primary_directions,
    instrument_pointings,
    event_uids_for_debugging,
    logger,
):
    opj = os.path.join
    logger.info(__name__ + ": start corsika")
    work_dir = corsika_and_grid_work_dir
    os.makedirs(work_dir, exist_ok=True)

    with cpw.cherenkov.CherenkovEventTapeWriter(
        path=opj(work_dir, "cherenkov_pools.tar")
    ) as evttar, tarfile.open(
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
            logger.info(__name__ + ": corsika is ready")
            evttar.write_runh(runh=corsika_run.runh)

            for event_idx, corsika_event in enumerate(corsika_run):
                logger.debug(
                    xml(
                        "EventTime",
                        uid=bookkeeping.uid.make_uid_str(
                            run_id=corsika_primary_steering["run"]["run_id"],
                            event_id=event_idx + 1,
                        ),
                        status="corsika_start",
                    )
                )

                corsika_evth, cherenkov_reader = corsika_event

                cherenkov_storage_path = opj(
                    work_dir, "cherenkov_pool_storage.tar"
                )

                cherenkov_bunch_storage.write(
                    path=cherenkov_storage_path,
                    event_tape_cherenkov_reader=cherenkov_reader,
                )

                uid = nail_down_event_identity(
                    corsika_evth=corsika_evth,
                    event_idx=event_idx,
                    corsika_primary_steering=corsika_primary_steering,
                )

                logger.debug(
                    xml("EventTime", uid=uid["uid_str"], status="corsika_done")
                )

                primary_rec = make_primary_record(
                    uid=uid,
                    corsika_evth=corsika_evth,
                    corsika_primary_steering=corsika_primary_steering,
                    primary_directions=primary_directions,
                )
                evttab["primary"].append_record(primary_rec)

                instrument_pointing_rec = make_instrument_pointing_record(
                    uid=uid, instrument_pointings=instrument_pointings
                )
                evttab["instrument_pointing"].append_record(
                    instrument_pointing_rec
                )
                _ = instrument_pointing_rec.pop("idx")
                instrument_pointing = instrument_pointing_rec

                cherenkovsize_rec = (
                    cherenkov_bunch_storage.make_cherenkovsize_record(
                        path=cherenkov_storage_path
                    )
                )
                cherenkovsize_rec.update(uid["record"])
                evttab["cherenkovsize"].append_record(cherenkovsize_rec)

                logger.debug(
                    xml(
                        "EventTime",
                        uid=uid["uid_str"],
                        status="made_cherenkovsize_record",
                    )
                )

                if cherenkovsize_rec["num_bunches"] > 0:
                    cherenkovpool_rec = (
                        cherenkov_bunch_storage.make_cherenkovpool_record(
                            path=cherenkov_storage_path
                        )
                    )
                    cherenkovpool_rec.update(uid["record"])
                    evttab["cherenkovpool"].append_record(cherenkovpool_rec)

                    logger.debug(
                        xml(
                            "EventTime",
                            uid=uid["uid_str"],
                            status="made_cherenkovpool_record",
                        )
                    )

                    groundgrid_config = ground_grid.make_ground_grid_config(
                        bin_width_m=env["config"]["ground_grid"]["geometry"][
                            "bin_width_m"
                        ],
                        num_bins_each_axis=env["config"]["ground_grid"][
                            "geometry"
                        ]["num_bins_each_axis"],
                        cherenkov_pool_median_x_m=cherenkovpool_rec["x_p50_m"],
                        cherenkov_pool_median_y_m=cherenkovpool_rec["y_p50_m"],
                        prng=prng,
                    )

                    groundgrid = ground_grid.GroundGrid(
                        bin_width_m=groundgrid_config["bin_width_m"],
                        num_bins_each_axis=groundgrid_config[
                            "num_bins_each_axis"
                        ],
                        center_x_m=groundgrid_config["center_x_m"],
                        center_y_m=groundgrid_config["center_y_m"],
                    )

                    cherenkov_storage_infov_path = opj(
                        work_dir,
                        "cherenkov_pool_storage_in_field_of_view.tar",
                    )
                    cherenkov_bunch_storage.cut_in_field_of_view(
                        in_path=cherenkov_storage_path,
                        out_path=cherenkov_storage_infov_path,
                        pointing=instrument_pointing,
                        field_of_view_half_angle_rad=env["instrument"][
                            "field_of_view_half_angle_rad"
                        ],
                    )

                    logger.debug(
                        xml(
                            "EventTime",
                            uid=uid["uid_str"],
                            status="cut_in_field_of_view",
                        )
                    )

                    (
                        groundgrid_result,
                        groundgrid_histogram,
                    ) = ground_grid.assign3(
                        groundgrid=groundgrid,
                        cherenkov_bunch_storage_path=cherenkov_storage_infov_path,
                        threshold_num_photons=env["config"]["ground_grid"][
                            "threshold_num_photons"
                        ],
                        prng=prng,
                    )

                    groundgrid_rec = make_groundgrid_record(
                        uid=uid,
                        groundgrid_config=groundgrid_config,
                        groundgrid_result=groundgrid_result,
                        groundgrid=groundgrid,
                    )
                    evttab["groundgrid"].append_record(groundgrid_rec)

                    logger.debug(
                        xml(
                            "EventTime",
                            uid=uid["uid_str"],
                            status="made_ground_grid_histogram",
                        )
                    )

                    if groundgrid_result["choice"]:
                        cherenkov_bunches_in_choice = (
                            cherenkov_bunch_storage.read_sphere(
                                path=cherenkov_storage_infov_path,
                                sphere_obs_level_x_m=groundgrid_result[
                                    "choice"
                                ]["core_x_m"],
                                sphere_obs_level_y_m=groundgrid_result[
                                    "choice"
                                ]["core_y_m"],
                                sphere_radius_m=groundgrid[
                                    "bin_smallest_enclosing_radius_m"
                                ],
                            )
                        )

                        assert_expected_num_photons_in_choice(
                            threshold_num_photons=env["config"]["ground_grid"][
                                "threshold_num_photons"
                            ],
                            groundgrid_result=groundgrid_result,
                            groundgrid_histogram=groundgrid_histogram,
                            cherenkov_bunches_in_choice=cherenkov_bunches_in_choice,
                        )

                        logger.debug(
                            xml(
                                "EventTime",
                                uid=uid["uid_str"],
                                status="read_cherenkov_in_sphere",
                            )
                        )

                        cherenkov_bunches_in_instrument = transform_cherenkov_bunches.from_obervation_level_to_instrument(
                            cherenkov_bunches=cherenkov_bunches_in_choice,
                            instrument_pointing=instrument_pointing,
                            instrument_pointing_model=env["config"][
                                "pointing"
                            ]["model"],
                            instrument_x_m=groundgrid_result["choice"][
                                "core_x_m"
                            ],
                            instrument_y_m=groundgrid_result["choice"][
                                "core_y_m"
                            ],
                            speed_of_ligth_m_per_s=env["instrument"][
                                "local_speed_of_light_m_per_s"
                            ],
                        )
                        del cherenkov_bunches_in_choice

                        logger.debug(
                            xml(
                                "EventTime",
                                uid=uid["uid_str"],
                                status="transformed_cherenkov_into_instrument_frame",
                            )
                        )

                        core_rec = make_core_record(
                            uid=uid,
                            groundgrid_result_choice=groundgrid_result[
                                "choice"
                            ],
                        )
                        evttab["core"].append_record(core_rec)

                        EventTape_append_event(
                            evttar=evttar,
                            corsika_evth=corsika_evth,
                            cherenkov_bunches=cherenkov_bunches_in_instrument,
                            core_x_m=groundgrid_result["choice"]["core_x_m"],
                            core_y_m=groundgrid_result["choice"]["core_y_m"],
                        )

                        logger.debug(
                            xml(
                                "EventTime",
                                uid=uid["uid_str"],
                                status="wrote_cherenkov_for_instrument",
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

                        logger.debug(
                            xml(
                                "EventTime",
                                uid=uid["uid_str"],
                                status="wrote_cherenkov_for_instrument",
                            )
                        )

                        cherenkovsizepart_rec = cherenkov_bunch_storage.make_cherenkovsize_record(
                            cherenkov_bunches=cherenkov_bunches_in_instrument
                        )
                        cherenkovsizepart_rec.update(uid["record"])
                        evttab["cherenkovsizepart"].append_record(
                            cherenkovsizepart_rec
                        )

                        logger.debug(
                            xml(
                                "EventTime",
                                uid=uid["uid_str"],
                                status="made_cherenkovpartsize_record",
                            )
                        )

                        if cherenkovsizepart_rec["num_bunches"] > 0:
                            cherenkovpoolpart_rec = cherenkov_bunch_storage.make_cherenkovpool_record(
                                cherenkov_bunches=cherenkov_bunches_in_instrument
                            )
                            cherenkovpoolpart_rec.update(uid["record"])
                            evttab["cherenkovpoolpart"].append_record(
                                cherenkovpoolpart_rec
                            )
                            logger.debug(
                                xml(
                                    "EventTime",
                                    uid=uid["uid_str"],
                                    status="made_cherenkovpartpool_record",
                                )
                            )

                        logger.debug(
                            xml("EventTime", uid=uid["uid_str"], status="end")
                        )

    logger.info(__name__ + ": convert particle output from .dat to .tar")
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
        "record": {snt.IDX: uid},
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

    pd = primary_direction

    if pd["method"] == "magnetic_deflection_skymap" and not pd["cutoff"]:
        rec[
            "draw_primary_direction_method"
        ] = event_table.structure.METHOD_SKYMAP_VALID
    elif pd["method"] == "magnetic_deflection_skymap" and pd["cutoff"]:
        rec[
            "draw_primary_direction_method"
        ] = event_table.structure.METHOD_SKYMAP_CUTOFF_FALLBACK_FULL_SKY
    elif pd["method"] == "viewcone":
        rec[
            "draw_primary_direction_method"
        ] = event_table.structure.METHOD_VIEWCONE
    else:
        raise AssertionError("Can not assign draw_primary_direction_method")

    return rec


def make_instrument_pointing_record(uid, instrument_pointings):
    rec = uid["record"].copy()
    for key in ["azimuth_rad", "zenith_rad"]:
        rec[key] = instrument_pointings[uid["uid_str"]][key]
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
    return rec


def make_groundgrid_result_record(uid, groundgrid_result):
    rec = uid["record"].copy()
    rec["num_bins_above_threshold"] = groundgrid_result[
        "num_bins_above_threshold"
    ]
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


def make_evth_with_core(
    corsika_evth,
    core_x_m,
    core_y_m,
):
    evth = corsika_evth.copy()
    evth[cpw.I.EVTH.NUM_REUSES_OF_CHERENKOV_EVENT] = 1.0
    evth[cpw.I.EVTH.X_CORE_CM(reuse=1)] = cpw.M2CM * core_x_m
    evth[cpw.I.EVTH.Y_CORE_CM(reuse=1)] = cpw.M2CM * core_y_m
    return evth


def ImgRoiTar_append(imgroitar, uid, groundgrid_result, groundgrid_histogram):
    bb = outer_telescope_array.init_binning()

    dyn_roi = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=ground_grid.make_histogram2d_dtype()
    )
    for entry in groundgrid_histogram:
        dx = entry["x_bin"] - groundgrid_result["choice"]["bin_idx_x"]
        dy = entry["y_bin"] - groundgrid_result["choice"]["bin_idx_y"]
        if abs(dx <= 12) and abs(dy <= 12):
            dyn_roi.append_recarray(entry)
    roi = dyn_roi.to_recarray()

    tar_append.tar_append(
        tarout=imgroitar,
        filename=uid["uid_path"] + ".i4_i4_f8.gz",
        filebytes=gzip.compress(roi.tobytes()),
    )


def ImgTar_append(imgtar, uid, groundgrid_histogram):
    tar_append.tar_append(
        tarout=imgtar,
        filename=uid["uid_path"] + ".i4_i4_f8.gz",
        filebytes=gzip.compress(groundgrid_histogram.tobytes()),
    )


def assert_expected_num_photons_in_choice(
    threshold_num_photons,
    groundgrid_result,
    groundgrid_histogram,
    cherenkov_bunches_in_choice,
):
    num_photons_in_choice = float(threshold_num_photons)
    for entry in groundgrid_histogram:
        if (
            entry["x_bin"] == groundgrid_result["choice"]["bin_idx_x"]
            and entry["y_bin"] == groundgrid_result["choice"]["bin_idx_y"]
        ):
            num_photons_in_choice = entry["weight_photons"]

    assert (
        cherenkov_bunches_in_choice.shape[0] > num_photons_in_choice
    ), "Expected at least {:f} photons in sphere.".format(
        num_photons_in_choice
    )
