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
from .simulate_shower_and_collect_cherenkov_light_in_grid import (
    nail_down_event_identity,
)


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

    logger.info("read primaries_and_pointings.")
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

    logger.info("read cherenkovpools_md5.json.")
    with gzip.open(
        opj(
            env["work_dir"],
            "prm2cer",
            "simulate_shower_and_collect_cherenkov_light_in_grid",
            "cherenkovpools_md5.json.gz",
        ),
        "rt",
    ) as fin:
        cherenkovpools_md5 = json_utils.loads(fin.read())

    logger.info("init event_table.")
    evttab = snt.SparseNumericTable(index_key="uid")
    evttab = event_table.add_levels_from_path(
        evttab=evttab,
        path=os.path.join(
            env["work_dir"],
            "prm2cer",
            "simulate_shower_and_collect_cherenkov_light_in_grid",
            "event_table.snt.zip",
        ),
    )
    additional_level_keys = [
        "instrument_pointing",
        "cherenkovsizepart",
        "cherenkovpoolpart",
    ]
    for key in additional_level_keys:
        evttab = event_table.add_empty_level(evttab, key)

    logger.info("simulate showers.")
    evttab = corsika_second_run(
        env=env,
        prng=prng,
        evttab=evttab,
        corsika_and_grid_work_dir=module_work_dir,
        corsika_primary_steering=dpp["corsika_primary_steering"],
        primary_directions=dpp["primary_directions"],
        instrument_pointings=dpp["instrument_pointings"],
        cherenkovpools_md5=cherenkovpools_md5,
        logger=logger,
    )

    logger.info("write event_table.snt.zip.")
    event_table.write_certain_levels_to_path(
        evttab=evttab,
        path=os.path.join(module_work_dir, "event_table.snt.zip"),
        level_keys=additional_level_keys,
    )

    logger.info("done.")
    json_line_logger.shutdown(logger=logger)

    # tidy up and compress
    utils.gzip_file(opj(module_work_dir, "log.jsonl"))
    utils.gzip_file(opj(module_work_dir, "cherenkov_pools.tar"))
    os.remove(opj(module_work_dir, "corsika.stdout.txt"))
    os.remove(opj(module_work_dir, "corsika.stderr.txt"))
    os.remove(opj(module_work_dir, "particle_pools.dat"))


def corsika_second_run(
    env,
    prng,
    evttab,
    corsika_and_grid_work_dir,
    corsika_primary_steering,
    primary_directions,
    instrument_pointings,
    cherenkovpools_md5,
    logger,
):
    opj = os.path.join
    logger.info("Start corsika second run.")
    work_dir = corsika_and_grid_work_dir
    os.makedirs(work_dir, exist_ok=True)

    evttab_groundgrid_by_uid = {}
    for rec in evttab["groundgrid"]:
        evttab_groundgrid_by_uid[rec["uid"]] = rec

    evttab_groundgrid_choice_by_uid = {}
    for rec in evttab["groundgrid_choice"]:
        evttab_groundgrid_choice_by_uid[rec["uid"]] = rec

    with cpw.cherenkov.CherenkovEventTapeWriter(
        path=opj(work_dir, "cherenkov_pools.tar")
    ) as evttar, cpw.CorsikaPrimary(
        steering_dict=corsika_primary_steering,
        stdout_path=opj(work_dir, "corsika.stdout.txt"),
        stderr_path=opj(work_dir, "corsika.stderr.txt"),
        particle_output_path=opj(work_dir, "particle_pools.dat"),
    ) as corsika_run:
        logger.info("corsika is ready.")
        evttar.write_runh(runh=corsika_run.runh)

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

            evttab["instrument_pointing"].append_record(
                make_instrument_pointing_record(
                    uid=uid, instrument_pointings=instrument_pointings
                )
            )

            instrument_pointing = instrument_pointings[uid["uid_str"]]
            groundgrid_config = evttab_groundgrid_by_uid[uid["uid"]]
            groundgrid = ground_grid.GroundGrid(
                bin_width_m=groundgrid_config["bin_width_m"],
                num_bins_each_axis=groundgrid_config["num_bins_each_axis"],
                center_x_m=groundgrid_config["center_x_m"],
                center_y_m=groundgrid_config["center_y_m"],
            )

            cherenkovmd5 = hashlib.md5()
            if uid["uid"] in evttab_groundgrid_choice_by_uid:
                logger.debug(
                    json_line_logger.xml(
                        "EventTime",
                        uid=uid["uid_str"],
                        status="extract cherenkov light.",
                    )
                )

                groundgrid_choice = evttab_groundgrid_choice_by_uid[uid["uid"]]
                cherenkovsizepartstats = (
                    cherenkov_bunch_storage.CherenkovSizeStatistics()
                )
                cherenkovpoolpartstats = (
                    cherenkov_bunch_storage.CherenkovPoolStatistics()
                )

                custom_evth = make_evth_with_core(
                    corsika_evth=corsika_evth,
                    core_x_m=groundgrid_choice["core_x_m"],
                    core_y_m=groundgrid_choice["core_y_m"],
                )
                evttar.write_evth(evth=custom_evth)

                for cherenkov_block in cherenkov_block_reader:
                    cherenkovmd5.update(cherenkov_block.tobytes())
                    cherenkov_in_sphere_block = (
                        cherenkov_bunch_storage.cut_in_sphere(
                            cherenkov_bunches=cherenkov_block,
                            sphere_obs_level_x_m=groundgrid_choice["core_x_m"],
                            sphere_obs_level_y_m=groundgrid_choice["core_y_m"],
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

                    if cherenkov_in_sphere_block.shape[0] > 0:
                        cherenkov_in_instrument_block = transform_cherenkov_bunches.from_obervation_level_to_instrument(
                            cherenkov_bunches=cherenkov_in_sphere_block,
                            instrument_pointing=instrument_pointing,
                            instrument_pointing_model=env["config"][
                                "pointing"
                            ]["model"],
                            instrument_x_m=groundgrid_choice["core_x_m"],
                            instrument_y_m=groundgrid_choice["core_y_m"],
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
                    num_photons_in_groundgrid_bin=groundgrid_choice[
                        "bin_num_photons"
                    ],
                )

            else:
                logger.debug(
                    json_line_logger.xml(
                        "EventTime",
                        uid=uid["uid_str"],
                        status="discard cherenkov light.",
                    )
                )
                for cherenkov_block in cherenkov_block_reader:
                    cherenkovmd5.update(cherenkov_block.tobytes())

            assert (
                cherenkovpools_md5[uid["uid_str"]] == cherenkovmd5.hexdigest()
            ), "Expected md5 sum of cherenkov light to be equal."

            logger.debug(
                json_line_logger.xml(
                    "EventTime", uid=uid["uid_str"], status="stop."
                )
            )

    return evttab


def assert_expected_num_photons_in_sphere(
    num_photons_in_sphere,
    num_photons_in_groundgrid_bin,
):
    """
    The epsilon works around edge cases which are likely caused by numerics.
    A relative ratio of (1.0 + 1e-3) does not make a significant difference
    in the physics.
    Apparently, adding 1e-3 was not enough due to rounding issues for sizes
    in the larger than 1e4.
    """
    epsilon = 1e-3
    assert (
        num_photons_in_sphere * (1.0 + epsilon)
    ) >= num_photons_in_groundgrid_bin, (
        "Expected "
        "num_photons_in_sphere * (1 + 1e-3) >= num_photons_in_groundgrid_bin. "
        "But actual num_photons_in_sphere is "
        "{:e} and num_photons_in_groundgrid_bin is {:e}".format(
            num_photons_in_sphere, num_photons_in_groundgrid_bin
        )
    )


def make_instrument_pointing_record(uid, instrument_pointings):
    rec = uid["record"].copy()
    for key in ["azimuth_rad", "zenith_rad"]:
        rec[key] = instrument_pointings[uid["uid_str"]][key]
    return rec


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
