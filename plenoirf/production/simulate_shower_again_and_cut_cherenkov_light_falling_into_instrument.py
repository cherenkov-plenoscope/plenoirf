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
from .simulate_shower_and_collect_cherenkov_light_in_grid import (
    nail_down_event_identity,
)


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

    with open(
        opj(
            env["work_dir"],
            "plenoirf.production.simulate_shower_and_collect_cherenkov_light_in_grid",
            "cherenkovpools_md5.json",
        ),
        "rt",
    ) as fin:
        cherenkovpools_md5 = json_utils.loads(fin.read())

    evttab = {}
    evttab = event_table.add_levels_from_path(
        evttab=evttab,
        path=os.path.join(
            env["work_dir"],
            "plenoirf.production.simulate_shower_and_collect_cherenkov_light_in_grid",
            "event_table.tar",
        ),
    )
    evttab = event_table.add_empty_level(evttab, "instrument_pointing")
    evttab = event_table.add_empty_level(evttab, "cherenkovsizepart")
    evttab = event_table.add_empty_level(evttab, "cherenkovpoolpart")

    evttab = stage_two(
        env=env,
        prng=prng,
        evttab=evttab,
        corsika_and_grid_work_dir=corsika_and_grid_work_dir,
        corsika_primary_steering=dpp["corsika_primary_steering"],
        primary_directions=dpp["primary_directions"],
        instrument_pointings=dpp["instrument_pointings"],
        cherenkovpools_md5=cherenkovpools_md5,
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

    logger.info(__name__ + ": ... done.")


def stage_two(
    env,
    prng,
    evttab,
    corsika_and_grid_work_dir,
    corsika_primary_steering,
    primary_directions,
    instrument_pointings,
    cherenkovpools_md5,
    event_uids_for_debugging,
    logger,
):
    opj = os.path.join
    logger.info(__name__ + ": start corsika stage one")
    work_dir = corsika_and_grid_work_dir
    os.makedirs(work_dir, exist_ok=True)

    evttab_groundgrid_by_uid = {}
    for rec in evttab["groundgrid"]:
        evttab_groundgrid_by_uid[rec[snt.IDX]] = rec

    evttab_groundgrid_result_by_uid = {}
    for rec in evttab["groundgrid_result"]:
        evttab_groundgrid_result_by_uid[rec[snt.IDX]] = rec

    with cpw.cherenkov.CherenkovEventTapeWriter(
        path=opj(work_dir, "cherenkov_pools.tar")
    ) as evttar, cpw.CorsikaPrimary(
        steering_dict=corsika_primary_steering,
        stdout_path=opj(work_dir, "corsika.stdout.txt"),
        stderr_path=opj(work_dir, "corsika.stderr.txt"),
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

                    if cherenkov_in_sphere_block.shape[0] > 0:
                        cherenkov_in_instrument_block = transform_cherenkov_bunches.from_obervation_level_to_instrument(
                            cherenkov_bunches=cherenkov_in_sphere_block,
                            instrument_pointing=instrument_pointing,
                            instrument_pointing_model=env["config"][
                                "pointing"
                            ]["model"],
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

            assert (
                cherenkovpools_md5[uid["uid_str"]] == cherenkovmd5.hexdigest()
            )

    return evttab


def assert_expected_num_photons_in_sphere(
    num_photons_in_sphere,
    num_photons_in_groundgrid_bin,
):
    assert num_photons_in_sphere >= num_photons_in_groundgrid_bin


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
