import numpy as np
import os
import plenopy
import rename_after_writing as rnw
import sparse_numeric_table as snt
from .. import bookkeeping
from .. import event_table

# from . import simulate_hardware


def run_block(env, blk, seed, block_id, logger):
    opj = os.path.join
    logger.info(__name__ + ": start ...")

    block_id_str = "{:06d}".format(block_id)
    block_dir = opj(env["work_dir"], "blocks", block_id_str)
    sub_work_dir = opj(block_dir, __name__)

    if os.path.exists(sub_work_dir):
        logger.info(__name__ + ": already done. skip computation.")
        return

    os.makedirs(sub_work_dir)
    prng = np.random.Generator(np.random.PCG64(seed))

    evttab = snt.SparseNumericTable(index_key="uid")
    evttab = event_table.add_levels_from_path(
        evttab=evttab,
        path=opj(
            block_dir,
            "plenoirf.production.simulate_loose_trigger",
            "event_table.snt.zip",
        ),
    )
    evttab = event_table.add_empty_level(evttab, "features")

    evttab = extract_features(
        evttab=evttab,
        light_field_geometry=blk["light_field_geometry"],
        light_field_geometry_addon=blk["light_field_geometry_addon"],
        event_uid_strs_in_block=blk["event_uid_strs_in_block"][block_id_str],
        block_dir=block_dir,
        prng=prng,
        logger=logger,
    )

    event_table.write_certain_levels_to_path(
        evttab=evttab,
        path=opj(sub_work_dir, "event_table.snt.zip"),
        level_keys=["features"],
    )

    logger.info(__name__ + ": ... done.")


def extract_features(
    evttab,
    light_field_geometry,
    light_field_geometry_addon,
    event_uid_strs_in_block,
    block_dir,
    prng,
    logger,
):
    for ptp in evttab["pasttrigger"]:
        event_uid = ptp["uid"]

        merlict_event_id = simulate_hardware.make_merlict_event_id(
            event_uid=event_uid,
            event_uid_strs_in_block=event_uid_strs_in_block,
        )
        event_path = os.path.join(
            block_dir, "merlict", "{:d}".format(merlict_event_id)
        )
        event = plenopy.Event(
            path=event_path, light_field_geometry=light_field_geometry
        )
        simulate_hardware.assert_plenopy_event_has_uid(
            event=event, event_uid=event_uid
        )

        try:
            lfft = plenopy.features.extract_features(
                cherenkov_photons=event.cherenkov_photons,
                light_field_geometry=light_field_geometry,
                light_field_geometry_addon=light_field_geometry_addon,
                prng=prng,
            )
            lfft["uid"] = event_uid
            evttab["features"].append_record(lfft)
        except Exception as excep:
            logger.critical(
                "uid:{:d}, exception:{:s}".format(
                    event_uid,
                    str(excep),
                )
            )

    return evttab
