import numpy as np
import os
from os.path import join as opj
import plenopy
import rename_after_writing as rnw
import sparse_numeric_table as snt
import json_line_logger

from .. import bookkeeping
from .. import event_table
from .. import utils


def run(env, part, seed):
    name = __name__.split(".")[-1]
    module_work_dir = opj(env["work_dir"], part, name)

    if os.path.exists(module_work_dir):
        return

    os.makedirs(module_work_dir)
    logger = json_line_logger.LoggerFile(opj(module_work_dir, "log.jsonl"))
    logger.info(name)
    logger.info(f"seed: {seed:d}")

    prng = np.random.Generator(np.random.PCG64(seed))

    evttab = snt.SparseNumericTable(index_key="uid")
    evttab = event_table.add_levels_from_path(
        evttab=evttab,
        path=opj(
            env["work_dir"],
            "cer2cls",
            "simulate_instrument_and_reconstruct_cherenkov",
            "event_table.snt.zip",
        ),
    )
    evttab = event_table.add_empty_level(evttab, "features")

    evttab = extract_features(
        evttab=evttab,
        light_field_geometry=env["light_field_geometry"],
        light_field_geometry_addon=env["light_field_geometry_addon"],
        reconstructed_cherenkov_path=opj(
            env["work_dir"],
            "cer2cls",
            "simulate_instrument_and_reconstruct_cherenkov",
            "reconstructed_cherenkov.loph.tar",
        ),
        prng=prng,
        logger=logger,
    )

    event_table.write_certain_levels_to_path(
        evttab=evttab,
        path=opj(module_work_dir, "event_table.snt.zip"),
        level_keys=["features"],
    )

    logger.info("done.")
    json_line_logger.shutdown(logger=logger)
    utils.gzip_file(opj(module_work_dir, "log.jsonl"))


def extract_features(
    evttab,
    light_field_geometry,
    light_field_geometry_addon,
    reconstructed_cherenkov_path,
    prng,
    logger,
):
    with plenopy.photon_stream.loph.LopfTarReader(
        reconstructed_cherenkov_path
    ) as lin:
        for event in lin:
            event_uid, event_loph = event

            cherenkov_photons = plenopy.classify.RawPhotons.from_lopf(
                loph=event_loph,
                light_field_geometry=light_field_geometry,
            )

            try:
                lfft = plenopy.features.extract_features(
                    cherenkov_photons=cherenkov_photons,
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
