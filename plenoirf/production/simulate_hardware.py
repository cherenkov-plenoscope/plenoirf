import os
from os import path as op
from os.path import join as opj

import merlict_development_kit_python as mlidev
import rename_after_writing as rnw
import json_utils


def run_block(env, blk, block_id, logger):
    env = simulate_hardware(env=env, block_id=block_id)
    make_debug_output(env=env, block_id=block_id)
    return env


def simulate_hardware(env, block_id):
    mlidev_cfg_path = opj(
        env["work_dir"], "merlict_plenoscope_propagator_config.json"
    )
    if not os.path.exists(mlidev_cfg_path):
        with rnw.open(mlidev_cfg_path, "wt") as f:
            f.write(
                json_utils.dumps(
                    env["config"]["merlict_plenoscope_propagator_config"],
                    indent=4,
                )
            )

    block_dir = opj(env["work_dir"], "blocks", "{:06d}".format(block_id))

    light_field_geometry_path = opj(
        env["plenoirf_dir"],
        "plenoptics",
        "instruments",
        env["instrument_key"],
        "light_field_geometry",
    )

    rc = mlidev.plenoscope_propagator.plenoscope_propagator(
        corsika_run_path=opj(block_dir, "cherenkov_pools.tar"),
        output_path=opj(block_dir, "merlict"),
        light_field_geometry_path=light_field_geometry_path,
        merlict_plenoscope_propagator_config_path=mlidev_cfg_path,
        random_seed=env["run_id"],
        photon_origins=True,
        stdout_path=opj(block_dir, "merlict.stdout.txt"),
        stderr_path=opj(block_dir, "merlict.stderr.txt"),
    )
    assert rc == 0, "Expected merlict's return code to be zero."

    return env


def make_debug_output(env, block_id):
    block_id_str = "{:06d}".format(block_id)
    uids_in_block = env["run"]["uids_in_cherenkov_pool_blocks"][block_id_str]
    for event_uid in env["run"]["event_uids_for_debugging"]:
        if event_uid in uids_in_block:
            print("Do some debug I guess?", event_uid, block_id)
