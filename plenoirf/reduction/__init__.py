import os
from os.path import join as opj
import glob

from .. import configuration
from .. import bookkeeping
from .. import production
from .. import utils

from . import memory
from . import by_run
from . import by_topic


def reduce_item(
    instrument_site_particle_dir,
    item_key,
    memory_config=None,
):
    memory_config = memory.make_config_if_None(memory_config)

    run_ids = list_run_ids_ready_for_reduction(
        map_dir=opj(instrument_site_particle_dir, "map"),
        checkpoint_keys=production.list_checkpoint_keys(),
    )

    if item_key == "event_table.snt.zip":
        recude_event_table(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            memory_config=memory_config,
        )
    elif item_key == "reconstructed_cherenkov.loph.tar":
        reduce_reconstructed_cherenkov(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            memory_config=memory_config,
        )
    elif item_key == "ground_grid_intensity.zip":
        reduce_ground_grid_intensity(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            roi=False,
            only_past_trigger=True,
            memory_config=memory_config,
        )
    elif item_key == "ground_grid_intensity_roi.zip":
        reduce_ground_grid_intensity(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            roi=True,
            only_past_trigger=True,
            memory_config=memory_config,
        )
    elif item_key == "benchmark.snt.zip":
        reduce_benchmarks(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            memory_config=memory_config,
        )
    elif item_key == "event_uids_for_debugging.txt":
        reduce_event_uids_for_debugging(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            memory_config=memory_config,
        )
    else:
        raise KeyError(f"No such item_key '{item_key}'.")


def _get_run_id_set_in_directory(path, filename_wildcardard="*.zip"):
    all_paths = glob.glob(opj(path, filename_wildcardard))

    run_ids = set()
    for a_path in all_paths:
        basename = os.path.basename(a_path)
        if len(basename) >= 6:
            first_six_chars = basename[0:6]
            if utils.can_be_interpreted_as_int(first_six_chars):
                run_id = int(first_six_chars)
                assert run_id not in run_ids
                run_ids.add(run_id)
    return run_ids


def list_run_ids_ready_for_reduction(map_dir, checkpoint_keys=None):
    if checkpoint_keys is None:
        checkpoint_keys = production.list_checkpoint_keys()

    run_ids_in_checkpoints = []
    for key in checkpoint_keys:
        run_ids_in_checkpoints.append(
            _get_run_id_set_in_directory(path=opj(map_dir, key))
        )

    run_ids = list(set.intersection(*run_ids_in_checkpoints))
    return sorted(run_ids)


def make_jobs(
    plenoirf_dir,
    config=None,
    lazy=True,
    memory_scheme="hpc-nfs",
):
    memory_config = memory.make_config(scheme=memory_scheme)

    if config is None:
        config = configuration.read(plenoirf_dir)

    production_final_checkpoint_key = production.list_checkpoint_keys()[-1]

    jobs = []
    for instrument_key in config["instruments"]:
        for site_key in config["sites"]["instruemnt_response"]:
            for particle_key in config["particles"]:
                instrument_site_particle_dir = opj(
                    plenoirf_dir,
                    "response",
                    instrument_key,
                    site_key,
                    particle_key,
                )

                map_dir, reduce_dir = _map_reduce_dirs(
                    instrument_site_particle_dir=instrument_site_particle_dir
                )
                if has_filenames_of_certain_pattern_in_path(
                    path=opj(
                        map_dir,
                        production_final_checkpoint_key,
                    ),
                    filename_wildcard="*.zip",
                ):
                    for item_key in list_items():
                        job_out_path = opj(reduce_dir, item_key)

                        if os.path.exists(job_out_path) and lazy:
                            print(f"skipping: '{job_out_path:s}'")
                        else:
                            job = {
                                "plenoirf_dir": plenoirf_dir,
                                "instrument_key": instrument_key,
                                "site_key": site_key,
                                "particle_key": particle_key,
                                "item_key": item_key,
                                "memory_config": memory_config,
                            }
                            jobs.append(job)
    return jobs


def has_filenames_of_certain_pattern_in_path(path, filename_wildcard):
    matching_paths = glob.glob(opj(path, filename_wildcard))
    return len(matching_paths) > 0


def run_job(job):
    instrument_site_particle_dir = opj(
        job["plenoirf_dir"],
        "response",
        job["instrument_key"],
        job["site_key"],
        job["particle_key"],
    )
    reduce_item(
        instrument_site_particle_dir=instrument_site_particle_dir,
        item_key=job["item_key"],
        memory_config=job["memory_config"],
    )
