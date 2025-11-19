import os
from os.path import join as opj
import glob
import datetime
import shutil

from sparse_numeric_table.files import (
    _split_into_chunks as _snt_split_into_chunks,
)
from .. import configuration
from .. import bookkeeping
from .. import production
from .. import utils

from . import memory
from . import by_run
from . import by_topic


def map_and_reduce_dirs(
    plenoirf_dir,
    instrument_key,
    site_key,
    particle_key,
):
    _d = opj(plenoirf_dir, "response", instrument_key, site_key, particle_key)
    return opj(_d, "map"), opj(_d, "reduce")


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


# by run
# ------


def _split_into_chunks(x, size):
    return _snt_split_into_chunks(x=x, chunk_size=size)


def _by_run_out_dirname_from_run_ids(run_ids):
    return f"{min(run_ids):06d}_to_{max(run_ids):06d}"


def _by_run_temporary_dir():
    return "__temporary_map_runs_to_topics__"


def _by_run_make_jobs_instrument_site_particle(
    plenoirf_dir,
    instrument_key,
    site_key,
    particle_key,
    num_runs_per_job,
    lazy,
):
    plenoirf_dir = os.path.abspath(plenoirf_dir)

    map_dir, reduce_dir = map_and_reduce_dirs(
        plenoirf_dir=plenoirf_dir,
        instrument_key=instrument_key,
        site_key=site_key,
        particle_key=particle_key,
    )

    all_run_ids = list_run_ids_ready_for_reduction(
        map_dir=map_dir,
    )

    chunks_run_ids = _split_into_chunks(
        x=all_run_ids,
        size=num_runs_per_job,
    )

    jobs = []
    for chunk_run_ids in chunks_run_ids:
        job = {}
        job["plenoirf_dir"] = plenoirf_dir
        job["instrument_key"] = instrument_key
        job["site_key"] = site_key
        job["particle_key"] = particle_key
        job["run_ids"] = chunk_run_ids
        job_dirname = _by_run_out_dirname_from_run_ids(chunk_run_ids)
        job["out_dir"] = opj(reduce_dir, _by_run_temporary_dir(), job_dirname)

        if os.path.exists(job["out_dir"]) and lazy:
            print(f"skipping: '{job['out_dir']:s}'")
        else:
            jobs.append(job)
    return jobs


def by_run_make_jobs(
    plenoirf_dir,
    config=None,
    num_runs_per_job=100,
    lazy=True,
    memory_scheme="hpc-nfs",
):
    memory_config = memory.make_config(scheme=memory_scheme)
    config = configuration.read_if_None(plenoirf_dir, config=config)
    assert num_runs_per_job > 0

    jobs = []
    for instrument_key in config["instruments"]:
        for site_key in config["sites"]["instruemnt_response"]:
            for particle_key in config["particles"]:
                _jobs = _by_run_make_jobs_instrument_site_particle(
                    plenoirf_dir=plenoirf_dir,
                    instrument_key=instrument_key,
                    site_key=site_key,
                    particle_key=particle_key,
                    num_runs_per_job=num_runs_per_job,
                    lazy=lazy,
                )

                for _job in _jobs:
                    _job["memory_config"] = memory_config

                jobs += _jobs
    return jobs


def by_run_run_job(job):
    return by_run.reduce(**job)


# by topic
# --------


def by_topic_make_jobs(
    plenoirf_dir,
    config=None,
    lazy=True,
    memory_scheme="hpc-nfs",
):
    memory_config = memory.make_config(scheme=memory_scheme)
    config = configuration.read_if_None(plenoirf_dir, config=config)

    jobs = []
    for instrument_key in config["instruments"]:
        for site_key in config["sites"]["instruemnt_response"]:
            for particle_key in config["particles"]:

                map_dir, reduce_dir = map_and_reduce_dirs(
                    plenoirf_dir=plenoirf_dir,
                    instrument_key=instrument_key,
                    site_key=site_key,
                    particle_key=particle_key,
                )
                temporary_run_reduction_dirs = (
                    _by_topic_find_temporary_run_reduction_dirs(
                        reduce_dir=reduce_dir
                    )
                )
                topics_in_paths = _by_topic_make_in_paths(
                    temporary_run_reduction_dirs=temporary_run_reduction_dirs
                )

                for topic_key in by_topic.list_topic_filenames():
                    if len(topics_in_paths[topic_key]) > 0:
                        job = {}
                        job["topic_key"] = topic_key
                        job["out_path"] = opj(reduce_dir, topic_key)
                        job["memory_config"] = memory_config
                        job["in_paths"] = topics_in_paths[topic_key]

                        if os.path.exists(job["out_path"]) and lazy:
                            print(f"skipping: '{job['out_path']:s}'")
                        else:
                            jobs.append(job)
    return jobs


def by_topic_run_job(job):
    return by_topic.merge_topic(**job)


def _by_topic_find_temporary_run_reduction_dirs(reduce_dir):
    return glob.glob(
        os.path.join(
            reduce_dir,
            _by_run_temporary_dir(),
            "*_to_*",
        )
    )


def _by_topic_make_in_paths(temporary_run_reduction_dirs):
    topic_paths = {}
    for filename in by_topic.list_topic_filenames():
        topic_paths[filename] = []

    valid = {}
    for reduction_dir in temporary_run_reduction_dirs:
        valid[reduction_dir] = True

    for reduction_dir in temporary_run_reduction_dirs:
        for filename in by_topic.list_topic_filenames():
            topic_path = os.path.join(reduction_dir, filename)
            if not os.path.exists(topic_path):
                valid[reduction_dir] = False

    for filename in by_topic.list_topic_filenames():
        for reduction_dir in temporary_run_reduction_dirs:
            if valid[reduction_dir]:
                topic_path = os.path.join(reduction_dir, filename)
                topic_paths[filename].append(topic_path)

    return topic_paths


# ----


def remove_temporary_reduce_dirs(plenoirf_dir, config=None, dry_run=False):
    config = configuration.read_if_None(plenoirf_dir, config=config)

    for instrument_key in config["instruments"]:
        for site_key in config["sites"]["instruemnt_response"]:
            for particle_key in config["particles"]:
                _, reduce_dir = map_and_reduce_dirs(
                    plenoirf_dir=plenoirf_dir,
                    instrument_key=instrument_key,
                    site_key=site_key,
                    particle_key=particle_key,
                )
                tmp_reduce_dir = os.path.join(
                    reduce_dir, _by_run_temporary_dir()
                )
                if os.path.exists(tmp_reduce_dir):
                    print(f"removing '{tmp_reduce_dir:s}'.")
                    if not dry_run:
                        shutil.rmtree(tmp_reduce_dir)


def is_complete(plenoirf_dir, config=None):
    config = configuration.read_if_None(plenoirf_dir, config=config)

    flag = True
    for instrument_key in config["instruments"]:
        for site_key in config["sites"]["instruemnt_response"]:
            for particle_key in config["particles"]:
                _, reduce_dir = map_and_reduce_dirs(
                    plenoirf_dir=plenoirf_dir,
                    instrument_key=instrument_key,
                    site_key=site_key,
                    particle_key=particle_key,
                )
                for filename in by_topic.list_topic_filenames():
                    topic_path = os.path.join(reduce_dir, filename)
                    if not os.path.exists(topic_path):
                        print(f"missing: '{topic_path:s}'.")
                        flag = False
    return flag


def hide_existing_reduce_dirs(plenoirf_dir, suffix=None, dry_run=False):
    """
    Hides the existing 'reduce' dirs to '.reduce.{suffix}'.
    """
    config = configuration.read_if_None(plenoirf_dir, config=None)

    if suffix is None:
        now = datetime.datetime.now()
        suffix = now.isoformat().replace(":", "-")

    for instrument_key in config["instruments"]:
        for site_key in config["sites"]["instruemnt_response"]:
            for particle_key in config["particles"]:
                _, reduce_dir = map_and_reduce_dirs(
                    plenoirf_dir=plenoirf_dir,
                    instrument_key=instrument_key,
                    site_key=site_key,
                    particle_key=particle_key,
                )
                isp_dir, _ = os.path.split(reduce_dir)

                if os.path.exists(reduce_dir):
                    old_reduce_dir = os.path.join(
                        isp_dir, f".reduce.{suffix:s}"
                    )
                    assert not os.path.exists(old_reduce_dir)

                    print(f"rename '{reduce_dir:s}' to '{old_reduce_dir:s}'.")
                    if not dry_run:
                        os.rename(reduce_dir, old_reduce_dir)
