import os
from os import path as op
from os.path import join as opj
import magnetic_deflection
import tempfile
import json_line_logger

from .. import bookkeeping
from .. import configurating


def make_jobs(production_dir):
    lock = magnetic_deflection.allsky.production.Production(
        os.path.join(path, "lock")
    )


def make_example_job(
    plenoirf_dir,
    run_id=1337,
    site_key="chile",
    particle_key="electron",
    num_showers=16,
):
    config = configurating.read(plenoirf_dir=plenoirf_dir)

    job = {}
    job["run_id"] = run_id
    job["random_seed"] = bookkeeping.random_seed_offset.make_random_seed(
        site_key=site_key,
        particle_key=particle_key,
        corsika_run_id=job["run_id"],
        sites=config["sites"]["instruemnt_response"],
        particles=config["particles"],
    )
    job["plenoirf_dir"] = plenoirf_dir
    job["site_key"] = site_key
    job["particle_key"] = particle_key
    job["num_showers"] = num_showers
    job["log_path"] = None
    return job


def run_job(job):
    with tempfile.TemporaryDirectory(suffix="-plenoirf") as tmp_dir:
        return run_job_in_dir(job=job, tmp_dir=tmp_dir)


def run_job_in_dir(job, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
