import os
from os import path as op
from os.path import join as opj
import magnetic_deflection
import tempfile
import json_line_logger

from .. import bookkeeping
from .. import configurating
from . import random

def make_jobs(production_dir):
    lock = magnetic_deflection.allsky.production.Production(
        os.path.join(path, "lock")
    )


def make_example_job(
    plenoirf_dir,
    run_id=1337,
    site_key="chile",
    particle_key="electron",
    instrument_key="diag9_default_default",
    num_showers=16,
):
    config = configurating.read(plenoirf_dir=plenoirf_dir)

    job = {}
    job["run_id"] = run_id
    job["plenoirf_dir"] = plenoirf_dir
    job["site_key"] = site_key
    job["particle_key"] = particle_key
    job["instrument_key"] = instrument_key
    job["num_showers"] = num_showers
    job["debug_probability"] = 1e-2
    return job


def run_job(job):
    with tempfile.TemporaryDirectory(suffix="-plenoirf") as tmp_dir:
        return run_job_in_dir(job=job, tmp_dir=tmp_dir)


def run_job_in_dir(job, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)

    prng = np.random.Generator(np.random.PCG64(job["run_id"]))



    """
    site_particle_magnetic_deflection = magnetic_deflection.allsky.AllSky(
        opj(job["plenoirf_dir"], "magnetic_deflection", job["site_key"], job["particle_key"])
    )

    drw = random.draw_events(
        run_id=job["run_id"],
        site_particle_magnetic_deflection=site_particle_magnetic_deflection,
        instrument_pointing_range=,
        instrument_field_of_view_half_angle_rad=,
        num_events,
    )
    """