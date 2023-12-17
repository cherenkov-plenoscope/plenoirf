import os
from os import path as op
from os.path import join as opj
import magnetic_deflection
import tempfile
import json_line_logger as jlog
import copy

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
    num_events=16,
):
    job = {}
    job["run_id"] = run_id
    job["plenoirf_dir"] = plenoirf_dir
    job["site_key"] = site_key
    job["particle_key"] = particle_key
    job["instrument_key"] = instrument_key
    job["num_events"] = num_events
    job["debug_probability"] = 1e-2
    return job


def run_job(job):
    with tempfile.TemporaryDirectory(suffix="-plenoirf") as tmp_dir:
        return run_job_in_dir(job=job, tmp_dir=tmp_dir)


def run_job_in_dir(job, tmp_dir):
    paths = make_paths(job=job)

    os.makedirs(paths["stage_dir"])

    logger_path = op.join(paths["stage_dir"], "log.jsonl")
    logger = jlog.LoggerFile(path=logger_path + ".tmp")
    logger.info("starting")

    logger.debug("making tmp_dir: {:s}".format(tmp_dir))
    os.makedirs(tmp_dir, exist_ok=True)

    logger.debug("reading plenoirf config")
    config = configurating.read(plenoirf_dir=plenoirf_dir)
    logger.debug("reading light-field camera config")
    light_field_camera_config = read_light_field_camera_config(
        plenoirf_dir=job["plenoirf_dir"],
        instrument_key=job["instrument_key"],
    )

    logger.debug("initializing prng with seed: {:d}".format(job["run_id"]))
    prng = np.random.Generator(np.random.PCG64(job["run_id"]))

    logger.debug("initializing this run's pointing-range")
    pointing_range = make_pointing_range_for_run(config=config, prng=prng)

    with jlog.TimeDelta(logger, "draw_primary_and_pointing"):
        _allsky = magnetic_deflection.allsky.AllSky(
            paths["magnetic_deflection_allsky"]
        )
        site = copy.deepcopy(_allsky.config["site"])
        particle = copy.deepcopy(_allsky.config["particle"])

        drw = random.draw_primaries_and_pointings(
            prng=prng,
            run_id=job["run_id"],
            site_particle_magnetic_deflection=_allsky,
            pointing_range=pointing_range,
            field_of_view_half_angle_rad=(
                0.5
                * np.deg2rad(light_field_camera_config["max_FoV_diameter_deg"])
            ),
            num_events=job["num_events"],
        )

    logger.info("ending")
    logger.debug("moving log.json to final path")
    rnw.move(logger_path + ".tmp", logger_path)
    return 1


def read_light_field_camera_config(plenoirf_dir, instrument_key):
    return merlict_development_kit_python.plenoscope_propagator.read_plenoscope_geometry(
        merlict_scenery_path=opj(
            plenoirf_dir,
            "plenoptics",
            "instruments",
            instrument_key,
            "light_field_geometry",
            "input",
            "scenery",
            "scenery.json",
        )
    )


def make_paths(
    plenoirf_dir,
    site_key,
    particle_key,
    instrument_key,
):
    paths = {}
    paths["plenoirf_dir"] = plenoirf_dir
    paths["stage_dir"] = opj(
        plenoirf_dir,
        "response",
        instrument_key,
        site_key,
        particle_key,
        "stage",
    )
    paths["magnetic_deflection_allsky"] = opj(
        plenoirf_dir, "magnetic_deflection", site_key, particle_key
    )
    paths["light_field_calibration"] = opj(
        plenoirf_dir,
        "plenoptics",
        "instruments",
        instrument_key,
        "light_field_geometry",
    )


def make_pointing_range_for_run(config, prng):
    total_range = acr.pointing_range.PointingRange_from_cone(
        azimuth_rad=0.0,
        zenith_rad=0.0,
        half_angel_rad=config["pointing"]["range"]["max_zenith_distance_rad"],
    )
    ptg = acr.pointing_range.draw_pointing(
        pointing_range=total_range, prng=prng
    )
    return acr.pointing_range.PointingRange_from_cone(
        azimuth_rad=ptg["azimuth_rad"],
        zenith_rad=ptg["zenith_rad"],
        half_angel_rad=config["pointing"]["range"]["run_half_angle_rad"],
    )
