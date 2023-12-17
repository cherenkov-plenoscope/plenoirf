from . import example

import os
from os import path as op
from os.path import join as opj
import magnetic_deflection
import tempfile
import json_line_logger


def init(production_dir):
    pass


def make_jobs(production_dir):
    lock = magnetic_deflection.allsky.production.Production(
        os.path.join(path, "lock")
    )


def make_random_seed_offset(particle_key, site_key, particle_keys, site_keys):
    site_keys = list(site_keys)
    particle_keys = list(particle_keys)
    assert len(particle_keys) <= 10
    assert len(site_keys) <= 10
    sorted_site_keys = sorted(site_keys)
    sorted_particle_keys = sorted(particle_keys)
    sidx = sorted_site_keys.index(site_key)
    pidx = sorted_particle_keys.index(particle_key)
    return 10 * sidx + pidx


def make_example_job(
    plenoirf_dir,
    run_id=1337,
    site_key="chile",
    particle_key="electron",
    num_showers=16,
):
    job["run_id"] = run_id
    job["random_seed_millions_to_billions"] = 137
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

    json_line_logger

    _assert_resources_exist(job=job)
    _make_output_dirs(job=job)
    _export_job_to_log_dir(job=job)

    log_path = op.join(job["log_dir"], _run_id_str(job) + "_runtime.jsonl")
    logger = jlogging.LoggerFile(path=log_path + ".tmp")
    logger.info("starting run")

    logger.info("init prng")
    prng = np.random.Generator(np.random.MT19937(seed=job["run_id"]))

    with jlogging.TimeDelta(logger, "draw_primary"):
        corsika_primary_steering = atmospheric_cherenkov_response.particles.draw_corsika_primary_steering(
            run_id=job["run_id"],
            site=job["site"],
            particle=job["particle"],
            site_particle_deflection=job["site_particle_deflection"],
            num_events=job["num_air_showers"],
            prng=prng,
        )

    if job["tmp_dir"] is None:
        tmp_dir = tempfile.mkdtemp(prefix="plenoscope_irf_")
    else:
        tmp_dir = op.join(job["tmp_dir"], _run_id_str(job))
        os.makedirs(tmp_dir, exist_ok=True)
    logger.info("make tmp_dir: {:s}".format(tmp_dir))

    tabrec = _init_table_records()

    with jlogging.TimeDelta(logger, "corsika_and_grid"):
        (
            cherenkov_pools_path,
            tabrec,
        ) = _run_corsika_and_grid_and_output_to_tmp_dir(
            job=job,
            prng=prng,
            tmp_dir=tmp_dir,
            corsika_primary_steering=corsika_primary_steering,
            tabrec=tabrec,
        )

    with jlogging.TimeDelta(logger, "particlepool"):
        tabrec = _populate_particlepool(job, tabrec)

    with jlogging.TimeDelta(logger, "merlict"):
        detector_responses_path = _run_merlict(
            job=job,
            cherenkov_pools_path=cherenkov_pools_path,
            tmp_dir=tmp_dir,
        )

    if not job["keep_tmp"]:
        os.remove(cherenkov_pools_path)

    with jlogging.TimeDelta(logger, "read_geometry"):
        light_field_geometry = pl.LightFieldGeometry(
            path=job["light_field_geometry_path"]
        )
        trigger_geometry = pl.trigger.geometry.read(
            path=job["trigger_geometry_path"]
        )

    with jlogging.TimeDelta(logger, "pass_loose_trigger"):
        tabrec, table_past_trigger, tmp_past_trigger_dir = _run_loose_trigger(
            job=job,
            tabrec=tabrec,
            detector_responses_path=detector_responses_path,
            light_field_geometry=light_field_geometry,
            trigger_geometry=trigger_geometry,
            tmp_dir=tmp_dir,
        )

    with jlogging.TimeDelta(logger, "export grid region-of-interest"):
        _export_grid_region_of_interest_if_passed_loose_trigger(
            job=job,
            tabrec=tabrec,
            tmp_dir=tmp_dir,
        )

    with jlogging.TimeDelta(logger, "classify_cherenkov"):
        tabrec = _classify_cherenkov_photons(
            job=job,
            tabrec=tabrec,
            tmp_dir=tmp_dir,
            table_past_trigger=table_past_trigger,
            light_field_geometry=light_field_geometry,
            trigger_geometry=trigger_geometry,
        )

    with jlogging.TimeDelta(logger, "extract_features"):
        tabrec = _extract_features(
            tabrec=tabrec,
            light_field_geometry=light_field_geometry,
            table_past_trigger=table_past_trigger,
            prng=prng,
        )

    with jlogging.TimeDelta(logger, "estimate_primary_trajectory"):
        tabrec = _estimate_primary_trajectory(
            job=job,
            tmp_dir=tmp_dir,
            light_field_geometry=light_field_geometry,
            tabrec=tabrec,
        )

    with jlogging.TimeDelta(logger, "export_event_table"):
        _export_event_table(job=job, tmp_dir=tmp_dir, tabrec=tabrec)

    if not job["keep_tmp"]:
        shutil.rmtree(tmp_dir)
    logger.info("ending run")
    nfs.move(log_path + ".tmp", log_path)
