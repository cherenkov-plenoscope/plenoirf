from .version import __version__


# from . import analysis
from . import features

# from . import instrument_response
# from . import create_test_tables

from . import reconstruction
from . import summary
from . import utils
from . import outer_telescope_array
from . import tar_append
from . import other_instruments

from . import provenance
from . import bookkeeping
from . import production
from . import debugging
from . import configuration
from . import event_table
from . import ground_grid
from . import configfile
from . import seeding
from . import logging
from . import benchmarking
from . import reduction

import os
import numpy as np
from os import path as op
from os.path import join as opj
import glob
import random

import plenopy
import plenoptics
import magnetic_deflection
import json_line_logger
import json_utils
from binning_utils.power10 import lower_bin_edge as power10_to_GeV


def init(plenoirf_dir):
    """
    Initializes a directory in where the instrument response function of an
    atmospheric Cherenkov instrument is estimated.

    Parameters
    ----------
    plenoirf_dir : str
        Path of the new plenoirf directory.

    Directory structure
    -------------------
    |-> plenoirf_dir
        |-> magnetic_deflection
        |   |-> namibia
        |   |   |-> gamma
        |   |   |-> electron
        |   |   |-> proton
        |   |   |-> helium
        |   |-> ...
        |
        |-> plenoptics
        |   |-> instruments
        |       |-> diag9_default_default
        |       |-> diag9_perlin55mm_gentle
        |       |-> ...
        |
        |-> response
        |   |-> namibia
        |   |   |-> gamma
        |   |   |   |-> diag9_default_default

    """
    plenoirf_dir = op.abspath(plenoirf_dir)
    os.makedirs(plenoirf_dir, exist_ok=True)

    provenance.gather_and_bumb(
        path=opj(plenoirf_dir, "provenance", "init.tar")
    )
    configuration.write_default(plenoirf_dir=plenoirf_dir)
    config = configuration.read(plenoirf_dir=plenoirf_dir)

    plenoptics.init(
        work_dir=opj(plenoirf_dir, "plenoptics"),
        random_seed=config["plenoptics"]["random_seed"],
        minimal=config["plenoptics"]["minimal"],
    )

    PORTAL_MIRROR_DIAMETER_M = 71.0
    PORTAL_FOV_HALF_ANGLE_RAD = np.deg2rad(3.25)

    magnetic_deflection.site_particle_organizer.init(
        work_dir=opj(plenoirf_dir, "magnetic_deflection"),
        site_keys=config["sites"]["magnetic_deflection"],
        particle_keys=config["particles"],
        energy_start_GeV=power10_to_GeV(
            **config["magnetic_deflection"]["energy_start_GeV_power10"]
        ),
        energy_stop_GeV=power10_to_GeV(
            **config["magnetic_deflection"]["energy_stop_GeV_power10"]
        ),
        energy_num_bins=32,
        energy_power_slope=-1.5,
        **magnetic_deflection.site_particle_organizer.guess_sky_faces_sky_vertices_and_groun_bin_area(
            field_of_view_half_angle_rad=PORTAL_FOV_HALF_ANGLE_RAD,
            mirror_diameter_m=PORTAL_MIRROR_DIAMETER_M,
        ),
    )

    configuration.version_control.init(plenoirf_dir=plenoirf_dir)


def run(
    plenoirf_dir, pool, logger=None, max_num_runs=None, skip_to_plenoirf=False
):
    """
    Run all simulations.

    Parameters
    ----------
    plenoirf_dir : str
        Path to the plenoirf_dir initialized with init(plenoirf_dir).
    pool : e.g. multiprocessing.Pool
        Parallel compute pool which must provide a map() function.
    """
    if logger is None:
        logger = json_line_logger.LoggerStdout()

    assert configuration.version_control.is_clean(plenoirf_dir=plenoirf_dir)

    logger.debug("gather provenance")
    provenance.gather_and_bumb(path=opj(plenoirf_dir, "provenance", "run.tar"))

    logger.debug("read config")
    config = configuration.read(plenoirf_dir=plenoirf_dir)

    if not skip_to_plenoirf:
        while magnetic_deflection.site_particle_organizer.needs_to_run(
            work_dir=opj(plenoirf_dir, "magnetic_deflection"),
            num_showers_target=config["magnetic_deflection"][
                "num_showers_target"
            ],
        ):
            logger.info("Produce more showers in magnetic_deflection.")
            magnetic_deflection.site_particle_organizer.run(
                work_dir=opj(plenoirf_dir, "magnetic_deflection"),
                pool=pool,
                num_runs=config["magnetic_deflection"]["run"]["num_runs"],
                num_showers_per_run=config["magnetic_deflection"]["run"][
                    "num_showers_per_run"
                ],
                num_showers_target=config["magnetic_deflection"][
                    "num_showers_target"
                ],
            )
        logger.info("magnetic_deflection production is complete")
        if not os.path.exists(
            opj(plenoirf_dir, "magnetic_deflection", "plot")
        ):
            magnetic_deflection.site_particle_organizer.run_plot(
                work_dir=opj(plenoirf_dir, "magnetic_deflection"),
                pool=pool,
            )
        logger.info("magnetic_deflection is complete")

        plenoptics.run(
            work_dir=opj(plenoirf_dir, "plenoptics"),
            pool=pool,
            logger=logger,
        )
        logger.info("plenoptics is complete")

        logger.info("estimating trigger_geometry.")
        os.makedirs(opj(plenoirf_dir, "trigger_geometry"), exist_ok=True)
        for instrumnet_key in config["instruments"]:
            logger.info(
                "estimating trigger_geometry for {:s}".format(instrumnet_key)
            )
            production.sum_trigger.make_write_and_plot_sum_trigger_geometry(
                trigger_geometry_path=opj(
                    plenoirf_dir,
                    "trigger_geometry",
                    instrumnet_key
                    + plenopy.trigger.geometry.suggested_filename_extension(),
                ),
                sum_trigger_config=config["sum_trigger"],
                light_field_calibration_path=opj(
                    plenoirf_dir,
                    "plenoptics",
                    "instruments",
                    instrumnet_key,
                    "light_field_geometry",
                ),
                logger=logger,
            )

        logger.info("trigger_geometry complete")

    logger.info("Populating the instrument response function")
    jobs = population_make_jobs(plenoirf_dir=plenoirf_dir, config=config)
    logger.info("Total of {:d} jobs is missing".format(len(jobs)))
    random.shuffle(jobs)

    if max_num_runs is not None:
        assert max_num_runs > 0
        jobs = jobs[0:max_num_runs]

    logger.info("Submitting {:d} jobs".format(len(jobs)))
    _population_register_jobs(
        plenoirf_dir=plenoirf_dir, jobs=jobs, logger=logger
    )
    results = pool.map(production.run_job, jobs)


def population_make_jobs(plenoirf_dir, config=None):
    if config is None:
        config = configuration.read(plenoirf_dir)

    run_id_range = bookkeeping.run_id_range.read_from_configfile()

    target = config["population_target"]
    part = config["population_partitioning"]

    jobs = []
    for instrument_key in target:
        for site_key in target[instrument_key]:
            for particle_key in target[instrument_key][site_key]:
                num_showers_target = target[instrument_key][site_key][
                    particle_key
                ]["num_showers_thrown"]
                num_runs = (
                    num_showers_target / part["num_showers_per_corsika_run"]
                )
                num_runs = int(np.ceil(num_runs))

                run_id_start = run_id_range["start"]
                run_id_stop = run_id_start + num_runs
                run_id_stop = min([run_id_stop, run_id_range["stop"]])

                jobs += _make_missing_jobs_instrument_site_particle(
                    plenoirf_dir=plenoirf_dir,
                    config=config,
                    instrument_key=instrument_key,
                    site_key=site_key,
                    particle_key=particle_key,
                    run_id_start=run_id_start,
                    run_id_stop=run_id_stop,
                )
    return jobs


def _population_register_jobs(plenoirf_dir, jobs, logger=None):
    if logger is None:
        logger = json_line_logger.LoggerStdout()

    runs = {}
    for job in jobs:
        key = (job["instrument_key"], job["site_key"], job["particle_key"])
        if key not in runs:
            runs[key] = []
        else:
            runs[key].append(job["run_id"])

    for key in runs:
        instrument_key, site_key, particle_key = key
        map_dir = os.path.join(
            plenoirf_dir,
            "response",
            instrument_key,
            site_key,
            particle_key,
            "map",
        )
        logger.info(
            f"Register {len(runs[key]):d} jobs in '"
            f"{instrument_key:s},"
            f"{site_key:s},"
            f"{particle_key:s}"
            f"'."
        )
        os.makedirs(map_dir, exist_ok=True)

        with bookkeeping.run_id_register.Register(map_dir=map_dir) as reg:
            reg.add_run_ids(runs[key])


def reset_run_id_register(plenoirf_dir, config=None):
    if config is None:
        config = configuration.read(plenoirf_dir)
    target = config["population_target"]

    for instrument_key in target:
        for site_key in target[instrument_key]:
            for particle_key in target[instrument_key][site_key]:
                map_dir = os.path.join(
                    plenoirf_dir,
                    "response",
                    instrument_key,
                    site_key,
                    particle_key,
                    "map",
                )
                os.makedirs(map_dir, exist_ok=True)

                with bookkeeping.run_id_register.Register(
                    map_dir=map_dir
                ) as reg:
                    reg.reset()


def _make_missing_jobs_instrument_site_particle(
    plenoirf_dir,
    config,
    instrument_key,
    site_key,
    particle_key,
    run_id_start,
    run_id_stop,
):
    p = config["population_partitioning"]

    run_ids = _make_missing_run_ids_instrument_site_particle(
        plenoirf_dir=plenoirf_dir,
        instrument_key=instrument_key,
        site_key=site_key,
        particle_key=particle_key,
        run_id_start=run_id_start,
        run_id_stop=run_id_stop,
    )
    run_ids = sorted(run_ids)

    jobs = []
    for run_id in run_ids:
        job = {}
        job["run_id"] = run_id
        job["plenoirf_dir"] = plenoirf_dir
        job["site_key"] = site_key
        job["particle_key"] = particle_key
        job["instrument_key"] = instrument_key
        job["num_events"] = p["num_showers_per_corsika_run"]
        job["max_num_events_in_merlict_run"] = p["num_showers_per_merlict_run"]
        job["debugging_figures"] = False
        jobs.append(job)
    return jobs


def _make_missing_run_ids_instrument_site_particle(
    plenoirf_dir,
    instrument_key,
    site_key,
    particle_key,
    run_id_start,
    run_id_stop,
):
    assert run_id_start >= bookkeeping.uid.RUN_ID_LOWER
    assert run_id_stop <= bookkeeping.uid.RUN_ID_UPPER

    map_dir = os.path.join(
        plenoirf_dir,
        "response",
        instrument_key,
        site_key,
        particle_key,
        "map",
    )

    if os.path.exists(map_dir):
        with bookkeeping.run_id_register.Register(map_dir=map_dir) as reg:
            existing_run_ids = set(reg.get_run_ids())
    else:
        existing_run_ids = set()

    missing_run_ids = []
    for run_id in np.arange(run_id_start, run_id_stop):
        if run_id not in existing_run_ids:
            missing_run_ids.append(run_id)

    return missing_run_ids


def benchmark(pool, out_path, num_runs):
    """
    benchmarks the compute infrastructure and writes results to out_path jsonl.
    """

    jobs = []
    for i in range(num_runs):
        jobs.append(None)

    results = pool.map(production.benchmark_compute_environment.run_job, jobs)

    with json_utils.lines.open(out_path, mode="w") as jlout:
        for result in results:
            jlout.write(result)


def reduce(plenoirf_dir, config=None, pool=None, use_tmp_dir=True):
    if pool is None:
        pool = utils.SerialPool()

    jobs = reduction.make_jobs(
        plenoirf_dir=plenoirf_dir, config=config, use_tmp_dir=use_tmp_dir
    )
    _ = pool.map(reduction.run_job, jobs)
