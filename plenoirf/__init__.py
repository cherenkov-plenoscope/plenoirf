from .version import __version__
from . import summary
from . import analysis
from . import features
from . import table
from . import instrument_response
from . import provenance
from . import create_test_tables
from . import reconstruction
from . import utils
from . import production
from . import other_instruments
from . import unique
from . import outer_telescope_array

import os
import numpy as np
from os import path as op
from os.path import join as opj
import shutil
import subprocess
import multiprocessing
import random
import glob
import tempfile
import pandas as pd
import tarfile
import io
import pkg_resources

import json_utils
import binning_utils
import plenopy as pl
import sparse_numeric_table as spt
import magnetic_deflection as mdfl
import gamma_ray_reconstruction as gamrec
import json_line_logger as jlogging
import rename_after_writing as rnw
import atmospheric_cherenkov_response
import merlict_development_kit_python


def make_example_executable_paths():
    return {
        "corsika_primary_path": opj(
            "build",
            "corsika",
            "modified",
            "corsika-75600",
            "run",
            "corsika75600Linux_QGSII_urqmd",
        ),
        "merlict_plenoscope_propagator_path": opj(
            "build", "merlict", "merlict-plenoscope-propagation"
        ),
        "merlict_plenoscope_calibration_map_path": opj(
            "build", "merlict", "merlict-plenoscope-calibration-map"
        ),
        "merlict_plenoscope_calibration_reduce_path": opj(
            "build", "merlict", "merlict-plenoscope-calibration-reduce"
        ),
        "merlict_plenoscope_raw_photon_propagation_path": opj(
            "build", "merlict", "merlict-plenoscope-raw-photon-propagation"
        ),
    }


def make_example_config_file_paths():
    return {
        "merlict_plenoscope_propagator_config_path": opj(
            "resources", "acp", "merlict_propagation_config.json"
        ),
        "merlict_plenoscope_propagator_without_night_sky_background_config_path": opj(
            "resources",
            "acp",
            "merlict_propagation_config_no_night_sky_background.json",
        ),
        "plenoscope_scenery_path": opj("resources", "acp", "71m", "scenery"),
    }


def make_example_config():
    cfg = {}
    cfg["light_field_geometry"] = {
        "num_photons_per_block": 4 * 1000 * 1000,
        "num_blocks": 360,
    }
    cfg["mount"] = "cable_robot_mount"
    cfg["plenoscope_pointing"] = atmospheric_cherenkov_response.pointing.init(
        azimuth_deg=0.0, zenith_deg=0.0
    )
    cfg["sites"] = {}
    for pk in ["namibia", "chile"]:
        cfg["sites"][pk] = atmospheric_cherenkov_response.sites.init(pk)

    cfg["particles"] = {}
    for pk in ["gamma", "electron", "proton", "helium"]:
        cfg["particles"][pk] = atmospheric_cherenkov_response.particles.init(
            pk
        )

    cfg["particles"][
        "grid"
    ] = atmospheric_cherenkov_response.demonstration.GRID

    cfg["sum_trigger"] = {
        "object_distances_m": [
            5000.0,
            6164.0,
            7600.0,
            9369.0,
            11551.0,
            14240.0,
            17556.0,
            21644.0,
            26683.0,
            32897.0,
            40557.0,
            50000.0,
        ],
        "threshold_pe": 105,
        "integration_time_slices": 10,
        "image": {
            "image_outer_radius_deg": 3.25 - 0.033335,
            "pixel_spacing_deg": 0.06667,
            "pixel_radius_deg": 0.146674,
            "max_number_nearest_lixel_in_pixel": 7,
        },
    }

    cfg["cherenkov_classification"] = {
        "region_of_interest": {
            "time_offset_start_s": -10e-9,
            "time_offset_stop_s": 10e-9,
            "direction_radius_deg": 2.0,
            "object_distance_offsets_m": [
                4000.0,
                2000.0,
                0.0,
                -2000.0,
            ],
        },
        "min_num_photons": 17,
        "neighborhood_radius_deg": 0.075,
        "direction_to_time_mixing_deg_per_s": 0.375e9,
    }

    cfg["reconstruction"] = {
        "trajectory": gamrec.trajectory.v2020dec04iron0b.config.make_example_config_for_71m_plenoscope(
            fov_radius_deg=3.25
        ),
    }

    cfg["raw_sensor_response"] = {
        "skip_num_events": 50,
    }

    cfg["runs"] = {
        "gamma": {"num": 64, "first_run_id": 1},
        "electron": {"num": 64, "first_run_id": 1},
        "proton": {"num": 64, "first_run_id": 1},
        "helium": {"num": 64, "first_run_id": 1},
    }
    cfg["num_airshowers_per_run"] = 100

    cfg["magnetic_deflection"] = {
        "num_energy_supports": 512,
        "max_energy_GeV": 64,
    }

    cfg["artificial_core_limitation"] = {
        "gamma": None,
        "electron": None,
        "proton": None,
        "helium": None,
    }

    return cfg


def init(run_dir, config=None, config_file_paths=None):
    run_dir = op.abspath(run_dir)
    os.makedirs(run_dir)
    os.makedirs(opj(run_dir, "input"))

    if config == None:
        config = make_example_config()

    json_utils.write(
        path=opj(run_dir, "input", "config.json" + "tmp"),
        out_dict=config,
    )
    rnw.move(
        opj(run_dir, "input", "config.json" + "tmp"),
        opj(run_dir, "input", "config.json"),
    )

    if config_file_paths == None:
        config_file_paths = make_example_config_file_paths()

    rnw.copy(
        src=config_file_paths["plenoscope_scenery_path"],
        dst=opj(run_dir, "input", "scenery"),
    )
    rnw.copy(
        src=config_file_paths["merlict_plenoscope_propagator_config_path"],
        dst=opj(run_dir, "input", "merlict_propagation_config.json"),
    )
    rnw.copy(
        src=config_file_paths[
            "merlict_plenoscope_propagator_without_night_sky_background_config_path"
        ],
        dst=opj(
            run_dir,
            "input",
            "merlict_plenoscope_propagator_without_night_sky_background_config.json",
        ),
    )


def _estimate_magnetic_deflection_of_air_showers(
    config, run_dir, map_and_reduce_pool, logger
):
    logger.info("Estimating magnetic deflection.")
    mdfl_dir = opj(run_dir, "magnetic_deflection")

    if op.exists(mdfl_dir):
        mdflconfig = mdfl.read_config(work_dir=mdfl_dir)

        for particle in config["particles"]:
            assert particle in mdflconfig["particles"]
        for site in config["sites"]:
            assert site in mdflconfig["sites"]
        np.testing.assert_almost_equal(
            config["plenoscope_pointing"]["azimuth_deg"],
            mdflconfig["pointing"]["azimuth_deg"],
            decimal=2,
        )
        np.testing.assert_almost_equal(
            config["plenoscope_pointing"]["zenith_deg"],
            mdflconfig["pointing"]["zenith_deg"],
            decimal=2,
        )
    else:
        mdfl.init(
            work_dir=mdfl_dir,
            particles=config["particles"],
            sites=config["sites"],
            pointing=config["plenoscope_pointing"],
            max_energy=config["magnetic_deflection"]["max_energy_GeV"],
            num_energy_supports=config["magnetic_deflection"][
                "num_energy_supports"
            ],
        )

        jobs = mdfl.make_jobs(work_dir=mdfl_dir)
        _ = map_and_reduce_pool.map(mdfl.map_and_reduce.run_job, jobs)
        mdfl.reduce(work_dir=mdfl_dir, logger=logger)


def _estimate_light_field_geometry_of_plenoscope(
    config,
    run_dir,
    map_and_reduce_pool,
    executables,
    logger,
    make_plots=True,
):
    logger.info("Estimating light-field-geometry.")

    if op.exists(opj(run_dir, "light_field_geometry")):
        assert utils.contains_same_bytes(
            opj(run_dir, "input", "scenery", "scenery.json"),
            opj(
                run_dir,
                "light_field_geometry",
                "input",
                "scenery",
                "scenery.json",
            ),
        )
    else:
        with tempfile.TemporaryDirectory(
            prefix="light_field_geometry_", dir=run_dir
        ) as map_dir:
            lfg_jobs = merlict_development_kit_python.light_field_calibration.make_jobs(
                merlict_map_path=executables[
                    "merlict_plenoscope_calibration_map_path"
                ],
                scenery_path=opj(run_dir, "input", "scenery"),
                map_dir=map_dir,
                num_photons_per_block=config["light_field_geometry"][
                    "num_photons_per_block"
                ],
                num_blocks=config["light_field_geometry"]["num_blocks"],
                random_seed=0,
            )
            _ = map_and_reduce_pool.map(
                merlict_development_kit_python.light_field_calibration.run_job, lfg_jobs
            )
            merlict_development_kit_python.light_field_calibration.reduce(
                merlict_reduce_path=executables[
                    "merlict_plenoscope_calibration_reduce_path"
                ],
                map_dir=map_dir,
                out_dir=opj(run_dir, "light_field_geometry"),
            )

    if not op.exists(opj(run_dir, "light_field_geometry", "plot")):
        if make_plots:
            logger.info("Plotting light-field-geometry.")
            lfg = pl.LightFieldGeometry(opj(run_dir, "light_field_geometry"))
            pl.plot.light_field_geometry.save_all(
                light_field_geometry=lfg,
                out_dir=opj(run_dir, "light_field_geometry", "plot"),
            )


def _estimate_trigger_geometry_of_plenoscope(
    config,
    run_dir,
    logger,
):
    logger.info("Estimating trigger-geometry.")
    if not op.exists(opj(run_dir, "trigger_geometry")):
        light_field_geometry = pl.LightFieldGeometry(
            path=opj(run_dir, "light_field_geometry")
        )
        img = config["sum_trigger"]["image"]
        trigger_image_geometry = (
            pl.trigger.geometry.init_trigger_image_geometry(
                image_outer_radius_rad=np.deg2rad(
                    img["image_outer_radius_deg"]
                ),
                pixel_spacing_rad=np.deg2rad(img["pixel_spacing_deg"]),
                pixel_radius_rad=np.deg2rad(img["pixel_radius_deg"]),
                max_number_nearest_lixel_in_pixel=img[
                    "max_number_nearest_lixel_in_pixel"
                ],
            )
        )
        trigger_geometry = pl.trigger.geometry.init_trigger_geometry(
            light_field_geometry=light_field_geometry,
            trigger_image_geometry=trigger_image_geometry,
            object_distances=config["sum_trigger"]["object_distances_m"],
        )
        pl.trigger.geometry.write(
            trigger_geometry=trigger_geometry,
            path=opj(run_dir, "trigger_geometry"),
        )
        tss = pl.trigger.geometry.init_summation_statistics(
            trigger_geometry=trigger_geometry
        )
        plenopy_trigger_script_plot_path = pkg_resources.resource_filename(
            "plenopy", os.path.join("trigger", "scripts", "plot.py")
        )
        try:
            logger.info("plotting trigger-geometry")
            subprocess.call(
                [
                    "python",
                    plenopy_trigger_script_plot_path,
                    os.path.join(run_dir, "trigger_geometry"),
                ]
            )
        except:
            logger.info("failed to plot trigger-geometry")


def _populate_table_of_thrown_air_showers(
    config,
    run_dir,
    map_and_reduce_pool,
    executables,
    tmp_absdir,
    date_dict,
    KEEP_TMP,
    LAZY_REDUCTION,
    logger,
):
    logger.info("Estimating instrument-response.")
    table_absdir = opj(run_dir, "event_table")
    os.makedirs(table_absdir, exist_ok=True)

    prov = provenance.make_provenance()
    prov = provenance.add_corsika(
        prov=prov, corsika_primary_path=executables["corsika_primary_path"]
    )

    logger.info("Write provenance.")
    json_utils.write(
        path=opj(table_absdir, "provenance.json"),
        out_dict=prov,
    )

    deflection = mdfl.read_deflection(
        work_dir=opj(run_dir, "magnetic_deflection"),
        style="dict",
    )

    irf_jobs = []
    for site_key in config["sites"]:
        site_absdir = opj(table_absdir, site_key)
        if op.exists(site_absdir):
            continue
        os.makedirs(site_absdir, exist_ok=True)

        for particle_key in config["particles"]:
            site_particle_absdir = opj(site_absdir, particle_key)
            if op.exists(site_particle_absdir):
                continue
            os.makedirs(site_particle_absdir, exist_ok=True)

            run_id = config["runs"][particle_key]["first_run_id"]
            for job_idx in np.arange(config["runs"][particle_key]["num"]):
                assert run_id > 0

                irf_job = instrument_response.make_job_dict(
                    run_dir=run_dir,
                    production_key="event_table",
                    run_id=run_id,
                    site_key=site_key,
                    particle_key=particle_key,
                    config=config,
                    deflection_table=deflection,
                    num_air_showers=config["num_airshowers_per_run"],
                    corsika_primary_path=executables["corsika_primary_path"],
                    merlict_plenoscope_propagator_path=executables[
                        "merlict_plenoscope_propagator_path"
                    ],
                    tmp_dir=tmp_absdir,
                    keep_tmp_dir=KEEP_TMP,
                    date_dict=date_dict,
                )

                run_id += 1
                irf_jobs.append(irf_job)

    random.shuffle(irf_jobs)
    _ = map_and_reduce_pool.map(instrument_response.run_job, irf_jobs)

    logger.info("Reduce instrument-response.")

    for site_key in config["sites"]:
        for particle_key in config["particles"]:
            instrument_response.reduce(
                run_dir=run_dir,
                production_key="event_table",
                site_key=site_key,
                particle_key=particle_key,
                logger=logger,
                LAZY=LAZY_REDUCTION,
            )


def run(
    run_dir,
    map_and_reduce_pool,
    executables=None,
    TMP_DIR_ON_WORKERNODE=True,
    KEEP_TMP=False,
    LAZY_REDUCTION=False,
    logger=jlogging.LoggerStdout(),
):
    map_and_reduce_pool = jlogging.MapAndReducePoolWithLogger(
        pool=map_and_reduce_pool,
        logger=logger,
    )

    date_dict = provenance.get_time_dict_now()
    logger.info("Start run()")

    run_dir = op.abspath(run_dir)

    if executables == None:
        executables = make_example_executable_paths()

    for exe_path in executables:
        executables[exe_path] = op.abspath(executables[exe_path])

    if TMP_DIR_ON_WORKERNODE:
        tmp_absdir = None
        logger.info("Use tmp_dir on workernodes.")
    else:
        tmp_absdir = opj(run_dir, "tmp")
        os.makedirs(tmp_absdir, exist_ok=True)
        logger.info("Use tmp_dir in out_dir {:s}.".format(tmp_absdir))

    logger.info("Read config")
    config = json_utils.read(opj(run_dir, "input", "config.json"))

    _estimate_magnetic_deflection_of_air_showers(
        config=config,
        run_dir=run_dir,
        map_and_reduce_pool=map_and_reduce_pool,
        logger=logger,
    )

    _estimate_light_field_geometry_of_plenoscope(
        config=config,
        run_dir=run_dir,
        map_and_reduce_pool=map_and_reduce_pool,
        executables=executables,
        logger=logger,
    )

    _estimate_trigger_geometry_of_plenoscope(
        config=config,
        run_dir=run_dir,
        logger=logger,
    )

    _populate_table_of_thrown_air_showers(
        config=config,
        run_dir=run_dir,
        map_and_reduce_pool=map_and_reduce_pool,
        executables=executables,
        tmp_absdir=tmp_absdir,
        KEEP_TMP=KEEP_TMP,
        date_dict=date_dict,
        LAZY_REDUCTION=LAZY_REDUCTION,
        logger=logger,
    )

    logger.info("End run().")
