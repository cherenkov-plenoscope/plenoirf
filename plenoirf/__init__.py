from .version import __version__


# from . import analysis
# from . import features
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

import os
from os import path as op
from os.path import join as opj

import plenoptics
import magnetic_deflection
import json_line_logger


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
    makesuredirs(plenoirf_dir)

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
        energy_start_GeV=config["energy_range"]["energy_start_GeV"],
        energy_stop_GeV=config["magnetic_deflection"]["energy_stop_GeV"],
        energy_num_bins=32,
        energy_power_slope=-1.5,
        **mdfl.site_particle_organizer.guess_sky_faces_sky_vertices_and_groun_bin_area(
            field_of_view_half_angle_rad=PORTAL_FOV_HALF_ANGLE_RAD,
            mirror_diameter_m=PORTAL_MIRROR_DIAMETER_M,
        ),
    )

    configuration.version_control.init(plenoirf_dir=plenoirf_dir)


def run(plenoirf_dir, pool, logger=None):
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

    while magnetic_deflection.site_particle_organizer.needs_to_run(
        work_dir=opj(plenoirf_dir, "magnetic_deflection"),
        num_showers_target=config["magnetic_deflection"]["num_showers_target"],
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
    logger.info("magnetic_deflection is complete")

    plenoptics.run(
        work_dir=opj(plenoirf_dir, "plenoptics"),
        pool=pool,
        logger=logger,
    )
    logger.info("plenoptics is complete")

    logger.info("estimating sum-trigger geometry.")
    for ikey in config["instruments"]:
        logger.info("estimating sum-trigger geometry for {:s}".format(ikey))
        production.sum_trigger.make_write_and_plot_sum_trigger_geometry(
            path=opj(plenoirf_dir, "trigger_geometry", ikey),
            sum_trigger_config=config["sum_trigger"],
            light_field_calibration_path=opj(
                plenoirf_dir,
                "plenoptics",
                "instruments",
                ikey,
                "light_field_geometry",
            ),
            logger=logger,
        )

    logger.info("sum-trigger geometry complete")


def makesuredirs(path):
    os.makedirs(path, exist_ok=True)
