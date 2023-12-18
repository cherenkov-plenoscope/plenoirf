from .version import __version__

# from . import summary
# from . import analysis
# from . import features
# from . import table
# from . import instrument_response


# from . import create_test_tables
# from . import reconstruction
from . import utils
from . import outer_telescope_array
from . import tar_append
from . import other_instruments

from . import provenance
from . import bookkeeping
from . import producing
from . import configurating

import os
from os import path as op
from os.path import join as opj

import plenoptics
import magnetic_deflection
import json_line_logger


def init(plenoirf_dir, build_dir="build"):
    """
    Initializes a directory in where the instrument response function of an
    atmospheric Cherenkov instrument is estimated.

    Parameters
    ----------
    plenoirf_dir : str
        Path of the new plenoirf directory.
    build_dir : str
        Path to the build directory where the CORSIKA and merlict executables
        are located.

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
    configurating.write_default(plenoirf_dir=plenoirf_dir, build_dir=build_dir)
    config = configurating.read(plenoirf_dir=plenoirf_dir)

    plenoptics.init(
        work_dir=opj(plenoirf_dir, "plenoptics"),
        random_seed=config["plenoptics"]["random_seed"],
        minimal=config["plenoptics"]["minimal"],
    )

    magnetic_deflection.init(
        work_dir=opj(plenoirf_dir, "magnetic_deflection"),
        energy_stop_GeV=config["magnetic_deflection"]["energy_stop_GeV"],
        site_keys=config["sites"]["magnetic_deflection"],
        particle_keys=config["particles"],
        corsika_primary_path=config["executables"]["corsika_primary_path"],
    )

    configurating.version_control.init(plenoirf_dir=plenoirf_dir)


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

    assert configurating.version_control.is_clean(plenoirf_dir=plenoirf_dir)

    logger.debug("gather provenance")
    provenance.gather_and_bumb(path=opj(plenoirf_dir, "provenance", "run.tar"))

    logger.debug("read config")
    config = configurating.read(plenoirf_dir=plenoirf_dir)

    while magnetic_deflection.needs_to_run(
        work_dir=opj(plenoirf_dir, "magnetic_deflection"),
        num_showers_target=config["magnetic_deflection"]["num_showers_target"],
    ):
        logger.info("Produce more showers in magnetic_deflection.")
        magnetic_deflection.run(
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
        producing.sum_trigger.make_write_and_plot_sum_trigger_geometry(
            path=opj(plenoirf_dir, "sum_trigger_geometry", ikey),
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
