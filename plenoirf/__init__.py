from .version import __version__

# from . import summary
# from . import analysis
# from . import features
# from . import table
# from . import instrument_response
from . import provenance

# from . import create_test_tables
# from . import reconstruction
from . import utils

# from . import production
from . import other_instruments
from . import unique
from . import outer_telescope_array
from . import configurating
from . import tar_append

import os
from os import path as op
from os.path import join as opj

import rename_after_writing as rnw
import json_utils
import plenoptics
import magnetic_deflection
import json_line_logger


def init(run_dir, build_dir="build"):
    prov = provenance.make_provenance()

    run_dir = op.abspath(run_dir)
    makesuredirs(run_dir)

    makesuredirs(opj(run_dir, "provenance"))
    provenance.tar_open_append_close(
        path=opj(run_dir, "provenance", "init.tar"),
        provenance=prov,
    )

    makesuredirs(opj(run_dir, "config"))
    with rnw.open(opj(run_dir, "config", "executables.json"), "wt") as f:
        f.write(
            json_utils.dumps(
                configurating.make_executables_paths(build_dir=build_dir),
                indent=4,
            )
        )

    with rnw.open(opj(run_dir, "config", "sites.json"), "wt") as f:
        f.write(json_utils.dumps(configurating.make_sites(), indent=4))

    with rnw.open(opj(run_dir, "config", "particles.json"), "wt") as f:
        f.write(json_utils.dumps(configurating.make_particles(), indent=4))

    with rnw.open(
        opj(run_dir, "config", "magnetic_deflection.json"), "wt"
    ) as f:
        f.write(
            json_utils.dumps(
                configurating.make_magnetic_deflection(), indent=4
            )
        )

    with rnw.open(opj(run_dir, "config", "plenoptics.json"), "wt") as f:
        f.write(json_utils.dumps(configurating.make_plenoptics(), indent=4))

    config = configurating.read(run_dir=run_dir)

    plenoptics.init(
        work_dir=opj(run_dir, "plenoptics"),
        random_seed=config["plenoptics"]["random_seed"],
        minimal=config["plenoptics"]["minimal"],
    )

    magnetic_deflection.init(
        work_dir=opj(run_dir, "magnetic_deflection"),
        energy_stop_GeV=config["magnetic_deflection"]["energy_stop_GeV"],
        site_keys=config["sites"]["magnetic_deflection"],
        particle_keys=config["particles"],
        corsika_primary_path=config["executables"]["corsika_primary_path"],
    )


def run(run_dir, pool, logger=None):
    """
    Run all simulations.

    Parameters
    ----------
    run_dir : str
        Path to the run_dir initialized with init(run_dir).
    pool : e.g. multiprocessing.Pool
        Parallel compute pool which must provide a map() function.
    """
    if logger is None:
        logger = json_line_logger.LoggerStdout()


def makesuredirs(path):
    os.makedirs(path, exist_ok=True)
