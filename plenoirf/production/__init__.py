from . import example

import os
import magnetic_deflection


def init(production_dir):
    pass


def make_jobs(production_dir):
    lock = magnetic_deflection.allsky.production.Production(
        os.path.join(path, "lock")
    )


def run_job(job):
    pass
