from .. import benchmarking
from .. import provenance

import os
import gzip
from os.path import join as opj
import json_utils
import rename_after_writing as rnw
import json_line_logger


def run(env):
    module_work_dir = opj(env["work_dir"], __name__)

    if os.path.exists(module_work_dir):
        return

    out = provenance.make_provenance()

    os.makedirs(module_work_dir)
    with rnw.Path(opj(module_work_dir, "provenance.json.gz")) as opath:
        with gzip.open(opath, "wt") as f:
            f.write(json_utils.dumps(out))
