import os
from os.path import join as opj
import tarfile
import numpy as np
import gzip
import hashlib

import json_line_logger

from ... import utils


def run(env, seed):
    module_work_dir = opj(env["work_dir"], __name__)

    if os.path.exists(module_work_dir):
        return

    os.makedirs(module_work_dir)
    logger = json_line_logger.LoggerFile(opj(module_work_dir, "log.jsonl"))
    logger.info(__name__)
    logger.info(f"seed: {seed:d}")

    prng = np.random.Generator(np.random.PCG64(seed))

    logger.info("done.")
    json_line_logger.shutdown(logger=logger)

    # tidy up and compress
    utils.gzip_file(opj(module_work_dir, "log.jsonl"))
