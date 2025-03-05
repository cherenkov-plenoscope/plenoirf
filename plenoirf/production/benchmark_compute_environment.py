from .. import benchmarking
from .. import provenance

import os
import json_utils
import tempfile
import rename_after_writing as rnw
import json_line_logger


def run(env, logger):
    logger.info(__name__ + ": start ...")

    outpath = os.path.join(env["work_dir"], "benchmark.json")
    if os.path.exists(outpath):
        logger.info(__name__ + ": already done. skip computation.")
        return

    out = {}
    out["disk_write_rate"] = benchmarking.disk_write_rate()
    logger.debug(
        json_line_logger.xml(
            "disk_write_rate",
            rate_1k_MB_per_s=out["disk_write_rate"]["1k"]["rate_MB_per_s"][
                "avg"
            ],
            rate_1M_MB_per_s=out["disk_write_rate"]["1M"]["rate_MB_per_s"][
                "avg"
            ],
            rate_100M_MB_per_s=out["disk_write_rate"]["100M"]["rate_MB_per_s"][
                "avg"
            ],
        )
    )

    out["disk_create_write_close_open_read_remove_latency"] = (
        benchmarking.disk_create_write_close_open_read_remove_latency()
    )
    logger.debug(
        json_line_logger.xml(
            "disk_create_write_close_open_read_remove_latency",
            time=out["disk_create_write_close_open_read_remove_latency"][
                "avg"
            ],
        )
    )

    out["corsika"] = benchmarking.corsika()
    logger.debug(
        json_line_logger.xml(
            "corsika_benchmark",
            total=out["corsika"]["total"],
            initializing=out["corsika"]["initializing"],
            md5=out["corsika"]["md5"],
            energy_rate_GeV_per_s=out["corsika"]["energy_rate_GeV_per_s"],
            cherenkov_bunch_rate_per_s=out["corsika"][
                "cherenkov_bunch_rate_per_s"
            ],
        )
    )

    with rnw.open(outpath, "wt") as f:
        f.write(json_utils.dumps(out))

    logger.info(__name__ + ": ... done.")


def run_job(job):
    """
    To benchmark the compute infrustructure without running the actual
    simulations.
    """
    logger = json_line_logger.LoggerStdout()
    out = {}
    with json_line_logger.TimeDelta(logger, "provenance"):
        out["provenance"] = provenance.make_provenance()
    with json_line_logger.TimeDelta(logger, "benchmark"):
        with tempfile.TemporaryDirectory() as tmp:
            run(env={"work_dir": tmp}, logger=logger)
            with open(os.path.join(tmp, "benchmark.json"), "rt") as fin:
                out["benchmark"] = json_utils.loads(fin.read())
    return out
