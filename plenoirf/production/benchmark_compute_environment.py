from .. import benchmarking
import os
import json_utils
import tempfile
import rename_after_writing as rnw
from json_line_logger import xml


def run(env, logger):
    out = {}
    out["tmp"] = tempfile.gettempdir()

    out["disk_write_rate"] = benchmarking.disk_write_rate(path=out["tmp"])
    logger.debug(
        xml(
            "disk_write_rate",
            rate_1k_MB_per_s=out["disk_write_rate"]["k"]["rate_MB_per_s"][
                "avg"
            ],
            rate_1M_MB_per_s=out["disk_write_rate"]["M"]["rate_MB_per_s"][
                "avg"
            ],
            rate_1G_MB_per_s=out["disk_write_rate"]["G"]["rate_MB_per_s"][
                "avg"
            ],
        )
    )

    out[
        "disk_create_write_close_open_read_remove_latency"
    ] = benchmarking.disk_create_write_close_open_read_remove_latency(
        path=out["tmp"]
    )
    logger.debug(
        xml(
            "disk_create_write_close_open_read_remove_latency",
            time=out["disk_create_write_close_open_read_remove_latency"][
                "avg"
            ],
        )
    )

    out["corsika"] = benchmarking.corsika(path=out["tmp"])
    logger.debug(
        xml(
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

    with rnw.open(os.path.join(env["work_dir"], "benchmark.json"), "wt") as f:
        f.write(json_utils.dumps(out))
