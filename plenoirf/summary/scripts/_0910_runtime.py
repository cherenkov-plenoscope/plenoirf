#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import sparse_numeric_table as snt
import json_utils
import zipfile


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)


for pk in res.PARTICLES:
    table_path = opj(res.response_path(particle_key=pk), "benchmark.snt.zip")
    hostnames_path = table_path + ".hostname_hashes.json"

    with snt.open(table_path, "r") as arc:
        table = arc.query()

    with open(hostnames_path, "r") as f:
        hostnames = json_utils.loads(f.read())

    by_hostname = {}

    benchmark_keys = set(table["benchmark"].dtype.names)
    benchmark_keys.remove("run_id")
    benchmark_keys.remove("hostname_hash")
    benchmark_keys.remove("time_unix_s")

    for hostname in hostnames:
        hostname_hash = hostnames[hostname]
        mask = table["benchmark"]["hostname_hash"] == hostname_hash

        by_hostname[hostname] = {}
        for key in benchmark_keys:
            by_hostname[hostname][key] = {
                "p16": np.percentile(a=table["benchmark"][key][mask], q=0.16),
                "p50": np.percentile(a=table["benchmark"][key][mask], q=0.50),
                "p84": np.percentile(a=table["benchmark"][key][mask], q=0.84),
            }

    with open(
        opj(res.paths["out_dir"], f"benchmarks_by_hostname.json"), "wt"
    ) as f:
        f.write(json_utils.dumps(by_hostname, indent=4))

    key = "corsika_energy_rate_GeV_per_s_avg"
    report = "CORSIKA proton shower performance\n"
    report += "--------------------------------\n"
    report += f"{'hostname':20s} {'production rate':s}\n"
    report += f"{'':20s} {'/ GeV s^{-1}':s}\n"
    for hostname in by_hostname:
        avg = by_hostname[hostname][key]["p50"]
        std = (
            by_hostname[hostname][key]["p84"]
            - by_hostname[hostname][key]["p16"]
        )
        if std > 0:
            report += f"{hostname:20s} {avg:.2f} +- {std:.2f}\n"

    with open(opj(res.paths["out_dir"], f"{key:s}.txt"), "wt") as f:
        f.write(report)

    PERCENTILES = ["p16", "p50", "p84"]
    # make table
    with open(opj(res.paths["out_dir"], f"benchmarks.csv"), "wt") as f:
        # header line
        head = []
        head.append("hostname")
        for benchmark_key in benchmark_keys:
            for per in PERCENTILES:
                head.append(f"{benchmark_key:s} {per:s}")
        f.write(str.join(",", head) + "\n")

        for hostname in by_hostname:
            line = [hostname]
            for benchmark_key in benchmark_keys:
                for per in PERCENTILES:
                    val = by_hostname[hostname][benchmark_key][per]
                    line.append(f"{val:f}")
            f.write(str.join(",", line) + "\n")
res.stop()
