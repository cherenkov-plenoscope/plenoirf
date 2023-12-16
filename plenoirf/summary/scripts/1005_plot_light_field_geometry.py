#!/usr/bin/python
import sys
import plenoirf as irf
import os
import pkg_resources
import subprocess

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

script_path = pkg_resources.resource_filename(
    "plenoptics",
    os.path.join("scripts", "plot_beams_statistics.py"),
)
subprocess.call(
    [
        "python",
        script_path,
        "--light_field_geometry_path",
        os.path.join(pa["run_dir"], "light_field_geometry"),
        "--out_dir",
        pa["out_dir"],
    ]
)
