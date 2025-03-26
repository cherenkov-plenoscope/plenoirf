#!/usr/bin/python
import sys
import plenoirf as irf
import os
from os.path import join as opj
from importlib import resources as importlib_resources
import subprocess

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

script_path = opj(
    importlib_resources.files("plenoptics"),
    "scripts",
    "plot_beams_statistics.py",
)

subprocess.call(
    [
        "python",
        script_path,
        "--light_field_geometry_path",
        opj(paths["plenoirf_dir"], "light_field_geometry"),
        "--out_dir",
        paths["out_dir"],
    ]
)
