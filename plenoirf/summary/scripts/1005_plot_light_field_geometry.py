#!/usr/bin/python
import sys
import plenoirf as irf
import os
from os.path import join as opj
from importlib import resources as importlib_resources
import subprocess

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

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
        opj(
            res.paths["plenoirf_dir"],
            "plenoptics",
            "instruments",
            res.instrument_key,
            "light_field_geometry",
        ),
        "--out_dir",
        res.paths["out_dir"],
    ]
)

res.stop()
