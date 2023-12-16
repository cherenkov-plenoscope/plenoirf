#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
from plenoirf.analysis import spectral_energy_distribution as sed_styles
import os
import sebastians_matplotlib_addons as seb
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.set_xlim([1e-1, 1e4])
ax.set_ylim([1e0, 1e7])
ax.loglog()
ax.set_xlabel("energy / GeV")
ax.set_ylabel("observation-time / s")
fig.savefig(os.path.join(pa["out_dir"], "energy_vs_observation-time.jpg"))
seb.close(fig)
