#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
from plenoirf.analysis import spectral_energy_distribution as sed_styles
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.set_xlim([1e-1, 1e4])
ax.set_ylim([1e0, 1e7])
ax.loglog()
ax.set_xlabel("energy / GeV")
ax.set_ylabel("observation-time / s")
fig.savefig(opj(res.paths["out_dir"], "energy_vs_observation-time.jpg"))
sebplt.close(fig)

res.stop()
