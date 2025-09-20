#!/usr/bin/python
import sys
import plenoirf as irf
import os
from os.path import join as opj
import json_utils
import cosmic_fluxes
import sebastians_matplotlib_addons as sebplt

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt)

for key in ["16:9", "1:1", "16:6", "4:3"]:

    fstyle, axspan = irf.summary.figure.style(key=key)
    fig = sebplt.figure(fstyle)
    ax = sebplt.add_axes(fig=fig, span=axspan)
    ax.set_xlabel(r"$x$ / 1")
    ax.set_ylabel(r"$y$ / 1")
    fig.savefig(opj(res.paths["out_dir"], f"{key:s}.jpg"))
    sebplt.close(fig)

res.stop()
