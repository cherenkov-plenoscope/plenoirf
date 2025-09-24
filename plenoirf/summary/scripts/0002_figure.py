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


label_dir = opj(res.paths["out_dir"], "labels")
os.makedirs(label_dir, exist_ok=True)


def make_line_label(path, **kwargs):
    fig = sebplt.figure(style={"rows": 40, "cols": 120, "fontsize": 1})
    ax = sebplt.add_axes(fig=fig, span=[0, 0, 1, 1], style=sebplt.AXES_BLANK)
    ax.plot([-0.9, 0.9], [0, 0], **kwargs)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    fig.savefig(path)
    sebplt.close(fig)


colors = [
    "lightgrey",
    irf.other_instruments.fermi_lat.COLOR,
    irf.other_instruments.cherenkov_telescope_array_south.COLOR,
    irf.other_instruments.portal.COLOR,
]
for pk in irf.summary.figure.PARTICLE_COLORS:
    colors.append(irf.summary.figure.PARTICLE_COLORS[pk])

linestyles = ["-", "--", ":", "-."]

for color in colors:
    for linestyle in linestyles:
        fname = f"{color:s}_{linestyle:s}.jpg"
        make_line_label(
            path=opj(label_dir, fname),
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )


res.stop()
