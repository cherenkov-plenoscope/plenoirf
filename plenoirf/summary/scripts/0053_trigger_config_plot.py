#!/usr/bin/python
import sys
import plenoirf as irf
import os
from os.path import join as opj
import numpy as np
import sebastians_matplotlib_addons as sebplt


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

zenith_bin = res.zenith_binning(key="3_bins_per_45deg")
trigger = res.trigger


fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
x_zd_rad = np.linspace(0, np.deg2rad(45), 1337)
ax.plot(
    np.rad2deg(x_zd_rad),
    irf.light_field_trigger.get_trigger_threshold_corrected_for_pointing_zenith(
        pointing_zenith_rad=x_zd_rad,
        trigger=trigger,
        nominal_threshold_pe=trigger["threshold_pe"],
    ),
    color="black",
)
ax.set_xlabel(r"instrument pointing zenith / (1$^{\circ}$)")
ax.set_ylabel("trigger threshold / photo electrons")
ax.set_xlim([0, 50])
fig.savefig(
    opj(
        res.paths["out_dir"],
        "threshold_vs_zenith.jpg",
    )
)
sebplt.close(fig)



res.analysis["trigger"]["modus"]["accepting"]["threshold_accepting_over_rejecting"]

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
x_zd_rad = np.linspace(0, np.deg2rad(45), 1337)
ax.plot(
    np.rad2deg(x_zd_rad),
    irf.light_field_trigger.get_trigger_threshold_corrected_for_pointing_zenith(
        pointing_zenith_rad=x_zd_rad,
        trigger=trigger,
        nominal_threshold_pe=trigger["threshold_pe"],
    ),
    color="black",
)
ax.set_xlabel(r"instrument pointing zenith / (1$^{\circ}$)")
ax.set_ylabel("trigger threshold / photo electrons")
ax.set_xlim([0, 50])
fig.savefig(
    opj(
        res.paths["out_dir"],
        "threshold_vs_zenith.jpg",
    )
)
sebplt.close(fig)


res.stop()
