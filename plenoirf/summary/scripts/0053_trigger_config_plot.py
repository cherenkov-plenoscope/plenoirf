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


fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
accepting_response_pe = np.geomspace(1e2, 1e6, 1337)
ax.plot(
    accepting_response_pe,
    irf.light_field_trigger.get_accepting_over_rejecting(
        trigger=trigger,
        accepting_response_pe=accepting_response_pe,
    ),
    color="black",
)
ax.set_xlabel("accepting focus response / photo electrons")
ax.set_ylabel("accepting over rejecting ratio / 1")
ax.set_xlim([min(accepting_response_pe), max(accepting_response_pe)])
ax.semilogx()
ax.axvline(x=trigger["threshold_pe"], color="gray", linestyle="--")
fig.savefig(
    opj(
        res.paths["out_dir"],
        "accepting_over_rejecting_ratio_vs_accepting_response.jpg",
    )
)
sebplt.close(fig)


res.stop()
