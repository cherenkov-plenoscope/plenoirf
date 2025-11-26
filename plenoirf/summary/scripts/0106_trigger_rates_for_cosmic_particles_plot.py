#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

fine_energy_bin = res.energy_binning(key="interpolation")
zenith_bin = res.zenith_binning("once")

cosmic_rates = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0105_trigger_rates_for_cosmic_particles")
)

mean_key = "mean"
unc_key = "absolute_uncertainty"

trigger_thresholds = np.array(
    res.analysis["trigger"][res.site_key]["ratescan_thresholds_pe"]
)
analysis_trigger_threshold = res.analysis["trigger"][res.site_key][
    "threshold_pe"
]

tt = 0
for tt, trigger_threshold in enumerate(trigger_thresholds):
    if trigger_threshold == analysis_trigger_threshold:
        break

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    sfig, sax = irf.summary.figure.style(key="4:3")
    fig = sebplt.figure(sfig)
    ax = sebplt.add_axes(fig=fig, span=sax)
    sebplt.add_axes_zenith_range_indicator(
        fig=fig,
        span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
        zenith_bin_edges_rad=zenith_bin["edges"],
        zenith_bin=zd,
        fontsize=6,
    )

    text_y = 0.7
    for pk in res.PARTICLES:
        dRdE = cosmic_rates[zk][pk]["differential_rate"][mean_key]
        dRdE_au = cosmic_rates[zk][pk]["differential_rate"][unc_key]

        ax.plot(
            fine_energy_bin["centers"],
            dRdE[tt, :],
            color=res.PARTICLE_COLORS[pk],
            label=pk,
        )
        ax.fill_between(
            x=fine_energy_bin["centers"],
            y1=dRdE[tt, :] - dRdE_au[tt, :],
            y2=dRdE[tt, :] + dRdE_au[tt, :],
            facecolor=res.PARTICLE_COLORS[pk],
            alpha=0.2,
            linewidth=0.0,
        )
        ax.text(
            0.5,
            0.1 + text_y,
            pk,
            color=res.PARTICLE_COLORS[pk],
            transform=ax.transAxes,
        )
        ir = cosmic_rates[zk][pk]["integral_rate"][mean_key][tt]
        ir_abs_unc = cosmic_rates[zk][pk]["integral_rate"][unc_key][tt]
        ax.text(
            0.65,
            0.1 + text_y,
            r"{: 8.1f} $\pm${: 6.1f} s$^{{-1}}$".format(ir, ir_abs_unc),
            color="k",
            transform=ax.transAxes,
        )
        text_y += 0.06

    ax.set_xlabel("energy / GeV")
    ax.set_ylabel("differential trigger rate /\ns$^{-1}$ (GeV)$^{-1}$")
    ax.loglog()
    ax.set_xlim(fine_energy_bin["limits"])
    ax.set_ylim([1e-3, 1e4])
    fig.savefig(
        opj(res.paths["out_dir"], f"differential_trigger_rate_{zk:s}.jpg")
    )
    sebplt.close(fig)

res.stop()
