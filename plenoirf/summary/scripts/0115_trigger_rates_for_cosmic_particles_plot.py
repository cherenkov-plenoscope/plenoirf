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

fine_energy_bin = res.energy_binning(key="60_bins_per_decade")
zenith_bin = res.zenith_binning("3_bins_per_45deg")

cosmic_rates = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0105_trigger_rates_for_cosmic_particles")
)

TRIGGER_MODI = json_utils.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger", "trigger_modi.json")
)

mean_key = "mean"
unc_key = "absolute_uncertainty"

trigger_thresholds = np.array(
    res.analysis["trigger"]["ratescan_thresholds_pe"]
)
analysis_trigger_threshold = res.analysis["trigger"]["threshold_pe"]


for tk in TRIGGER_MODI:

    tt = irf.utils.find_closest_index_in_array_for_value(
        arr=trigger_thresholds,
        val=analysis_trigger_threshold,
    )
    if tk == "far_accepting_focus":
        tt += 18  # roughly same total trigger rate

    os.makedirs(opj(res.paths["out_dir"], tk))
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
            dRdE = cosmic_rates[zk][pk][tk]["differential_rate"][mean_key]
            dRdE_au = cosmic_rates[zk][pk][tk]["differential_rate"][unc_key]

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
            ir = cosmic_rates[zk][pk][tk]["integral_rate"][mean_key][tt]
            ir_abs_unc = cosmic_rates[zk][pk][tk]["integral_rate"][unc_key][tt]
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
            opj(
                res.paths["out_dir"],
                tk,
                f"differential_trigger_rate_{zk:s}.jpg",
            )
        )
        sebplt.close(fig)

res.stop()
