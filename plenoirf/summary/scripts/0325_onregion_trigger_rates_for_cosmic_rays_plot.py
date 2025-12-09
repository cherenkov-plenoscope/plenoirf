#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import lima1983analysis
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

ONREGION_TYPES = res.analysis["on_off_measuremnent"]["onregion_types"]

onregion_rates = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"],
        "0320_onregion_trigger_rates_for_cosmic_rays",
    )
)

fine_energy_bin = res.energy_binning(key="interpolation")
zenith_bin = res.zenith_binning("3_bins_per_45deg")

mean_key = "mean"
unc_key = "absolute_uncertainty"

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    for ok in ONREGION_TYPES:
        sfig, sax = irf.summary.figure.style("4:3")
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
            dRdE = onregion_rates[zk][ok][pk]["differential_rate"][mean_key]
            dRdE_au = onregion_rates[zk][ok][pk]["differential_rate"][unc_key]
            ax.plot(
                fine_energy_bin["centers"],
                dRdE,
                color=res.PARTICLE_COLORS[pk],
            )
            ax.fill_between(
                x=fine_energy_bin["centers"],
                y1=dRdE - dRdE_au,
                y2=dRdE + dRdE_au,
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
            ir = onregion_rates[zk][ok][pk]["integral_rate"][mean_key]
            ir_abs_unc = onregion_rates[zk][ok][pk]["integral_rate"][unc_key]
            ax.text(
                0.65,
                0.1 + text_y,
                r"{: 8.1f} $\pm${: 6.1f} s$^{{-1}}$".format(ir, ir_abs_unc),
                color="k",
                transform=ax.transAxes,
            )
            text_y += 0.06

        ax.set_xlim(fine_energy_bin["limits"])
        ax.set_ylim([1e-5, 1e3])
        ax.loglog()
        ax.set_xlabel("energy / GeV")
        ax.set_ylabel(r"differential rate / s$^{-1}$ (GeV)$^{-1}$")
        fig.savefig(
            opj(
                res.paths["out_dir"],
                f"{zk:s}_{ok:s}_differential_event_rates.jpg",
            )
        )
        sebplt.close(fig)

res.stop()
