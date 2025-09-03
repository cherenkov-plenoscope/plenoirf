#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import lima1983analysis
import cosmic_fluxes
import json_utils
import binning_utils
import propagate_uncertainties
import copy
import lima1983analysis

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

ONREGION_TYPES = res.analysis["on_off_measuremnent"]["onregion_types"]

onregion_rates = json_utils.tree.read(
    opj(
        res.paths["analysis_dir"],
        "0320_onregion_trigger_rates_for_cosmic_rays",
    )
)
onregion_acceptance = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0300_onregion_trigger_acceptance")
)

energy_bin = res.energy_binning(key="trigger_acceptance_onregion")
energy_fine_bin = res.energy_binning(key="interpolation")
zenith_bin = res.zenith_binning("once")

PULSARS = cosmic_fluxes.pulsars.list_pulsar_names()


def array_to_txt(arr, path):
    with open(path, "wt") as f:
        for val in arr:
            f.write("{:e}\n".format(val))


with open(opj(res.paths["out_dir"], "README.md"), "wt") as f:
    ccc = """
    E: energy.
    A: Effective area for gamma-rays after all cuts in on-region.
    B: Rate of detected background (mostly cosmic-rays)
    K: Flux of pulsar.
    dKdE: Differential flux of pulsar w.r.t. energy.
    R: Rate of detected gamma-rays.
    dRdE: Differential rate of detected gamma-rays w.r.t. energy.
    """
    f.write(ccc)

array_to_txt(
    arr=energy_fine_bin["edges"],
    path=opj(res.paths["out_dir"], "E_bin_edges_GeV.txt"),
)

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    zk_dir = opj(res.paths["out_dir"], zk)
    os.makedirs(zk_dir, exist_ok=True)

    for ok in ONREGION_TYPES:
        zk_ok_dir = opj(zk_dir, ok)
        os.makedirs(zk_ok_dir, exist_ok=True)

        A_gamma = onregion_acceptance[zk][ok]["gamma"]["point"]["mean"]
        A_gamma_fine_m2 = np.interp(
            x=energy_fine_bin["centers"],
            xp=energy_bin["centers"],
            fp=A_gamma,
        )
        array_to_txt(arr=A_gamma_fine_m2, path=opj(zk_ok_dir, "A_m2.txt"))

        # background
        # ----------
        rates_cosmic_rays_per_s = []
        rates_cosmic_rays_per_s_au = []
        rates_of_cosmic_rays = {}
        for pk in res.COSMIC_RAYS:
            rate = onregion_rates[zk][ok][pk]["integral_rate"]["mean"]
            rate_au = onregion_rates[zk][ok][pk]["integral_rate"][
                "absolute_uncertainty"
            ]
            rates_of_cosmic_rays[pk] = rate
            rates_cosmic_rays_per_s.append(rate)
            rates_cosmic_rays_per_s_au.append(rate_au)
        (
            rate_cosmic_rays_per_s,
            rate_cosmic_rays_per_s_au,
        ) = propagate_uncertainties.sum(
            x=rates_cosmic_rays_per_s, x_au=rates_cosmic_rays_per_s_au
        )
        array_to_txt(
            arr=[rate_cosmic_rays_per_s],
            path=opj(zk_ok_dir, "B_per_s.txt"),
        )
        for pk in res.COSMIC_RAYS:
            array_to_txt(
                arr=[rates_of_cosmic_rays[pk]],
                path=opj(zk_ok_dir, "B_{:s}_per_s.txt".format(pk)),
            )

        for pk in PULSARS:
            zk_ok_pk_dir = opj(zk_ok_dir, pk)

            print(zk, ok, pk)

            try:
                pulsar = irf.analysis.pulsar_timing.ppog_init_from_profiles(
                    energy_bin_edges=energy_fine_bin["edges"],
                    pulsar_name=pk,
                )
            except KeyError:
                continue

            os.makedirs(zk_ok_pk_dir, exist_ok=True)
            dKdE_per_m2_per_s_per_GeV = pulsar["differential_flux_vs_energy"]

            array_to_txt(
                arr=dKdE_per_m2_per_s_per_GeV,
                path=opj(zk_ok_pk_dir, "dKdE_per_m2_per_s_per_GeV.txt"),
            )
            fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
            ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
            sebplt.add_axes_zenith_range_indicator(
                fig=fig,
                span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
                zenith_bin_edges_rad=zenith_bin["edges"],
                zenith_bin=zd,
                fontsize=6,
            )
            ax.plot(
                energy_fine_bin["edges"][0:-1],
                dKdE_per_m2_per_s_per_GeV,
                "k-",
            )
            ax.loglog()
            _ymax = np.max(dKdE_per_m2_per_s_per_GeV)
            _ymax = 10 ** np.ceil(np.log10(_ymax))
            ax.set_ylim([1e-4 * _ymax, _ymax])
            ax.set_xlabel(r"E$\,/\,$GeV")
            ax.set_ylabel(
                r"$\frac{\mathrm{d\,K}}{\mathrm{d\,E}}$ / m$^{-2}$ s$^{-1}$ (GeV)$^{-1}$"
            )
            fig.savefig(opj(zk_ok_pk_dir, "dKdE_per_m2_per_s_per_GeV.jpg"))
            sebplt.close(fig)

            dRdE_per_s_per_GeV = np.zeros(energy_fine_bin["num"])
            for ebin in range(energy_fine_bin["num"]):
                dRdE_per_s_per_GeV[ebin] = (
                    A_gamma_fine_m2[ebin] * dKdE_per_m2_per_s_per_GeV[ebin]
                )

            array_to_txt(
                arr=dRdE_per_s_per_GeV,
                path=opj(zk_ok_pk_dir, "dRdE_per_s_per_GeV.txt"),
            )
            fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
            ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
            sebplt.add_axes_zenith_range_indicator(
                fig=fig,
                span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
                zenith_bin_edges_rad=zenith_bin["edges"],
                zenith_bin=zd,
                fontsize=6,
            )

            ax.plot(
                energy_fine_bin["edges"][0:-1],
                dRdE_per_s_per_GeV,
                "k-",
            )
            ax.loglog()
            _ymax = np.max(dRdE_per_s_per_GeV)
            _ymax = 10 ** np.ceil(np.log10(_ymax))
            ax.set_ylim([1e-4 * _ymax, _ymax])
            ax.set_xlabel(r"energy$\,/\,$GeV")
            ax.set_ylabel(
                r"$\frac{\mathrm{d\,R}}{\mathrm{d\,E}}$ / s$^{-1}$ (GeV)$^{-1}$"
            )
            fig.savefig(opj(zk_ok_pk_dir, "dRdE_per_s_per_GeV.jpg"))
            sebplt.close(fig)

            R_per_s = 0.0
            for ebin in range(energy_fine_bin["num"]):
                R_per_s += (
                    dRdE_per_s_per_GeV[ebin] * energy_fine_bin["widths"][ebin]
                )
            array_to_txt(arr=[R_per_s], path=opj(zk_ok_pk_dir, "R_per_s.txt"))

res.stop()
