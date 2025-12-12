#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import binning_utils
import flux_sensitivity
import spectral_energy_distribution_units as sed
from plenoirf.analysis import spectral_energy_distribution as sed_styles
import cosmic_fluxes
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils
import scipy.spatial

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

fermi = irf.other_instruments.fermi_lat
cta = irf.other_instruments.cherenkov_telescope_array_south
portal = irf.other_instruments.portal


ONREGION_TYPES = res.analysis["on_off_measuremnent"]["onregion_types"]


# GAMMA RAY BURSTS
# -----------------------
#
# GRB 190829A
# GRB 090902B
# GRB 190114C
# GRB 130427A
# GRB 221009A


def estimate_max_photon_rate_vs_observation_time(
    grb_light_curve,
    energy_start_GeV,
    energy_stop_GeV,
):
    assert energy_start_GeV > 0
    assert energy_stop_GeV > 0
    assert energy_stop_GeV > energy_start_GeV

    _time_s = grb_light_curve["time_since_T0_s"]
    _energy_GeV = grb_light_curve["energy_GeV"]
    mask = np.logical_and(
        _energy_GeV >= energy_start_GeV,
        _energy_GeV <= energy_stop_GeV,
    )
    time_s = _time_s[mask]

    return irf.analysis.light_curve.estimate_max_rate_vs_observation_time(
        t=time_s
    )


def make_x_limits_ticks_mayor_ticks_minor(
    observation_time_start_decade,
    observation_time_stop_decade,
):
    x_start_decade = observation_time_start_decade
    x_stop_decade = observation_time_stop_decade

    xlim_s = np.array([10.0**x_start_decade, 10.0**x_stop_decade])
    xticks_s = 10.0 ** np.arange(x_start_decade, x_stop_decade + 1)
    _xticks_minor_s = [
        [s * 10.0**i for s in range(1, 10)]
        for i in range(x_start_decade, x_stop_decade)
    ]
    xticks_minor_s = []
    for _ixtm in _xticks_minor_s:
        xticks_minor_s += _ixtm

    return xlim_s, xticks_s, xticks_minor_s


def load_fermi_lat_dFdE_vs_t(energy_range):
    fls = (
        irf.other_instruments.fermi_lat.flux_sensitivity_vs_observation_time_vs_energy()
    )
    assert fls["dnde"]["unit"] == "cm-2 MeV-1 ph s-1"
    odnde = {"dnde": {}}
    odnde["dnde"]["value"] = fls["dnde"]["value"] * 1e4 * 1e3
    odnde["dnde"]["unit"] = "m-2 GeV ph s-1"
    odnde["energy_bin_edges"] = {}
    odnde["energy_bin_edges"]["value"] = (
        fls["energy_bin_edges"]["value"] * 1e-3
    )
    odnde["energy_bin_edges"]["unit"] = "GeV"
    odnde["observation_times"] = fls["observation_times"]
    lo_ebin = binning_utils.find_bin_with_start_stop_in_edges(
        bin_edges=odnde["energy_bin_edges"]["value"],
        start=energy_range["start_GeV"],
        stop=energy_range["stop_GeV"],
    )

    out = {}
    out["differential_flux_per_m2_per_s_per_GeV"] = odnde["dnde"]["value"][
        :, lo_ebin
    ]
    out["observation_times_s"] = odnde["observation_times"]["value"]
    return out


def load_portal_dFdE_vs_t(energy_range, zk, ok, dk, sysuncix):
    energy_bin = res.energy_binning(key="5_bins_per_decade")
    enidx = binning_utils.find_bin_with_start_stop_in_edges(
        bin_edges=energy_bin["edges"],
        start=energy_range["start_GeV"],
        stop=energy_range["stop_GeV"],
    )
    dS = json_utils.tree.Tree(
        opj(res.paths["analysis_dir"], "0540_diffsens_estimate")
    )
    dFdE = dS[zk][ok][dk]["differential_flux"][:, :, sysuncix]

    out = {}
    out["differential_flux_per_m2_per_s_per_GeV"] = dFdE[enidx, :]
    out["observation_times_s"] = dS[zk][ok][dk]["observation_times"]
    return out


def load_crab_nebula_dFdE_vs_t(
    energy_range,
    observation_time_limits,
):
    crab_nebula = cosmic_fluxes.read_crab_nebula_flux_from_resources()
    dFdE = np.interp(
        x=energy_range["pivot_GeV"],
        xp=crab_nebula["energy"]["values"],
        fp=crab_nebula["differential_flux"]["values"],
    )
    out = {}
    out["observation_times_s"] = np.array(observation_time_limits)
    out["differential_flux_per_m2_per_s_per_GeV"] = dFdE * np.ones(
        len(out["observation_times_s"])
    )
    return out


def load_grb_F_vs_t(grb_key, energy_range, min_observation_time_s):
    assert min_observation_time_s > 0

    grb_light_curve = irf.other_instruments.fermi_lat.gamma_ray_burst_light_curve_1GeV_regime(
        grb_key=grb_key
    )

    (grb_max_rate_per_s, grb_observation_time_s) = (
        estimate_max_photon_rate_vs_observation_time(
            grb_light_curve=grb_light_curve,
            energy_start_GeV=energy_range["start_GeV"],
            energy_stop_GeV=energy_range["stop_GeV"],
        )
    )

    ttt = grb_observation_time_s.tolist()
    rrr = grb_max_rate_per_s.tolist()

    ttt += [min_observation_time_s, min_observation_time_s]
    rrr += [min(rrr), max(rrr)]

    points = np.array([ttt, rrr]).T

    hull = scipy.spatial.ConvexHull(points)
    outer_points = hull.points[hull.vertices]
    return {
        "hull": {
            "observation_times_s": outer_points[:, 0],
            "flux_per_m2_per_s": outer_points[:, 1],
        },
        "edge": {
            "observation_times_s": grb_observation_time_s,
            "flux_per_m2_per_s": grb_max_rate_per_s,
        },
    }


def add_plot_component_crab_nebula_reference_flux(
    plot_components, crab_nebula_dFdE_vs_t
):
    for i in range(4):
        scale_factor = np.power(10.0, (-1) * i)
        com = {}
        com["observation_time"] = crab_nebula_dFdE_vs_t["observation_times_s"]
        com["differential_flux"] = (
            scale_factor
            * crab_nebula_dFdE_vs_t["differential_flux_per_m2_per_s_per_GeV"]
        )
        com["label"] = "{:1.1e} Crab".format(scale_factor) if i == 0 else None
        com["color"] = "black"
        com["alpha"] = 0.25 / (1.0 + i)
        com["linestyle"] = "--"
        plot_components.append(com.copy())


def add_plot_component_fermi_lat_from_funk2013comparison(
    plot_components,
    energy_range,
):
    if energy_range["start_GeV"] <= 25.0 < energy_range["stop_GeV"]:
        _fermi_lat = fermi.sensitivity_vs_observation_time(energy_GeV=25.0)
        com = {}
        com["observation_time"] = _fermi_lat["observation_time"]["values"]
        com["differential_flux"] = _fermi_lat["differential_flux"]["values"]
        com["label"] = fermi.LABEL + "-funk2013comparison"
        com["color"] = fermi.COLOR
        com["alpha"] = 1.0
        com["linestyle"] = "--"
        plot_components.append(com)


def add_plot_component_fermi_lat(
    plot_components,
    fermi_lat_dFdE_vs_t,
):
    com = {}
    com["observation_time"] = fermi_lat_dFdE_vs_t["observation_times_s"]
    com["differential_flux"] = fermi_lat_dFdE_vs_t[
        "differential_flux_per_m2_per_s_per_GeV"
    ]
    com["label"] = fermi.LABEL + "-mueller"
    com["color"] = fermi.COLOR
    com["alpha"] = 1.0
    com["linestyle"] = "-"
    plot_components.append(com)


def add_plot_component_cta_south_from_funk2013comparison(
    plot_components,
    energy_range,
):
    if energy_range["start_GeV"] <= 25.0 < energy_range["stop_GeV"]:
        _cta_south = irf.other_instruments.cherenkov_telescope_array_south.sensitivity_vs_observation_time(
            energy_GeV=25.0
        )
        com = {}
        com["observation_time"] = _cta_south["observation_time"]["values"]
        com["differential_flux"] = _cta_south["differential_flux"]["values"]
        com["label"] = cta.LABEL + "-funk2013comparison"
        com["color"] = cta.COLOR
        com["alpha"] = 1.0
        com["linestyle"] = "--"
        plot_components.append(com)


def powerquantile(x, q=1.5):
    return binning_utils.power.space(min(x), max(x), size=3, power_slope=-q)[1]


# load
# ----
zenith_bin = res.zenith_binning("3_bins_per_45deg")
ZENITH_ZD_ZK = [(zd, f"zd{zd:d}") for zd in range(zenith_bin["num"])]


diff_sens_scenario = res.analysis["differential_sensitivity"][
    "gamma_ray_effective_area_scenario"
]

energy_ranges = {
    "portal": {
        "start_GeV": binning_utils.power10.lower_bin_edge(
            decade=0, bin=2, num_bins_per_decade=5
        ),
        "stop_GeV": binning_utils.power10.lower_bin_edge(
            decade=0, bin=3, num_bins_per_decade=5
        ),
    },
    "cta": {
        "start_GeV": binning_utils.power10.lower_bin_edge(
            decade=1, bin=1, num_bins_per_decade=5
        ),
        "stop_GeV": binning_utils.power10.lower_bin_edge(
            decade=1, bin=2, num_bins_per_decade=5
        ),
    },
}
for energy_range_key in energy_ranges:
    energy_ranges[energy_range_key]["pivot_GeV"] = np.geomspace(
        energy_ranges[energy_range_key]["start_GeV"],
        energy_ranges[energy_range_key]["stop_GeV"],
        3,
    )[1]
    energy_ranges[energy_range_key]["width_GeV"] = (
        energy_ranges[energy_range_key]["stop_GeV"]
        - energy_ranges[energy_range_key]["start_GeV"]
    )

portal_systematic_uncertainties = res.analysis["on_off_measuremnent"][
    "systematic_uncertainties"
]

xlim_s, xticks_s, xticks_minor_s = make_x_limits_ticks_mayor_ticks_minor(
    observation_time_start_decade=-3,
    observation_time_stop_decade=7,
)

y_lim_per_m2_per_s_per_GeV = np.array([1e-6, 1e0])

plot_funk2013comparison = True

for energy_range_key in energy_ranges:
    energy_range = energy_ranges[energy_range_key]

    fermi_lat_dFdE_vs_t = load_fermi_lat_dFdE_vs_t(energy_range=energy_range)
    crab_nebula_dFdE_vs_t = load_crab_nebula_dFdE_vs_t(
        energy_range=energy_range,
        observation_time_limits=xlim_s,
    )

    for zd, zk in ZENITH_ZD_ZK:
        for ok in ONREGION_TYPES:
            for dk in flux_sensitivity.differential.SCENARIOS:
                os.makedirs(
                    opj(res.paths["out_dir"], zk, ok, dk), exist_ok=True
                )

                print(energy_range_key, zk, ok, dk)

                plot_components = []

                add_plot_component_crab_nebula_reference_flux(
                    plot_components=plot_components,
                    crab_nebula_dFdE_vs_t=crab_nebula_dFdE_vs_t,
                )

                add_plot_component_fermi_lat(
                    plot_components=plot_components,
                    fermi_lat_dFdE_vs_t=fermi_lat_dFdE_vs_t,
                )

                if plot_funk2013comparison:
                    add_plot_component_fermi_lat_from_funk2013comparison(
                        plot_components=plot_components,
                        energy_range=energy_range,
                    )

                    add_plot_component_cta_south_from_funk2013comparison(
                        plot_components=plot_components,
                        energy_range=energy_range,
                    )

                # Portal Cherenkov plenoscope
                # ---------------------------
                for sysuncix in range(len(portal_systematic_uncertainties)):
                    portal_dFdE_vs_t = load_portal_dFdE_vs_t(
                        energy_range=energy_range,
                        zk=zk,
                        ok=ok,
                        dk=dk,
                        sysuncix=sysuncix,
                    )

                    if sysuncix == 0:
                        _alpha = 0.5
                        _linestyle = ":"
                    else:
                        _alpha = 1.0
                        _linestyle = "-"

                    com = {}
                    com["observation_time"] = portal_dFdE_vs_t[
                        "observation_times_s"
                    ]
                    com["differential_flux"] = portal_dFdE_vs_t[
                        "differential_flux_per_m2_per_s_per_GeV"
                    ]
                    com["label"] = (
                        f"{portal.LABEL:s} sys.: {portal_systematic_uncertainties[sysuncix]:.1e}"
                    )
                    com["color"] = portal.COLOR
                    com["alpha"] = _alpha
                    com["linestyle"] = _linestyle
                    plot_components.append(com)

                # figure
                # ------

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

                for com in plot_components:
                    ax.plot(
                        com["observation_time"],
                        com["differential_flux"],
                        label=com["label"],
                        color=com["color"],
                        alpha=com["alpha"],
                        linestyle=com["linestyle"],
                    )

                if energy_range["stop_GeV"] < 10.0:
                    # GRB light curve max photon rate
                    grb_key = "GRB090902B"
                    grb_color = "black"
                    grb_F_vs_t = load_grb_F_vs_t(
                        grb_key=grb_key,
                        energy_range=energy_range,
                        min_observation_time_s=xlim_s[0],
                    )
                    ax.fill(
                        grb_F_vs_t["hull"]["observation_times_s"],
                        grb_F_vs_t["hull"]["flux_per_m2_per_s"]
                        / energy_range["width_GeV"],
                        alpha=0.25,
                        color=grb_color,
                    )
                    ax.plot(
                        grb_F_vs_t["edge"]["observation_times_s"],
                        grb_F_vs_t["edge"]["flux_per_m2_per_s"]
                        / energy_range["width_GeV"],
                        alpha=1,
                        color=grb_color,
                    )
                    ax.text(
                        s=grb_key,
                        x=powerquantile(
                            grb_F_vs_t["hull"]["observation_times_s"], q=1.5
                        ),
                        y=powerquantile(
                            grb_F_vs_t["hull"]["flux_per_m2_per_s"]
                            / energy_range["width_GeV"],
                            q=3.5,
                        ),
                    )

                ax.set_xlim(xlim_s)
                ax.set_ylim(y_lim_per_m2_per_s_per_GeV)
                ax.loglog()
                ax.set_xticks(xticks_s)
                ax.set_xticks(xticks_minor_s, minor=True)
                # ax.legend(loc="best", fontsize=10)
                ax.set_xlabel("observation time / s")
                ax.set_ylabel(
                    sed_styles.PLENOIRF_SED_STYLE["y_label"]
                    + " / "
                    + sed_styles.PLENOIRF_SED_STYLE["y_unit"]
                )

                fig.savefig(
                    opj(
                        res.paths["out_dir"],
                        zk,
                        ok,
                        dk,
                        "differential_flux_sensitivity_vs_obseravtion_time_{:d}MeV.jpg".format(
                            int(energy_range["pivot_GeV"] * 1e3)
                        ),
                    )
                )
                sebplt.close(fig)

res.stop()
