#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import lima1983analysis
import cosmic_fluxes
import json_utils
import binning_utils
import propagate_uncertainties
import copy
import lima1983analysis

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]
COSMIC_RAYS = copy.deepcopy(PARTICLES)
COSMIC_RAYS.pop("gamma")

onregion_rates = json_utils.tree.read(
    os.path.join(
        pa["summary_dir"], "0320_onregion_trigger_rates_for_cosmic_rays"
    )
)
onregion_acceptance = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

energy_binning = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]
energy_fine_bin = energy_binning["interpolation"]


prng = np.random.Generator(
    np.random.generator.PCG64(sum_config["random_seed"])
)

pulsar_name = "J0614-3329"
pulsar = irf.analysis.pulsar_timing.ppog_init_from_profiles(
    energy_bin_edges=energy_fine_bin["edges"],
    pulsar_name=pulsar_name,
)


fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.plot(
    pulsar["energy_bin_edges"][0:-1],
    pulsar["differential_flux_vs_energy"],
    "k-",
)
ax.loglog()
_ymax = np.max(pulsar["differential_flux_vs_energy"])
ax.set_ylim([1e-6 * _ymax, _ymax])
ax.set_xlabel("energy / GeV")
ax.set_ylabel(
    r"$\frac{\mathrm{d\,flux}}{\mathrm{d\,energy}}$ / m$^{-2}$ s$^{-1}$ (GeV)$^{-1}$"
)
fig.savefig(
    os.path.join(pa["out_dir"], "pulsar_{:s}_flux.jpg".format(pulsar_name))
)
seb.close(fig)


fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.plot(
    binning_utils.centers(bin_edges=pulsar["phase_bin_edges"]) / (2 * np.pi),
    pulsar["relative_amplitude_vs_phase"],
    color="k",
    linestyle="-",
)
ax.set_xlim([0, 1])
ax.set_xlabel(r"phase / 2$\pi$")
ax.set_ylabel(r"relative / 1")
fig.savefig(
    os.path.join(
        pa["out_dir"], "pulsar_{:s}_phaseogram.jpg".format(pulsar_name)
    )
)
seb.close(fig)


fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.plot(
    pulsar["relative_amplitude_vs_phase_cdf"],
    pulsar["phase_bin_edges"] / (2 * np.pi),
    color="k",
    linestyle="-",
)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel(r"cummulative distribution function / 1")
ax.set_ylabel(r"phase / 2$\pi$")
fig.savefig(
    os.path.join(
        pa["out_dir"],
        "pulsar_{:s}_phaseogram_cummulative_distribution_function.jpg".format(
            pulsar_name
        ),
    )
)
seb.close(fig)


TEST_DRAW_RANDOM_PHASE = False
if TEST_DRAW_RANDOM_PHASE:
    arrival_phases = irf.analysis.pulsar_timing.ppog_draw_phase(
        pulsar, prng, 100 * 1000
    )

    arrival_phases_counts = np.histogram(
        arrival_phases,
        bins=pulsar["phase_bin_edges"],
    )[0]
    arrival_phases_counts = arrival_phases_counts / np.sum(
        arrival_phases_counts
    )

    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    seb.ax_add_histogram(
        ax=ax,
        bin_edges=pulsar["phase_bin_edges"] / (2 * np.pi),
        bincounts=pulsar["relative_amplitude_vs_phase"],
        linestyle="-",
        linecolor="k",
        draw_bin_walls=True,
    )
    seb.ax_add_histogram(
        ax=ax,
        bin_edges=pulsar["phase_bin_edges"] / (2 * np.pi),
        bincounts=arrival_phases_counts,
        linestyle=":",
        linecolor="r",
        draw_bin_walls=True,
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel(r"phase / 2$\pi$")
    ax.set_ylabel(r"relative / 1")
    fig.savefig(
        os.path.join(
            pa["out_dir"],
            "pulsar_{:s}_phaseogram_test_draw.jpg".format(pulsar_name),
        )
    )
    seb.close(fig)


run_observation_time_s = 3600


def expose_to_background(
    observation_time_s,
    background_rate_per_s,
    prng,
):
    background_intensity = observation_time_s * background_rate_per_s
    background_intensity = int(np.round(background_intensity))
    assert background_intensity > 1e3, "intensity is low -> rounding error."
    background_arrival_phases = prng.uniform(
        low=0.0,
        high=(2 * np.pi),
        size=background_intensity,
    )
    return background_arrival_phases


def draw_time_to_detect_next_gamma_ray(gamma_rate_per_s, prng, num):
    return -np.log(prng.uniform(size=num)) / gamma_rate_per_s


def expose_to_pulsar(
    observation_time_s,
    gamma_rate_per_s,
    pulsar,
    prng,
):
    tt = 0.0
    intensity = 0
    while tt < observation_time_s:
        tt += draw_time_to_detect_next_gamma_ray(
            gamma_rate_per_s=gamma_rate_per_s, prng=prng, num=1
        )
        intensity += 1

    gamma_ray_phases = irf.analysis.pulsar_timing.ppog_draw_phase(
        ppog=pulsar,
        prng=prng,
        num=intensity,
    )

    return gamma_ray_phases


def histogram_runs(runs, num_phase_bins=None, a=1e-2):
    if num_phase_bins == None:
        num_events = 0
        for run in runs:
            num_events += len(run["gamma_phases"])
            num_events += len(run["background_phases"])
        num_phase_bins = int(np.round(a * np.sqrt(num_events)))

    assert num_phase_bins >= 1
    phase_bin = binning_utils.Binning(
        np.linspace(0, 2 * np.pi, num_phase_bins + 1)
    )

    exposure_time = 0.0
    gam_hist = np.zeros(num_phase_bins, dtype=np.int)
    bkg_hist = np.zeros(num_phase_bins, dtype=np.int)

    for run in runs:
        exposure_time += run["exposure_time"]
        gam_hist += np.histogram(run["gamma_phases"], bins=phase_bin["edges"])[
            0
        ]
        bkg_hist += np.histogram(
            run["background_phases"], bins=phase_bin["edges"]
        )[0]

    out = {}
    out["exposure_time"] = exposure_time
    out["phaseogram"] = {}
    out["phaseogram"]["bin"] = phase_bin
    out["phaseogram"]["intensity"] = {}
    out["phaseogram"]["intensity"]["unit"] = "1"
    out["phaseogram"]["intensity"] = {}
    out["phaseogram"]["intensity"]["gamma"] = gam_hist
    out["phaseogram"]["intensity"]["gamma_au"] = np.sqrt(gam_hist)

    out["phaseogram"]["intensity"]["background"] = bkg_hist
    out["phaseogram"]["intensity"]["background_au"] = np.sqrt(bkg_hist)

    out["phaseogram"]["intensity"]["total"] = gam_hist + bkg_hist
    out["phaseogram"]["intensity"]["total_au"] = np.sqrt(
        out["phaseogram"]["intensity"]["total"]
    )

    i = out["phaseogram"]["intensity"]

    tt = out["exposure_time"] / phase_bin["num"]

    out["phaseogram"]["rate"] = {}
    out["phaseogram"]["rate"]["unit"] = "s$^{-1}$"

    out["phaseogram"]["rate"]["gamma"] = i["gamma"] / tt
    out["phaseogram"]["rate"]["gamma_au"] = i["gamma_au"] / tt

    out["phaseogram"]["rate"]["background"] = i["background"] / tt
    out["phaseogram"]["rate"]["background_au"] = i["background_au"] / tt

    out["phaseogram"]["rate"]["total"] = i["total"] / tt
    out["phaseogram"]["rate"]["total_au"] = i["total_au"] / tt
    return out


onregion_alpha = {
    "small": 1 / 9,
    "medium": 1 / 3,
    "large": 1 / 1,
}

PULSARS = cosmic_fluxes.pulsars.list_pulsar_names()


for sk in SITES:
    sk_dir = os.path.join(pa["out_dir"], sk)
    for ok in ONREGION_TYPES:
        sk_ok_dir = os.path.join(sk_dir, ok)
        os.makedirs(sk_ok_dir, exist_ok=True)

        # background
        # ----------
        rates_cosmic_rays_per_s = []
        rates_cosmic_rays_per_s_au = []
        rates_of_cosmic_rays = {}
        for pk in COSMIC_RAYS:
            rate = onregion_rates[sk][ok][pk]["integral_rate"]["mean"]
            rate_au = onregion_rates[sk][ok][pk]["integral_rate"][
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

        json_utils.write(
            os.path.join(sk_ok_dir, "rates_of_cosmic_rays.json"),
            rates_of_cosmic_rays,
            indent=4,
        )

        # signal
        # ------
        A_gamma = onregion_acceptance[sk][ok]["gamma"]["point"]["mean"]
        A_gamma_fine_m2 = np.interp(
            x=energy_fine_bin["centers"],
            xp=energy_bin["centers"],
            fp=A_gamma,
        )

        dKdE_per_m2_per_s_per_GeV = pulsar["differential_flux_vs_energy"]

        dRdE_per_s_per_GeV = np.zeros(energy_fine_bin["num_bins"])
        for ebin in range(energy_fine_bin["num_bins"]):
            dRdE_per_s_per_GeV[ebin] = (
                A_gamma_fine_m2[ebin] * dKdE_per_m2_per_s_per_GeV[ebin]
            )

        fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
        ax.plot(
            energy_fine_bin["centers"],
            dRdE_per_s_per_GeV,
            color="k",
            linestyle="-",
        )
        dRdE_max = np.max(dRdE_per_s_per_GeV)
        ax.set_ylim([1e-4 * dRdE_max, 2 * dRdE_max])
        ax.loglog()
        ax.set_xlabel("energy / GeV")
        ax.set_ylabel(
            r"$\frac{\mathrm{d\,rate}}{\mathrm{d\,energy}}$ / s$^{-1}$ (GeV)$^{-1}$"
        )
        fig.savefig(
            os.path.join(
                sk_ok_dir,
                "{:s}_onregion-{:s}_differential_rate_in_onregion.jpg".format(
                    sk, ok
                ),
            )
        )
        seb.close(fig)

        rate_gamma_rays_per_s = 0.0
        for ebin in range(energy_fine_bin["num_bins"]):
            rate_gamma_rays_per_s += (
                dRdE_per_s_per_GeV[ebin] * energy_fine_bin["widths"][ebin]
            )

        json_utils.write(
            os.path.join(sk_ok_dir, "rate_of_gamma_rays.json"),
            {"gamma": rate_gamma_rays_per_s},
            indent=4,
        )

        rate_min = rate_cosmic_rays_per_s - 2 * rate_gamma_rays_per_s
        rate_max = rate_cosmic_rays_per_s + 12 * rate_gamma_rays_per_s

        runs = []
        while len(runs) < 128:
            print("runs: ", len(runs))
            run = {}
            run["gamma_phases"] = expose_to_pulsar(
                observation_time_s=run_observation_time_s,
                gamma_rate_per_s=rate_gamma_rays_per_s,
                pulsar=pulsar,
                prng=prng,
            )

            run["background_phases"] = expose_to_background(
                observation_time_s=run_observation_time_s,
                background_rate_per_s=rate_cosmic_rays_per_s,
                prng=prng,
            )
            run["exposure_time"] = run_observation_time_s
            runs.append(run)

            if np.mod(np.log(len(runs)) / np.log(2), 1) <= 1e-6:
                hist = histogram_runs(runs, a=1e-2)

                fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
                ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
                seb.ax_add_histogram(
                    ax=ax,
                    bin_edges=hist["phaseogram"]["bin"]["edges"] / (2 * np.pi),
                    bincounts=hist["phaseogram"]["rate"]["total"],
                    bincounts_lower=hist["phaseogram"]["rate"]["total"]
                    - hist["phaseogram"]["rate"]["total_au"],
                    bincounts_upper=hist["phaseogram"]["rate"]["total"]
                    + hist["phaseogram"]["rate"]["total_au"],
                    linestyle="-",
                    linecolor="k",
                    face_color="k",
                    face_alpha=0.2,
                    draw_bin_walls=False,
                )
                ax.set_ylim([rate_min, rate_max])
                ax.set_xlim([0, 1])
                ax.set_xlabel(r"phase / 2$\pi$")
                ax.set_ylabel(r"rate / s$^{-1}$")
                fig.savefig(
                    os.path.join(
                        sk_ok_dir,
                        "{:s}_onregion-{:s}_phaseogram_exposure-time-{:03d}h.jpg".format(
                            sk, ok, len(runs)
                        ),
                    )
                )
                seb.close(fig)

                # significance
                # ------------
                num_phase_bins = hist["phaseogram"]["bin"]["num"]
                sign = np.zeros(num_phase_bins)

                alpha = onregion_alpha[ok]
                B_on = np.zeros(num_phase_bins)
                B_off = np.zeros(num_phase_bins)
                C = rate_cosmic_rays_per_s * hist["exposure_time"]
                C_std = np.sqrt(C)

                for b in range(num_phase_bins):
                    B_on[b] = int(
                        np.round(prng.normal(loc=C, scale=np.sqrt(C_std)))
                    )
                    B_off[b] = int(
                        np.round(
                            prng.normal(
                                loc=C / alpha, scale=np.sqrt(C_std / alpha)
                            )
                        )
                    )

                for b in range(num_phase_bins):
                    S = hist["phaseogram"]["intensity"]["gamma"][b]
                    B = hist["phaseogram"]["intensity"]["background"][b]
                    try:
                        sign[b] = lima1983analysis.estimate_S_eq17(
                            N_on=S + B_on[b], N_off=B_off[b], alpha=alpha
                        )
                    except AssertionError as err:
                        print(err)
                        sign[b] = float("nan")

                fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
                ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
                seb.ax_add_histogram(
                    ax=ax,
                    bin_edges=hist["phaseogram"]["bin"]["edges"] / (2 * np.pi),
                    bincounts=sign,
                    linestyle="-",
                    linecolor="k",
                    draw_bin_walls=True,
                )
                ax.set_ylim([-0.1, 5.1])
                ax.set_xlim([0, 1])
                ax.set_xlabel(r"phase / 2$\pi$")
                ax.set_ylabel("significance / 1\n(Li and Ma, 1983, Eq.17)")
                fig.savefig(
                    os.path.join(
                        sk_ok_dir,
                        "{:s}_onregion-{:s}_phaseogram_exposure-time-{:03d}h_significance_lima1983.jpg".format(
                            sk, ok, len(runs)
                        ),
                    )
                )
                seb.close(fig)

        print(
            "Site: ",
            sk,
            ", Size of on-region: ",
            ok,
            ", Rate cosmic-rays: ",
            rate_cosmic_rays_per_s,
            "+-",
            rate_cosmic_rays_per_s_au,
            "s^{-1}",
            ", Area gamma-rays at 1GeV ",
            A_gamma[2],
            "m^{2}",
            ", pulsar_name ",
            pulsar_name,
        )
