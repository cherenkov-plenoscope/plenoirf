#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import pylatex as ltx
import warnings
import json_utils
import io


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

latex_geometry_options = {
    "paper": "a4paper",
    # "paperwidth": "18cm",
    # "paperheight": "32cm",
    "head": "0cm",
    "left": "2cm",
    "right": "2cm",
    "top": "0cm",
    "bottom": "2cm",
    "includehead": True,
    "includefoot": True,
}


PARTICLES = res.PARTICLES
COSMIC_RAYS = res.COSMIC_RAYS
SED_STYLE_KEY = "portal"
OUTER_ARRAY_KEY = "ring-mst"

ok = ["small", "medium", "large"][0]
dk = "bell_spectrum"

energy_bin = res.energy_binning(key="point_spread_function")
zenith_bin = res.zenith_binning(key="once")


def noesc(text):
    return ltx.utils.NoEscape(text)


def ppath(*args):
    p1 = opj(*args)
    p2 = os.path.normpath(p1)
    return os.path.abspath(p2)


def get_total_trigger_rate_at_analysis_threshold(trigger_rates_by_origin):
    tti = trigger_rates_by_origin["analysis_trigger_threshold_idx"]
    total_rate_per_s = {}
    for zd in range(zenith_bin["num"]):
        zk = f"zd{zd:d}"

        total_rate_per_s[zk] = float(
            trigger_rates_by_origin["origins"]["night_sky_background"][tti]
        )
        for pk in trigger_rates_by_origin["origins"][zk]:
            total_rate_per_s[zk] += trigger_rates_by_origin["origins"][zk][pk][
                tti
            ]

    return total_rate_per_s


def verbatim(string):
    return r"\begin{verbatim}" + r"{:s}".format(string) + r"\end{verbatim}"


provenance = res.read_provenance()

total_trigger_rate_per_s = get_total_trigger_rate_at_analysis_threshold(
    trigger_rates_by_origin=json_utils.tree.read(
        ppath(res.paths["analysis_dir"], "0131_trigger_rates_total")
    )["trigger_rates_by_origin"]
)
"""
total_trigger_rate_per_s_ltx = irf.utils.latex_scientific(
    real=total_trigger_rate_per_s, format_template="{:.3e}"
)
"""

doc = ltx.Document(
    default_filepath=opj(
        res.paths["analysis_dir"], f"{res.production_dirname:s}"
    ),
    documentclass="article",
    document_options=[],
    geometry_options=latex_geometry_options,
    font_size="small",
    page_numbers=True,
)

doc.preamble.append(ltx.Package("multicol"))
doc.preamble.append(ltx.Package("lipsum"))
doc.preamble.append(ltx.Package("float"))
doc.preamble.append(ltx.Package("verbatim"))

doc.preamble.append(
    ltx.Command(
        "title",
        noesc(r"Simulating the Cherenkov Plenoscope"),
    )
)
doc.preamble.append(
    ltx.Command("author", "Sebastian A. Mueller and Werner Hofmann")
)
doc.preamble.append(ltx.Command("date", ""))

doc.append(noesc(r"\maketitle"))
doc.append(noesc(r"\begin{multicols}{2}"))

with doc.create(ltx.Section("Version", numbering=False)):
    _basic_version_str = irf.provenance.make_basic_version_str(
        production_dirname=res.production_dirname,
        production_provenance=production_provenance,
        analysis_provenance=analysis_provenance,
    )
    doc.append(noesc(verbatim(_basic_version_str)))

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            opj(
                res.paths["starter_kit_dir"],
                "portal-corporate-identity",
                "images",
                "side_total_from_distance.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            noesc(
                r"Portal, a Cherenkov-Plenoscope "
                r"to observe gamma-rays with energies as low as 1\,GeV."
            )
        )
"""
with doc.create(ltx.Section("Performance", numbering=False)):
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0550_diffsens_plot",
                sk,
                ok,
                SED_STYLE_KEY,
                "{:s}_{:s}_{:s}_differential_sensitivity_sed_style_{:s}_0180s.jpg".format(
                    sk, ok, dk, SED_STYLE_KEY
                ),
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            noesc(
                r"Differential sensitivity for a point-like source of gamma-rays. "
                r"Fermi-LAT \cite{wood2016fermiperformance} in orange. "
                r"CTA-south in blue based on the public instrument-response \cite{cta2018baseline}. "
            )
        )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0610_sensitivity_vs_observation_time",
                sk,
                ok,
                dk,
                "sensitivity_vs_obseravtion_time_{:d}MeV.jpg".format(2500),
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            noesc(
                r"Sensitivity vs. observation-time at 2.5\,GeV. "
                r"Fermi-LAT in orange, and "
                r"Portal in black (dotted has $1\times{}10^{-3}$ sys.)."
            )
        )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0230_point_spread_function",
                "{:s}_gamma.jpg".format(sk),
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            noesc(
                r"Angular resolution. "
                r"Fermi-LAT \cite{wood2016fermiperformance} in orange, "
                r"CTA-south \cite{cta2018baseline} in blue, and "
                r"Portal in black."
            )
        )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0066_energy_estimate_quality",
                "{:s}_gamma_resolution.jpg".format(sk),
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            noesc(
                r"Energy resolution. "
                r"Fermi-LAT \cite{wood2016fermiperformance} in orange. "
                r"CTA-south \cite{cta2018baseline} in blue. "
            )
        )

    doc.append(
        noesc(
            r"The Crab Nebula's gamma-ray-flux \cite{aleksic2015measurement} "
            r"\mbox{(100\%, 10\%, 1\%, and 0.1\%)} is shown in fading gray dashes. "
        )
    )

# doc.append(noesc(r"\columnbreak"))

with doc.create(ltx.Section("Site", numbering=False)):
    doc.append(sk)
    doc.append(
        noesc(
            verbatim(
                irf.utils.dict_to_pretty_str(
                    irf_config["config"]["sites"][sk]
                )
            )
        )
    )
    doc.append(
        noesc(
            r"Flux of airshowers (not cosmic particles) are estimated "
            r"based on the "
            r"fluxes of cosmic protons \cite{aguilar2015precision}, "
            r"electrons and positrons \cite{aguilar2014precision}, and "
            r"helium \cite{patrignani2017helium}."
        )
    )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0016_flux_of_airshowers_plot",
                sk + "_airshower_differential_flux.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            "Flux of airshowers (not particles) at the site. "
            "This includes airshowers below the geomagnetic-cutoff created by secondary, terrestrial particles."
        )

trgstr = irf.light_field_trigger.make_trigger_modus_str(
    analysis_trigger=sum_config["trigger"][sk],
    production_trigger=irf_config["config"]["sum_trigger"],
)

with doc.create(ltx.Section("Trigger", numbering=False)):
    doc.append(noesc(verbatim(trgstr)))
    doc.append(
        noesc(
            "Trigger-rate during observation is $\\approx{"
            + total_trigger_rate_per_s_ltx
            + r"}\,$s$^{-1}$"
        )
    )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0130_trigger_ratescan_plot",
                sk + "_ratescan.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            "Ratescan. For low thresholds the rates seem "
            "to saturate. This is because of limited statistics. "
            "The rates are expected to raise further."
        )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0071_trigger_probability_vs_cherenkov_size_plot",
                sk + "_trigger_probability_vs_cherenkov_size.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            "Trigger-probability vs. true Cherenkov-size in photo-sensors."
        )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0075_trigger_probability_vs_cherenkov_density_on_ground_plot",
                "{:s}_passing_trigger.jpg".format(sk),
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            "Trigger-probability vs. Cherenkov-density on ground."
        )

with doc.create(ltx.Section("Acceptance at Trigger", numbering=False)):
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0101_trigger_acceptance_for_cosmic_particles_plot",
                sk + "_diffuse.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption("Effective acceptance for a diffuse source.")
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0101_trigger_acceptance_for_cosmic_particles_plot",
                sk + "_point.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption("Effective area for a point like source.")

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0106_trigger_rates_for_cosmic_particles_plot",
                sk + "_differential_trigger_rate.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            noesc(
                r"Trigger-rate on gamma-ray-source {:s}".format(
                    sum_config["gamma_ray_reference_source"]["name_3fgl"]
                )
                + r"\cite{acero2015fermi3fglcatalog}. "
                + r"Entire field-of-view."
            )
        )

with doc.create(
    ltx.Section("Cherenkov- and Night-Sky-Light", numbering=False)
):
    doc.append("Finding Cherenkov-photons in the pool of nigth-sky-light.")
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0060_cherenkov_photon_classification_plot",
                sk + "_gamma_confusion.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            "Size-confusion of Cherenkov-photons emitted in airshowers initiated by gamma-rays."
        )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0060_cherenkov_photon_classification_plot",
                sk + "_gamma_sensitivity_vs_true_energy.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            "Classification-power for Cherenkov-photons emitted in airshowers initiated by gamma-rays."
        )

with doc.create(ltx.Section("Energy", numbering=False)):
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0066_energy_estimate_quality",
                sk + "_gamma.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption("Energy-confusion for gamma-rays.")

with doc.create(ltx.Section("Angular resolution", numbering=False)):
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0213_trajectory_benchmarking",
                "{:s}_gamma_psf_image_all.jpg".format(sk),
            ),
            width=noesc(r"1.0\linewidth"),
        )

with doc.create(ltx.Section("Acceptance after all Cuts", numbering=False)):
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0301_onregion_trigger_acceptance_plot",
                "{:s}_{:s}_diffuse.jpg".format(sk, ok),
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption("Effective acceptance for a diffuse source.")

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0301_onregion_trigger_acceptance_plot",
                "{:s}_{:s}_point.jpg".format(sk, ok),
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption("Effective area for a point like source.")

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0325_onregion_trigger_rates_for_cosmic_rays_plot",
                "{:s}_{:s}_differential_event_rates.jpg".format(sk, ok),
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption(
            "Final rates in on-region while observing {:s}".format(
                sum_config["gamma_ray_reference_source"]["name_3fgl"]
            )
        )

with doc.create(ltx.Section("Timing", numbering=False)):
    doc.append(
        "Performace to reconstruct a gamma-ray's time of arrival. "
        "Nothing fancy, just the time of the trigger without applying geometric corrections. "
    )
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0905_reconstructing_gamma_arrival_time",
                "{:s}_gamma_arrival_time_spread.jpg".format(sk),
            ),
            width=noesc(r"1.0\linewidth"),
        )

with doc.create(ltx.Section("Quality", numbering=False)):
    doc.append(
        "The quality of the instrument-response-function. "
        "Is the scatter-angle large enough? "
        "Here scatter-angle is the angle between the particle's direction "
        "and the direction a particle must have to see its "
        "Cherenkov-light in the center of the instrument's field-of-view."
    )

    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0108_trigger_rates_for_cosmic_particles_vs_max_scatter_angle_plot",
                "{:s}_trigger-rate_vs_scatter.jpg".format(sk),
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption("Trigger-rate vs. max. scatter-angle.")
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0108_trigger_rates_for_cosmic_particles_vs_max_scatter_angle_plot",
                "{:s}_diff-trigger-rate_vs_scatter.jpg".format(sk),
            ),
            width=noesc(r"1.0\linewidth"),
        )
        fig.add_caption("Diff. trigger-rate vs. max. scatter-angle.")

    for cosmic_key in COSMIC_RAYS:
        with doc.create(ltx.Figure(position="H")) as fig:
            fig.add_image(
                ppath(
                    paths["analysis_dir"],
                    "0108_trigger_rates_for_cosmic_particles_vs_max_scatter_angle_plot",
                    "{:s}_{:s}_diff-trigger-rate_vs_scatter_vs_energy".format(
                        sk, cosmic_key
                    ),
                ),
                width=noesc(r"1.0\linewidth"),
            )
            fig.add_caption(
                "{:s}. Diff. trigger-rate vs. max. scatter-angle vs. energy.".format(
                    cosmic_key
                )
            )

with doc.create(ltx.Section("Optical performance", numbering=False)):
    doc.append(
        "Statistics of the plenoscope's optical beams. Gray area contains 90%. Dashed, vertical line is median."
    )
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "1005_plot_light_field_geometry",
                "solid_angles_log.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "1005_plot_light_field_geometry",
                "areas_log.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "1005_plot_light_field_geometry",
                "time_spreads_log.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "1005_plot_light_field_geometry",
                "efficiencies_log.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )
    doc.append(
        noesc(
            r"Performance to estimate the depth of a bright, point-like source. Dotted lines mark the $g_+$ and $g_-$ bounds."
        )
    )
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "1210_demonstrate_resolution_of_depth",
                "relative_depth_reco_vs_true.jpg",
            ),
            width=noesc(r"1.0\linewidth"),
        )

with doc.create(
    ltx.Section("Outer array to veto hadrons", numbering=False)
):
    doc.append(
        "Explore an outer array of 'small' telescopes to veto "
        "hadronic showers with large impact distances."
    )
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0820_passing_trigger_of_outer_array_of_small_telescopes",
                "array_configuration_{:s}.jpg".format(OUTER_ARRAY_KEY),
            ),
            width=noesc(r"1.0\linewidth"),
        )
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0820_passing_trigger_of_outer_array_of_small_telescopes",
                "{:s}_{:s}_telescope_trigger_probability.jpg".format(
                    sk, OUTER_ARRAY_KEY
                ),
            ),
            width=noesc(r"1.0\linewidth"),
        )
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0821_passing_trigger_of_outer_array_of_small_telescopes_plot",
                "{:s}_{:s}.jpg".format(sk, OUTER_ARRAY_KEY),
            ),
            width=noesc(r"1.0\linewidth"),
        )

with doc.create(ltx.Section("Compute-time", numbering=False)):
    doc.append(
        "Relative compute-time for protons, which are the bulk. "
        "Small contributions are either quick to compute or do not occur often."
    )
    with doc.create(ltx.Figure(position="H")) as fig:
        fig.add_image(
            ppath(
                paths["analysis_dir"],
                "0910_runtime",
                "{:s}_proton_relative_runtime.jpg".format(sk),
            ),
            width=noesc(r"1.0\linewidth"),
        )

"""
doc.append(noesc(r"\bibliographystyle{apalike}"))
doc.append(
    noesc(
        r"\bibliography{"
        + opj(
            res.paths["starter_kit_dir"], "sebastians_references", "references"
        )
        + "}"
    )
)

doc.append(noesc(r"\end{multicols}{2}"))
doc.generate_pdf(clean_tex=False)

res.stop()
