#!/usr/bin/python
import sys
import plenoirf as irf
import os
from os.path import join as opj
import json_utils
import cosmic_fluxes
import sebastians_matplotlib_addons as sebplt
import photon_spectra

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt)


def get_function_by_name(funtions, name):
    for f in funtions:
        if f["name"] == name:
            return f
    return None


nsb = res.config["merlict_plenoscope_propagator_config"][
    "night_sky_background_ligth"
]
pec = res.config["merlict_plenoscope_propagator_config"][
    "photo_electric_converter"
]
mir = get_function_by_name(
    funtions=res.instrument["scenery"]["functions"],
    name="mirror_reflectivity_vs_wavelength",
)

sun_light = photon_spectra.sunlight_at_sea_level.init()
assert sun_light["units"][0] == "m"
assert sun_light["units"][1] == "m^{-2} s^{-1} m^{-1}"

sfig, sax = irf.summary.figure.style("4:3")
fig = sebplt.figure(sfig)
ax = sebplt.add_axes(fig=fig, span=sax)
ax.plot(
    sun_light["wavelength"],
    sun_light["value"],
    color="peru",
)
ax.set_xlabel("wavelength / m")
ax.set_ylabel(r"differential photon flux / s$^{-1}$ m$^{-2}$ m$^{-1}$")
ax.semilogy()
ax.set_xlim([200e-9, 800e-9])
fig.savefig(opj(res.paths["out_dir"], "sun_light_at_sea_level.jpg"))
sebplt.close(fig)

# Check powerdensity is about 800 W m^{-2} as sea level
h_planck_J_per_Hz = 6.626e-34
c0_m_per_s = 299_792_458

power_density_at_sea_level_W_per_m2 = 0.0
wavelengths_m = np.linspace(
    sun_light["wavelength"].min(),
    sun_light["wavelength"].max(),
    13337,
)
for l in range(len(wavelengths_m) - 1):
    photon_wavelength_m = wavelengths_m[l]
    delta_wavelength_m = wavelengths_m[l + 1] - wavelengths_m[l]
    photon_energy_J = (h_planck_J_per_Hz * c0_m_per_s) / photon_wavelength_m
    diff_flux_per_s_per_m2_per_m = np.interp(
        x=photon_wavelength_m,
        xp=sun_light["wavelength"],
        fp=sun_light["value"],
    )
    flux_per_s_per_m2 = diff_flux_per_s_per_m2_per_m * delta_wavelength_m
    delta_power_density_W_per_m2 = flux_per_s_per_m2 * photon_energy_J
    power_density_at_sea_level_W_per_m2 += delta_power_density_W_per_m2

assert 700 < power_density_at_sea_level_W_per_m2 < 1000

# STAR

MAGNITUDE_SUN = -27.0
MAGNITUDE_STAR = +2.0  # brightest star in field of view (on average)
factor_sun_over_star_brightness = irf.utils.astronomic_magnitude_to_brightness(
    MAGNITUDE_SUN
) / irf.utils.astronomic_magnitude_to_brightness(MAGNITUDE_STAR)


sfig, sax = irf.summary.figure.style("4:3")
fig = sebplt.figure(sfig)
ax = sebplt.add_axes(fig=fig, span=sax)
ax.plot(
    sun_light["wavelength"],
    sun_light["value"] / factor_sun_over_star_brightness,
    color="peru",
)
ax.set_xlabel("wavelength / m")
ax.set_ylabel(r"differential photon flux / s$^{-1}$ m$^{-2}$ m$^{-1}$")
ax.semilogy()
ax.set_xlim([200e-9, 800e-9])
fig.savefig(opj(res.paths["out_dir"], "star_at_sea_level.jpg"))
sebplt.close(fig)


# expected rate in Portal light field beam
PORTAL_MIRROR_AREA_M2 = 4_000
PORTAL_NUM_PAXEL = 61
avg_aperture_area_of_portal_beam_m2 = PORTAL_MIRROR_AREA_M2 / PORTAL_NUM_PAXEL
fresnel_transmission_entering_lenses = 0.9

common_wavelength_m = np.linspace(200e-9, 800e-9, 13337)
common_mirror_reflectivity_1 = (
    photon_spectra.utils.make_values_for_common_wavelength(
        wavelength=mir["argument_versus_value"][:, 0],
        value=mir["argument_versus_value"][:, 1],
        common_wavelength=common_wavelength_m,
    )
)
common_pde_1 = photon_spectra.utils.make_values_for_common_wavelength(
    wavelength=pec["quantum_efficiency_vs_wavelength"][:, 0],
    value=pec["quantum_efficiency_vs_wavelength"][:, 1],
    common_wavelength=common_wavelength_m,
)
common_diff_sun_flux_per_s_per_m2_per_m = (
    photon_spectra.utils.make_values_for_common_wavelength(
        wavelength=sun_light["wavelength"],
        value=sun_light["value"],
        common_wavelength=common_wavelength_m,
    )
)
common_diff_star_flux_per_s_per_m2_per_m = (
    common_diff_sun_flux_per_s_per_m2_per_m / factor_sun_over_star_brightness
)

common_star_star_light_field_beam_diff_rate_per_s_per_m = (
    common_mirror_reflectivity_1
    * fresnel_transmission_entering_lenses
    * common_pde_1
    * common_diff_star_flux_per_s_per_m2_per_m
    * avg_aperture_area_of_portal_beam_m2
)


sfig, sax = irf.summary.figure.style("4:3")
fig = sebplt.figure(sfig)
ax = sebplt.add_axes(fig=fig, span=sax)
ax.plot(
    common_wavelength_m,
    common_star_star_light_field_beam_diff_rate_per_s_per_m,
    color="black",
)
ax.set_xlabel("wavelength / m")
ax.set_ylabel(r"differential rate / s$^{-1}$ m$^{-1}$")
ax.semilogy()
ax.set_xlim([200e-9, 800e-9])
fig.savefig(
    opj(
        res.paths["out_dir"],
        "light_field_beam_diff_rate_from_star.jpg",
    )
)
sebplt.close(fig)


# integrate rate in light field beam
light_field_beam_rate_focus_to_infinity_per_s = np.nanmean(
    common_star_star_light_field_beam_diff_rate_per_s_per_m
) * (common_wavelength_m.max() - common_wavelength_m.min())


# http://www.stargazing.net/david/constel/howmanystars.html
# Data is based on the Tycho Catalogue which was obtained from page VII of the
# Millennium Star Atlas, Volume I, Sky Publishing Corporation and European Space
# Agency. The Tycho Catalog is believed to be 99.9 percent complete to magnitude
# 10.0 and 90 percent complete to magnitude 10.5. Table data for magnitudes 11
# to 20 are projected on the average increased of 291%. 291 % is the average
# increase of stars between magnitudes 6 to 7, 7 to 8, 8 to 9, and 9 to 10.
_star_population = np.asarray(
    [
        [-1.46, 1],
        [-1, 2],
        [0, 8],
        [1, 22],
        [2, 93],
        [3, 283],
        [4, 893],
        [5, 2822],
        [6, 8768],
        [7, 26533],
        [8, 77627],
        [9, 217689],
        [10, 626883],
        [11, 1823573],
        [12, 5304685],
        [13, 15431076],
        [14, 44888260],
        [15, 130577797],
        [16, 379844556],
        [17, 1104949615],
        [18, 3214245496],
        [19, 9350086162],
        [20, 27198952706],
    ]
)
star_population = {
    "magnitude": _star_population[:, 0],
    "num_stars": _star_population[:, 1],
}

sfig, sax = irf.summary.figure.style("4:3")
fig = sebplt.figure(sfig)
ax = sebplt.add_axes(fig=fig, span=sax)
ax.plot(
    star_population["magnitude"],
    star_population["num_stars"],
    color="black",
)
ax.set_xlabel("magnitude of star")
ax.set_ylabel("num. stars")
ax.semilogy()
ax.set_xlim(-2, 11)
fig.savefig(
    opj(
        res.paths["out_dir"],
        "star_population.jpg",
    )
)
sebplt.close(fig)

res.stop()
