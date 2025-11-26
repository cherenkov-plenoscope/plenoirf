#!/usr/bin/python
import sys
import plenoirf as irf
import os
from os.path import join as opj
import json_utils
import cosmic_fluxes
import sebastians_matplotlib_addons as sebplt

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

sfig, sax = irf.summary.figure.style("4:3")

fig = sebplt.figure(sfig)
ax = sebplt.add_axes(fig=fig, span=sax)
ax.plot(
    nsb["flux_vs_wavelength"][:, 0],
    nsb["flux_vs_wavelength"][:, 1],
    color="deepskyblue",
)
ax.set_xlabel("wavelength / m")
ax.set_ylabel(
    r"differential photon flux / s$^{-1}$ m$^{-2}$ (sr)$^{-1}$ m$^{-1}$"
)
ax.semilogy()
ax.set_xlim([200e-9, 800e-9])
fig.savefig(opj(res.paths["out_dir"], "night_sky_background.jpg"))
sebplt.close(fig)

pec = res.config["merlict_plenoscope_propagator_config"][
    "photo_electric_converter"
]

sfig, sax = irf.summary.figure.style("21:9")

fig = sebplt.figure(sfig)
ax = sebplt.add_axes(fig=fig, span=sax)
ax.plot(
    pec["quantum_efficiency_vs_wavelength"][:, 0],
    pec["quantum_efficiency_vs_wavelength"][:, 1],
    color="gray",
)
ax.set_xlabel("wavelength / m")
ax.set_ylabel("photo electric\ndetection efficiency / 1")
ax.set_xlim([200e-9, 800e-9])
ax.set_ylim([0, 0.5])
fig.savefig(opj(res.paths["out_dir"], "photo_electric_converter.jpg"))
sebplt.close(fig)


mir = get_function_by_name(
    funtions=res.instrument["scenery"]["functions"],
    name="mirror_reflectivity_vs_wavelength",
)

fig = sebplt.figure(sfig)
ax = sebplt.add_axes(fig=fig, span=sax)
ax.plot(
    mir["argument_versus_value"][:, 0],
    mir["argument_versus_value"][:, 1],
    color="gray",
)
ax.set_xlabel("wavelength / m")
ax.set_ylabel("mirror reflectivity / 1")
ax.set_ylim([0.5, 1])
ax.set_xlim([200e-9, 800e-9])
fig.savefig(opj(res.paths["out_dir"], "mirror_reflectivity.jpg"))
sebplt.close(fig)


lns = get_function_by_name(
    funtions=res.instrument["scenery"]["functions"],
    name="lens_refraction_vs_wavelength",
)

fig = sebplt.figure(sfig)
ax = sebplt.add_axes(fig=fig, span=sax)
ax.plot(
    lns["argument_versus_value"][:, 0],
    lns["argument_versus_value"][:, 1],
    color="gray",
)
ax.set_xlabel("wavelength / m")
ax.set_ylabel("lens refractivity / 1")
ax.set_ylim([1.4, 1.6])
ax.set_xlim([200e-9, 800e-9])
fig.savefig(opj(res.paths["out_dir"], "lens_refractivity.jpg"))
sebplt.close(fig)

res.stop()
