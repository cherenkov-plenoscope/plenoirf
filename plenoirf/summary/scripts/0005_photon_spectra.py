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

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.plot(
    nsb["flux_vs_wavelength"][:, 0],
    nsb["flux_vs_wavelength"][:, 1],
    color="black",
)
ax.set_xlabel("wavelength / m")
ax.set_ylabel(r"differential flux / s$^{-1}$ m$^{-2}$ (sr)$^{-1}$ m$^{-1}$")
ax.semilogy()
ax.set_xlim([200e-9, 800e-9])
fig.savefig(opj(res.paths["out_dir"], "night_sky_background.jpg"))
sebplt.close(fig)

pec = res.config["merlict_plenoscope_propagator_config"][
    "photo_electric_converter"
]

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.plot(
    pec["quantum_efficiency_vs_wavelength"][:, 0],
    pec["quantum_efficiency_vs_wavelength"][:, 1],
    color="black",
)
ax.set_xlabel("wavelength / m")
ax.set_ylabel("efficiency / 1")
ax.set_xlim([200e-9, 800e-9])
ax.set_ylim([0, 0.5])
fig.savefig(opj(res.paths["out_dir"], "photo_electric_converter.jpg"))
sebplt.close(fig)


mir = get_function_by_name(
    funtions=res.instrument["scenery"]["functions"],
    name="mirror_reflectivity_vs_wavelength",
)

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.plot(
    mir["argument_versus_value"][:, 0],
    mir["argument_versus_value"][:, 1],
    color="black",
)
ax.set_xlabel("wavelength / m")
ax.set_ylabel("reflectivity / 1")
ax.set_ylim([0.5, 1])
ax.set_xlim([200e-9, 800e-9])
fig.savefig(opj(res.paths["out_dir"], "mirror_reflectivity.jpg"))
sebplt.close(fig)


lns = get_function_by_name(
    funtions=res.instrument["scenery"]["functions"],
    name="lens_refraction_vs_wavelength",
)

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.plot(
    lns["argument_versus_value"][:, 0],
    lns["argument_versus_value"][:, 1],
    color="black",
)
ax.set_xlabel("wavelength / m")
ax.set_ylabel("refractivity / 1")
ax.set_ylim([1.4, 1.6])
ax.set_xlim([200e-9, 800e-9])
fig.savefig(opj(res.paths["out_dir"], "lens_refractivity.jpg"))
sebplt.close(fig)


fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.plot(
    lns["argument_versus_value"][:, 0],
    lns["argument_versus_value"][:, 1],
    color="black",
    linestyle=":",
    label="lens refractivity",
)
ax.plot(
    mir["argument_versus_value"][:, 0],
    mir["argument_versus_value"][:, 1],
    color="black",
    linestyle="--",
    label="mirror reflectivity",
)
ax.plot(
    pec["quantum_efficiency_vs_wavelength"][:, 0],
    pec["quantum_efficiency_vs_wavelength"][:, 1],
    color="black",
    linestyle="-",
    label="photo electric efficiency",
)
ax.legend(bbox_to_anchor=(0.5, 0.5))
ax.set_xlabel("wavelength / m")
ax.set_ylim([0.0, 1.6])
ax.set_xlim([200e-9, 800e-9])
fig.savefig(opj(res.paths["out_dir"], "mirror_lens_photo.jpg"))
sebplt.close(fig)

res.stop()
