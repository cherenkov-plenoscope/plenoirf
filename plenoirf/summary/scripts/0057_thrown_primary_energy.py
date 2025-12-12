#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import cosmic_fluxes
import json_utils
import sparse_numeric_table as snt
import spherical_coordinates
import solid_angle_utils
import binning_utils
import spherical_histogram
import sebastians_matplotlib_addons as sebplt


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

energy_bin = res.energy_binning(key="10_bins_per_decade")
zenith_bin = res.zenith_binning(key="3_bins_per_45deg")

passing_trigger = res.read_passed_trigger(
    opj(res.paths["analysis_dir"], "0055_passing_trigger"),
    trigger_mode_key="far_accepting_focus_and_near_rejecting_focus",
)

MIN_COUNT = 1  # float32 vs. float64 uncertainty???


def zero_if_less_equal(x, value):
    mask = x <= value
    x[mask] = 0
    return x


intensity_passing_trigger = {}
for pk in res.PARTICLES:
    intensity_passing_trigger[pk] = np.zeros(energy_bin["num"], dtype=int)
    for enbin in range(energy_bin["num"]):
        uid_passed_trigger = passing_trigger[pk].uid(
            energy_bin_indices=[enbin]
        )
        intensity_passing_trigger[pk][enbin] = uid_passed_trigger.shape[0]
    intensity_passing_trigger[pk] = zero_if_less_equal(
        intensity_passing_trigger[pk], MIN_COUNT
    )

intensity_thrown = {}
for pk in res.PARTICLES:
    intensity_thrown[pk] = np.zeros(energy_bin["num"], dtype=int)
    for enbin in range(energy_bin["num"]):
        intensity_thrown[pk][enbin] = res.event_table(
            particle_key=pk
        ).population(energy_bin_indices=[enbin])
    intensity_thrown[pk] = zero_if_less_equal(intensity_thrown[pk], MIN_COUNT)


intensity_ratio = {}
for pk in res.PARTICLES:
    with np.errstate(divide="ignore"):
        intensity_ratio[pk] = (
            intensity_passing_trigger[pk] / intensity_thrown[pk]
        )

population = {
    "thrown": {"x": intensity_thrown, "label": "num. thrown / 1"},
    "trigger": {"x": intensity_passing_trigger, "label": "num. trigger / 1"},
    "ratio": {
        "x": intensity_ratio,
        "label": "num. trigger over num. thrown / 1",
    },
}

for population_key in population:
    sfig, sax = irf.summary.figure.style("16:9")
    fig = sebplt.figure(sfig)
    ax = sebplt.add_axes(fig=fig, span=sax)
    for pk in res.PARTICLES:
        sebplt.ax_add_histogram(
            ax=ax,
            bin_edges=energy_bin["edges"],
            bincounts=population[population_key]["x"][pk],
            linestyle="-",
            linecolor=res.PARTICLE_COLORS[pk],
            linealpha=1.0,
            bincounts_upper=None,
            bincounts_lower=None,
            face_color=res.PARTICLE_COLORS[pk],
            face_alpha=0.3,
        )
    ax.loglog()
    ax.set_xlim(energy_bin["limits"])
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel(population[population_key]["label"])
    fig.savefig(
        opj(res.paths["out_dir"], f"population_{population_key:s}.jpg")
    )
    sebplt.close(fig)

res.stop()
