#!/usr/bin/python
import sys
import copy
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import numpy as np
import json_utils
import binning_utils
import rename_after_writing as rnw
import sebastians_matplotlib_addons as sebplt

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)


def make_binning(script_resources):
    res = script_resources
    zenith_bin = res.zenith_binning("twice")

    _energy_bin = res.energy_binning(key="trigger_acceptance_onregion")
    energy_bin = binning_utils.Binning(
        bin_edges=np.geomspace(
            _energy_bin["start"], _energy_bin["stop"], zenith_bin["num"] + 1
        )
    )
    altitude_bin = binning_utils.Binning(
        bin_edges=np.geomspace(10e3, 20e3, energy_bin["num"] + 1)
    )
    bins = irf.summary.estimator.binning.Bins(
        zenith_rad=zenith_bin["edges"],
        energy_GeV=energy_bin["edges"],
        altitude_m=altitude_bin["edges"],
    )
    binning_dir = opj(res.paths["out_dir"], "binning")
    bins.to_path(binning_dir)
    return irf.summary.estimator.binning.Bins.from_path(binning_dir)


def make_passing_cuts(script_resources, particles):
    res = script_resources
    cache_dir = opj(res.paths["out_dir"], "passing_cuts")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

        passing_trigger = json_utils.tree.read(
            opj(res.paths["analysis_dir"], "0055_passing_trigger")
        )
        passing_quality = json_utils.tree.read(
            opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
        )

        for pk in particles:
            passing_cuts_pk = snt.logic.intersection(
                passing_trigger[pk]["uid"],
                passing_quality[pk]["uid"],
            )
            with rnw.open(opj(cache_dir, pk + ".json"), "wt") as f:
                f.write(json_utils.dumps({"uid": passing_cuts_pk}))
    return json_utils.tree.read(cache_dir)


bins = make_binning(script_resources=res)

SIGNAL = ["gamma"]
BACKGROUND = ["proton", "helium"]

json_utils.write(
    opj(res.paths["out_dir"], "signal_and_background.json"),
    {"signal": SIGNAL, "background": BACKGROUND},
)

uid_passing_cuts = make_passing_cuts(
    script_resources=res, particles=SIGNAL + BACKGROUND
)

asi_dir = opj(res.paths["out_dir"], "assignment")
os.makedirs(asi_dir, exist_ok=True)
asi_sig_dir = opj(asi_dir, "signal")
os.makedirs(asi_sig_dir, exist_ok=True)
asi_bgr_dir = opj(asi_dir, "background")
os.makedirs(asi_bgr_dir, exist_ok=True)

for pk in SIGNAL:
    asi_sig_pk_dir = opj(asi_sig_dir, pk)
    os.makedirs(asi_sig_pk_dir, exist_ok=True)

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={
                "primary": ["uid", "energy_GeV"],
                "instrument_pointing": ["uid", "zenith_rad"],
                "cherenkovpool": ["uid", "z_emission_p50_m"],
            },
            indices=uid_passing_cuts[pk]["uid"],
            sort=True,
        )

    assignment_raw = (
        irf.summary.estimator.binning.assign_uids_to_zenith_energy_altitude(
            event_table=event_table,
            bins=bins,
        )
    )
    assignment_smooth = irf.summary.estimator.binning.smoothen_uid_assign_zenith_energy_altitude(
        assignment=assignment_raw,
        bins=bins,
    )
    json_utils.write(opj(asi_sig_pk_dir, "raw.json"), assignment_raw)
    json_utils.write(opj(asi_sig_pk_dir, "smooth.json"), assignment_smooth)


for pk in BACKGROUND:
    asi_bgr_pk_dir = opj(asi_bgr_dir, pk)
    os.makedirs(asi_bgr_pk_dir, exist_ok=True)

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={
                "instrument_pointing": ["uid", "zenith_rad"],
            },
            indices=uid_passing_cuts[pk]["uid"],
        )
    assignment_raw = irf.summary.estimator.binning.assign_uids_zenith(
        event_table=event_table,
        bins=bins,
    )
    assignment_smooth = (
        irf.summary.estimator.binning.smoothen_uid_assign_zenith(
            assignment=assignment_raw,
            bins=bins,
        )
    )
    json_utils.write(opj(asi_bgr_pk_dir, "raw.json"), assignment_raw)
    json_utils.write(opj(asi_bgr_pk_dir, "smooth.json"), assignment_smooth)

# PLOT
# ====

foo = json_utils.tree.read(opj(res.paths["out_dir"]))

exposure = irf.summary.estimator.binning.len_cube(
    foo["assignment"]["signal"]["gamma"]["raw"]
)

plot_dir = opj(res.paths["out_dir"], "plot")
os.makedirs(plot_dir, exist_ok=True)

for zz in range(bins.zenith["num"]):
    fig = sebplt.figure(style=sebplt.FIGURE_1_1)
    ax_c = sebplt.add_axes(fig=fig, span=[0.2, 0.2, 0.75, 0.75])
    pcm = ax_c.pcolormesh(
        bins.energy["edges"],
        bins.altitude["edges"],
        np.transpose(exposure[zz]),
        cmap="Greys",
        norm=sebplt.plt_colors.PowerNorm(
            gamma=0.5, vmin=0, vmax=np.max(exposure)
        ),
    )
    ax_c.set_ylabel("altitude / m")
    ax_c.set_xlabel("energy / GeV")
    ax_c.loglog()
    sebplt.ax_add_grid(ax_c)
    fig.savefig(opj(plot_dir, f"gamma_exposure_zenith{zz:02d}.jpg"))
    sebplt.close(fig)

res.stop()
