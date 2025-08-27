#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import plenopy as pl
import sebastians_matplotlib_addons as sebplt
import json_utils


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

passing_trigger = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)
passing_trajectory_quality = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0059_passing_trajectory_quality")
)
reconstructed_energy = json_utils.tree.read(
    opj(
        res.paths["analysis_dir"],
        "0065_learning_airshower_maximum_and_energy",
        "RandomForest",
    ),
)

# energy
# ------
energy_bin = res.energy_binning(key="trigger_acceptance_onregion")

containment_percents = [68, 95]
num_containment_fractions = len(containment_percents)

mk = "energy"

cta = irf.other_instruments.cherenkov_telescope_array_south
fermi = irf.other_instruments.fermi_lat


pk = "gamma"

pk_dir = opj(res.paths["out_dir"], pk)
os.makedirs(pk_dir, exist_ok=True)

with res.open_event_table(particle_key=pk) as arc:
    event_table = arc.query(
        levels_and_columns={
            "primary": "__all__",
            "instrument_pointing": "__all__",
            "groundgrid_choice": "__all__",
            "reconstructed_trajectory": "__all__",
            "features": "__all__",
        }
    )

uid_common = snt.logic.intersection(
    passing_trigger[pk]["uid"],
    passing_quality[pk]["uid"],
    passing_trajectory_quality[pk]["trajectory_quality"]["uid"],
    reconstructed_energy[pk][mk]["uid"],
)
event_table = snt.logic.cut_and_sort_table_on_indices(
    table=event_table,
    common_indices=uid_common,
    level_keys=[
        "primary",
        "instrument_pointing",
        "groundgrid_choice",
        "reconstructed_trajectory",
        "features",
    ],
)

rectab = irf.reconstruction.trajectory_quality.make_rectangular_table(
    event_table=event_table,
    instrument_pointing_model=res.config["pointing"]["model"],
)

_true_energy = rectab["primary/energy_GeV"]
_reco_energy = irf.analysis.energy.align_on_idx(
    input_idx=reconstructed_energy[pk][mk]["uid"],
    input_values=reconstructed_energy[pk][mk]["energy_GeV"],
    target_idxs=rectab["uid"],
)

energy = {
    "true": _true_energy,
    "reco": _reco_energy,
}
enekey = "true"

theta_deg = np.abs(np.rad2deg(rectab["trajectory/theta_rad"]))

out = {}
out["comment"] = "theta is angle between true and reco. direction of source. "
out["energy_bin_edges_GeV"] = energy_bin["edges"]
for con in range(num_containment_fractions):
    tkey = "theta{:02d}".format(containment_percents[con])
    out[tkey + "_rad"] = np.nan * np.ones(shape=energy_bin["num"])
    out[tkey + "_relunc"] = np.nan * np.ones(shape=energy_bin["num"])

for ebin in range(energy_bin["num"]):
    energy_bin_start = energy_bin["edges"][ebin]
    energy_bin_stop = energy_bin["edges"][ebin + 1]
    energy_bin_mask = np.logical_and(
        energy[enekey] >= energy_bin_start,
        energy[enekey] < energy_bin_stop,
    )
    num_events = np.sum(energy_bin_mask)
    energy_bin_theta_deg = theta_deg[energy_bin_mask]

    for con in range(num_containment_fractions):
        (
            t_deg,
            t_relunc,
        ) = irf.analysis.gamma_direction.estimate_containment_radius(
            theta_deg=energy_bin_theta_deg,
            psf_containment_factor=1e-2 * containment_percents[con],
        )

        tkey = "theta{:02d}".format(containment_percents[con])
        out[tkey + "_rad"][ebin] = np.deg2rad(t_deg)
        out[tkey + "_relunc"][ebin] = t_relunc

json_utils.write(
    opj(
        pk_dir,
        "angular_resolution.json".format(containment_percents[con]),
    ),
    out,
)

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

con = 0
tt_deg = np.rad2deg(out["theta68_rad"])
tt_relunc = out["theta68_relunc"]
sebplt.ax_add_histogram(
    ax=ax,
    bin_edges=energy_bin["edges"],
    bincounts=tt_deg,
    linestyle="-",
    linecolor="k",
    linealpha=1.0,
    bincounts_upper=tt_deg * (1 + tt_relunc),
    bincounts_lower=tt_deg * (1 - tt_relunc),
    face_color="k",
    face_alpha=0.1,
    label="Portal",
)

ax.plot(
    cta.angular_resolution()["reconstructed_energy"]["values"],
    np.rad2deg(cta.angular_resolution()["angular_resolution_68"]["values"]),
    color=cta.COLOR,
    label=cta.LABEL,
)

ax.plot(
    fermi.angular_resolution()["reconstructed_energy"]["values"],
    np.rad2deg(fermi.angular_resolution()["angular_resolution_68"]["values"]),
    color=fermi.COLOR,
    label=fermi.LABEL,
)

ax.set_xlim([1e-1, 1e4])
ax.set_ylim([1e-2, 1e1])
# ax.legend(loc="best", fontsize=10)
ax.loglog()
enelabels = {"true": "", "reco": "reco. "}
ax.set_xlabel(enelabels[enekey] + r"energy$\,/\,$GeV")
ax.set_ylabel(r"$\Theta{}$ 68%$\,/\,$1$^\circ{}$")

fig.savefig(opj(res.paths["out_dir"], pk + ".jpg"))
sebplt.close(fig)

res.stop()
