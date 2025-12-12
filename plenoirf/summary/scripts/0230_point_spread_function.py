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

passing_trigger = res.read_passed_trigger(
    opj(res.paths["analysis_dir"], "0055_passing_trigger"),
    trigger_mode_key="far_accepting_focus_and_near_rejecting_focus",
)
passing_quality = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)
passing_trajectory_quality = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0059_passing_trajectory_quality")
)
reconstructed_energy = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"],
        "0065_learning_airshower_maximum_and_energy",
        "MultiLayerPerceptron",
    ),
)


OTHERS = {
    "hofmann2006performance_without_earth_magnetic_field": {
        "energy_GeV": [966.7, 95.855, 29.055, 9.7492],
        "resolution_deg": [0.005758, 0.019627, 0.043256, 0.113806],
    },
    "hofmann2006performance_with_earth_magnetic_field": {
        "energy_GeV": [991.57, 98.321, 29.803],
        "resolution_deg": [0.007771, 0.030353, 0.07359],
    },
    "aharonian2001": {
        "energy_GeV": [
            1.0,
            26.615384615384617,
            52.23076923076923,
            77.84615384615384,
            103.46153846153847,
            129.0769230769231,
            154.69230769230768,
            180.30769230769232,
            205.92307692307693,
            231.53846153846155,
            257.1538461538462,
            282.7692307692308,
            308.38461538461536,
            334.0,
            359.61538461538464,
            385.2307692307693,
            410.84615384615387,
            436.46153846153845,
            462.0769230769231,
            487.6923076923077,
            513.3076923076924,
            538.9230769230769,
            564.5384615384615,
            590.1538461538462,
            615.7692307692307,
            641.3846153846155,
            667.0,
            692.6153846153846,
            718.2307692307693,
            743.8461538461538,
            769.4615384615386,
            795.0769230769231,
            820.6923076923077,
            846.3076923076923,
            871.9230769230769,
            897.5384615384617,
            923.1538461538462,
            948.7692307692308,
            974.3846153846154,
            1000.0,
        ],
        "resolution_deg": [
            0.8,
            0.21529645969169975,
            0.16440666441902504,
            0.14015065213387187,
            0.12507727594342896,
            0.11448551584721958,
            0.10648844852826889,
            0.10015768663339873,
            0.09497471072280389,
            0.09062347881600298,
            0.08689859951464854,
            0.0836598587191486,
            0.08080772970324002,
            0.07826929108524781,
            0.07598969157862483,
            0.07392674759302115,
            0.07204739731793279,
            0.07032530029747029,
            0.06873916861471656,
            0.06727157949409515,
            0.06590811305182008,
            0.06463671475139736,
            0.06344721634929626,
            0.06233097068551779,
            0.061280569603181324,
            0.06028962347655631,
            0.05935258701878154,
            0.05846462028565158,
            0.05762147675052878,
            0.05681941241875672,
            0.056055111451627816,
            0.05532562486103442,
            0.05462831963809155,
            0.05396083627525927,
            0.05332105308921749,
            0.05270705609116027,
            0.05211711341078358,
            0.05154965348046649,
            0.05100324634178947,
            0.050476587558415456,
        ],
    },
}


# energy
# ------
energy_bin = res.energy_binning(key="trigger_acceptance_onregion")

containment_percents = [68, 95]
num_containment_fractions = len(containment_percents)

mk = "energy_GeV"

cta = irf.other_instruments.cherenkov_telescope_array_south
fermi = irf.other_instruments.fermi_lat
portal = irf.other_instruments.portal


pk = "gamma"

pk_dir = opj(res.paths["out_dir"], pk)
os.makedirs(pk_dir, exist_ok=True)

uid_common = snt.logic.intersection(
    passing_trigger[pk]["uid"],
    passing_quality[pk]["uid"],
    passing_trajectory_quality[pk]["trajectory_quality"]["uid"],
    reconstructed_energy[pk][mk]["uid"],
)

with res.open_event_table(particle_key=pk) as arc:
    event_table = arc.query(
        levels_and_columns={
            "primary": "__all__",
            "instrument_pointing": "__all__",
            "groundgrid_choice": "__all__",
            "reconstructed_trajectory": "__all__",
            "features": "__all__",
        },
        indices=uid_common,
        sort=True,
    )

rectab = irf.reconstruction.trajectory_quality.make_rectangular_table(
    event_table=event_table,
    instrument_pointing_model=res.config["pointing"]["model"],
)
del event_table

_true_energy = rectab["primary/energy_GeV"]
_reco_energy = irf.analysis.energy.align_on_idx(
    input_idx=reconstructed_energy[pk][mk]["uid"],
    input_values=reconstructed_energy[pk][mk][mk],
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

sfig, sax = irf.summary.figure.style("4:3")
fig = sebplt.figure(sfig)
ax = sebplt.add_axes(fig=fig, span=sax)

con = 0
tt_deg = np.rad2deg(out["theta68_rad"])
tt_relunc = out["theta68_relunc"]
sebplt.ax_add_histogram(
    ax=ax,
    bin_edges=energy_bin["edges"],
    bincounts=tt_deg,
    linestyle="-",
    linecolor=portal.COLOR,
    linealpha=1.0,
    bincounts_upper=tt_deg * (1 + tt_relunc),
    bincounts_lower=tt_deg * (1 - tt_relunc),
    face_color=portal.COLOR,
    face_alpha=0.1,
    label=portal.LABEL,
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

ax.plot(
    OTHERS["hofmann2006performance_with_earth_magnetic_field"]["energy_GeV"],
    OTHERS["hofmann2006performance_with_earth_magnetic_field"][
        "resolution_deg"
    ],
    color="black",
    alpha=0.5,
    linestyle="--",
    label="Hofmann, `limits', 2006",
)

ax.plot(
    OTHERS["aharonian2001"]["energy_GeV"],
    OTHERS["aharonian2001"]["resolution_deg"],
    color="black",
    alpha=0.5,
    linestyle=":",
    label="Aharonian et. al., `5@5', 2001",
)

ax.set_xlim([1e-1, 1e4])
ax.set_ylim([1e-2, 1e1])
# ax.legend(loc="best", fontsize=10)
ax.loglog()
enelabels = {"true": "", "reco": "reco. "}
ax.set_xlabel(enelabels[enekey] + r"energy$\,/\,$GeV")
ax.set_ylabel(r"direction containment 68%$\,/\,$1$^\circ{}$")

fig.savefig(opj(res.paths["out_dir"], pk + ".jpg"))
sebplt.close(fig)

res.stop()
