#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import plenopy as pl
import sebastians_matplotlib_addons as seb
import json_utils


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

passing_trigger = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)
passing_trajectory_quality = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0059_passing_trajectory_quality")
)
reconstructed_energy = json_utils.tree.read(
    os.path.join(
        pa["summary_dir"], "0065_learning_airshower_maximum_and_energy"
    ),
)

# energy
# ------
energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

containment_percents = [68, 95]
num_containment_fractions = len(containment_percents)

mk = "energy"

cta = irf.other_instruments.cherenkov_telescope_array_south
fermi = irf.other_instruments.fermi_lat


def align_on_idx(input_idx, input_values, target_idxs):
    Q = {}
    for ii in range(len(input_idx)):
        Q[input_idx[ii]] = input_values[ii]
    aligned_values = np.nan * np.ones(target_idxs.shape[0])
    for ii in range(target_idxs.shape[0]):
        aligned_values[ii] = Q[target_idxs[ii]]
    return aligned_values


for sk in irf_config["config"]["sites"]:
    pk = "gamma"

    site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
    os.makedirs(site_particle_dir, exist_ok=True)

    event_table = spt.read(
        path=os.path.join(
            pa["run_dir"], "event_table", sk, pk, "event_table.tar"
        ),
        structure=irf.table.STRUCTURE,
    )
    idx_valid = spt.intersection(
        [
            passing_trigger[sk][pk]["idx"],
            passing_quality[sk][pk]["idx"],
            passing_trajectory_quality[sk][pk]["idx"],
            reconstructed_energy[sk][pk][mk]["idx"],
        ]
    )
    event_table = spt.cut_and_sort_table_on_indices(
        table=event_table,
        common_indices=idx_valid,
    )

    rectab = irf.reconstruction.trajectory_quality.make_rectangular_table(
        event_table=event_table,
        plenoscope_pointing=irf_config["config"]["plenoscope_pointing"],
    )

    _true_energy = rectab["primary/energy_GeV"]
    _reco_energy = align_on_idx(
        input_idx=reconstructed_energy[sk][pk][mk]["idx"],
        input_values=reconstructed_energy[sk][pk][mk]["energy"],
        target_idxs=rectab["idx"],
    )

    energy = {
        "true": _true_energy,
        "reco": _reco_energy,
    }
    enekey = "true"

    theta_deg = np.abs(np.rad2deg(rectab["trajectory/theta_rad"]))

    out = {}
    out[
        "comment"
    ] = "theta is angle between true and reco. direction of source. "
    out["energy_bin_edges_GeV"] = energy_bin["edges"]
    for con in range(num_containment_fractions):
        tkey = "theta{:02d}".format(containment_percents[con])
        out[tkey + "_rad"] = np.nan * np.ones(shape=energy_bin["num_bins"])
        out[tkey + "_relunc"] = np.nan * np.ones(shape=energy_bin["num_bins"])

    for ebin in range(energy_bin["num_bins"]):
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
        os.path.join(
            site_particle_dir,
            "angular_resolution.json".format(containment_percents[con]),
        ),
        out,
    )

    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

    con = 0
    tt_deg = np.rad2deg(out["theta68_rad"])
    tt_relunc = out["theta68_relunc"]
    seb.ax_add_histogram(
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
        np.rad2deg(
            cta.angular_resolution()["angular_resolution_68"]["values"]
        ),
        color=cta.COLOR,
        label=cta.LABEL,
    )

    ax.plot(
        fermi.angular_resolution()["reconstructed_energy"]["values"],
        np.rad2deg(
            fermi.angular_resolution()["angular_resolution_68"]["values"]
        ),
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

    fig.savefig(os.path.join(pa["out_dir"], sk + "_" + pk + ".jpg"))
    seb.close(fig)
