#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import pandas
import numpy as np
import sklearn
import pickle
import json_utils
import confusion_matrix
import binning_utils
from sklearn import neural_network
from sklearn import ensemble
from sklearn import model_selection
from sklearn import utils
from sklearn.preprocessing import StandardScaler
import scipy.signal
import sebastians_matplotlib_addons as sebplt

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

transformed_features_dir = opj(
    res.paths["analysis_dir"], "0062_transform_features"
)
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
energy_population_function = json_utils.read(
    opj(
        res.paths["analysis_dir"],
        "0065_01_energy_population_function",
        "energy_population_function.json",
    )
)
energy_bin = res.energy_binning(key="5_bins_per_decade")
random_seed = res.analysis["random_seed"]


def _energy_to_sx(e):
    pop = energy_population_function
    return (np.log10(e) - pop["log10_shift"]) / pop["log10_scale"]


def energy_to_flat(e):
    pop = energy_population_function
    sx = _energy_to_sx(e=e)
    return np.interp(xp=pop["xp"], fp=pop["fp"], x=sx)


def _sx_to_energy(sx):
    pop = energy_population_function
    return 10 ** ((sx * pop["log10_scale"]) + pop["log10_shift"])


def flat_to_energy(f):
    pop = energy_population_function
    sx = np.interp(xp=pop["fp"], fp=pop["xp"], x=f)
    return _sx_to_energy(sx=sx)


def test_energy_to_flat(max_rel_delta_energy=0.05):
    E_original = energy_bin["centers"]
    sx = _energy_to_sx(e=E_original)
    assert np.all(sx <= 1.0)
    assert np.all(sx >= 0.0)
    E_back = _sx_to_energy(sx=sx)
    delta = np.abs(E_original - E_back) / E_original
    assert np.all(delta < max_rel_delta_energy)

    flat = energy_to_flat(e=E_original)
    assert np.all(flat <= 1.0)
    assert np.all(flat >= 0.0)
    E_back = flat_to_energy(f=flat)
    delta = np.abs(E_original - E_back) / E_original
    assert np.all(delta < max_rel_delta_energy)


def altitude_to_flat(a):
    return np.log10(a)


def flat_to_altitude(f):
    return 10**f


APPLY_INVERSE_POPULATION_WEIGHTS = True

targets = {
    "energy_GeV": {
        "column": 0,
        "to_flat": energy_to_flat,
        "from_flat": flat_to_energy,
    },
    "z_emission_p50_m": {
        "column": 1,
        "to_flat": altitude_to_flat,
        "from_flat": flat_to_altitude,
    },
}

test_energy_to_flat()


def read_event_frame(
    res,
    particle_key,
    run_dir,
    transformed_features_dir,
    train_test,
):
    pk = particle_key

    uid_train_and_test = snt.logic.union(
        train_test[pk]["train"],
        train_test[pk]["test"],
    )

    event_table = res.event_table(particle_key=pk).query(
        levels_and_columns={
            "primary": ["uid", "energy_GeV"],
            "cherenkovpool": ["uid", "z_emission_p50_m"],
        },
        indices=uid_train_and_test,
    )

    transformed_features_path = opj(
        transformed_features_dir,
        pk,
        "transformed_features.zip",
    )
    with snt.open(transformed_features_path, "r") as arc:
        event_table["transformed_features"] = arc.query()[
            "transformed_features"
        ]

    out = {}
    for kk in ["test", "train"]:
        table_kk = snt.logic.cut_and_sort_table_on_indices(
            table=event_table,
            common_indices=train_test[pk][kk],
        )
        out[kk] = snt.logic.make_rectangular_DataFrame(table_kk)

    return out


FEATURE_KEYS = [
    "num_photons",
    "image_smallest_ellipse_object_distance",
    "image_smallest_ellipse_solid_angle",
    "image_smallest_ellipse_half_depth",
    "paxel_intensity_peakness_mean_over_std",
    "combi_shower_density",
    "combi_shower_volume",
    "reconstructed_trajectory_cx_rad",
    "reconstructed_trajectory_cy_rad",
    "reconstructed_trajectory_x_m",
    "reconstructed_trajectory_y_m",
    "combi_A",
    "combi_B",
    "combi_C",
    # "reconstructed_trajectory_hypot_cx_cy",
    # "reconstructed_trajectory_hypot_x_y",
]


def make_x_y_arrays(event_frame):
    f = event_frame

    xll = []
    for key in FEATURE_KEYS:
        xll.append(f[f"transformed_features/{key:s}"].values)
    x = np.array(xll).T

    y = np.array(
        [
            targets["energy_GeV"]["to_flat"](f["primary/energy_GeV"].values),
            targets["z_emission_p50_m"]["to_flat"](
                f["cherenkovpool/z_emission_p50_m"].values
            ),
        ]
    ).T
    return x, y


NUM_BOOTSTRIPS = 10

passing = {}
for pk in res.PARTICLES:
    passing[pk] = snt.logic.intersection(
        passing_trigger[pk]["uid"],
        passing_quality[pk]["uid"],
        passing_trajectory_quality[pk]["uid"],
    )

num_features = len(FEATURE_KEYS)


REGRESSORS = {}
REGRESSORS["MultiLayerPerceptron"] = {
    "constructor": sklearn.neural_network.MLPRegressor,
    "kwargs": {
        "hidden_layer_sizes": (
            3 * num_features,
            5 * num_features,
            7 * num_features,
            5 * num_features,
            3 * num_features,
            1 * num_features,
        ),
        "alpha": 0.025,
        "random_state": random_seed,
        "verbose": True,
    },
}
REGRESSORS["RandomForest"] = {
    "constructor": sklearn.ensemble.RandomForestRegressor,
    "kwargs": {
        "random_state": random_seed,
        "n_estimators": 32 * num_features,
        "verbose": True,
    },
}


for bootstrip in range(NUM_BOOTSTRIPS):
    print("bootstrip", bootstrip)
    bootstrip_dir = opj(res.paths["cache_dir"], f"bootstrip-{bootstrip:02d}")

    if os.path.exists(bootstrip_dir):
        continue

    os.makedirs(bootstrip_dir, exist_ok=True)

    # make train_test seperation for this bootstrip
    # ---------------------------------------------
    train_test = {}
    for pk in res.PARTICLES:
        train_test[pk] = {}
        if pk == "gamma":
            (train_test[pk]["train"], train_test[pk]["test"]) = (
                irf.bootstripping.train_test_split(
                    x=passing[pk],
                    bootstrip=bootstrip,
                    num_bootstrips=NUM_BOOTSTRIPS,
                )
            )
            print(
                pk,
                f"train: {train_test[pk]['train'].shape[0]:d}, "
                f"test: {train_test[pk]['test'].shape[0]:d}.",
            )
        else:
            train_test[pk]["train"] = []
            train_test[pk]["test"] = np.array(passing[pk])

    particle_frames = {}
    for pk in res.PARTICLES:
        particle_frames[pk] = read_event_frame(
            res=res,
            particle_key=pk,
            run_dir=res.paths["plenoirf_dir"],
            transformed_features_dir=transformed_features_dir,
            train_test=train_test,
        )

    MA = {}
    for pk in res.PARTICLES:
        MA[pk] = {}
        for mk in ["test", "train"]:
            MA[pk][mk] = {}
            MA[pk][mk]["x"], MA[pk][mk]["y"] = make_x_y_arrays(
                event_frame=particle_frames[pk][mk]
            )
            if pk == "gamma" and mk == "train":
                MA[pk][mk]["w"] = irf.features.scaling.make_sample_weights(
                    x=particle_frames[pk][mk]["primary/energy_GeV"],
                    bin_edges=energy_bin["edges"],
                )

    # train model on gamma only
    # -------------------------
    assert MA["gamma"]["train"]["x"].shape[1] == num_features

    models = {}
    for mk in REGRESSORS:
        models[mk] = REGRESSORS[mk]["constructor"](**REGRESSORS[mk]["kwargs"])

    _X_shuffle, _y_shuffle, _w_shuffle = sklearn.utils.shuffle(
        MA["gamma"]["train"]["x"],
        MA["gamma"]["train"]["y"],
        MA["gamma"]["train"]["w"],
        random_state=random_seed,
    )

    for mk in REGRESSORS:
        model_dir = opj(bootstrip_dir, mk)
        os.makedirs(model_dir, exist_ok=True)

        if APPLY_INVERSE_POPULATION_WEIGHTS:
            models[mk].fit(
                X=_X_shuffle, y=_y_shuffle, sample_weight=_w_shuffle
            )
        else:
            models[mk].fit(X=_X_shuffle, y=_y_shuffle)

        model_path = opj(model_dir, mk + ".pkl")
        with open(model_path, "wb") as fout:
            fout.write(pickle.dumps(models[mk]))

        for pk in res.PARTICLES:
            pk_dir = opj(model_dir, pk)
            os.makedirs(pk_dir, exist_ok=True)

            _y_score = models[mk].predict(MA[pk]["test"]["x"])

            for tk in targets:
                y_score = targets[tk]["from_flat"](
                    _y_score[:, targets[tk]["column"]]
                )
                out = {}
                out["comment"] = "Reconstructed from the test-set."
                out["learner"] = mk
                out[tk] = y_score
                out["uid"] = np.array(particle_frames[pk]["test"]["uid"])

                json_utils.write(opj(pk_dir, tk + ".json"), out)

# read bootstrippings
# -------------------
results = {}
for mk in REGRESSORS:
    results[mk] = {}

    for pk in res.PARTICLES:
        results[mk][pk] = {}

        for tk in targets:
            results[mk][pk][tk] = {}

            for bootstrip in range(NUM_BOOTSTRIPS):
                path = opj(
                    res.paths["cache_dir"],
                    f"bootstrip-{bootstrip:02d}",
                    mk,
                    pk,
                    tk + ".json",
                )
                boot = json_utils.read(path)

                for i in range(len(boot["uid"])):
                    _uid = boot["uid"][i]
                    _tgt = boot[tk][i]

                    if _uid in results[mk][pk][tk]:
                        results[mk][pk][tk][_uid].append(_tgt)
                    else:
                        results[mk][pk][tk][_uid] = [_tgt]

# assert is complete
# ------------------
for mk in results:
    for pk in results[mk]:
        for tk in results[mk][pk]:
            assert len(results[mk][pk][tk]) == len(passing[pk])
            for uid in results[mk][pk][tk]:
                if pk == "gamma":
                    assert len(results[mk][pk][tk][uid]) == 1
                else:
                    assert len(results[mk][pk][tk][uid]) == NUM_BOOTSTRIPS


# combine bootstrippings
# ----------------------
out = {}
for mk in results:
    out[mk] = {}
    for pk in results[mk]:
        out[mk][pk] = {}

        out[mk][pk] = np.recarray(
            shape=len(passing[pk]),
            dtype=[
                ("uid", "<u8"),
                ("energy_GeV", "<f4"),
                ("z_emission_p50_m", "<f4"),
            ],
        )

        for iii, uid in enumerate(passing[pk]):
            out[mk][pk]["uid"][iii] = uid
            for tk in results[mk][pk]:
                out[mk][pk][tk][iii] = np.median(results[mk][pk][tk][uid])


# EXPORT
# ======
for mk in results:
    mk_dir = opj(res.paths["out_dir"], mk)
    for pk in results[mk]:
        mk_pk_dir = opj(mk_dir, pk)
        os.makedirs(mk_pk_dir, exist_ok=True)

        for tk in targets:
            ooo = {}
            ooo[tk] = out[mk][pk][tk]
            ooo["uid"] = out[mk][pk]["uid"]
            json_utils.write(opj(mk_pk_dir, tk + ".json"), ooo)


def read_true_energy(uid):
    table = res.event_table(particle_key="gamma").query(
        levels_and_columns={"primary": ["uid", "energy_GeV"]},
        indices=uid,
        sort=True,
    )
    np.testing.assert_array_equal(uid, table["primary"]["uid"])
    return table["primary"]["energy_GeV"]


def smooth1d(x, num_iterations=1000):
    x = np.array(x, dtype=float)
    NUM = x.shape[0]
    for jj in range(num_iterations):
        for i_start in range(NUM - 1):
            i_stop = i_start + 1
            if x[i_start] >= x[i_stop]:
                i_before = i_start - 1
                i_after = i_stop + 1
                if i_before >= 0:
                    x[i_start] = np.mean([x[i_start], x[i_before]])
                if i_after < NUM:
                    x[i_stop] = np.mean([x[i_stop], x[i_after]])
    return x


def add_points_in_between(xp, fp, num_points=10):
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    num = xp.shape[0]
    assert fp.shape[0] == num
    x = []
    f = []
    for j_start in range(num - 1):
        j_stop = j_start + 1
        for ii in range(num_points):
            weight_stop = ii / num_points
            weight_start = 1.0 - weight_stop
            inter_x = xp[j_start] * weight_start + xp[j_stop] * weight_stop
            inter_f = fp[j_start] * weight_start + fp[j_stop] * weight_stop
            x.append(inter_x)
            f.append(inter_f)

    return np.array(x), np.array(f)


def w_p95(i, fine_num):
    return np.interp(
        xp=[0.0, 0.7, 0.9, 1.0],
        fp=[0.0, 0.0, 1.0, 1.0],
        x=i / (fine_num),
    )


def w_p05(i, fine_num):
    return np.interp(
        xp=[0.0, 0.1, 0.3, 1.0],
        fp=[1.0, 1.0, 0.0, 0.0],
        x=i / (fine_num),
    )


def w_max(i, fine_num):
    return np.interp(
        xp=[0.0, 0.1, 0.3, 0.7, 0.9, 1.0],
        fp=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        x=i / (fine_num),
    )


min_number_samples = 1

kernerl1d = np.array([1, 2, 5, 2, 1])
kernerl1d = kernerl1d / np.sum(kernerl1d)

matrix_kernel = np.array([[1, 2, 1], [2, 5, 2], [1, 2, 1]])
matrix_kernel = matrix_kernel / np.sum(matrix_kernel)
fine_energy_bin = binning_utils.Binning(
    np.geomspace(
        energy_bin["start"], energy_bin["stop"], energy_bin["num"] * 2 + 1
    )
)

gam = {}
for mk in REGRESSORS:
    gam[mk] = {}
    reco = json_utils.read(
        opj(res.paths["out_dir"], mk, "gamma", "energy_GeV.json")
    )
    gam[mk]["uid"] = reco["uid"]
    gam[mk]["reco"] = reco["energy_GeV"]
    gam[mk]["true"] = read_true_energy(uid=gam[mk]["uid"])

    gam[mk]["matrix"] = confusion_matrix.init(
        ax0_key="true",
        ax0_values=gam[mk]["true"],
        ax0_bin_edges=fine_energy_bin["edges"],
        ax1_key="reco",
        ax1_values=gam[mk]["reco"],
        ax1_bin_edges=fine_energy_bin["edges"],
        min_exposure_ax0=min_number_samples,
        default_low_exposure=0.0,
    )

    gam[mk]["matrix"]["convolve"] = scipy.signal.convolve2d(
        gam[mk]["matrix"]["counts"],
        matrix_kernel,
        mode="same",
    )

    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    _pcm_confusion = ax.pcolormesh(
        gam[mk]["matrix"]["ax0_bin_edges"],
        gam[mk]["matrix"]["ax1_bin_edges"],
        np.transpose(gam[mk]["matrix"]["convolve"]),
        cmap="Greys",
        norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
    )
    ax.loglog()
    ax.set_aspect("equal")
    ax.set_ylim(energy_bin["limits"])
    ax.set_xlim(energy_bin["limits"])
    ax.set_xlabel("true energy / GeV")
    ax.set_ylabel("reco. energy / GeV")
    fig.savefig(opj(res.paths["out_dir"], f"{mk:s}_polyfit.jpg"))
    sebplt.close(fig)

    fine_one_edges = np.linspace(
        0, fine_energy_bin["num"], fine_energy_bin["num"] + 1
    )

    f_reco = []
    f_true = []
    con = gam[mk]["matrix"]["convolve"]
    fine_num = fine_energy_bin["num"]
    for i in range(fine_energy_bin["num"]):
        con_slice = con[i, :]
        if np.sum(con_slice) > 0:
            f_true.append(i)

            p05 = binning_utils.quantile(
                bin_edges=fine_one_edges,
                bin_counts=con_slice,
                q=0.05,
            )
            p95 = binning_utils.quantile(
                bin_edges=fine_one_edges,
                bin_counts=con_slice,
                q=0.95,
            )
            pmax = np.argmax(con_slice)
            _w_p05 = w_p05(i=i, fine_num=fine_num)
            _w_max = w_max(i=i, fine_num=fine_num)
            _w_p95 = w_p95(i=i, fine_num=fine_num)
            _w_sum = _w_p05 + _w_max + _w_p95
            assert 0.95 < _w_sum < 1.05
            f_reco.append(_w_p05 * p05 + _w_max * pmax + _w_p95 * p95)

    f_reco_smooth = smooth1d(
        x=f_reco,
        num_iterations=1_000,
    )

    gam[mk]["f_true"], gam[mk]["f_reco"] = add_points_in_between(
        xp=f_true,
        fp=f_reco_smooth,
        num_points=3,
    )

    gam[mk]["f_reco_smooth"] = np.array(gam[mk]["f_reco"])

    for i in range(50):
        gam[mk]["f_reco_smooth"] = scipy.signal.convolve(
            gam[mk]["f_reco_smooth"], kernerl1d, mode="same"
        )
        for j in range(3):
            gam[mk]["f_reco_smooth"][j] = gam[mk]["f_reco"][j]
            gam[mk]["f_reco_smooth"][-j] = gam[mk]["f_reco"][-j]
        """
        gam[mk]["f_reco_smooth"] = smooth1d(
            x=gam[mk]["f_reco"],
            num_iterations=100,
        )
        """

    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    ax.plot(
        gam[mk]["f_true"],
        gam[mk]["f_reco"],
        color="black",
    )
    ax.plot(
        gam[mk]["f_true"],
        gam[mk]["f_reco_smooth"],
        color="red",
    )
    ax.set_aspect("equal")
    ax.set_ylim([0, fine_energy_bin["num"]])
    ax.set_xlim([0, fine_energy_bin["num"]])
    ax.set_xlabel("true energy")
    ax.set_ylabel("reco. energy")
    fig.savefig(opj(res.paths["out_dir"], f"{mk:s}_f.jpg"))
    sebplt.close(fig)

    gam[mk]["fE_true"] = np.geomspace(
        energy_bin["start"],
        energy_bin["stop"],
        gam[mk]["f_true"].shape[0],
    )
    gam[mk]["fE_reco_smooth"] = np.interp(
        xp=gam[mk]["f_true"],
        fp=gam[mk]["fE_true"],
        x=gam[mk]["f_reco_smooth"],
    )

    gam[mk]["reco2"] = np.interp(
        xp=gam[mk]["fE_reco_smooth"],
        fp=gam[mk]["fE_true"],
        x=gam[mk]["reco"],
    )

    gam[mk]["matrix2"] = confusion_matrix.init(
        ax0_key="true",
        ax0_values=gam[mk]["true"],
        ax0_bin_edges=fine_energy_bin["edges"],
        ax1_key="reco2",
        ax1_values=gam[mk]["reco2"],
        ax1_bin_edges=fine_energy_bin["edges"],
        min_exposure_ax0=min_number_samples,
        default_low_exposure=0.0,
    )

    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    _pcm_confusion = ax.pcolormesh(
        gam[mk]["matrix2"]["ax0_bin_edges"],
        gam[mk]["matrix2"]["ax1_bin_edges"],
        np.transpose(gam[mk]["matrix2"]["counts_normalized_on_ax0"]),
        cmap="Greys",
        norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
    )
    ax.loglog()
    ax.set_aspect("equal")
    ax.set_ylim(energy_bin["limits"])
    ax.set_xlim(energy_bin["limits"])
    ax.set_xlabel("true energy / GeV")
    ax.set_ylabel("reco. energy / GeV")
    fig.savefig(opj(res.paths["out_dir"], f"{mk:s}_reco2.jpg"))
    sebplt.close(fig)


# EXPORT TRANSFORM
# ================
for mk in gam:
    mk_dir = opj(res.paths["out_dir"], mk + "Transformed")
    for pk in results[mk]:
        mk_pk_dir = opj(mk_dir, pk)
        os.makedirs(mk_pk_dir, exist_ok=True)

        for tk in targets:
            ooo = {}
            ooo["uid"] = out[mk][pk]["uid"]
            if tk == "energy_GeV":
                ooo[tk] = np.interp(
                    xp=gam[mk]["fE_reco_smooth"],
                    fp=gam[mk]["fE_true"],
                    x=out[mk][pk][tk],
                )
                # Fall back when out of range
                mask_too_low = ooo[tk] < energy_bin["start"]
                mask_too_high = ooo[tk] > energy_bin["stop"]
                mask_out_of_range = np.logical_and(mask_too_low, mask_too_high)
                ooo[tk][mask_out_of_range] = out[mk][pk][tk][mask_out_of_range]
            else:
                ooo[tk] = out[mk][pk][tk]
            json_utils.write(opj(mk_pk_dir, tk + ".json"), ooo)


"""
def merge_machine_lerners(energy_resolutions, reco_energies, energy_bin):
    NUM_REGRESSORS = len(energy_resolutions)
    assert len(reco_energies) == NUM_REGRESSORS
    NUM_EVENTS = reco_energies[0].shape[0]
    for i in range(NUM_REGRESSORS):
        assert len(reco_energies[i]) == NUM_EVENTS

    reco_energy_bin_assignment = -1 * np.ones(shape=NUM_EVENTS)
    median_reco_energy = np.median(np.asarray(reco_energies), axis=0)

    assert median_reco_energy.shape == (NUM_EVENTS,)

    reco_energy_bin_assignment = -1 + np.digitize(
        median_reco_energy, bins=energy_bin["edges"]
    )

    resos = np.nan * np.ones(shape=(NUM_EVENTS, NUM_REGRESSORS))
    for e in range(NUM_EVENTS):
        for l in range(NUM_REGRESSORS):
            ebin = reco_energy_bin_assignment[e]
            if 0 <= ebin < energy_bin["num"]:
                resos[e, l] = energy_resolutions[l][ebin]

    mask = np.zeros(shape=NUM_EVENTS, dtype=int)

    for e in range(NUM_EVENTS):
        try:
            lmin = np.nanargmin(resos[e, :])
            mask[e] = lmin
        except ValueError as err:
            mask[e] = -1

    return mask


def apply_machine_lerner_merge(mask, reco_x):
    NUM_REGRESSORS = len(reco_x)
    assert NUM_REGRESSORS > 1
    NUM_EVENTS = reco_x[0].shape[0]

    out_x = np.nan * np.ones(shape=NUM_EVENTS)
    for e in range(NUM_EVENTS):
        if mask[e] == -1:
            x_vals = [reco_x[l][e] for l in range(NUM_REGRESSORS)]
            out_x = np.median(x_vals)
        else:
            l = mask[e]
            out_x = reco_x[l]
    return out_x


# benchmark learners on gamma
# ---------------------------
REGRESSORS = ["MultiLayerPerceptron", "RandomForest"]
pk = "gamma"
with res.open_event_table(particle_key=pk) as arc:
    _event_table = arc.query(
        levels_and_columns={"primary": ["uid", "energy_GeV"]}
    )
gamma_primary = snt.logic.cut_and_sort_table_on_indices(
    table=_event_table,
    common_indices=passing[pk],
)["primary"]

true_energy = gamma_primary["energy_GeV"]

gamma_energy_resolution = {}
reco_energies = {}
for mk in REGRESSORS:
    gamma_energy_resolution[mk] = {}

    reco_energies[mk] = irf.analysis.energy.align_on_idx(
        input_idx=out[mk][pk]["uid"],
        input_values=out[mk][pk]["energy_GeV"],
        target_idxs=gamma_primary["uid"],
    )

    (
        gamma_energy_resolution[mk]["deltaE_over_E"],
        gamma_energy_resolution[mk]["deltaE_over_E_relunc"],
    ) = irf.analysis.energy.estimate_energy_resolution_vs_reco_energy(
        true_energy=true_energy,
        reco_energy=reco_energies[mk],
        reco_energy_bin_edges=energy_bin["edges"],
        containment_fraction=0.68,
    )
"""

# create existing output format
# -----------------------------
"""
combined_dir = opj(res.paths["out_dir"], "combined")
for pk in res.PARTICLES:
    os.makedirs(opj(combined_dir, pk), exist_ok=True)


    merging_mask = merge_machine_lerners(
        energy_resolutions=[
            gamma_energy_resolution["MultiLayerPerceptron"]["deltaE_over_E"],
            gamma_energy_resolution["RandomForest"]["deltaE_over_E"],
        ],
        reco_energies=[
            out["MultiLayerPerceptron"][pk]["energy_GeV"],
            out["RandomForest"][pk]["energy_GeV"],
        ],
        energy_bin=energy_bin,
    )

    reco_energy = apply_machine_lerner_merge(
        mask=merging_mask,
        reco_x=(
            out["MultiLayerPerceptron"][pk]["energy_GeV"],
            out["RandomForest"][pk]["energy_GeV"],
        ),
    )

    reco_z_emission_p50_m = apply_machine_lerner_merge(
        mask=merging_mask,
        reco_x=(
            out["MultiLayerPerceptron"][pk]["z_emission_p50_m"],
            out["RandomForest"][pk]["z_emission_p50_m"],
        ),
    )

    np.testing.assert_array_equal(
        out["RandomForest"][pk]["uid"],
        out["MultiLayerPerceptron"][pk]["uid"],
    )
    uids = out["RandomForest"][pk]["uid"]

    ooo = {}
    ooo["energy_GeV"] = np.mean(
        [
            out["MultiLayerPerceptron"][pk]["energy_GeV"],
            out["RandomForest"][pk]["energy_GeV"],
        ],
        axis=0,
    )
    ooo["uid"] = uids
    json_utils.write(opj(combined_dir, pk, "energy_GeV" + ".json"), ooo)

    ooo = {}
    ooo["z_emission_p50_m"] = np.mean(
        [
            out["MultiLayerPerceptron"][pk]["z_emission_p50_m"],
            out["RandomForest"][pk]["z_emission_p50_m"],
        ],
        axis=0,
    )
    ooo["uid"] = uids
    json_utils.write(opj(combined_dir, pk, "z_emission_p50_m" + ".json"), ooo)
"""

res.stop()
