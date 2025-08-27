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
from sklearn import neural_network
from sklearn import ensemble
from sklearn import model_selection
from sklearn import utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

transformed_features_dir = opj(
    res.paths["analysis_dir"], "0062_transform_features"
)
passing_trigger = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)
passing_trajectory_quality = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0059_passing_trajectory_quality")
)
energy_bin = res.energy_binning(key="trigger_acceptance_onregion")
random_seed = res.analysis["random_seed"]

targets = {
    "energy_GeV": {
        "column": 0,
    },
    "z_emission_p50_m": {
        "column": 1,
    },
}

min_number_samples = 100


def read_event_frame(
    res,
    particle_key,
    run_dir,
    transformed_features_dir,
    train_test,
):
    pk = particle_key

    with res.open_event_table(particle_key=pk) as arc:
        airshower_table = arc.query(
            levels_and_columns={
                "primary": ["uid", "energy_GeV"],
                "cherenkovpool": ["uid", "z_emission_p50_m"],
                "reconstructed_trajectory": "__all__",
            }
        )

    transformed_features_path = opj(
        transformed_features_dir,
        pk,
        "transformed_features.zip",
    )
    with snt.open(transformed_features_path, "r") as arc:
        airshower_table["transformed_features"] = arc.query()[
            "transformed_features"
        ]

    EXT_STRUCTRURE = irf.features.init_all_features_structure()

    out = {}
    for kk in ["test", "train"]:
        table_kk = snt.logic.cut_and_sort_table_on_indices(
            table=airshower_table,
            common_indices=train_test[pk][kk],
            index_key="uid",
        )
        out[kk] = snt.logic.make_rectangular_DataFrame(
            table_kk,
            index_key="uid",
        )

    return out


def make_x_y_arrays(event_frame):
    f = event_frame

    reco_radius_core_m = np.hypot(
        f["reconstructed_trajectory/x_m"],
        f["reconstructed_trajectory/y_m"],
    )

    norm_reco_radius_core_m = reco_radius_core_m / 640.0

    reco_theta_rad = np.hypot(
        f["reconstructed_trajectory/cx_rad"],
        f["reconstructed_trajectory/cy_rad"],
    )
    norm_reco_theta_rad = reco_theta_rad / np.deg2rad(3.5)

    x = np.array(
        [
            f["transformed_features/num_photons"].values,
            f[
                "transformed_features/image_smallest_ellipse_object_distance"
            ].values,
            f[
                "transformed_features/image_smallest_ellipse_solid_angle"
            ].values,
            f["transformed_features/image_smallest_ellipse_half_depth"].values,
            # f["transformed_features/combi_A"].values,
            # f["transformed_features/combi_B"].values,
            # f["transformed_features/combi_C"].values,
            f[
                "transformed_features/paxel_intensity_peakness_std_over_mean"
            ].values,
            norm_reco_radius_core_m,
            norm_reco_theta_rad,
            f["transformed_features/combi_image_infinity_std_density"].values,
            f[
                "transformed_features/combi_paxel_intensity_median_hypot"
            ].values,
            f["transformed_features/combi_diff_image_and_light_front"].values,
        ]
    ).T
    y = np.array(
        [
            np.log10(f["primary/energy_GeV"].values),
            np.log10(f["cherenkovpool/z_emission_p50_m"].values),
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

for bootstrip in range(NUM_BOOTSTRIPS):
    print("bootstrip", bootstrip)
    bootstrip_dir = opj(res.paths["out_dir"], f"bootstrip-{bootstrip:02d}")

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

    # train model on gamma only
    # -------------------------
    num_features = MA["gamma"]["train"]["x"].shape[1]
    models = {}

    models["MultiLayerPerceptron"] = sklearn.neural_network.MLPRegressor(
        solver="lbfgs",
        alpha=1e-2,
        hidden_layer_sizes=(3 * num_features),
        random_state=random_seed,
        verbose=False,
        max_iter=5000,
        learning_rate_init=0.1,
    )
    models["RandomForest"] = sklearn.ensemble.RandomForestRegressor(
        random_state=random_seed,
        n_estimators=10,
    )

    _X_shuffle, _y_shuffle = sklearn.utils.shuffle(
        MA["gamma"]["train"]["x"],
        MA["gamma"]["train"]["y"],
        random_state=random_seed,
    )

    for mk in models:
        model_dir = opj(bootstrip_dir, mk)
        os.makedirs(model_dir, exist_ok=True)

        models[mk].fit(_X_shuffle, _y_shuffle)

        model_path = opj(model_dir, mk + ".pkl")
        with open(model_path, "wb") as fout:
            fout.write(pickle.dumps(models[mk]))

        for pk in res.PARTICLES:
            pk_dir = opj(model_dir, pk)
            os.makedirs(pk_dir, exist_ok=True)

            _y_score = models[mk].predict(MA[pk]["test"]["x"])

            for tk in targets:
                y_score = 10 ** _y_score[:, targets[tk]["column"]]

                out = {}
                out["comment"] = "Reconstructed from the test-set."
                out["learner"] = mk
                out[tk] = y_score
                out["uid"] = np.array(particle_frames[pk]["test"]["uid"])

                json_utils.write(opj(pk_dir, tk + ".json"), out)

# read bootstrippings
# -------------------
results = {}
for mk in ["MultiLayerPerceptron", "RandomForest"]:
    results[mk] = {}

    for pk in res.PARTICLES:
        results[mk][pk] = {}

        for tk in targets:
            results[mk][pk][tk] = {}

            for bootstrip in range(NUM_BOOTSTRIPS):
                path = opj(
                    res.paths["out_dir"],
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

combined_dir = opj(res.paths["out_dir"], "combined")
for pk in res.PARTICLES:
    os.makedirs(opj(combined_dir, pk), exist_ok=True)

    """
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
    """
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


res.stop()
