#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
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

train_test = json_utils.tree.read(
    os.path.join(
        res.paths["analysis_dir"],
        "0030_splitting_train_and_test_sample",
    )
)
transformed_features_dir = os.path.join(
    res.paths["analysis_dir"], "0062_transform_features"
)
passing_trigger = json_utils.tree.read(
    os.path.join(res.paths["analysis_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    os.path.join(res.paths["analysis_dir"], "0056_passing_basic_quality")
)
passing_trajectory = json_utils.tree.read(
    os.path.join(res.paths["analysis_dir"], "0059_passing_trajectory_quality")
)

random_seed = res.analysis["random_seed"]

PARTICLES = res.PARTICLES
NON_GAMMA_PARTICLES = dict(PARTICLES)
NON_GAMMA_PARTICLES.pop("gamma")

targets = {
    "energy": {
        "power_index": 0,
        "start": 1e-1,
        "stop": 1e3,
        "num_bins": 20,
        "label": "energy",
        "unit": "GeV",
    },
    "airshower_maximum": {
        "power_index": 1,
        "start": 7.5e3,
        "stop": 25e3,
        "num_bins": 20,
        "label": "airshower maximum",
        "unit": "m",
    },
}

min_number_samples = 100


def read_event_frame(
    res,
    particle_key,
    run_dir,
    transformed_features_dir,
    passing_trigger,
    passing_quality,
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

    transformed_features_path = os.path.join(
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
        uids_valid_kk = snt.logic.intersection(
            [
                passing_trigger[pk]["uid"],
                passing_quality[pk]["uid"],
                passing_trajectory[pk]["uid"],
                train_test[pk][kk],
            ]
        )
        table_kk = snt.logic.cut_and_sort_table_on_indices(
            table=airshower_table,
            common_indices=uids_valid_kk,
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
            norm_reco_radius_core_m,
            norm_reco_theta_rad,
            # f["transformed_features/combi_image_infinity_std_density"].values,
            # f[
            #    "transformed_features/combi_paxel_intensity_median_hypot"
            # ].values,
            # f["transformed_features/combi_diff_image_and_light_front"].values,
        ]
    ).T
    y = np.array(
        [
            np.log10(f["primary/energy_GeV"].values),
            np.log10(f["cherenkovpool/z_emission_p50_m"].values),
        ]
    ).T
    return x, y


train_test_gamma_energy = {}
for pk in PARTICLES:
    if pk == "gamma":
        train_test_gamma_energy[pk] = {}
        train_test_gamma_energy[pk]["train"] = train_test[pk]["train"]
        train_test_gamma_energy[pk]["test"] = train_test[pk]["test"]
    else:
        train_test_gamma_energy[pk] = {}
        train_test_gamma_energy[pk]["train"] = []
        train_test_gamma_energy[pk]["test"] = np.concatenate(
            [train_test[pk]["train"], train_test[pk]["test"]]
        )


particle_frames = {}
for pk in PARTICLES:
    particle_frames[pk] = read_event_frame(
        res=res,
        particle_key=pk,
        run_dir=res.paths["plenoirf_dir"],
        transformed_features_dir=transformed_features_dir,
        passing_trigger=passing_trigger,
        passing_quality=passing_quality,
        train_test=train_test_gamma_energy,
    )

# prepare sets
# ------------
MA = {}
for pk in PARTICLES:
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
    hidden_layer_sizes=(num_features, num_features, num_features),
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
    models[mk].fit(_X_shuffle, _y_shuffle)

    model_path = os.path.join(res.paths["out_dir"], mk + ".pkl")
    with open(model_path, "wb") as fout:
        fout.write(pickle.dumps(models[mk]))

    for pk in PARTICLES:
        _y_score = models[mk].predict(MA[pk]["test"]["x"])

        for tk in targets:
            y_true = 10 ** MA[pk]["test"]["y"][:, targets[tk]["power_index"]]
            y_score = 10 ** _y_score[:, targets[tk]["power_index"]]

            out = {}
            out["comment"] = "Reconstructed from the test-set."
            out["learner"] = mk
            out[tk] = y_score
            out["unit"] = targets[tk]["unit"]
            out["uid"] = np.array(particle_frames[pk]["test"]["uid"])

            pk_dir = os.path.join(res.paths["out_dir"], pk)
            os.makedirs(pk_dir, exist_ok=True)
            json_utils.write(os.path.join(pk_dir, tk + ".json"), out)

res.stop()
