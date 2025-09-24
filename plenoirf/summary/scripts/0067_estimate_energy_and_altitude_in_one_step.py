#!/usr/bin/python
import sys
import copy
import plenoirf as irf
import confusion_matrix
import sparse_numeric_table as snt
import rename_after_writing as rnw
import os
from os.path import join as opj
import pandas
import numpy as np
import pickle
import json_utils
import binning_utils
import sebastians_matplotlib_addons as sebplt

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

"""
transformed_features_dir = opj(
    res.paths["analysis_dir"], "0062_transform_features"
)

zenith_bin = res.zenith_binning("twice")

_energy_bin = res.energy_binning(key="trigger_acceptance_onregion")
energy_bin = binning_utils.Binning(
    bin_edges=np.geomspace(
        _energy_bin["start"], _energy_bin["stop"], zenith_bin["num"] + 1
    )
)

altitude_bin = binning_utils.Binning(
    bin_edges=np.geomspace(5e3, 25e3, energy_bin["num"] + 1)
)


def make_passing_cuts(script_resources, particles):
    res = script_resources
    cache_dir = opj(res.paths["out_dir"], "passing_cuts.__cache__")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

        passing_trigger = json_utils.tree.Tree(
            opj(res.paths["analysis_dir"], "0055_passing_trigger")
        )
        passing_quality = json_utils.tree.Tree(
            opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
        )

        for pk in particles:
            passing_cuts_pk = snt.logic.intersection(
                passing_trigger[pk]["uid"],
                passing_quality[pk]["uid"],
            )
            with rnw.open(opj(cache_dir, pk + ".json"), "wt") as f:
                f.write(json_utils.dumps({"uid": passing_cuts_pk}))
    return json_utils.tree.Tree(cache_dir)


bins = irf.summary.estimator.Bins(
    zenith_bin["edges"], energy_bin["edges"], altitude_bin["edges"]
)


def make_overlaps(size):
    overlaps = []
    for ipivot in range(size):
        indices = []
        weights = []
        for ii in range(ipivot - 1, ipivot + 2):
            if ii >= 0 and ii < size:
                indices.append(ii)
                weights.append(1.0 if ii == ipivot else 0.5)
        overlaps.append({"indices": indices, "weights": weights})
    return overlaps


def smoothen_uid_assign_zenith_energy_altitude(
    assignment,
    zenith_bin,
    zenith_bin_overlaps,
    energy_bin,
    energy_bin_overlaps,
    altitude_bin,
    altitude_bin_overlaps,
):
    smo = irf.summary.all_in_one_estimator.make_cube_of_lists(
        shape=(zenith_bin["num"], energy_bin["num"], altitude_bin["num"]),
        default=None,
    )
    for zdp in range(zenith_bin["num"]):
        for enp in range(energy_bin["num"]):
            for alp in range(altitude_bin["num"]):

                smo[zdp][enp][alp] = set()
                for zd in zenith_bin_overlaps[zdp]["indices"]:
                    for en in energy_bin_overlaps[enp]["indices"]:
                        for al in altitude_bin_overlaps[alp]["indices"]:
                            to_add = set(assignment[zd][en][al])
                            smo[zdp][enp][alp] = set.union(
                                smo[zdp][enp][alp],
                                to_add,
                            )
                smo[zdp][enp][alp] = np.asarray(
                    list(smo[zdp][enp][alp]), dtype=int
                )
                smo[zdp][enp][alp] = np.sort(smo[zdp][enp][alp])
    return smo


def smoothen_uid_assign_zenith(
    assignment,
    zenith_bin,
    zenith_bin_overlaps,
):
    smo = [None for zd in range(zenith_bin["num"])]
    for zdp in range(zenith_bin["num"]):

        smo[zdp] = set()
        for zd in zenith_bin_overlaps[zdp]["indices"]:
            to_add = set(assignment[zd])
            smo[zdp] = set.union(smo[zdp], to_add)

        smo[zdp] = np.asarray(list(smo[zdp]), dtype=int)
        smo[zdp] = np.sort(smo[zdp])
    return smo


SIGNAL = ["gamma"]
BACKGROUND = ["proton", "helium"]

FINAL_UIDS = make_passing_cuts(
    script_resources=res, particles=SIGNAL + BACKGROUND
)
zenith_bin_overlaps = make_overlaps(size=zenith_bin["num"])
energy_bin_overlaps = make_overlaps(size=energy_bin["num"])
altitude_bin_overlaps = make_overlaps(size=altitude_bin["num"])


signal_raw_assignment = {}
signal_assignment = {}
for pk in SIGNAL:
    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={
                "primary": ["uid", "energy_GeV"],
                "instrument_pointing": ["uid", "zenith_rad"],
                "cherenkovpool": ["uid", "z_emission_p50_m"],
            },
            indices=FINAL_UIDS[pk],
            sort=True,
        )

    signal_raw_assignment[pk] = (
        irf.summary.all_in_one_estimator.assign_uids_to_zenith_energy_altitude(
            event_table=event_table,
            bins=bins,
        )
    )
    signal_assignment[pk] = smoothen_uid_assign_zenith_energy_altitude(
        assignment=signal_raw_assignment[pk],
        zenith_bin=zenith_bin,
        zenith_bin_overlaps=zenith_bin_overlaps,
        energy_bin=energy_bin,
        energy_bin_overlaps=energy_bin_overlaps,
        altitude_bin=altitude_bin,
        altitude_bin_overlaps=altitude_bin_overlaps,
    )


background_raw_assignment = {}
background_assignment = {}
for pk in BACKGROUND:
    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={
                "instrument_pointing": ["uid", "zenith_rad"],
            }
        )
        event_table = snt.logic.cut_table_on_indices(
            event_table,
            common_indices=FINAL_UIDS[pk],
        )
    background_raw_assignment[pk] = irf.summary.all_in_one_estimator.assign_uids_zenith(
        event_table=event_table,
        bins=bins,
    )
    background_assignment[pk] = smoothen_uid_assign_zenith(
        assignment=background_raw_assignment[pk],
        zenith_bin=zenith_bin,
        zenith_bin_overlaps=zenith_bin_overlaps,
    )


def make_signal_X_y_arrays(
    signal_assignment, zd, en, al, bs, signal_feature_frame, num_bootstrips
):
    uids_train, uids_test = irf.bootstripping.train_test_split(
        x=signal_assignment[zd][en][al],
        bootstrip=bs,
        num_bootstrips=num_bootstrips,
    )
    mask_train = snt.logic.make_mask_of_right_in_left(
        left_indices=signal_feature_frame["uid"],
        right_indices=uids_train,
    )
    X_train = signal_feature_frame[mask_train]
    X_train = X_train.drop(columns=["uid"])
    y_train = np.ones(shape=X_train.shape[0])

    mask_test = snt.logic.make_mask_of_right_in_left(
        left_indices=signal_feature_frame["uid"],
        right_indices=uids_test,
    )
    X_test = signal_feature_frame[mask_test]

    return X_train, y_train, X_test


def make_X_array(feature_frame, uids_train, uids_test):
    mask_train = snt.logic.make_mask_of_right_in_left(
        left_indices=feature_frame["uid"],
        right_indices=uids_train,
    )
    X_train = feature_frame[mask_train]

    mask_test = snt.logic.make_mask_of_right_in_left(
        left_indices=feature_frame["uid"],
        right_indices=uids_test,
    )
    X_test = feature_frame[mask_test]

    X_train = X_train.drop(columns=["uid"])
    X_test = X_test.drop(columns=["uid"])
    return X_train, X_test


def make_background_X_y_arrays(background_feature_frames):
    bff = background_feature_frames
    X = pandas.concat([bff["proton"], bff["helium"]])
    X = X.drop(columns=["uid"])
    y = np.zeros(shape=X.shape[0])
    return X, y


def make_feature_frame(res, pk, uid_cut):
    transformed_features_path = opj(
        transformed_features_dir,
        pk,
        "transformed_features.zip",
    )
    with snt.open(transformed_features_path, "r") as arc:
        table = arc.query()

    table = snt.logic.cut_and_sort_table_on_indices(
        table=table, common_indices=uid_cut
    )
    frame = snt.logic.make_rectangular_DataFrame(table)
    frame = frame.drop(
        columns=[
            "transformed_features/aperture_num_islands_watershed_rel_thr_2",
            "transformed_features/aperture_num_islands_watershed_rel_thr_4",
            "transformed_features/aperture_num_islands_watershed_rel_thr_8",
        ]
    )
    return frame


feature_frames = {}
for pk in SIGNAL + BACKGROUND:
    feature_frames[pk] = make_feature_frame(
        res=res, pk=pk, uid_cut=FINAL_UIDS[pk]
    )


MIN_NUM_SIGNAL = 25
num_bootstrips = 10
num_features = 25

mlpclassifier_kwargs = {
    "solver": "lbfgs",
    "alpha": 0.0001,
    "hidden_layer_sizes": (num_features, num_features, num_features),
    "random_state": res.analysis["random_seed"],
    "verbose": False,
    "max_iter": 200,
    "learning_rate_init": 0.001,
}

background_X_all, background_y_all = make_background_X_y_arrays(
    background_feature_frames=feature_frames,
)


for zd in range(zenith_bin["num"]):
    for en in range(energy_bin["num"]):
        for al in range(altitude_bin["num"]):

            num_signal = len(signal_assignment["gamma"][zd][en][al])
            if num_signal > MIN_NUM_SIGNAL:
                mlpc_zd_en_zl_dir = opj(
                    res.paths["out_dir"],
                    "mlpc",
                    f"zd{zd:02d}",
                    f"en{en:02d}",
                    f"al{al:02}",
                )
                if not os.path.exists(mlpc_zd_en_zl_dir):

                    y_proba_bootstripping_mean = {}
                    for pk in SIGNAL + BACKGROUND:
                        y_proba_bootstripping_mean[pk] = {}

                    for bs in range(num_bootstrips):
                        mlpc_zd_en_zl_bs_dir = opj(
                            mlpc_zd_en_zl_dir, f"bs{bs:02d}"
                        )

                        print(zd, en, al, bs, "num", num_signal)
                        os.makedirs(mlpc_zd_en_zl_bs_dir)

                        uids = {"gamma": {}}
                        uids["gamma"]["train"], _ = (
                            irf.bootstripping.train_test_split(
                                x=signal_assignment["gamma"][zd][en][al],
                                bootstrip=bs,
                                num_bootstrips=num_bootstrips,
                            )
                        )
                        uids["gamma"]["test"] = np.asarray(
                            list(
                                set.difference(
                                    set(FINAL_UIDS["gamma"]),
                                    set(uids["gamma"]["train"]),
                                )
                            )
                        )
                        assert (
                            len(
                                set.intersection(
                                    set(uids["gamma"]["test"]),
                                    set(uids["gamma"]["train"]),
                                )
                            )
                            == 0
                        )

                        for ck in BACKGROUND:
                            uids[ck] = {}
                            uids[ck]["train"], uids[ck]["test"] = (
                                irf.bootstripping.train_test_split(
                                    x=FINAL_UIDS[ck],
                                    bootstrip=bs,
                                    num_bootstrips=num_bootstrips,
                                )
                            )

                        with rnw.open(
                            opj(mlpc_zd_en_zl_bs_dir, "uid.json"), "wt"
                        ) as f:
                            f.write(json_utils.dumps(uids))

                        X = {}
                        for pk in SIGNAL + BACKGROUND:
                            X[pk] = {}
                            X[pk]["train"], X[pk]["test"] = make_X_array(
                                feature_frame=feature_frames[pk],
                                uids_train=uids[pk]["train"],
                                uids_test=uids[pk]["test"],
                            )
                        y = {}
                        for pk in SIGNAL:
                            y[pk] = {}
                            y[pk]["train"] = np.ones(
                                shape=X[pk]["train"].shape[0]
                            )
                        for pk in BACKGROUND:
                            y[pk] = {}
                            y[pk]["train"] = np.zeros(
                                shape=X[pk]["train"].shape[0]
                            )

                        X_train = pandas.concat([X[pk]["train"] for pk in X])
                        y_train = np.concat([y[pk]["train"] for pk in y])

                        X_train_shuffle, y_train_shuffle = (
                            sklearn.utils.shuffle(
                                X_train,
                                y_train,
                                random_state=res.analysis["random_seed"],
                            )
                        )

                        mlpc = sklearn.neural_network.MLPClassifier(
                            **mlpclassifier_kwargs
                        )
                        mlpc.fit(X_train_shuffle, y_train_shuffle)

                        mlpc_path = opj(mlpc_zd_en_zl_bs_dir, "mlpc.pkl")
                        with rnw.open(mlpc_path, "wb") as fout:
                            fout.write(pickle.dumps(mlpc))

                        for pk in SIGNAL + BACKGROUND:
                            _y_proba_pk = mlpc.predict_proba(X[pk]["test"])
                            assert (
                                _y_proba_pk.shape[0]
                                == uids[pk]["test"].shape[0]
                            )

                            assert (
                                snt.logic.intersection(
                                    [uids[pk]["test"], uids[pk]["train"]]
                                ).shape[0]
                                == 0
                            )

                            for iii, uid in enumerate(uids[pk]["test"]):
                                _y_proba_pk_to_be_signal = _y_proba_pk[iii][0]
                                assert uid not in uids[pk]["train"]

                                if uid not in y_proba_bootstripping_mean[pk]:
                                    y_proba_bootstripping_mean[pk][
                                        int(uid)
                                    ] = [iii]
                                else:
                                    y_proba_bootstripping_mean[pk][
                                        int(uid)
                                    ].append(iii)

                    for pk in SIGNAL + BACKGROUND:
                        arr = snt.SparseNumericTable(
                            dtypes={
                                "y_proba": [
                                    ("uid", "<u8"),
                                    ("p50", "<f4"),
                                    ("s68", "<f4"),
                                ]
                            },
                            index_key="uid",
                        )
                        for uid in y_proba_bootstripping_mean[pk]:
                            yyy = y_proba_bootstripping_mean[pk][uid]
                            assert len(yyy) == num_bootstrips
                            print("yyy", yyy, np.percentile(yyy, q=50))
                            arr["y_proba"].append(
                                {
                                    "uid": uid,
                                    "p50": np.percentile(yyy, 50),
                                    "s68": np.percentile(yyy, 50 + 34)
                                    - np.percentile(yyy, 50 - 34),
                                }
                            )

                        with snt.open(
                            opj(
                                mlpc_zd_en_zl_dir,
                                f"{pk:s}_y_proba_bootstripping_mean.snt.zip",
                            ),
                            "w",
                            dtypes_and_index_key_from=arr,
                        ) as f:
                            f.append_table(arr)
"""
res.stop()

# ConvergenceWarning
