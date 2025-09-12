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
import datetime

#
import sklearn
import sklearn.neural_network
import sklearn.ensemble
import sklearn.model_selection
import sklearn.utils

#
import sebastians_matplotlib_addons as sebplt

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

transformed_features_dir = opj(
    res.paths["analysis_dir"], "0062_transform_features"
)


def make_signal_X_y_arrays(
    signal_assignment, zd, en, al, bs, signal_feature_frame, num_bootstrips
):
    uid_train, uid_test = irf.bootstripping.train_test_split(
        x=signal_assignment[zd][en][al],
        bootstrip=bs,
        num_bootstrips=num_bootstrips,
    )
    mask_train = snt.logic.make_mask_of_right_in_left(
        left_indices=signal_feature_frame["uid"],
        right_indices=uid_train,
    )
    X_train = signal_feature_frame[mask_train]
    X_train = X_train.drop(columns=["uid"])
    y_train = np.ones(shape=X_train.shape[0])

    mask_test = snt.logic.make_mask_of_right_in_left(
        left_indices=signal_feature_frame["uid"],
        right_indices=uid_test,
    )
    X_test = signal_feature_frame[mask_test]

    return X_train, y_train, X_test


def make_X_array(feature_frame, uid):
    mask = snt.logic.make_mask_of_right_in_left(
        left_indices=feature_frame["uid"],
        right_indices=uid,
    )
    X = feature_frame[mask]

    X = X.drop(columns=["uid"])
    return X


def make_background_X_y_arrays(background_feature_frames):
    bff = background_feature_frames
    xlist = []
    for pk in background_feature_frames:
        xlist.append(bff[pk])
    X = pandas.concat(xlist)
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
        table = arc.query(indices=uid_cut)

    frame = snt.logic.make_rectangular_DataFrame(table)
    frame = frame.drop(
        columns=[
            "transformed_features/aperture_num_islands_watershed_rel_thr_2",
            "transformed_features/aperture_num_islands_watershed_rel_thr_4",
            "transformed_features/aperture_num_islands_watershed_rel_thr_8",
        ]
    )
    return frame


def assignment_union_en_al(en_al_assignment):
    out = []
    for en in range(len(en_al_assignment)):
        al_assignment = en_al_assignment[en]
        for al in range(len(al_assignment)):
            out.append(al_assignment[al])
    return snt.logic.union(*out)


bins = irf.summary.estimator.binning.Bins.from_path(
    path=opj(res.paths["analysis_dir"], "0081_estimator_binning", "binning")
)
signal_and_background = json_utils.read(
    opj(
        res.paths["analysis_dir"],
        "0081_estimator_binning",
        "signal_and_background.json",
    )
)
SIGNAL = signal_and_background["signal"]
BACKGROUND = signal_and_background["background"]

passing_cuts = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0081_estimator_binning", "passing_cuts")
)

signal_assignment = json_utils.tree.read(
    opj(
        res.paths["analysis_dir"],
        "0081_estimator_binning",
        "assignment",
        "signal",
    )
)

background_assignment = json_utils.tree.read(
    opj(
        res.paths["analysis_dir"],
        "0081_estimator_binning",
        "assignment",
        "background",
    )
)

feature_frames = {}
feature_frames["signal"] = {}
for pk in SIGNAL:
    feature_frames["signal"][pk] = make_feature_frame(
        res=res, pk=pk, uid_cut=passing_cuts[pk]["uid"]
    )
feature_frames["background"] = {}
for pk in BACKGROUND:
    feature_frames["background"][pk] = make_feature_frame(
        res=res, pk=pk, uid_cut=passing_cuts[pk]["uid"]
    )


MIN_NUM_SIGNAL = 25
NUM_BOOTSTRIPS = 3
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

rk = "raw"

for zd in range(bins.zenith["num"]):
    for en in range(bins.energy["num"]):
        for al in range(bins.altitude["num"]):

            num_signal = 0
            for sk in SIGNAL:
                num_signal += len(signal_assignment[sk][rk][zd][en][al])

            print(
                f"zd{zd:02d} en{en:02d} al{al:02d}, num. signal: {num_signal:d}"
            )

            if num_signal <= MIN_NUM_SIGNAL:
                continue

            esti_zd_en_zl_dir = opj(
                res.paths["out_dir"],
                "estimator",
                f"zd{zd:02d}",
                f"en{en:02d}",
                f"al{al:02d}",
            )

            for bs in range(NUM_BOOTSTRIPS):
                bs_dir = opj(esti_zd_en_zl_dir, f"bs{bs:02d}")
                bs_tmp_dir = bs_dir + ".part"

                if os.path.exists(bs_dir):
                    continue

                print(f"bootstrip {bs:d} of {NUM_BOOTSTRIPS:d}")
                os.makedirs(bs_tmp_dir)

                # train, test, 'probe' split
                # --------------------------
                uids = {}
                uids["signal"] = {}
                for sk in SIGNAL:
                    uids["signal"][sk] = {}
                    (
                        uids["signal"][sk]["train"],
                        uids["signal"][sk]["test"],
                    ) = irf.bootstripping.train_test_split(
                        x=signal_assignment[sk][rk][zd][en][al],
                        bootstrip=bs,
                        num_bootstrips=NUM_BOOTSTRIPS,
                    )

                    _uids_same_zd = assignment_union_en_al(
                        en_al_assignment=signal_assignment[sk][rk][zd],
                    )
                    uids["signal"][sk]["probe"] = snt.logic.difference(
                        _uids_same_zd, uids["signal"][sk]["train"]
                    )

                uids["background"] = {}
                for bg in BACKGROUND:
                    uids["background"][bg] = {}
                    (
                        uids["background"][bg]["train"],
                        uids["background"][bg]["test"],
                    ) = irf.bootstripping.train_test_split(
                        x=background_assignment[bg][rk][zd],
                        bootstrip=bs,
                        num_bootstrips=NUM_BOOTSTRIPS,
                    )

                    _uids_same_zd = background_assignment[bg][rk][zd]
                    uids["background"][bg]["probe"] = snt.logic.difference(
                        _uids_same_zd, uids["background"][bg]["train"]
                    )

                json_utils.write(
                    opj(bs_tmp_dir, "uids.json"),
                    uids,
                )

                # Make training X and y arrays
                # ----------------------------

                X_train = []
                y_train = []
                for pk in SIGNAL:
                    X_pk = make_X_array(
                        feature_frame=feature_frames["signal"][pk],
                        uid=uids["signal"][pk]["train"],
                    )
                    y_pk = np.ones(shape=X_pk.shape[0])
                    X_train.append(X_pk)
                    y_train.append(y_pk)

                for pk in BACKGROUND:
                    X_pk = make_X_array(
                        feature_frame=feature_frames["background"][pk],
                        uid=uids["background"][pk]["train"],
                    )
                    y_pk = np.zeros(shape=X_pk.shape[0])
                    X_train.append(X_pk)
                    y_train.append(y_pk)

                X_train = pandas.concat(X_train)
                y_train = np.concat(y_train)

                X_train_shuffle, y_train_shuffle = sklearn.utils.shuffle(
                    X_train,
                    y_train,
                    random_state=res.analysis["random_seed"],
                )
                del X_train, y_train

                # train classifier
                # ----------------

                print(
                    f"Training classifier on {X_train_shuffle.shape[0]:d} samples"
                )

                _T_start = datetime.datetime.now()

                classifier = sklearn.neural_network.MLPClassifier(
                    **mlpclassifier_kwargs
                )
                classifier.fit(X_train_shuffle, y_train_shuffle)

                _T_stop = datetime.datetime.now()
                _dT = _T_stop - _T_start

                mlpc_path = opj(bs_tmp_dir, "classifier.pkl")
                with rnw.open(mlpc_path, "wb") as fout:
                    fout.write(pickle.dumps(classifier))

                print(f"Training took {_dT.total_seconds():.1f}s")
                json_utils.write(
                    opj(bs_tmp_dir, "classifier.pkl.time.json"),
                    {"timedelta_s": _dT.total_seconds()},
                )

                # Make probing X arrays
                # ---------------------

                # apply classifier
                # ----------------
                y_probability = {}
                for pk in SIGNAL:
                    X_probe = make_X_array(
                        feature_frame=feature_frames["signal"][pk],
                        uid=uids["signal"][pk]["probe"],
                    )
                    y_probability[pk] = classifier.predict_proba(X_probe)
                for pk in BACKGROUND:
                    X_probe = make_X_array(
                        feature_frame=feature_frames["background"][pk],
                        uid=uids["background"][pk]["probe"],
                    )
                    y_probability[pk] = classifier.predict_proba(X_probe)

                json_utils.write(
                    opj(bs_tmp_dir, "y_probability.json"), y_probability
                )

                os.rename(src=bs_tmp_dir, dst=bs_dir)


"""
                y_proba_bootstripping_mean = {}
                for pk in SIGNAL + BACKGROUND:
                    y_proba_bootstripping_mean[pk] = {}

                for bs in range(num_bootstrips):
                    mlpc_zd_en_zl_bs_dir = opj(
                        esti_zd_en_zl_dir, f"bs{bs:02d}"
                    )

                    print(zd, en, al, bs, "num", num_signal)

                    uids = {"gamma": {}}
                    uids["gamma"]["train"], _ = (
                        irf.bootstripping.train_test_split(
                            x=signal_assignment["gamma"][rk][zd][en][
                                al
                            ],
                            bootstrip=bs,
                            num_bootstrips=num_bootstrips,
                        )
                    )
                    uids["gamma"]["test"] = np.asarray(
                        list(
                            set.difference(
                                set(passing_cuts["gamma"]["uid"]),
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
                                x=passing_cuts[ck]["uid"],
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
                            uid_train=uids[pk]["train"],
                            uid_test=uids[pk]["test"],
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
                            esti_zd_en_zl_dir,
                            f"{pk:s}_y_proba_bootstripping_mean.snt.zip",
                        ),
                        "w",
                        dtypes_and_index_key_from=arr,
                    ) as f:
                        f.append_table(arr)
"""

res.stop()

# ConvergenceWarning
