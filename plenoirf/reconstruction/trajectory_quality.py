from .. import table
from .. import analysis

import numpy as np
import airshower_template_generator as atg
import sparse_numeric_table as spt


def make_rectangular_table(event_table, plenoscope_pointing):
    tab = spt.cut_and_sort_table_on_indices(
        table=event_table,
        common_indices=event_table["reconstructed_trajectory"][spt.IDX],
    )
    df = spt.make_rectangular_DataFrame(tab)

    cx, cy = analysis.gamma_direction.momentum_to_cx_cy_wrt_aperture(
        momentum_x_GeV_per_c=df["primary/momentum_x_GeV_per_c"],
        momentum_y_GeV_per_c=df["primary/momentum_y_GeV_per_c"],
        momentum_z_GeV_per_c=df["primary/momentum_z_GeV_per_c"],
        plenoscope_pointing=plenoscope_pointing,
    )
    df["true_trajectory/cx_rad"] = cx
    df["true_trajectory/cy_rad"] = cy
    df["true_trajectory/x_m"] = -df["core/core_x_m"]
    df["true_trajectory/y_m"] = -df["core/core_y_m"]
    df["true_trajectory/r_m"] = np.hypot(
        df["true_trajectory/x_m"], df["true_trajectory/y_m"]
    )

    df["reconstructed_trajectory/r_m"] = np.hypot(
        df["reconstructed_trajectory/x_m"], df["reconstructed_trajectory/y_m"]
    )

    # w.r.t. source
    # -------------
    c_para, c_perp = atg.projection.project_light_field_onto_source_image(
        cer_cx_rad=df["reconstructed_trajectory/cx_rad"],
        cer_cy_rad=df["reconstructed_trajectory/cy_rad"],
        cer_x_m=0.0,
        cer_y_m=0.0,
        primary_cx_rad=df["true_trajectory/cx_rad"],
        primary_cy_rad=df["true_trajectory/cy_rad"],
        primary_core_x_m=df["true_trajectory/x_m"],
        primary_core_y_m=df["true_trajectory/y_m"],
    )

    df["trajectory/theta_para_rad"] = c_para
    df["trajectory/theta_perp_rad"] = c_perp

    df["trajectory/theta_rad"] = np.hypot(
        df["reconstructed_trajectory/cx_rad"] - df["true_trajectory/cx_rad"],
        df["reconstructed_trajectory/cy_rad"] - df["true_trajectory/cy_rad"],
    )

    df["features/image_half_depth_shift_c"] = np.hypot(
        df["features/image_half_depth_shift_cx"],
        df["features/image_half_depth_shift_cy"],
    )

    return df.to_records(index=False)


QUALITY_FEATURES = {
    "reconstructed_trajectory/r_m": {
        "scale": "linear",
        "trace": [
            [0, 0.25],
            [50, 0.8],
            [175, 1.0],
            [200, 0.8],
            [350, 0.25],
            [640, 0.0],
        ],
        "weight": 1.0,
    },
    "features/num_photons": {
        "scale": "log10",
        "trace": [
            [1, 0.0],
            [4, 1.0],
        ],
        "weight": 0.0,
    },
    "features/image_half_depth_shift_c": {
        "scale": "linear",
        "trace": [
            [0.0, 0.0],
            [1.5e-3, 1.0],
        ],
        "weight": 0.0,
    },
    "features/image_smallest_ellipse_solid_angle": {
        "scale": "log10",
        "trace": [
            [-7, 0.0],
            [-5, 1.0],
        ],
        "weight": 0.0,
    },
}


def estimate_trajectory_quality(event_frame, quality_features):
    weight_sum = 0.0
    quality = np.zeros(event_frame["idx"].shape[0])
    for qf_key in quality_features:
        weight_sum += quality_features[qf_key]["weight"]

    for qf_key in quality_features:
        qf = quality_features[qf_key]

        if qf["scale"] == "linear":
            w = event_frame[qf_key]
        elif qf["scale"] == "log10":
            w = np.log10(event_frame[qf_key])
        else:
            assert False, "Scaling unknown"

        trace = np.array(qf["trace"])
        q_comp = np.interp(x=w, xp=trace[:, 0], fp=trace[:, 1])
        q_comp *= qf["weight"] / weight_sum
        quality += q_comp
    return quality
