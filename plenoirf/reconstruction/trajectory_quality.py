from .. import event_table as table
from .. import analysis
from .. import production

import numpy as np
import airshower_template_generator as atg
import sparse_numeric_table as snt
import homogeneous_transformation
import atmospheric_cherenkov_response
import corsika_primary as cpw


def from_observation_level_to_instrument(
    particle_momentum,
    particle_core_position,
    instrument_pointing,
    instrument_pointing_model,
):
    cm2m = 1e2
    BUNCH = cpw.cherenkov_bunches.BUNCH
    particle_bunches = np.zeros(shape=(1, len(BUNCH.DTYPE)), dtype=np.float32)
    particle_bunches[0, BUNCH.X_CM] = particle_core_position[0] * cm2m
    particle_bunches[0, BUNCH.Y_CM] = particle_core_position[1] * cm2m
    particle_bunches[0, BUNCH.UX_1] = 0.0
    particle_bunches[0, BUNCH.VY_1] = 0.0
    particle_bunches[0, TIME_NS] = 0.0
    particle_bunches[0, EMISSOION_ALTITUDE_ASL_CM] = 0.0
    particle_bunches[0, BUNCH_SIZE_1] = 0.0
    particle_bunches[0, WAVELENGTH_NM] = 0.0

    production.transform_cherenkov_bunches.from_obervation_level_to_instrument(
        cherenkov_bunches=particle_bunches,
        instrument_pointing=instrument_pointing,
        instrument_pointing_model=instrument_pointing_model,
        instrument_x_m=0.0,
        instrument_y_m=0.0,
        speed_of_ligth_m_per_s=1.0,
    )


def make_rectangular_table(event_table, instrument_pointing_model):
    tab = snt.logic.cut_and_sort_table_on_indices(
        table=event_table,
        common_indices=event_table["reconstructed_trajectory"]["uid"],
    )
    df = snt.logic.make_rectangular_DataFrame(tab, index_key="uid")

    df["reconstructed_trajectory/r_m"] = np.hypot(
        df["reconstructed_trajectory/x_m"],
        df["reconstructed_trajectory/y_m"],
    )

    df["features/image_half_depth_shift_c"] = np.hypot(
        df["features/image_half_depth_shift_cx"],
        df["features/image_half_depth_shift_cy"],
    )

    """
    cx, cy = analysis.gamma_direction.momentum_to_cx_cy_wrt_aperture(
        momentum_x_GeV_per_c=df["primary/momentum_x_GeV_per_c"],
        momentum_y_GeV_per_c=df["primary/momentum_y_GeV_per_c"],
        momentum_z_GeV_per_c=df["primary/momentum_z_GeV_per_c"],
        plenoscope_pointing=instrument_pointing,
    )
    df["true_trajectory/cx_rad"] = cx
    df["true_trajectory/cy_rad"] = cy
    df["true_trajectory/x_m"] = -df["core/core_x_m"]
    df["true_trajectory/y_m"] = -df["core/core_y_m"]
    df["true_trajectory/r_m"] = np.hypot(
        df["true_trajectory/x_m"], df["true_trajectory/y_m"]
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
    """

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
    quality = np.zeros(event_frame["uid"].shape[0])
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
