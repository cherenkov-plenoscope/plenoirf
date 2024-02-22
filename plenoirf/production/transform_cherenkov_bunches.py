import homogeneous_transformation
import atmospheric_cherenkov_response as acr
import spherical_coordinates
import corsika_primary as cpw
import plenopy
import numpy as np


def from_obervation_level_to_instrument(
    cherenkov_bunches,
    instrument_pointing,
    instrument_pointing_model,
    instrument_x_m,
    instrument_y_m,
    speed_of_ligth_m_per_s,
):
    CM_PER_M = 1e2
    S_PER_NS = 1e-9

    instrument_x_cm = CM_PER_M * instrument_x_m
    instrument_y_cm = CM_PER_M * instrument_y_m
    speed_of_ligth_cm_per_ns = CM_PER_M * S_PER_NS * speed_of_ligth_m_per_s

    rot_civil = acr.pointing.pointing_to_civil_rotation(
        pointing=instrument_pointing,
        mount=instrument_pointing_model,
    )
    t_civil = {
        "rot": rot_civil,
        "pos": [instrument_x_cm, instrument_y_cm, 0],
    }
    homtra_cm = homogeneous_transformation.compile(t_civil=t_civil)

    return transform_cherenkov_bunches(
        cherenkov_bunches=cherenkov_bunches,
        homtra_cm=homtra_cm,
        speed_of_ligth_cm_per_ns=speed_of_ligth_cm_per_ns,
    )


def transform_cherenkov_bunches(
    cherenkov_bunches, homtra_cm, speed_of_ligth_cm_per_ns
):
    cer = cherenkov_bunches.copy()

    # bunches to rays
    cer_dir = make_momentum_direction(cherenkov_bunches=cer)

    cer_sup_cm = make_support(cherenkov_bunches=cer)

    # transform
    (
        t_cer_sup_cm,
        t_cer_dir,
    ) = homogeneous_transformation.transform_ray_inverse(
        t=homtra_cm, ray_supports=cer_sup_cm, ray_directions=cer_dir
    )

    d_cm = distance_to_reach_xy_plane(z=t_cer_sup_cm[:, 2], cz=t_cer_dir[:, 2])

    d_time_ns = d_cm / speed_of_ligth_cm_per_ns

    t_cer_support_on_aperture_plane_cm = t_cer_sup_cm + np.multiply(
        t_cer_dir, d_cm[:, np.newaxis]
    )

    # rays to bunches
    cer[:, cpw.I.BUNCH.UX_1] = t_cer_dir[:, 0]
    cer[:, cpw.I.BUNCH.VY_1] = t_cer_dir[:, 1]
    cer[:, cpw.I.BUNCH.X_CM] = t_cer_support_on_aperture_plane_cm[:, 0]
    cer[:, cpw.I.BUNCH.Y_CM] = t_cer_support_on_aperture_plane_cm[:, 1]
    cer[:, cpw.I.BUNCH.TIME_NS] += d_time_ns
    return cer


def make_support(cherenkov_bunches):
    x = cherenkov_bunches[:, cpw.I.BUNCH.X_CM]
    y = cherenkov_bunches[:, cpw.I.BUNCH.Y_CM]
    z = np.zeros(cherenkov_bunches.shape[0])
    return np.c_[x, y, z]


def make_momentum_direction(cherenkov_bunches):
    TOWARDS_XY_PLANE = -1.0
    ux = cherenkov_bunches[:, cpw.I.BUNCH.UX_1]
    vy = cherenkov_bunches[:, cpw.I.BUNCH.VY_1]
    z = spherical_coordinates.restore_cz(cx=ux, cy=vy)
    return np.c_[cx, cy, TOWARDS_XY_PLANE * cz]


def distance_to_reach_xy_plane(cz, z):
    d = (-1.0) * z / cz
    return d
