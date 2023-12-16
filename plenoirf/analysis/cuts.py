import sparse_numeric_table as spt
import spherical_coordinates
import numpy as np


def cut_primary_direction_within_angle(
    primary_table,
    radial_angle_deg,
    azimuth_deg,
    zenith_deg,
):
    delta_deg = np.rad2deg(spherical_coordinates.angle_between_az_zd(
        azimuth1_rad=primary_table["azimuth_rad"],
        zenith1_rad=primary_table["zenith_rad"],
        azimuth2_rad=np.deg2rad(azimuth_deg),
        zenith2_rad=np.deg2rad(zenith_deg),
    ))
    inside = delta_deg <= radial_angle_deg
    idxs_inside = primary_table[spt.IDX][inside]
    return idxs_inside


def cut_quality(
    feature_table,
    max_relative_leakage,
    min_reconstructed_photons,
):
    ft = feature_table
    # size
    # ----
    mask_sufficient_size = ft["num_photons"] >= min_reconstructed_photons
    idxs_sufficient_size = ft[spt.IDX][mask_sufficient_size]

    # leakage
    # -------
    relative_leakage = (
        ft["image_smallest_ellipse_num_photons_on_edge_field_of_view"]
        / ft["num_photons"]
    )
    mask_acceptable_leakage = relative_leakage <= max_relative_leakage
    idxs_acceptable_leakage = ft[spt.IDX][mask_acceptable_leakage]

    return spt.intersection([idxs_sufficient_size, idxs_acceptable_leakage])
