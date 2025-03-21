import sparse_numeric_table as snt
import spherical_coordinates
import numpy as np


def cut_primary_direction_within_angle(
    event_table,
    max_angle_between_primary_and_pointing_rad,
):
    assert max_angle_between_primary_and_pointing_rad > 0.0

    np.testing.assert_array_equal(
        event_table["primary"]["uid"],
        event_table["instrument_pointing"]["uid"],
    )

    delta_rad = spherical_coordinates.angle_between_az_zd(
        azimuth1_rad=event_table["primary"]["azimuth_rad"],
        zenith1_rad=event_table["primary"]["zenith_rad"],
        azimuth2_rad=event_table["instrument_pointing"]["azimuth_rad"],
        zenith2_rad=event_table["instrument_pointing"]["zenith_rad"],
    )

    mask_inside = delta_rad <= max_angle_between_primary_and_pointing_rad
    uids_inside = event_table["primary"]["uid"][mask_inside]
    return uids_inside


def cut_quality(
    feature_table,
    max_relative_leakage,
    min_reconstructed_photons,
):
    ft = feature_table
    # size
    # ----
    mask_sufficient_size = ft["num_photons"] >= min_reconstructed_photons
    uids_sufficient_size = ft["uid"][mask_sufficient_size]

    # leakage
    # -------
    relative_leakage = (
        ft["image_smallest_ellipse_num_photons_on_edge_field_of_view"]
        / ft["num_photons"]
    )
    mask_acceptable_leakage = relative_leakage <= max_relative_leakage
    uids_acceptable_leakage = ft["uid"][mask_acceptable_leakage]

    return snt.logic.intersection(
        [uids_sufficient_size, uids_acceptable_leakage]
    )
