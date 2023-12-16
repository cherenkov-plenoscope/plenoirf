import plenoirf


def test_guess_num_off_regions():
    assert 3 == plenoirf.summary.guess_num_offregions(
        fov_radius_deg=3.25,
        gamma_resolution_radius_at_energy_threshold_deg=0.8,
        onregion_radius_deg=0.8,
        fraction_of_fov_being_useful=0.33,
    )

    assert 12 == plenoirf.summary.guess_num_offregions(
        fov_radius_deg=3.25,
        gamma_resolution_radius_at_energy_threshold_deg=0.8,
        onregion_radius_deg=0.4,
        fraction_of_fov_being_useful=0.33,
    )

    assert 50 == plenoirf.summary.guess_num_offregions(
        fov_radius_deg=3.25,
        gamma_resolution_radius_at_energy_threshold_deg=0.8,
        onregion_radius_deg=0.2,
        fraction_of_fov_being_useful=0.33,
    )

    assert 1 == plenoirf.summary.guess_num_offregions(
        fov_radius_deg=3.25,
        gamma_resolution_radius_at_energy_threshold_deg=0.8,
        onregion_radius_deg=0.8,
        fraction_of_fov_being_useful=0.12,
    )

    assert 5 == plenoirf.summary.guess_num_offregions(
        fov_radius_deg=3.25,
        gamma_resolution_radius_at_energy_threshold_deg=0.8,
        onregion_radius_deg=0.4,
        fraction_of_fov_being_useful=0.12,
    )

    assert 18 == plenoirf.summary.guess_num_offregions(
        fov_radius_deg=3.25,
        gamma_resolution_radius_at_energy_threshold_deg=0.8,
        onregion_radius_deg=0.2,
        fraction_of_fov_being_useful=0.12,
    )
