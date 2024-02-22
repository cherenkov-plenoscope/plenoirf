import plenoirf
import spherical_coordinates
import corsika_primary as cpw
import atmospheric_cherenkov_response as acr
import numpy as np


def draw_cherenkov_bunches_from_point_source(
    prng,
    instrument_sphere_x_m=0,
    instrument_sphere_y_m=0,
    instrument_sphere_radius_m=10,
    source_azimuth_rad=0,
    source_zenith_rad=0,
    source_distance_to_instrument_m=1e4,
    speed_of_ligth_m_per_s=3e8,
    size=1000,
):
    return cpw.testing.draw_cherenkov_bunches_from_point_source(
        instrument_sphere_x_cm=1e2 * instrument_sphere_x_m,
        instrument_sphere_y_cm=1e2 * instrument_sphere_y_m,
        instrument_sphere_radius_cm=1e2 * instrument_sphere_radius_m,
        source_azimuth_rad=source_azimuth_rad,
        source_zenith_rad=source_zenith_rad,
        source_distance_to_instrument_cm=1e2 * source_distance_to_instrument_m,
        prng=prng,
        size=size,
        speed_of_ligth_cm_per_ns=1e2 * 1e-9 * speed_of_ligth_m_per_s,
    )


def median_x_m(bunches):
    return 1e-2 * np.median(bunches[:, cpw.I.BUNCH.X_CM])


def median_y_m(bunches):
    return 1e-2 * np.median(bunches[:, cpw.I.BUNCH.Y_CM])


def median_cx_cy(bunches):
    MOMENTUM_TO_INCIDENT = -1.0
    cx = MOMENTUM_TO_INCIDENT * np.median(bunches[:, cpw.I.BUNCH.UX_1])
    cy = MOMENTUM_TO_INCIDENT * np.median(bunches[:, cpw.I.BUNCH.VY_1])
    return cx, cy


def angle_between_bunches_and_pointing(bunches, pointing):
    ccx, ccy = median_cx_cy(bunches=bunches)
    pcx, pcy = spherical_coordinates.az_zd_to_cx_cy(
        azimuth_rad=pointing["azimuth_rad"],
        zenith_rad=pointing["zenith_rad"],
    )
    return spherical_coordinates.angle_between_cx_cy(
        cx1=ccx,
        cy1=ccy,
        cx2=pcx,
        cy2=pcy,
    )


def angle_between_bunches_and_zaxis(bunches):
    ccx, ccy = median_cx_cy(bunches=bunches)
    pcx, pcy = 0.0, 0.0
    return spherical_coordinates.angle_between_cx_cy(
        cx1=ccx,
        cy1=ccy,
        cx2=pcx,
        cy2=pcy,
    )


def time_sprad_on_xy_plane_ns(bunches):
    return np.std(bunches[:, cpw.I.BUNCH.TIME_NS])


def draw_instrument_pointings(prng, max_zenith_rad, size):
    pointing_range = acr.pointing_range.PointingRange_from_cone(
        half_angel_rad=max_zenith_rad,
    )
    pointings = []
    for i in range(size):
        pointing = acr.pointing_range.draw_pointing(
            pointing_range=pointing_range, prng=prng
        )
        pointings.append(pointing)
    return pointings


def draw_instrument_positions(prng, low, high, size):
    return prng.uniform(low=low, high=high, size=(size, 2))


def test_no_translation_no_pointing():
    SPEED_OF_LIGTH_M_PER_S = 3e8

    prng = np.random.Generator(np.random.PCG64(123))

    bunches = draw_cherenkov_bunches_from_point_source(
        prng=prng, speed_of_ligth_m_per_s=SPEED_OF_LIGTH_M_PER_S
    )

    instrument_pointing = {"azimuth_rad": 0.0, "zenith_rad": 0.0}
    instrument_pointing_model = "cable_robot"
    instrument_x_m = 0.0
    instrument_y_m = 0.0

    bunches_T = plenoirf.production.transform_cherenkov_bunches.from_obervation_level_to_instrument(
        cherenkov_bunches=bunches,
        instrument_pointing=instrument_pointing,
        instrument_pointing_model=instrument_pointing_model,
        instrument_x_m=instrument_x_m,
        instrument_y_m=instrument_y_m,
        speed_of_ligth_m_per_s=SPEED_OF_LIGTH_M_PER_S,
    )

    np.testing.assert_array_almost_equal(bunches, bunches_T)


def test_only_pointing():
    BUNCH = cpw.I.BUNCH
    SPEED_OF_LIGTH_M_PER_S = 3e8

    NUM_POINTINGS = 10
    prng = np.random.Generator(np.random.PCG64(123))

    instrument_pointing_model = "cable_robot"
    instrument_pointings = draw_instrument_pointings(
        prng=prng,
        max_zenith_rad=np.deg2rad(60),
        size=NUM_POINTINGS,
    )

    time_spreads_on_instrument_aperture_plane_ns = []

    for instrument_pointing in instrument_pointings:
        bunches = draw_cherenkov_bunches_from_point_source(
            source_azimuth_rad=instrument_pointing["azimuth_rad"],
            source_zenith_rad=instrument_pointing["zenith_rad"],
            prng=prng,
            speed_of_ligth_m_per_s=SPEED_OF_LIGTH_M_PER_S,
        )
        assert angle_between_bunches_and_pointing(
            bunches=bunches, pointing=instrument_pointing
        ) < np.deg2rad(3.0)
        dt_ns = time_sprad_on_xy_plane_ns(bunches=bunches)

        bunches_T = plenoirf.production.transform_cherenkov_bunches.from_obervation_level_to_instrument(
            cherenkov_bunches=bunches,
            instrument_pointing=instrument_pointing,
            instrument_pointing_model=instrument_pointing_model,
            instrument_x_m=0.0,
            instrument_y_m=0.0,
            speed_of_ligth_m_per_s=SPEED_OF_LIGTH_M_PER_S,
        )

        I_NOT_EFFECTED = [
            BUNCH.EMISSOION_ALTITUDE_ASL_CM,
            BUNCH.BUNCH_SIZE_1,
            BUNCH.WAVELENGTH_NM,
        ]

        for II in I_NOT_EFFECTED:
            np.testing.assert_array_almost_equal(
                bunches[:, II], bunches_T[:, II]
            )

        assert np.abs(median_x_m(bunches_T)) < 1.0
        assert np.abs(median_y_m(bunches_T)) < 1.0

        assert angle_between_bunches_and_zaxis(bunches=bunches_T) < np.deg2rad(
            3.0
        )

        dt_T_ns = time_sprad_on_xy_plane_ns(bunches=bunches_T)
        time_spreads_on_instrument_aperture_plane_ns.append(dt_T_ns)
        assert dt_ns > dt_T_ns

    assert np.mean(time_spreads_on_instrument_aperture_plane_ns) < 1e-2


def test_only_translation():
    BUNCH = cpw.I.BUNCH
    SPEED_OF_LIGTH_M_PER_S = 3e8

    NUM_TRANSLATIONS = 10
    prng = np.random.Generator(np.random.PCG64(123))
    instrument_xys_m = draw_instrument_positions(
        prng=prng, low=-1e3, high=1e3, size=NUM_TRANSLATIONS
    )

    instrument_pointing = {"azimuth_rad": 0.0, "zenith_rad": 0.0}
    instrument_pointing_model = "cable_robot"

    for instrument_xy_m in instrument_xys_m:
        instrument_x_m, instrument_y_m = instrument_xy_m

        bunches = draw_cherenkov_bunches_from_point_source(
            instrument_sphere_x_m=instrument_x_m,
            instrument_sphere_y_m=instrument_y_m,
            prng=prng,
            speed_of_ligth_m_per_s=SPEED_OF_LIGTH_M_PER_S,
        )

        assert np.abs(median_x_m(bunches) - instrument_x_m) < 1.0
        assert np.abs(median_y_m(bunches) - instrument_y_m) < 1.0

        dt_ns = time_sprad_on_xy_plane_ns(bunches=bunches)

        bunches_T = plenoirf.production.transform_cherenkov_bunches.from_obervation_level_to_instrument(
            cherenkov_bunches=bunches,
            instrument_pointing=instrument_pointing,
            instrument_pointing_model=instrument_pointing_model,
            instrument_x_m=instrument_x_m,
            instrument_y_m=instrument_y_m,
            speed_of_ligth_m_per_s=SPEED_OF_LIGTH_M_PER_S,
        )

        I_NOT_EFFECTED = [
            BUNCH.UX_1,
            BUNCH.VY_1,
            BUNCH.TIME_NS,
            BUNCH.EMISSOION_ALTITUDE_ASL_CM,
            BUNCH.BUNCH_SIZE_1,
            BUNCH.WAVELENGTH_NM,
        ]

        for II in I_NOT_EFFECTED:
            np.testing.assert_array_almost_equal(
                bunches[:, II], bunches_T[:, II]
            )

        assert np.abs(median_x_m(bunches_T)) < 1.0
        assert np.abs(median_y_m(bunches_T)) < 1.0

        dt_T_ns = time_sprad_on_xy_plane_ns(bunches=bunches_T)
        np.testing.assert_approx_equal(dt_ns, dt_T_ns)


def test_both_translation_and_pointing():
    BUNCH = cpw.I.BUNCH
    SPEED_OF_LIGTH_M_PER_S = 3e8

    NUM_TRANSLATIONS = 10
    NUM_POINTINGS = 10

    prng = np.random.Generator(np.random.PCG64(123))

    instrument_pointing_model = "cable_robot"
    instrument_xys_m = draw_instrument_positions(
        prng=prng, low=-1e3, high=1e3, size=NUM_TRANSLATIONS
    )
    instrument_pointings = draw_instrument_pointings(
        prng=prng,
        max_zenith_rad=np.deg2rad(60),
        size=NUM_POINTINGS,
    )
    time_spreads_on_instrument_aperture_plane_ns = []

    for instrument_xy_m in instrument_xys_m:
        instrument_x_m, instrument_y_m = instrument_xy_m
        for instrument_pointing in instrument_pointings:
            bunches = draw_cherenkov_bunches_from_point_source(
                source_azimuth_rad=instrument_pointing["azimuth_rad"],
                source_zenith_rad=instrument_pointing["zenith_rad"],
                instrument_sphere_x_m=instrument_x_m,
                instrument_sphere_y_m=instrument_y_m,
                prng=prng,
                speed_of_ligth_m_per_s=SPEED_OF_LIGTH_M_PER_S,
            )

            assert np.abs(median_x_m(bunches) - instrument_x_m) < 1.0
            assert np.abs(median_y_m(bunches) - instrument_y_m) < 1.0
            assert angle_between_bunches_and_pointing(
                bunches=bunches, pointing=instrument_pointing
            ) < np.deg2rad(3.0)
            dt_ns = time_sprad_on_xy_plane_ns(bunches=bunches)

            bunches_T = plenoirf.production.transform_cherenkov_bunches.from_obervation_level_to_instrument(
                cherenkov_bunches=bunches,
                instrument_pointing=instrument_pointing,
                instrument_pointing_model=instrument_pointing_model,
                instrument_x_m=instrument_x_m,
                instrument_y_m=instrument_y_m,
                speed_of_ligth_m_per_s=SPEED_OF_LIGTH_M_PER_S,
            )

            I_NOT_EFFECTED = [
                BUNCH.EMISSOION_ALTITUDE_ASL_CM,
                BUNCH.BUNCH_SIZE_1,
                BUNCH.WAVELENGTH_NM,
            ]

            for II in I_NOT_EFFECTED:
                np.testing.assert_array_almost_equal(
                    bunches[:, II], bunches_T[:, II]
                )

            assert np.abs(median_x_m(bunches_T)) < 1.0
            assert np.abs(median_y_m(bunches_T)) < 1.0
            assert angle_between_bunches_and_zaxis(
                bunches=bunches_T
            ) < np.deg2rad(3.0)

            dt_T_ns = time_sprad_on_xy_plane_ns(bunches=bunches_T)
            time_spreads_on_instrument_aperture_plane_ns.append(dt_T_ns)
            assert dt_ns > dt_T_ns

    assert np.mean(time_spreads_on_instrument_aperture_plane_ns) < 1e-2


def test_timing():
    BUNCH = cpw.I.BUNCH
    SPEED_OF_LIGTH_M_PER_S = 3e8

    instrument_pointing = {"azimuth_rad": 0.0, "zenith_rad": np.deg2rad(45)}
    instrument_pointing_model = "cable_robot"
    instrument_x_m = 0.0
    instrument_y_m = 0.0

    prng = np.random.Generator(np.random.PCG64(123))

    bunches = draw_cherenkov_bunches_from_point_source(
        source_azimuth_rad=instrument_pointing["azimuth_rad"],
        source_zenith_rad=instrument_pointing["zenith_rad"],
        instrument_sphere_x_m=0.0,
        instrument_sphere_y_m=0.0,
        instrument_sphere_radius_m=10.0,
        prng=prng,
        speed_of_ligth_m_per_s=SPEED_OF_LIGTH_M_PER_S,
    )

    # light is coming from pos x
    bunches_median_cx, bunches_median_cy = median_cx_cy(bunches=bunches)
    assert 40 < np.rad2deg(bunches_median_cx) < 50
    assert -1 < np.rad2deg(bunches_median_cy) < 1

    mask_neg_x = bunches[:, BUNCH.X_CM] < -5.0
    time_ns_at_neg_x = np.median(bunches[mask_neg_x, BUNCH.TIME_NS])

    mask_pos_x = bunches[:, BUNCH.X_CM] > 5.0
    time_ns_at_pos_x = np.median(bunches[mask_pos_x, BUNCH.TIME_NS])

    # light arrives earlier in pos x and later in neg x
    assert time_ns_at_pos_x < time_ns_at_neg_x
    assert np.abs(time_ns_at_pos_x - time_ns_at_neg_x) > 15

    # now transform
    bunches_T = plenoirf.production.transform_cherenkov_bunches.from_obervation_level_to_instrument(
        cherenkov_bunches=bunches,
        instrument_pointing=instrument_pointing,
        instrument_pointing_model=instrument_pointing_model,
        instrument_x_m=instrument_x_m,
        instrument_y_m=instrument_y_m,
        speed_of_ligth_m_per_s=SPEED_OF_LIGTH_M_PER_S,
    )

    # light is coming from almost z
    bunches_T_median_cx, bunches_T_median_cy = median_cx_cy(bunches=bunches_T)
    assert -1 < np.rad2deg(bunches_T_median_cx) < 1
    assert -1 < np.rad2deg(bunches_T_median_cy) < 1

    t_mask_neg_x = bunches_T[:, BUNCH.X_CM] < -5.0
    t_time_ns_at_neg_x = np.median(bunches_T[t_mask_neg_x, BUNCH.TIME_NS])

    t_mask_pos_x = bunches_T[:, BUNCH.X_CM] > 5.0
    t_time_ns_at_pos_x = np.median(bunches_T[t_mask_pos_x, BUNCH.TIME_NS])

    # almost no difference in arrival time
    assert np.abs(t_time_ns_at_pos_x - t_time_ns_at_neg_x) < 0.001
