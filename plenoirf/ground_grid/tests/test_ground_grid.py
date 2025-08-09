import plenoirf.ground_grid
import numpy as np
import corsika_primary as cpw
import pytest


def test_init():
    prng = np.random.Generator(np.random.PCG64(1))

    config = {
        "bin_width_m": 1.0,
        "num_bins_each_axis": 3,
        "center_x_m": 0.0,
        "center_y_m": 0.0,
    }
    groundgrid = plenoirf.ground_grid.GroundGrid(**config)
    SIZE = 1_000

    GGH = plenoirf.ground_grid.GGH()

    for event in range(20):
        GGH.init_groundgrid(groundgrid=groundgrid)

        GGH.assign_cherenkov_bunches(
            cherenkov_bunches=cpw.testing.draw_cherenkov_bunches_from_point_source(
                instrument_sphere_x_cm=0.0,
                instrument_sphere_y_cm=0.0,
                instrument_sphere_radius_cm=50.0,
                source_azimuth_rad=0.0,
                source_zenith_rad=0.0,
                source_distance_to_instrument_cm=1e4,
                prng=prng,
                size=SIZE,
                bunch_size_low=0.9,
                bunch_size_high=1.0,
            )
        )

        histogram = GGH.get_histogram()

        np.testing.assert_almost_equal(
            actual=np.sum(histogram["weight_photons"]),
            desired=0.95 * SIZE,
            decimal=-1,
        )
        np.testing.assert_array_equal(
            actual=histogram["x_bin"],
            desired=np.ones(1),
        )
        np.testing.assert_array_equal(
            actual=histogram["y_bin"],
            desired=np.ones(1),
        )
    GGH.close()


def bunches_in_grid(x_m, y_m):
    assert len(x_m) == len(y_m)
    SIZE = len(x_m)
    BUNCH = cpw.cherenkov_bunches.BUNCH
    bunches = np.zeros(shape=(SIZE, len(BUNCH.DTYPE)), dtype=np.float32)
    bunches[:, BUNCH.X_CM] = 1e2 * x_m
    bunches[:, BUNCH.Y_CM] = 1e2 * y_m

    bunches[:, BUNCH.UX_1] = 0.0
    bunches[:, BUNCH.X_CM] = 0.0

    bunches[:, BUNCH.TIME_NS] = 0.0
    bunches[:, BUNCH.EMISSOION_ALTITUDE_ASL_CM] = 1e2 * 1e4
    bunches[:, BUNCH.BUNCH_SIZE] = 1.0
    bunches[:, BUNCH.WAVELENGTH_NM] = 533.0
    return bunches


def test_zenith_45deg():
    prng = np.random.Generator(np.random.PCG64(1))

    config = {
        "bin_width_m": 1.0,
        "num_bins_each_axis": 101,
        "center_x_m": 0.0,
        "center_y_m": 0.0,
    }
    groundgrid = plenoirf.ground_grid.GroundGrid(**config)
    BLOCK_SIZE = 10_000
    NUM_BLOCKS = 10
    ZENITH_RAD = np.deg2rad(45.0)

    illium_radius_m = 50

    GGH = plenoirf.ground_grid.GGH()
    GGH.init_groundgrid(groundgrid=groundgrid)

    for azimuth_rad in np.linspace(0.0, np.pi, NUM_BLOCKS, endpoint=False):
        GGH.assign_cherenkov_bunches(
            cherenkov_bunches=cpw.testing.draw_cherenkov_bunches_from_point_source(
                instrument_sphere_x_cm=0.0,
                instrument_sphere_y_cm=0.0,
                instrument_sphere_radius_cm=illium_radius_m * 1e2,
                source_azimuth_rad=azimuth_rad,
                source_zenith_rad=ZENITH_RAD,
                source_distance_to_instrument_cm=1e8,
                prng=prng,
                size=BLOCK_SIZE,
                bunch_size_low=0.9,
                bunch_size_high=1.0,
            )
        )
    histogram = GGH.get_histogram()
    GGH.close()

    np.testing.assert_almost_equal(
        actual=np.sum(histogram["weight_photons"])
        / (BLOCK_SIZE * NUM_BLOCKS)
        * np.cos(ZENITH_RAD),
        desired=0.95,
        decimal=1,
    )

    illum_area_m2 = np.pi * illium_radius_m**2
    np.testing.assert_almost_equal(
        actual=len(histogram) / illum_area_m2,
        desired=1.0,
        decimal=-1,
    )


def test_massive_fill():
    prng = np.random.Generator(np.random.PCG64(1))

    config = {
        "bin_width_m": 1.0,
        "num_bins_each_axis": 1000,
        "center_x_m": 0.0,
        "center_y_m": 0.0,
    }
    groundgrid = plenoirf.ground_grid.GroundGrid(**config)
    BLOCK_SIZE = 10_000
    NUM_BLOCKS = 100

    illium_radius_m = 500

    GGH = plenoirf.ground_grid.GGH()
    GGH.init_groundgrid(groundgrid=groundgrid)

    for block in range(NUM_BLOCKS):
        GGH.assign_cherenkov_bunches(
            cherenkov_bunches=cpw.testing.draw_cherenkov_bunches_from_point_source(
                instrument_sphere_x_cm=0.0,
                instrument_sphere_y_cm=0.0,
                instrument_sphere_radius_cm=illium_radius_m * 1e2,
                source_azimuth_rad=0.0,
                source_zenith_rad=0.0,
                source_distance_to_instrument_cm=1e8,
                prng=prng,
                size=BLOCK_SIZE,
                bunch_size_low=0.9,
                bunch_size_high=1.0,
            )
        )
    histogram = GGH.get_histogram()
    GGH.close()

    np.testing.assert_almost_equal(
        actual=np.sum(histogram["weight_photons"]) / (BLOCK_SIZE * NUM_BLOCKS),
        desired=0.95,
        decimal=1,
    )

    illum_area_m2 = np.pi * illium_radius_m**2
    np.testing.assert_almost_equal(
        actual=len(histogram) / illum_area_m2,
        desired=1.0,
        decimal=-1,
    )
