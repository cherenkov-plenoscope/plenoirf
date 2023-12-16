import plenoirf
import numpy as np


def test_point_in_alligned_ellipse():
    MAYOR = 2
    MINOR = 1

    def inside(x, y):
        return plenoirf.reconstruction.onregion._is_point_inside_centered_ellipse_with_mayor_axis_on_x(
            point_x=x,
            point_y=y,
            ellipse_mayor_along_x_radius=MAYOR,
            ellipse_minor_along_y_radius=MINOR,
        )

    assert inside(0, 0)
    assert inside(1, 0)
    assert inside(2, 0)
    assert not inside(2.1, 0)

    assert inside(0, 0.5)
    assert inside(0, 1)
    assert not inside(0, 1.1)


def test_point_in_centered_ellipse():
    MAYOR = 2
    MINOR = 1

    def inside(x, y):
        return plenoirf.reconstruction.onregion._is_point_inside_ellipse(
            point_x=x,
            point_y=y,
            ellipse_center_x=0.0,
            ellipse_center_y=0.0,
            ellipse_mayor_radius=MAYOR,
            ellipse_minor_radius=MINOR,
            ellipse_main_axis_azimuth=0.0,
        )

    assert inside(0, 0)
    assert inside(1, 0)
    assert inside(2, 0)
    assert not inside(2.1, 0)

    assert inside(0, 0.5)
    assert inside(0, 1)
    assert not inside(0, 1.1)


def test_point_in_centered_rotated_ellipse():
    MAYOR = 2
    MINOR = 1
    MAYOR_AXIS_ON_Y = np.pi / 2

    def inside(x, y):
        return plenoirf.reconstruction.onregion._is_point_inside_ellipse(
            point_x=x,
            point_y=y,
            ellipse_center_x=0.0,
            ellipse_center_y=0.0,
            ellipse_mayor_radius=MAYOR,
            ellipse_minor_radius=MINOR,
            ellipse_main_axis_azimuth=MAYOR_AXIS_ON_Y,
        )

    assert inside(0, 0)
    assert inside(1, 0)
    assert not inside(1.1, 0)

    assert inside(0, 0.5)
    assert inside(0, 1)
    assert inside(0, 2)
    assert not inside(0, 2.1)
