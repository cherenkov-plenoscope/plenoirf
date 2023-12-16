import numpy as np
import sparse_numeric_table as spt
import shapely
from shapely import geometry as shapely_geometry
from .. import table as irf_table


def make_polygon(onregion, num_steps=1000):
    return _make_ellipse_polygon(
        center_x=onregion["ellipse_center_cx"],
        center_y=onregion["ellipse_center_cy"],
        mayor_radius=onregion["ellipse_mayor_radius"],
        minor_radius=onregion["ellipse_minor_radius"],
        main_axis_azimuth=onregion["ellipse_main_axis_azimuth"],
        num_steps=num_steps,
    )


def intersecting_area_of_polygons(a, b):
    poly_a = shapely_geometry.Polygon(shell=a)
    poly_b = shapely_geometry.Polygon(shell=b)
    poly_intersection_a_b = poly_a.intersection(poly_b)
    return poly_intersection_a_b.area


def make_circular_polygon(radius, num_steps=1000):
    poly = []
    for phi in np.linspace(0, np.pi * 2.0, num_steps, endpoint=False):
        poly.append([radius * np.cos(phi), radius * np.sin(phi)])
    return np.array(poly)


def _make_ellipse_polygon(
    center_x,
    center_y,
    mayor_radius,
    minor_radius,
    main_axis_azimuth,
    num_steps=1000,
):
    # polar coordinates
    theta = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)
    radius = (mayor_radius * minor_radius) / np.sqrt(
        (minor_radius * np.cos(theta)) ** 2
        + (mayor_radius * np.sin(theta)) ** 2
    )

    xs = center_x + np.cos(theta + main_axis_azimuth) * radius
    ys = center_y + np.sin(theta + main_axis_azimuth) * radius
    return np.array([xs, ys]).T


def estimate_onregion(
    reco_cx,
    reco_cy,
    reco_main_axis_azimuth,
    reco_num_photons,
    reco_core_radius,
    config,
):
    pivot_opening_angle = np.deg2rad(config["opening_angle_deg"])
    opening_angle_scaling = np.interp(
        x=reco_num_photons,
        xp=config["opening_angle_scaling"]["reco_num_photons_pe"],
        fp=config["opening_angle_scaling"]["scale"],
    )
    opening_angle = pivot_opening_angle * opening_angle_scaling

    ellipticity = np.interp(
        x=reco_core_radius,
        xp=config["ellipticity_scaling"]["reco_core_radius_m"],
        fp=config["ellipticity_scaling"]["scale"],
    )

    # constant solid angle: A = pi*(ellipticity*r)*(1/ellipticity*r)
    ellipse_mayor_radius = opening_angle * ellipticity
    ellipse_minor_radius = opening_angle / ellipticity
    ellipse_solid_angle = np.pi * ellipse_mayor_radius * ellipse_minor_radius

    return {
        "ellipse_center_cx": float(reco_cx),
        "ellipse_center_cy": float(reco_cy),
        "ellipse_main_axis_azimuth": float(reco_main_axis_azimuth),
        "ellipse_mayor_radius": ellipse_mayor_radius,
        "ellipse_minor_radius": ellipse_minor_radius,
        "ellipse_solid_angle": ellipse_solid_angle,
    }


def is_direction_inside(cx, cy, onregion):
    return _is_point_inside_ellipse(
        point_x=cx,
        point_y=cy,
        ellipse_center_x=onregion["ellipse_center_cx"],
        ellipse_center_y=onregion["ellipse_center_cy"],
        ellipse_mayor_radius=onregion["ellipse_mayor_radius"],
        ellipse_minor_radius=onregion["ellipse_minor_radius"],
        ellipse_main_axis_azimuth=onregion["ellipse_main_axis_azimuth"],
    )


def _is_point_inside_ellipse(
    point_x,
    point_y,
    ellipse_center_x,
    ellipse_center_y,
    ellipse_mayor_radius,
    ellipse_minor_radius,
    ellipse_main_axis_azimuth,
):
    tpx = point_x - ellipse_center_x
    tpy = point_y - ellipse_center_y

    a = -1.0 * ellipse_main_axis_azimuth
    rpx = tpx * np.cos(a) - tpy * np.sin(a)
    rpy = tpx * np.sin(a) + tpy * np.cos(a)

    return _is_point_inside_centered_ellipse_with_mayor_axis_on_x(
        point_x=rpx,
        point_y=rpy,
        ellipse_mayor_along_x_radius=ellipse_mayor_radius,
        ellipse_minor_along_y_radius=ellipse_minor_radius,
    )


def _is_point_inside_centered_ellipse_with_mayor_axis_on_x(
    point_x,
    point_y,
    ellipse_mayor_along_x_radius,
    ellipse_minor_along_y_radius,
):
    return (
        (point_x**2) / (ellipse_mayor_along_x_radius**2)
        + (point_y**2) / (ellipse_minor_along_y_radius**2)
    ) <= 1.0
