import numpy as np
import propagate_uncertainties as pru
import binning_utils
import datetime
import io
import tarfile
import scipy.interpolate
import json_utils
import warnings


def contains_same_bytes(path_a, path_b):
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        a_bytes = fa.read()
        b_bytes = fb.read()
        return a_bytes == b_bytes


def tar_append(tarout, file_name, file_bytes):
    with io.BytesIO() as buff:
        info = tarfile.TarInfo(file_name)
        info.size = buff.write(file_bytes)
        buff.seek(0)
        tarout.addfile(info, buff)


def power10space_bin_edges(binning, fine):
    assert fine > 0
    space = binning_utils.power10.space(
        start_decade=binning["start"]["decade"],
        start_bin=binning["start"]["bin"] * fine,
        stop_decade=binning["stop"]["decade"],
        stop_bin=binning["stop"]["bin"] * fine,
        num_bins_per_decade=binning["num_bins_per_decade"] * fine,
    )
    return space, len(space) - 1


def _divide_silent(numerator, denominator, default):
    valid = denominator != 0
    division = np.ones(shape=numerator.shape) * default
    division[valid] = numerator[valid] / denominator[valid]
    return division


_10s = 10
_1M = 60
_1h = _1M * 60
_1d = _1h * 24
_1w = _1d * 7
_1m = _1d * 30
_1y = 365 * _1d


def make_civil_times_points_in_quasi_logspace():
    """
    time-points from 1s to 100y in the civil steps of:
    s, m, h, d, week, Month, year, decade
    """

    times = []
    for _secs in np.arange(1, _10s, 1):
        times.append(_secs)
    for _10secs in np.arange(_10s, _1M, _10s):
        times.append(_10secs)
    for _mins in np.arange(_1M, _1h, _1M):
        times.append(_mins)
    for _hours in np.arange(_1h, _1d, _1h):
        times.append(_hours)
    for _days in np.arange(_1d, _1w, _1d):
        times.append(_days)
    for _weeks in np.arange(_1w, 4 * _1w, _1w):
        times.append(_weeks)
    for _months in np.arange(_1m, 12 * _1m, _1m):
        times.append(_months)
    for _years in np.arange(_1y, 10 * _1y, _1y):
        times.append(_years)
    for _decades in np.arange(10 * _1y, 100 * _1y, 10 * _1y):
        times.append(_decades)
    return times


def make_civil_time_str(time_s, format_seconds="{:f}"):
    try:
        years = int(time_s // _1y)
        tr = time_s - years * _1y

        days = int(tr // _1d)
        tr = tr - days * _1d

        hours = int(tr // _1h)
        tr = tr - hours * _1h

        minutes = int(tr // _1M)
        tr = tr - minutes * _1M

        s = ""
        if years:
            s += "{:d}y ".format(years)
        if days:
            s += "{:d}d ".format(days)
        if hours:
            s += "{:d}h ".format(hours)
        if minutes:
            s += "{:d}min ".format(minutes)
        if tr:
            s += (format_seconds + "s").format(tr)
        if s[-1] == " ":
            s = s[0:-1]
        return s
    except Exception as err:
        print(str(err))
        return (format_seconds + "s").format(time_s)


def find_closest_index_in_array_for_value(
    arr, val, max_rel_error=0.1, max_abs_error=None
):
    arr = np.array(arr)
    idx = np.argmin(np.abs(arr - val))
    if max_abs_error:
        assert np.abs(arr[idx] - val) < max_abs_error
    else:
        assert np.abs(arr[idx] - val) < max_rel_error * val
    return idx


def latex_scientific(
    real,
    format_template="{:e}",
    nan_template="nan",
    drop_mantisse_if_one=False,
):
    if real != real:
        return nan_template
    assert format_template.endswith("e}")
    s = format_template.format(real)

    pos_e = s.find("e")
    assert pos_e >= 0
    mantisse = s[0:pos_e]
    exponent = str(int(s[pos_e + 1 :]))
    if drop_mantisse_if_one and float(mantisse) == 1.0:
        out = r"10^{" + exponent + r"}"
    else:
        out = mantisse + r"\times{}10^{" + exponent + r"}"
    return out


def integrate_rate_where_known(dRdE, dRdE_au, E_edges):
    unknown = np.isnan(dRdE_au)

    _dRdE = dRdE.copy()
    _dRdE_au = dRdE_au.copy()

    _dRdE[unknown] = 0.0
    _dRdE_au[unknown] = 0.0

    T, T_au = pru.integrate(f=_dRdE, f_au=_dRdE_au, x_bin_edges=E_edges)
    return T, T_au


def filter_particles_with_electric_charge(particles):
    out = {}
    for pk in particles:
        if np.abs(particles[pk]["electric_charge_qe"]) > 0:
            out[pk] = dict(particles[pk])
    return out


def copy_square_selection_from_2D_array(img, ix, iy, r, fill=np.nan):
    """
    Returns a square selection from the input-image.

    Parameters
    ----------
    img : np.array 2D
        Input image to copy from.
    ix : int
        X central bin for selection in input-image
    iy : int
        Y central bin for selection in input-image
    r : int
        Radius of selection. Width of selection is (2*r + 1).
    fill : float/int
        If selection is outside of input-image, this value is written to
        output.
    """
    assert r >= 0
    out = fill * np.ones(shape=(2 * r + 1, 2 * r + 1), dtype=img.dtype)
    x_start = ix - r
    x_stop = ix + r + 1
    y_start = iy - r
    y_stop = iy + r + 1
    for ox, ix in enumerate(np.arange(x_start, x_stop)):
        for oy, iy in enumerate(np.arange(y_start, y_stop)):
            if ix >= 0 and ix < img.shape[0]:
                if iy >= 0 and iy < img.shape[1]:
                    out[ox, oy] = img[ix, iy]
    return out


def fill_nans_from_end(arr, val):
    for i in np.arange(len(arr) - 1, -1, -1):
        if np.isnan(arr[i]):
            arr[i] = val
        else:
            break
    return arr


def fill_nans_from_start(arr, val):
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            arr[i] = val
        else:
            break
    return arr


def gradient_in_bin_edges(x, bin_edges):
    assert len(bin_edges) - 1 == len(x)
    dx = np.nan * np.ones(len(x) - 1)
    for i in range(len(x) - 1):
        dx[i] = (x[i + 1] - x[i]) / bin_edges[i]
    return dx


def read_json_but_forgive(path, default={}):
    try:
        with open(path, "rt") as f:
            out = json_utils.loads(f.read())
    except Exception as e:
        print(e)
        warnings.warn("Failed to load '{:s}'".format(path))
        out = default
    return out


def dict_to_pretty_str(dictionary):
    ss = json_utils.dumps(dictionary, indent=2)
    ss = ss.replace('"', "")
    ss = ss.replace("{", "")
    ss = ss.replace("}", "")
    oss = io.StringIO()
    for line in ss.splitlines():
        if len(line) > 0:
            oss.write(line)
            oss.write("\n")
    oss.seek(0)
    return oss.read()


def ray_parameter_for_closest_distance_to_point(
    ray_support,
    ray_direction,
    point,
):
    """
    Returns parameter for ray to be at closest point.
    """
    # We create a plane orthogonal to this ray and containing the point
    # plane equation:
    #  d = x*a + y*b + z*c
    #
    # We set the normal vector n of the plane to the ray's direction vector:
    #  a=direction.x b=direction.y c=direction.z
    #
    # Now we insert the support vector of the frame into the plane eqaution:
    #  d = point.x*dirx + point.y*diry + point.z*dirz
    d = np.dot(ray_direction, point)

    # Insert the ray into plane equation and solve for the ray parameter.
    # The ray's direction is normalized, therefore: (direction * direction)=1

    return d - np.dot(ray_support, ray_direction)


def ray_at(ray_support, ray_direction, parameter):
    return ray_support + ray_direction * parameter
