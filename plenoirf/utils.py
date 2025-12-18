import numpy as np
import propagate_uncertainties as pru
import io
import os
import json_utils
import warnings
import gzip
import rename_after_writing as rnw


def gzip_file(path, ext=".gz", block_size=2**23):
    has_gz_ext = os.path.splitext(path)[0] == ext

    if has_gz_ext:
        warnings.warn(f"gzip on '{path:s}' already has '{ext:s}'.")
    if _has_gzip_signature_bytes(path):
        warnings.warn(f"gzip on '{path:s}' already has signature x'1f,8b'.")

    with rnw.Path(path + ext) as opath:
        with open(path, "rb") as fin:
            with gzip.open(opath, "wb") as fout:
                while True:
                    block = fin.read(block_size)
                    if len(block) == 0:
                        break
                    fout.write(block)
    os.remove(path)


def gunzip_file(path, ext=".gz", block_size=2**23):
    opath, actual_ext = os.path.splitext(path)

    if not actual_ext == ext:
        warnings.warn(f"gunzip on '{path:s}' without '{ext:s}'.")
    if not _has_gzip_signature_bytes(path):
        warnings.warn(f"gunzip on '{path:s}' without signature x'1f,8b'.")

    with rnw.Path(opath) as tmp_opath:
        with gzip.open(path, "rb") as fin:
            with open(tmp_opath, "wb") as fout:
                while True:
                    block = fin.read(block_size)
                    if len(block) == 0:
                        break
                    fout.write(block)
    os.remove(path)


def _has_gzip_signature_bytes(path):
    size = os.stat(path).st_size
    if size < 4:
        return False
    else:
        with open(path, "rb") as f:
            head = f.read(2)
        if head == b"\x1f\x8b":
            return True
        else:
            return False


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


def is_10th_part_in_current_decade(i):
    """
    Can be used to e.g. skip log messages.
    Motivated by CORSIKA's log.

    Parameters
    ----------
    i : int
        Number to be tested.

    Returns True for:
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 20, 30, 40, 50, 60, 70, 80, 90,
        100, 200, 300, 400, 500, 600, 700, 800, 900,
        1000, ...
    else:
        False
    """
    if i <= 0:
        return True
    else:
        m = 10 ** int(np.log10(i))
        return i % m == 0


def zipfile_write_dir_recursively(zipfile, filename, arcname):
    for root, dirs, files in os.walk(filename):
        for file in files:
            fname = os.path.join(root, file)
            relname = os.path.relpath(os.path.join(root, file), filename)
            aname = os.path.join(arcname, relname)
            zipfile.write(filename=fname, arcname=aname)


def SerialPool_if_None(pool):
    if pool is None:
        return SerialPool()
    else:
        return pool


class SerialPool:
    def __init__(self):
        pass

    def map(self, func, iterable):
        return [func(item) for item in iterable]

    def starmap(self, func, iterable):
        return [func(*item) for item in iterable]

    def __repr__(self):
        out = "{:s}()".format(self.__class__.__name__)
        return out


def find_limits(x, ignore_non_positive=False):
    not_nan = np.logical_not(np.isnan(x))
    not_inf = np.logical_not(np.isinf(x))
    valid = np.logical_and(not_nan, not_inf)

    if ignore_non_positive:
        is_positive = x > 0
        valid = np.logical_and(valid, is_positive)

    return np.min(x[valid]), np.max(x[valid])


def find_decade_power_limits(x):
    start, stop = find_limits(x=x, ignore_non_positive=True)
    return np.floor(np.log10(start)), np.ceil(np.log10(stop))


def find_decade_limits(x):
    start_power, stop_power = find_decade_power_limits(x=x)
    return 10**start_power, 10**stop_power


class open_and_read_into_memory_when_small_enough:
    """
    Helps when sequential read is cheap but seeking is expensive.
    This is often the case with bulk network storages on HPC clusters.
    """

    def __init__(self, path, mode="rb", size="64M"):
        """
        Open a file for reading.

        Parameters
        ----------
        path : str
            Path to be opened for reading.
        mode : str (default: rb)
            Read in binary 'rb' or text 'rt' mode.
        size : None or int or str
            If the file is larger than size, it is not read into memory.
            None == 0 == "0" are the same.
            Understands metric prefixes k, M, and G. size="64M" is equal to
            64_000_000.
        """
        self.path = path
        _filesize_bytes = os.stat(self.path).st_size
        if size is None:
            _size_in_bytes = 0
        elif can_be_interpreted_as_int(size):
            _size_in_bytes = int(size)
        else:
            _size_in_bytes = parse_metric_prefix(size)

        assert not "w" in mode, "Can only open in read mode='r'."

        if mode == "rb":
            self.mode = "b"
        elif mode == "rt":
            self.mode = "t"
        else:
            raise RuntimeError(f"mode='{mode:s} is not supported.'")

        self.in_memory = _filesize_bytes < _size_in_bytes

    def __enter__(self):
        if self.in_memory:
            if self.mode == "b":
                self.f = io.BytesIO()
                with open(self.path, "rb") as fin:
                    self.f.write(fin.read())
                self.f.seek(0)
            elif mode == "t":
                self.f = io.StringIO()
                with open(self.path, "rt") as fin:
                    self.f.write(fin.read())
                self.f.seek(0)
        else:
            self.f = open(self.path, "r" + self.mode)

        return self.f

    def close(self):
        if not self.in_memory:
            self.f.close()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def __repr__(self):
        return f"{self.__class__.__name__:s}()"


def can_be_interpreted_as_int(s):
    try:
        v = int(s)
        return True
    except ValueError:
        return False


def parse_metric_prefix(s):
    prefix = {"G": 1_000_000_000, "M": 1_000_000, "k": 1_000}

    out = None
    for key in prefix:
        if s.endswith(key):
            out = int(s[:-1]) * prefix[key]
    if out is None:
        out = int(s)

    return out


def astronomic_magnitude_to_brightness(magnitude):
    scale = 100 ** (1 / 5)
    return 1.0 / (scale ** (magnitude - 1))


def astronomic_brightness_to_magnitude(brightness):
    scale = 100 ** (1 / 5)

    def log_scale(x):
        return np.log(x) / np.log(scale)

    # brightness = 1.0 / ( scale ** (magnitude - 1))
    # brightness * ( scale ** (magnitude - 1)) = 1.0
    # scale ** (magnitude - 1) = 1.0 / brightness
    # magnitude - 1 = log_scale(1.0 / brightness)
    magnitude = 1.0 + log_scale(1.0 / brightness)
    return magnitude
