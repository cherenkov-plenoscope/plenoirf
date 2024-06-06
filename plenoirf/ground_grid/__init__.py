import ray_voxel_overlap
import numpy as np
import binning_utils
import corsika_primary as cpw
import spherical_coordinates
import tempfile
import json_utils
import os
import tarfile
import subprocess
import json_line_logger
import io

from .. import configfile


def make_ground_grid_config(
    bin_width_m,
    num_bins_each_axis,
    prng,
):
    assert bin_width_m > 0.0
    assert num_bins_each_axis > 0

    random_radius_m = (1 / 2) * bin_width_m

    random_shift_x_m = prng.uniform(-random_radius_m, random_radius_m)
    random_shift_y_m = prng.uniform(-random_radius_m, random_radius_m)

    return {
        "bin_width_m": bin_width_m,
        "num_bins_each_axis": num_bins_each_axis,
        "center_x_m": random_shift_x_m,
        "center_y_m": random_shift_y_m,
    }


def GroundGrid(
    bin_width_m,
    num_bins_each_axis,
    center_x_m,
    center_y_m,
):
    assert bin_width_m > 0.0
    assert num_bins_each_axis > 0

    gg = {}
    gg["bin_width_m"] = float(bin_width_m)
    gg["num_bins_each_axis"] = int(num_bins_each_axis)
    gg["center_x_m"] = float(center_x_m)
    gg["center_y_m"] = float(center_y_m)

    _bin_half_width_m = 0.5 * gg["bin_width_m"]
    gg["bin_smallest_enclosing_radius_m"] = np.sqrt(3 * _bin_half_width_m**2)

    _width = gg["bin_width_m"] * gg["num_bins_each_axis"]
    _num = 1 + gg["num_bins_each_axis"]
    for ax in ["x", "y"]:
        _center = gg["center_{:s}_m".format(ax)]
        _start = (-1 / 2) * _width + _center
        _stop = (1 / 2) * _width + _center
        _bin_edges = np.linspace(start=_start, stop=_stop, num=_num)
        gg["{:s}_bin".format(ax)] = binning_utils.Binning(bin_edges=_bin_edges)

    z_bin_edges = np.linspace(
        start=-(1 / 2) * gg["bin_width_m"],
        stop=(1 / 2) * gg["bin_width_m"],
        num=2,
    )
    gg["z_bin"] = binning_utils.Binning(bin_edges=z_bin_edges)

    # area
    # ----
    gg["num_bins_thrown"] = gg["num_bins_each_axis"] ** 2
    gg["bin_area_m2"] = gg["bin_width_m"] ** 2
    gg["area_thrown_m2"] = gg["num_bins_thrown"] * gg["bin_area_m2"]
    return gg


def make_result(
    groundgrid,
    groundgrid_histogram,
    threshold_num_photons,
    prng,
):
    bin_idxs_above_threshold = find_bin_idxs_above_or_equal_threshold3(
        grid_histogram=groundgrid_histogram,
        threshold_num_photons=threshold_num_photons,
    )

    out = {}
    out["num_bins_above_threshold"] = len(bin_idxs_above_threshold)

    if out["num_bins_above_threshold"] == 0:
        out["choice"] = None
    else:
        out["choice"] = draw_random_bin_choice3(
            groundgrid=groundgrid,
            groundgrid_histogram=groundgrid_histogram,
            bin_idxs_above_threshold=bin_idxs_above_threshold,
            prng=prng,
        )

    # compare to classic scatter algorithm as used in the
    # Cherenkov-Telescpe-Array
    # ---------------------------------------------------
    out["scatter_histogram"] = histogram_bins_in_scatter_radius(
        groundgrid=groundgrid,
        bin_idxs=bin_idxs_above_threshold,
    )

    return out


def find_bin_idxs_above_or_equal_threshold3(
    grid_histogram,
    threshold_num_photons,
):
    bin_idxs = {}
    for entry in grid_histogram:
        if entry["weight_photons"] >= threshold_num_photons:
            bin_idx = (entry["x_bin"], entry["y_bin"])
            bin_idxs[bin_idx] = entry["weight_photons"]
    return bin_idxs


def draw_random_bin_choice3(
    groundgrid,
    groundgrid_histogram,
    bin_idxs_above_threshold,
    prng,
):
    bin_idx = draw_bin_idx(bin_idxs=bin_idxs_above_threshold, prng=prng)

    cc = {}
    cc["bin_idx_x"] = bin_idx[0]
    cc["bin_idx_y"] = bin_idx[1]
    cc["core_x_m"] = groundgrid["x_bin"]["centers"][cc["bin_idx_x"]]
    cc["core_y_m"] = groundgrid["y_bin"]["centers"][cc["bin_idx_y"]]
    cc["bin_num_photons"] = bin_idxs_above_threshold[bin_idx]
    return cc


def draw_bin_idx(bin_idxs, prng):
    keys = list(bin_idxs.keys())
    return keys[prng.choice(a=len(keys))]


def histogram_bins_in_scatter_radius(groundgrid, bin_idxs):
    NUM_BINS = 16
    scatter_radius_bin_edges_m = radii_for_area_power_space(num_bins=NUM_BINS)

    scatter_radii_of_bins_above_threshold = []
    for bin_idx in bin_idxs:
        bin_x_m = groundgrid["x_bin"]["centers"][bin_idx[0]]
        bin_y_m = groundgrid["y_bin"]["centers"][bin_idx[1]]
        bin_radius_m = np.hypot(bin_x_m, bin_y_m)
        scatter_radii_of_bins_above_threshold.append(bin_radius_m)
    bin_counts = np.histogram(
        scatter_radii_of_bins_above_threshold,
        bins=scatter_radius_bin_edges_m,
    )[0]
    out = {
        "scatter_radius_bin_edges_m": scatter_radius_bin_edges_m,
        "bin_counts": bin_counts,
    }
    return out


def radii_for_area_power_space(start=1e6, factor=2.0, num_bins=16):
    radii = [0]
    for i in range(num_bins):
        area = start * factor**i
        r = np.sqrt(area) / np.pi
        radii.append(r)
    return np.array(radii)


def make_histogram2d_dtype():
    return [("x_bin", "i4"), ("y_bin", "i4"), ("weight_photons", "f8")]


def make_groundgrid_config_bytes(groundgrid):
    f = io.StringIO()
    M2CM = 1e2
    for dim in ["x", "y", "z"]:
        dimbin = "{:s}_bin".format(dim)
        f.write("{:d}\n".format(groundgrid[dimbin]["num"]))
        f.write("{:e}\n".format(M2CM * groundgrid[dimbin]["start"]))
        f.write("{:e}\n".format(M2CM * groundgrid[dimbin]["stop"]))
    f.seek(0)
    return str.encode(f.read())


def tar_header_bytes(name, size):
    info = tarfile.TarInfo()
    info.name = name
    info.size = size
    return info.tobuf()


def _fill_tar_buffer_with_zeros_untill_next_block(buff):
    TAR_BLOCK_SIZE = 512
    num_zeros = (
        TAR_BLOCK_SIZE - buff.tell() % TAR_BLOCK_SIZE
    ) % TAR_BLOCK_SIZE
    buff.write(num_zeros * b"\0")
    return buff


def tar_make_groundgrid_init_txt(groundgrid):
    buff = io.BytesIO()
    payload = make_groundgrid_config_bytes(groundgrid=groundgrid)
    buff.write(tar_header_bytes(name="init.txt", size=len(payload)))
    buff.write(payload)
    buff = _fill_tar_buffer_with_zeros_untill_next_block(buff=buff)
    buff.seek(0)
    return buff.read()


def tar_make_export_txt():
    buff = io.BytesIO()
    buff.write(tar_header_bytes(name="export.txt", size=0))
    buff.seek(0)
    return buff.read()


def tar_make_cer(cherenkov_bunches):
    buff = io.BytesIO()
    assert cherenkov_bunches.dtype == np.float32
    assert len(cherenkov_bunches.shape) == 2
    assert cherenkov_bunches.shape[1] == 8
    payload = cherenkov_bunches.flatten(order="c").tobytes()
    buff.write(tar_header_bytes(name="cer.x8.float32", size=len(payload)))
    buff.write(payload)
    buff = _fill_tar_buffer_with_zeros_untill_next_block(buff=buff)
    buff.seek(0)
    return buff.read()


def tar_make_zero_block():
    return 512 * b"\0"


def tar_read_data(fileobj, size):
    num_blocks = size // 512
    if size > num_blocks * 512:
        num_blocks += 1

    if num_blocks == 0:
        return b""
    else:
        buff = fileobj.read(num_blocks * 512)
        data = buff[0:size]
        return data


def tar_read_header(fileobj):
    return tarfile.TarInfo.frombuf(
        fileobj.read(512),
        encoding=tarfile.ENCODING,
        errors="surrogateescape",
    )


class GGH:
    def __init__(self):
        self.merlict_ground_grid_path = configfile.read()["ground_grid"]
        self.process = subprocess.Popen(
            self.merlict_ground_grid_path,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )

    def init_groundgrid(self, groundgrid):
        self.groundgrid_num_bins_each_axis = int(
            groundgrid["num_bins_each_axis"]
        )
        self.process.stdin.write(
            tar_make_groundgrid_init_txt(groundgrid=groundgrid)
        )
        self.process.stdin.flush()

    def assign_cherenkov_bunches(self, cherenkov_bunches):
        cer = tar_make_cer(cherenkov_bunches=cherenkov_bunches)
        self.process.stdin.write(cer)
        self.process.stdin.flush()

    def get_histogram(self):
        self.process.stdin.write(tar_make_export_txt())
        self.process.stdin.flush()
        info = tar_read_header(fileobj=self.process.stdout)
        assert info.name == "histogram.int32_int32_float64"
        buff = tar_read_data(fileobj=self.process.stdout, size=info.size)
        hist = np.frombuffer(buff, dtype=make_histogram2d_dtype())
        assert_histogram_in_limits(
            hist=hist,
            num_bins_each_axis=self.groundgrid_num_bins_each_axis,
        )
        return hist

    def close(self):
        NUM_ZERO_BLOCKS_AT_END_OF_TAR = 2
        for i in range(NUM_ZERO_BLOCKS_AT_END_OF_TAR):
            self.process.stdin.write(tar_make_zero_block())
        self.process.stdin.flush()

        for i in range(NUM_ZERO_BLOCKS_AT_END_OF_TAR):
            z = self.process.stdout.read(512)
            assert z == 512 * b"\x00"

        self.process.wait()
        assert self.process.returncode == 0


def assert_histogram_in_limits(hist, num_bins_each_axis):
    num = num_bins_each_axis
    if any(hist["x_bin"] < 0) or any(hist["x_bin"] >= num):
        raise AssertionError(
            "merlict_c89 ground_grid_main hist x_bin out of range"
        )
    if any(hist["y_bin"] < 0) or any(hist["y_bin"] >= num):
        raise AssertionError(
            "merlict_c89 ground_grid_main hist y_bin out of range"
        )
