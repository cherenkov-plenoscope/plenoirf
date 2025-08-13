import io
import os
import zipfile
import datetime
import random
import glob
import copy

import sparse_numeric_table as snt
import rename_after_writing as rnw

from .. import ground_grid
from .. import configuration
from .. import simulate_shower_and_collect_cherenkov_light_in_grid
from .. import utils
from ...version import __version__

from . import logfile


def read_groundgrid_choice_by_uid(fileobj):
    """
    Parameters
    ----------
    fileobj : file like
        File like plenoirf production run e.g. RRRRRR.prm2cer.zip

    Returns
    -------
    groundgrid choice by uid : dict
        Maps the airshower's uid to the groundgrid's choice
        "bin_idx_x", "bin_idx_y".
    """
    PATTERN = (
        "prm2cer/"
        "simulate_shower_and_collect_cherenkov_light_in_grid/"
        "event_table.snt.zip"
    )

    with zipfile.ZipFile(fileobj, "r") as zin:
        for fileitem in zin.filelist:
            if PATTERN in fileitem.filename:
                with zin.open(fileitem, "r") as fin:
                    with snt.open(file=fin, mode="r") as arc:
                        groundgrid_choice = arc.query(
                            levels_and_columns={
                                "groundgrid_choice": [
                                    "uid",
                                    "bin_idx_x",
                                    "bin_idx_y",
                                ]
                            }
                        )
                break

    out = {}
    for item in groundgrid_choice["groundgrid_choice"]:
        out[item["uid"]] = {}
        for key in ["bin_idx_x", "bin_idx_y"]:
            out[item["uid"]][key] = item[key]
    return out


def read_run_id_from_run_zip_fileobj(fileobj):
    run_ids = set()
    with zipfile.ZipFile(fileobj, "r") as zin:
        for fileitem in zin.filelist:
            run_ids.add(fileitem.filename[0:6])
    assert len(run_ids) == 1
    return int(run_ids.pop())


def assert_no_duplicate_bins_in_groundgrid_histogram2d(
    groundgrid_histogram2d_tar_bytes,
):
    """
    Parameters
    ----------
    groundgrid_histogram2d_tar_bytes : bytes
        The content of the tarfile containing the groundgrid's sparse 2D
        histograms.
    """
    buff = io.BytesIO()
    buff.write(groundgrid_histogram2d_tar_bytes)
    buff.seek(0)

    with ground_grid.histogram2d.TarReader(fileobj=buff) as tin:
        print("Checking for duplicates ... ", end="")
        for uid, hist2d in tin:
            ground_grid.histogram2d.assert_bins_unique(hist2d)
        print("Done.")


def fixing_groundgrid_histogram2d_tarfile(
    groundgrid_histogram2d_tar_bytes,
    groundgrid_choice_by_uid,
    groundgrid_num_bins_each_axis,
):
    """
    Parameters
    ----------
    groundgrid_histogram2d_tar_bytes : bytes
        The content of the tarfile containing the groundgrid's sparse 2D
        histograms.
    groundgrid_choice_by_uid : dict
        Mapping of uids to groundgrid's choices "bin_idx_x" and "bin_idx_y".
    groundgrid_num_bins_each_axis : int
        Taken from the plenoirf's config.

    Returns
    -------
    groundgrid_histogram2d_tar_bytes : bytes
        Same as input but with fixed content.
    """
    TarR = ground_grid.histogram2d.TarReader
    TarW = ground_grid.histogram2d.TarWriter

    out_buff = io.BytesIO()
    in_buff = io.BytesIO()
    in_buff.write(groundgrid_histogram2d_tar_bytes)
    in_buff.seek(0)

    print("Applying hotfix ... ", end="")
    with TarR(fileobj=in_buff) as tin, TarW(fileobj=out_buff) as tout:
        for uid, in_hist2d in tin:
            assert uid in groundgrid_choice_by_uid

            # APPLY FIX
            # ---------
            out_hist2d = simulate_shower_and_collect_cherenkov_light_in_grid.ImgRoiTar_apply_cut(
                groundgrid_histogram=in_hist2d,
                groundgrid_choice_bin_idx_x=groundgrid_choice_by_uid[uid][
                    "bin_idx_x"
                ],
                groundgrid_choice_bin_idx_y=groundgrid_choice_by_uid[uid][
                    "bin_idx_y"
                ],
            )
            out_hist2d = ground_grid.histogram2d.remove_duplicate_bins_hotfix_2025_08_08(
                out_hist2d
            )

            # ASSERT
            # ------
            ground_grid.histogram2d.assert_bins_in_limits(
                hist=out_hist2d,
                num_bins_each_axis=groundgrid_num_bins_each_axis,
            )
            ground_grid.histogram2d.assert_bins_unique(hist=out_hist2d)
            assert (
                out_hist2d.shape[0] <= 625
            ), "Expected at most 25x25=625 bins."
            tout.write(uid=uid, hist=out_hist2d)

    print("Done.")
    out_buff.seek(0)
    return out_buff.read()


def apply_fix(
    in_path,
    out_path,
    use_tmp_dir=True,
    num_bins_each_axis=None,
    plenoirf_dir=None,
):
    """
    Apply the hotfix to:
        1) Remove duplicate bin entries in the groundgrid
           histogram ROI storage.
        2) Remove bin entries which are outside of the Region-of-Interest (ROI).

    Parameters
    ----------

    in_path : str
        Path to the existing plenoirf production run.
        E.g. RRRRRR.prm2cer.zip
    out_path : str
        Path to where the fixed plenoirf production run will be writte.
        E.g. RRRRRR.prm2cer.zip.fix
    use_tmp_dir : bool
        If to use the local '/tmp' directory to write the output before copying
        it to its final 'out_path'.
    num_bins_each_axis : int or None
        If None, this will be parsed from the plenoirf_dir's config
    plenoirf_dir : str or None
        To read the plenoirf config in case num_bins_each_axis is None.
    """
    ZF = zipfile.ZipFile
    OP = utils.open_and_read_into_memory_when_small_enough

    if num_bins_each_axis is None:
        num_bins_each_axis = configuration.read(plenoirf_dir=plenoirf_dir)[
            "ground_grid"
        ]["geometry"]["num_bins_each_axis"]
    else:
        assert plenoirf_dir is None, (
            "Expected 'plenoirf_dir' is None when 'num_bins_each_axis' "
            "is given."
        )
    assert num_bins_each_axis > 0

    print(f"in_path           : {in_path:s}")
    print(f"out_path          : {out_path:s}")
    print(f"num_bins_each_axis: {num_bins_each_axis:d}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    FOUND = {"logfile": False, "ground_grid": False, "ground_grid_roi": False}

    with rnw.Path(out_path, use_tmp_dir=use_tmp_dir) as tmp_out_path, OP(
        in_path, size="128M"
    ) as in_file:
        groundgrid_choice_by_uid = read_groundgrid_choice_by_uid(
            fileobj=in_file
        )
        in_file.seek(0)
        run_id = read_run_id_from_run_zip_fileobj(fileobj=in_file)
        in_file.seek(0)

        LOG_PATH = f"{run_id:06d}/prm2cer/log.jsonl.gz"
        _pattern = (
            f"{run_id:06d}/"
            "prm2cer/"
            "simulate_shower_and_collect_cherenkov_light_in_grid/"
            "ground_grid_intensity"
        )
        GROUND_GRID_ROI_PATH = _pattern + "_roi.tar"
        GROUND_GRID_PATH = _pattern + ".tar"

        with ZF(in_file, "r") as zin, ZF(tmp_out_path, "w") as zout:
            for fileitem in zin.filelist:

                with zin.open(fileitem, "r") as fin:
                    payload = fin.read()

                if GROUND_GRID_PATH == fileitem.filename:
                    print("__Assert no duplicates in ground grid__")
                    assert_no_duplicate_bins_in_groundgrid_histogram2d(
                        groundgrid_histogram2d_tar_bytes=payload,
                    )
                    FOUND["ground_grid"] = True

                if GROUND_GRID_ROI_PATH == fileitem.filename:
                    print("__Fixing__")
                    payload = fixing_groundgrid_histogram2d_tarfile(
                        groundgrid_histogram2d_tar_bytes=payload,
                        groundgrid_choice_by_uid=groundgrid_choice_by_uid,
                        groundgrid_num_bins_each_axis=num_bins_each_axis,
                    )
                    print("__Assert no duplicates in ground grid ROI__")
                    assert_no_duplicate_bins_in_groundgrid_histogram2d(
                        groundgrid_histogram2d_tar_bytes=payload,
                    )
                    FOUND["ground_grid_roi"] = True

                if LOG_PATH == fileitem.filename:
                    with logfile.LoggerAppender(
                        payload=payload, mode="b|gz"
                    ) as logapp:
                        logapp.logger.info(
                            f"HOTFIX, plenoirf=v{__version__:s}"
                        )
                        payload = logapp.get_payload()
                    FOUND["logfile"] = True

                with zout.open(fileitem.filename, "w") as fout:
                    fout.write(payload)

    for key in FOUND:
        assert FOUND[key], f"Did not find '{key:s}' in '{in_path:s}'."

    in_size = os.stat(in_path).st_size
    out_size = os.stat(out_path).st_size
    delta_size = in_size - out_size
    relative_size = out_size / in_size

    print(f"Size reduction is {delta_size*1e-6:.1f}MBytes. ")
    print(f"Now {relative_size*1e2:.0f}% of original.")


def _make_jobs(plenoirf_dir, old_prm2cer_dirname):
    ujobs = []
    config = configuration.read(plenoirf_dir=plenoirf_dir)
    target = config["population_target"]
    groundgrid_num_bins_each_axis = config["ground_grid"]["geometry"][
        "num_bins_each_axis"
    ]

    prm2cer_ext = ".prm2cer.zip"

    for instrument_key in target:
        for site_key in target[instrument_key]:
            for particle_key in target[instrument_key][site_key]:

                map_dir = os.path.join(
                    plenoirf_dir,
                    "response",
                    instrument_key,
                    site_key,
                    particle_key,
                    "map",
                )
                old_prm2cer_dir = os.path.join(map_dir, old_prm2cer_dirname)

                possible_run_paths = glob.glob(
                    os.path.join(old_prm2cer_dir, "*" + prm2cer_ext)
                )
                for possible_run_path in possible_run_paths:
                    filename = os.path.basename(possible_run_path)
                    is6 = utils.can_be_interpreted_as_int(filename[0:6])
                    isE = filename[6:] == prm2cer_ext
                    if is6 and isE:
                        out_path = os.path.join(map_dir, "prm2cer", filename)
                        if not os.path.exists(out_path):
                            ujob = {
                                "groundgrid_num_bins_each_axis": groundgrid_num_bins_each_axis,
                                "in_path": possible_run_path,
                                "out_path": out_path,
                            }
                            ujobs.append(ujob)

    return ujobs


def make_jobs(plenoirf_dir, old_prm2cer_dirname, num_runs_per_job=100):
    """
    To apply the fix:
      1) Move the 'prm2cer' dirs manually to a new basename
         'old_prm2cer_dirname'.
      2) Call this function to make the jobs.
      3) call run_jobs(job)

    ----------
    plenoirf_dir : str
        Path to the plenoirf dir.
    old_prm2cer_dirname : str
        Directory basename where the old runs are in which need fixing.
    num_runs_per_job : int
        Bundle these many run fixes into one compute job.
    """
    ujobs = _make_jobs(
        plenoirf_dir=plenoirf_dir, old_prm2cer_dirname=old_prm2cer_dirname
    )
    random.shuffle(ujobs)
    jobs = _bundle_items(items=ujobs, num_items_per_bundle=num_runs_per_job)
    return jobs


def _bundle_items(items, num_items_per_bundle=100):
    assert num_items_per_bundle > 0
    bundles = []
    bundle = []
    for item in items:
        if len(bundle) >= num_items_per_bundle:
            bundles.append(copy.copy(bundle))
            bundle = [item]
        else:
            bundle.append(item)

    # the remains
    if len(bundle) > 0:
        bundles.append(bundle)
    return bundles


def run_job(job):
    for ujob in job:
        if not os.path.exists(ujob["out_path"]):
            apply_fix(
                in_path=ujob["in_path"],
                out_path=ujob["out_path"],
                use_tmp_dir=True,
                num_bins_each_axis=ujob["groundgrid_num_bins_each_axis"],
                plenoirf_dir=None,
            )
        else:
            print(f"Already done: '{ujob['out_path']:s}'.")
