import io
import os
import zipfile
import datetime

import sparse_numeric_table as snt
import rename_after_writing as rnw

from .. import ground_grid
from .. import configuration
from .. import simulate_shower_and_collect_cherenkov_light_in_grid

from . import logfile


def read_groundgrid_choice_by_uid(in_path):
    """
    Parameters
    ----------
    in_path : str
        Path to a plenoirf production run e.g. RRRRRR.prm2cer.zip

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

    with zipfile.ZipFile(in_path, "r") as zin:
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


def apply(plenoirf_dir, in_path, out_path, use_tmp_dir=True):
    """
    Apply the hotfix to:
        1) Remove duplicate bin entries in the groundgrid
           histogram ROI storage.
        2) Remove bin entries which are outside of the Region-of-Interest (ROI).

    Parameters
    ----------
    plenoirf_dir : str
        To read the plenoirf config.
    in_path : str
        Path to the existing plenoirf production run.
        E.g. RRRRRR.prm2cer.zip
    out_path : str
        Path to where the fixed plenoirf production run will be writte.
        E.g. RRRRRR.prm2cer.zip.fix
    use_tmp_dir : bool
        If to use the local '/tmp' directory to write the output before copying
        it to its final 'out_path'.
    """
    _pattern = (
        "prm2cer/"
        "simulate_shower_and_collect_cherenkov_light_in_grid/"
        "ground_grid_intensity"
    )
    GROUND_GRID_INTENSITY_ROI_PATTERN = _pattern + "_roi.tar"
    GROUND_GRID_INTENSITY_PATTERN = _pattern + ".tar"
    ZF = zipfile.ZipFile

    print(f"plenoirf_dir: {plenoirf_dir:s}")
    print(f"in_path     : {in_path:s}")
    print(f"out_path    : {out_path:s}")

    plenoirf_config = configuration.read(plenoirf_dir=plenoirf_dir)
    groundgrid_choice_by_uid = read_groundgrid_choice_by_uid(in_path=in_path)

    with rnw.Path(out_path, use_tmp_dir=use_tmp_dir) as tmp_out_path:
        with ZF(in_path, "r") as zin, ZF(tmp_out_path, "w") as zout:
            for fileitem in zin.filelist:

                with zin.open(fileitem, "r") as fin:
                    payload = fin.read()

                if GROUND_GRID_INTENSITY_PATTERN in fileitem.filename:
                    print("__Assert no duplicates in ground grid__")
                    assert_no_duplicate_bins_in_groundgrid_histogram2d(
                        groundgrid_histogram2d_tar_bytes=payload,
                    )

                if GROUND_GRID_INTENSITY_ROI_PATTERN in fileitem.filename:
                    print("__Fixing__")
                    payload = fixing_groundgrid_histogram2d_tarfile(
                        groundgrid_histogram2d_tar_bytes=payload,
                        groundgrid_choice_by_uid=groundgrid_choice_by_uid,
                        groundgrid_num_bins_each_axis=plenoirf_config[
                            "ground_grid"
                        ]["geometry"]["num_bins_each_axis"],
                    )
                    print("__Assert no duplicates in ground grid ROI__")
                    assert_no_duplicate_bins_in_groundgrid_histogram2d(
                        groundgrid_histogram2d_tar_bytes=payload,
                    )

                with zout.open(fileitem.filename, "w") as fout:
                    fout.write(payload)

            hotfix_loglist = logfile.loads_loglist_from_run_zipfile(zin)
            now = datetime.datetime.now().isoformat()
            hotfix_loglist.append(f"{now:s}, {__name__:s}")
            logfile.dumps_loglist_to_run_zipfile(zout, hotfix_loglist)

    in_size = os.stat(in_path).st_size
    out_size = os.stat(out_path).st_size
    delta_size = in_size - out_size
    relative_size = out_size / in_size

    print(f"Size reduction is {delta_size*1e-6:.1f}MBytes. ")
    print(f"Now {relative_size*1e2:.0f}% of original.")
