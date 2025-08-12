import plenoirf
import zipfile
import os
import argparse
import sparse_numeric_table as snt
import rename_after_writing as rnw
import gzip
import sequential_tar
import io

parser = argparse.ArgumentParser(
    prog="too_much_groundgrid_roi.py",
    description=("hotfix: too_much_groundgrid_roi."),
)
parser.add_argument(
    "plenoirf_dir",
    metavar="PLENOIRF_DIR",
    type=str,
)
parser.add_argument(
    "in_path",
    metavar="IN_PATH",
    type=str,
)
parser.add_argument(
    "out_path",
    nargs="?",
    metavar="OUT_PATH",
    default=None,
    type=str,
)
parser.add_argument(
    "--use_tmp_dir",
    action="store_true",
)

args = parser.parse_args()
in_path = args.in_path
plenoirf_dir = args.plenoirf_dir

if args.out_path is None:
    out_path = in_path + ".fix"
else:
    out_path = args.out_path

plenoirf.production.hotfix.groundgrid_roi_2025_08_08.apply(
    plenoirf_dir=plenoirf_dir,
    in_path=in_path,
    out_path=out_path,
)
