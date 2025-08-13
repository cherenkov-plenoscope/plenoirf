import plenoirf
import argparse


parser = argparse.ArgumentParser(
    prog="groundgrid_roi_2025_08_08.py",
    description=("groundgrid_roi_2025_08_08"),
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

plenoirf.production.hotfix.groundgrid_roi_2025_08_08.apply_fix(
    plenoirf_dir=plenoirf_dir,
    in_path=in_path,
    out_path=out_path,
    use_tmp_dir=args.use_tmp_dir,
)
