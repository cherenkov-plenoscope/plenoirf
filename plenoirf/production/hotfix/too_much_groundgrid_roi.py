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


ZF = zipfile.ZipFile
TF = sequential_tar.open


def read_groundgrid_choice_by_uid(in_path):
    PATTERN = "prm2cer/simulate_shower_and_collect_cherenkov_light_in_grid/event_table.snt.zip"
    with ZF(in_path, "r") as zin:
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


def find_out_if_duplicate_bins_in_groundgrid_histogram2d(
    groundgrid_histogram2d_tar_bytes,
):
    dtype = plenoirf.ground_grid.histogram2d.make_dtype()
    buff = io.BytesIO()
    buff.write(groundgrid_histogram2d_tar_bytes)

    with TF(fileobj=buff, mode="r") as tin:
        for item in tin:
            assert str.endswith(item.name, ".i4_i4_f8.gz")
            uid = plenoirf.bookkeeping.uid.make_uid(
                run_id=int(item.name[0:6]),
                event_id=int(item.name[7 : 7 + 6]),
            )
            payload_gz = item.read(mode="rb")
            payload = gzip.decompress(payload_gz)
            hist2d = np.frombuffer(payload, dtype=dtype)
            plenoirf.ground_grid.histogram2d.assert_bins_unique(hist2d)


def hotfix_2025_08_08_ground_grid_intensity_roi(
    groundgrid_histogram2d_tar_bytes,
    groundgrid_choice_by_uid,
    groundgrid_num_bins_each_axis,
):
    dtype = plenoirf.ground_grid.histogram2d.make_dtype()

    out_buff = io.BytesIO()
    in_buff = io.BytesIO()
    in_buff.write(groundgrid_histogram2d_tar_bytes)
    in_buff.seek(0)

    size_reduction = 0

    with TF(fileobj=in_buff, mode="r") as tin, TF(
        fileobj=out_buff, mode="w"
    ) as tout:
        for item in tin:
            assert str.endswith(item.name, ".i4_i4_f8.gz")
            uid = plenoirf.bookkeeping.uid.make_uid(
                run_id=int(item.name[0:6]),
                event_id=int(item.name[7 : 7 + 6]),
            )

            assert uid in groundgrid_choice_by_uid

            in_payload_gz = item.read(mode="rb")
            in_payload = gzip.decompress(in_payload_gz)

            in_hist2d = np.frombuffer(in_payload, dtype=dtype)

            # APPLY FIX
            # ---------
            out_hist2d = plenoirf.production.simulate_shower_and_collect_cherenkov_light_in_grid.ImgRoiTar_apply_cut(
                groundgrid_histogram=in_hist2d,
                groundgrid_choice_bin_idx_x=groundgrid_choice_by_uid[uid][
                    "bin_idx_x"
                ],
                groundgrid_choice_bin_idx_y=groundgrid_choice_by_uid[uid][
                    "bin_idx_y"
                ],
            )
            out_hist2d = plenoirf.ground_grid.histogram2d.remove_duplicate_bins_hotfix_2025_08_08(
                out_hist2d
            )

            # ASSERT
            # ------
            plenoirf.ground_grid.histogram2d.assert_bins_in_limits(
                hist=out_hist2d,
                num_bins_each_axis=groundgrid_num_bins_each_axis,
            )
            plenoirf.ground_grid.histogram2d.assert_bins_unique(
                hist=out_hist2d
            )

            out_payload = out_hist2d.tobytes()
            out_payload_gz = gzip.compress(out_payload)

            size_reduction += len(in_payload_gz) - len(out_payload_gz)
            tout.write(name=item.name, payload=out_payload_gz, mode="wb")

    print(f"Size reduction: {size_reduction*1e-6:.1f}MBytes")

    out_buff.seek(0)
    return out_buff.read()


plenoirf_config = plenoirf.configuration.read(plenoirf_dir)


_pattern = "prm2cer/simulate_shower_and_collect_cherenkov_light_in_grid/ground_grid_intensity"
GROUND_GRID_INTENSITY_ROI_PATTERN = _pattern + "_roi.tar"
GROUND_GRID_INTENSITY_PATTERN = _pattern + ".tar"

groundgrid_choice_by_uid = read_groundgrid_choice_by_uid(in_path=in_path)

with rnw.Path(out_path, use_tmp_dir=args.use_tmp_dir) as tmp_out_path:
    with ZF(in_path, "r") as zin, ZF(tmp_out_path, "w") as zout:
        for fileitem in zin.filelist:

            with zin.open(fileitem, "r") as fin:
                payload = fin.read()

            if GROUND_GRID_INTENSITY_PATTERN in fileitem.filename:
                find_out_if_duplicate_bins_in_groundgrid_histogram2d(
                    groundgrid_histogram2d_tar_bytes=payload,
                )

            if GROUND_GRID_INTENSITY_ROI_PATTERN in fileitem.filename:
                payload = hotfix_2025_08_08_ground_grid_intensity_roi(
                    groundgrid_histogram2d_tar_bytes=payload,
                    groundgrid_choice_by_uid=groundgrid_choice_by_uid,
                    groundgrid_num_bins_each_axis=plenoirf_config[
                        "ground_grid"
                    ]["geometry"]["num_bins_each_axis"],
                )
                find_out_if_duplicate_bins_in_groundgrid_histogram2d(
                    groundgrid_histogram2d_tar_bytes=payload,
                )

            with zout.open(fileitem.filename, "w") as fout:
                fout.write(payload)
