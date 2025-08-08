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


def hotfix_2025_08_08_ground_grid_intensity_roi(
    in_payload, groundgrid_choice_by_uid
):
    dtype = plenoirf.ground_grid.intensity.default_record_dtype()

    out_buff = io.BytesIO()
    in_buff = io.BytesIO()
    in_buff.write(in_payload)
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

            in_arr = np.frombuffer(in_payload, dtype=dtype)

            # FIX
            # ---
            out_arr = plenoirf.production.simulate_shower_and_collect_cherenkov_light_in_grid.ImgRoiTar_apply_cut(
                groundgrid_histogram=in_arr,
                groundgrid_choice_bin_idx_x=groundgrid_choice_by_uid[uid][
                    "bin_idx_x"
                ],
                groundgrid_choice_bin_idx_y=groundgrid_choice_by_uid[uid][
                    "bin_idx_y"
                ],
            )
            print(uid, len(out_arr))
            # count unique cells

            counts = {}
            for cell in out_arr:
                xy = (cell["x_bin"], cell["y_bin"])
                if xy in counts:
                    counts[xy] += 1
                else:
                    counts[xy] = 0
            for cell in counts:
                if counts[cell] > 1:
                    print(cell, counts[cell])

            out_payload = out_arr.tobytes()
            out_payload_gz = gzip.compress(out_payload)

            size_reduction += len(in_payload_gz) - len(out_payload_gz)

            tout.write(name=item.name, payload=out_payload_gz, mode="wb")

    print(f"Size reduction: {size_reduction*1e-6:.1f}MBytes")

    out_buff.seek(0)
    return out_buff.read()


GROUND_GRID_INTENSITY_ROI_PATTERN = "prm2cer/simulate_shower_and_collect_cherenkov_light_in_grid/ground_grid_intensity_roi.tar"
groundgrid_choice_by_uid = read_groundgrid_choice_by_uid(in_path=in_path)

with rnw.Path(out_path, use_tmp_dir=args.use_tmp_dir) as tmp_out_path:
    with ZF(in_path, "r") as zin, ZF(tmp_out_path, "w") as zout:
        for fileitem in zin.filelist:

            with zin.open(fileitem, "r") as fin:
                payload = fin.read()

            if GROUND_GRID_INTENSITY_ROI_PATTERN in fileitem.filename:
                payload = hotfix_2025_08_08_ground_grid_intensity_roi(
                    in_payload=payload,
                    groundgrid_choice_by_uid=groundgrid_choice_by_uid,
                )

            with zout.open(fileitem.filename, "w") as fout:
                fout.write(payload)
