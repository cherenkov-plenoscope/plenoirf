import zipfile
import tarfile
import os
import io
import glob
import rename_after_writing as rnw
import sparse_numeric_table as snt
import sequential_tar
import gzip
import plenopy
import json_utils
import dynamicsizerecarray

from .. import event_table
from .. import configuration
from .. import bookkeeping


def list_items():
    return [
        "event_table.snt.zip",
        "reconstructed_cherenkov.loph.tar",
        "ground_grid_intensity.zip",
        "ground_grid_intensity_roi.zip",
        "benchmark.snt.zip",
        "event_uids_for_debugging.txt",
    ]


def reduce_item(map_dir, out_path, item_key, use_tmp_dir=True):
    run_paths = glob.glob(os.path.join(map_dir, "*.zip"))
    if item_key == "event_table.snt.zip":
        recude_event_table(
            run_paths=run_paths, out_path=out_path, use_tmp_dir=use_tmp_dir
        )
    elif item_key == "reconstructed_cherenkov.loph.tar":
        reduce_reconstructed_cherenkov(
            run_paths=run_paths, out_path=out_path, use_tmp_dir=use_tmp_dir
        )
    elif item_key == "ground_grid_intensity.zip":
        reduce_ground_grid_intensity(
            run_paths=run_paths,
            out_path=out_path,
            roi=False,
            only_past_trigger=True,
            use_tmp_dir=use_tmp_dir,
        )
    elif item_key == "ground_grid_intensity_roi.zip":
        reduce_ground_grid_intensity(
            run_paths=run_paths,
            out_path=out_path,
            roi=True,
            only_past_trigger=True,
            use_tmp_dir=use_tmp_dir,
        )
    elif item_key == "benchmark.snt.zip":
        reduce_benchmarks(
            run_paths=run_paths, out_path=out_path, use_tmp_dir=use_tmp_dir
        )
    elif item_key == "event_uids_for_debugging.txt":
        reduce_event_uids_for_debugging(
            run_paths=run_paths, out_path=out_path, use_tmp_dir=use_tmp_dir
        )
    else:
        raise KeyError(f"No such item_key '{item_key}'.")


def make_jobs(plenoirf_dir, config=None, lazy=False, use_tmp_dir=True):
    if config is None:
        config = configuration.read(plenoirf_dir)

    jobs = []
    for instrument_key in config["instruments"]:
        for site_key in config["sites"]["instruemnt_response"]:
            for particle_key in config["particles"]:
                stage_dir = os.path.join(
                    plenoirf_dir,
                    "response",
                    instrument_key,
                    site_key,
                    particle_key,
                    "stage",
                )
                if has_filenames_of_certain_pattern_in_path(
                    path=stage_dir, filename_wildcard="*.zip"
                ):
                    for item_key in list_items():
                        if lazy:
                            job_out_path = os.path.join(
                                plenoirf_dir,
                                "response",
                                instrument_key,
                                site_key,
                                particle_key,
                                item_key,
                            )
                            if os.path.exists(job_out_path):
                                continue
                        job = {
                            "plenoirf_dir": plenoirf_dir,
                            "instrument_key": instrument_key,
                            "site_key": site_key,
                            "particle_key": particle_key,
                            "item_key": item_key,
                            "use_tmp_dir": use_tmp_dir,
                        }
                        jobs.append(job)
    return jobs


def has_filenames_of_certain_pattern_in_path(path, filename_wildcard):
    matching_paths = glob.glob(os.path.join(path, filename_wildcard))
    return len(matching_paths) > 0


def run_job(job):
    par_dir = os.path.join(
        job["plenoirf_dir"],
        "response",
        job["instrument_key"],
        job["site_key"],
        job["particle_key"],
    )
    with rnw.Path(os.path.join(par_dir, job["item_key"])) as out_path:
        reduce_item(
            map_dir=os.path.join(par_dir, "stage"),
            out_path=out_path,
            item_key=job["item_key"],
            use_tmp_dir=job["use_tmp_dir"],
        )


def zip_read_IO(file, internal_path, mode="rb"):
    with zipfile.ZipFile(file=file, mode="r") as zin:
        with zin.open(internal_path) as fin:
            if "|gz" in mode:
                block = gzip.decompress(fin.read())
            else:
                block = fin.read()
    if "t" in mode:
        buff = io.StringIO()
        buff.write(bytes.decode(block))
    elif "b" in mode:
        buff = io.BytesIO()
        buff.write(block)
    else:
        raise KeyError("mode must either be 'b' or 't'.")
    buff.seek(0)
    return buff


def recude_event_table(run_paths, out_path, use_tmp_dir=True):
    with rnw.Path(out_path, use_tmp_dir=use_tmp_dir) as tmp_path:
        with snt.open(
            tmp_path,
            mode="w",
            dtypes=event_table.structure.dtypes(),
            index_key=event_table.structure.UID_DTYPE[0],
            compress=True,
        ) as arc:
            for run_path in run_paths:
                run_basename = os.path.basename(run_path)
                run_id_str = os.path.splitext(run_basename)[0]
                buff = zip_read_IO(
                    file=run_path,
                    internal_path=os.path.join(
                        run_id_str, "event_table.snt.zip"
                    ),
                    mode="rb",
                )
                with snt.open(file=buff, mode="r") as part:
                    run_evttab = part.query()
                arc.append_table(run_evttab)


def reduce_reconstructed_cherenkov(run_paths, out_path, use_tmp_dir=True):
    with rnw.Path(out_path, use_tmp_dir=use_tmp_dir) as tmp_path:
        with plenopy.photon_stream.loph.LopfTarWriter(path=tmp_path) as lout:
            for run_path in run_paths:
                run_basename = os.path.basename(run_path)
                run_id_str = os.path.splitext(run_basename)[0]
                buff = zip_read_IO(
                    file=run_path,
                    internal_path=os.path.join(
                        run_id_str, "reconstructed_cherenkov.loph.tar"
                    ),
                    mode="rb",
                )
                with plenopy.photon_stream.loph.LopfTarReader(
                    fileobj=buff
                ) as lin:
                    for event in lin:
                        uid, phs = event
                        lout.add(uid=uid, phs=phs)


def reduce_ground_grid_intensity(
    run_paths, out_path, roi=False, only_past_trigger=False, use_tmp_dir=True
):
    with rnw.Path(out_path, use_tmp_dir=use_tmp_dir) as tmp_path:
        with zipfile.ZipFile(tmp_path, "w") as zout:
            for run_path in run_paths:
                run_basename = os.path.basename(run_path)
                run_id_str = os.path.splitext(run_basename)[0]

                if only_past_trigger:
                    buff = zip_read_IO(
                        file=run_path,
                        internal_path=os.path.join(
                            run_id_str, "event_table.snt.zip"
                        ),
                        mode="rb",
                    )
                    with snt.open(file=buff, mode="r") as part:
                        run_evttab = part.query()
                    past_trigger_uids = run_evttab["pasttrigger"]["uid"]

                suff = "_roi" if roi else ""
                internal_path = os.path.join(
                    run_id_str,
                    "plenoirf.production.simulate_shower_and_collect_cherenkov_light_in_grid",
                    f"ground_grid_intensity{suff:s}.tar",
                )
                buff = zip_read_IO(
                    file=run_path,
                    internal_path=internal_path,
                    mode="rb",
                )
                with sequential_tar.open(fileobj=buff, mode="r") as tarin:
                    for item in tarin:
                        run_id = int(item.name[0:6])
                        event_id = int(item.name[7 : 7 + 6])
                        event_uid = bookkeeping.uid.make_uid(
                            run_id=run_id,
                            event_id=event_id,
                        )
                        payload = item.read(mode="rb")
                        do_add = True
                        if only_past_trigger:
                            do_add = False
                            if event_uid in past_trigger_uids:
                                do_add = True

                        if do_add:
                            with zout.open(item.name, "w") as fout:
                                fout.write(payload)


def _make_benchmarks_dtype():
    dtype = [
        ("run_id", "<u8"),
        ("hostname_hash", "<i8"),
        ("time_unix_s", "<f8"),
        ("corsika_total_s", "<f4"),
        ("corsika_initializing_s", "<f4"),
        ("corsika_energy_rate_GeV_per_s_avg", "<f4"),
        ("corsika_energy_rate_GeV_per_s_std", "<f4"),
        ("corsika_cherenkov_bunch_rate_per_s_avg", "<f4"),
        ("corsika_cherenkov_bunch_rate_per_s_std", "<f4"),
        ("disk_write_rate_1k_rate_MB_per_s_avg", "<f4"),
        ("disk_write_rate_1k_rate_MB_per_s_std", "<f4"),
        ("disk_write_rate_1M_rate_MB_per_s_avg", "<f4"),
        ("disk_write_rate_1M_rate_MB_per_s_std", "<f4"),
        ("disk_write_rate_100M_rate_MB_per_s_avg", "<f4"),
        ("disk_write_rate_100M_rate_MB_per_s_std", "<f4"),
        ("disk_create_write_close_open_read_remove_latency_avg", "<f4"),
        ("disk_create_write_close_open_read_remove_latency_std", "<f4"),
    ]
    return dtype


def reduce_benchmarks(run_paths, out_path, use_tmp_dir=True):
    hostname_hashes = {}

    stats = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=_make_benchmarks_dtype()
    )

    for run_path in run_paths:
        run_basename = os.path.basename(run_path)
        run_id_str = os.path.splitext(run_basename)[0]

        buff = zip_read_IO(
            file=run_path,
            internal_path=os.path.join(
                run_id_str,
                "plenoirf.production.gather_and_export_provenance",
                "provenance.json.gz",
            ),
            mode="rt|gz",
        )
        item = json_utils.loads(buff.read())

        if item["hostname"] not in hostname_hashes:
            hostname_hashes[item["hostname"]] = hash(item["hostname"])

        buff = zip_read_IO(
            file=run_path,
            internal_path=os.path.join(
                run_id_str,
                "plenoirf.production.benchmark_compute_environment",
                "benchmark.json.gz",
            ),
            mode="rt|gz",
        )
        bench = json_utils.loads(buff.read())

        rec = {}
        rec["hostname_hash"] = hash(item["hostname"])
        rec["time_unix_s"] = item["time"]["unix"]
        rec["run_id"] = int(run_id_str)
        ccc = bench["corsika"]
        rec["corsika_total_s"] = ccc["total"]
        rec["corsika_initializing_s"] = ccc["initializing"]
        rec["corsika_energy_rate_GeV_per_s_avg"] = ccc[
            "energy_rate_GeV_per_s"
        ]["avg"]
        rec["corsika_energy_rate_GeV_per_s_std"] = ccc[
            "energy_rate_GeV_per_s"
        ]["std"]
        rec["corsika_cherenkov_bunch_rate_per_s_avg"] = ccc[
            "cherenkov_bunch_rate_per_s"
        ]["avg"]
        rec["corsika_cherenkov_bunch_rate_per_s_std"] = ccc[
            "cherenkov_bunch_rate_per_s"
        ]["avg"]
        ddd = bench["disk_write_rate"]
        rec["disk_write_rate_1k_rate_MB_per_s_avg"] = ddd["1k"][
            "rate_MB_per_s"
        ]["avg"]
        rec["disk_write_rate_1k_rate_MB_per_s_std"] = ddd["1k"][
            "rate_MB_per_s"
        ]["std"]
        rec["disk_write_rate_1M_rate_MB_per_s_avg"] = ddd["1M"][
            "rate_MB_per_s"
        ]["avg"]
        rec["disk_write_rate_1M_rate_MB_per_s_std"] = ddd["1M"][
            "rate_MB_per_s"
        ]["std"]
        rec["disk_write_rate_100M_rate_MB_per_s_avg"] = ddd["100M"][
            "rate_MB_per_s"
        ]["avg"]
        rec["disk_write_rate_100M_rate_MB_per_s_std"] = ddd["100M"][
            "rate_MB_per_s"
        ]["std"]
        ddd = bench["disk_create_write_close_open_read_remove_latency"]
        rec["disk_create_write_close_open_read_remove_latency_avg"] = ddd[
            "avg"
        ]
        rec["disk_create_write_close_open_read_remove_latency_std"] = ddd[
            "std"
        ]
        stats.append_record(rec)

    with rnw.Path(out_path, use_tmp_dir=use_tmp_dir) as tmp_path:
        with snt.open(
            file=tmp_path,
            mode="w",
            dtypes={"benchmark": _make_benchmarks_dtype()},
            index_key="run_id",
            compress=True,
        ) as fout:
            fout.append_table({"benchmark": stats})

    with rnw.Path(
        out_path + ".hostname_hashes.json", use_tmp_dir=use_tmp_dir
    ) as tmp_path:
        with open(tmp_path, mode="w") as fout:
            fout.write(json_utils.dumps(hostname_hashes))


def reduce_event_uids_for_debugging(run_paths, out_path, use_tmp_dir=True):
    with rnw.Path(out_path, use_tmp_dir=use_tmp_dir) as tmp_path:
        with open(
            tmp_path,
            "wt",
        ) as fout:
            for run_path in run_paths:
                run_basename = os.path.basename(run_path)
                run_id_str = os.path.splitext(run_basename)[0]

                buff = zip_read_IO(
                    file=run_path,
                    internal_path=os.path.join(
                        run_id_str,
                        "plenoirf.production.draw_event_uids_for_debugging",
                        "event_uids_for_debugging.json.gz",
                    ),
                    mode="rt|gz",
                )
                event_uids = json_utils.loads(buff.read())
                for event_uid in event_uids:
                    fout.write(f"{event_uid:012d}\n")
