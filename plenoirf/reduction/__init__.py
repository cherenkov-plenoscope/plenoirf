import zipfile
import tarfile
import os
from os.path import join as opj
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
from .. import production


def list_items():
    return [
        "event_table.snt.zip",
        "reconstructed_cherenkov.loph.tar",
        "ground_grid_intensity.zip",
        "ground_grid_intensity_roi.zip",
        "benchmark.snt.zip",
        "event_uids_for_debugging.txt",
    ]


def reduce_item(instrument_site_particle_dir, item_key, use_tmp_dir=True):
    run_ids = list_run_ids_ready_for_reduction(
        map_dir=opj(instrument_site_particle_dir, "map"),
        checkpoint_keys=production.list_checkpoint_keys(),
    )

    if item_key == "event_table.snt.zip":
        recude_event_table(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            use_tmp_dir=use_tmp_dir,
        )
    elif item_key == "reconstructed_cherenkov.loph.tar":
        reduce_reconstructed_cherenkov(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            use_tmp_dir=use_tmp_dir,
        )
    elif item_key == "ground_grid_intensity.zip":
        reduce_ground_grid_intensity(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            roi=False,
            only_past_trigger=True,
            use_tmp_dir=use_tmp_dir,
        )
    elif item_key == "ground_grid_intensity_roi.zip":
        reduce_ground_grid_intensity(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            roi=True,
            only_past_trigger=True,
            use_tmp_dir=use_tmp_dir,
        )
    elif item_key == "benchmark.snt.zip":
        reduce_benchmarks(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            use_tmp_dir=use_tmp_dir,
        )
    elif item_key == "event_uids_for_debugging.txt":
        reduce_event_uids_for_debugging(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            use_tmp_dir=use_tmp_dir,
        )
    else:
        raise KeyError(f"No such item_key '{item_key}'.")


def _get_run_id_set_in_directory(path, filename_wildcardard="*.zip"):
    all_paths = glob.glob(os.path.join(path, filename_wildcardard))

    run_ids = set()
    for a_path in all_paths:
        basename = os.path.basename(a_path)
        if len(basename) >= 6:
            first_six_chars = basename[0:6]
            if str_can_be_interpreted_as_int(first_six_chars):
                run_id = int(first_six_chars)
                assert run_id not in run_ids
                run_ids.add(run_id)
    return run_ids


def list_run_ids_ready_for_reduction(map_dir, checkpoint_keys=None):
    if checkpoint_keys is None:
        checkpoint_keys = production.list_checkpoint_keys()

    run_ids_in_checkpoints = []
    for key in checkpoint_keys:
        run_ids_in_checkpoints.append(
            _get_run_id_set_in_directory(path=os.path.join(map_dir, key))
        )

    run_ids = list(set.intersection(*run_ids_in_checkpoints))
    return sorted(run_ids)


def str_can_be_interpreted_as_int(s):
    try:
        v = int(s)
        return True
    except ValueError:
        return False


def make_jobs(plenoirf_dir, config=None, lazy=False, use_tmp_dir=True):
    if config is None:
        config = configuration.read(plenoirf_dir)

    jobs = []
    for instrument_key in config["instruments"]:
        for site_key in config["sites"]["instruemnt_response"]:
            for particle_key in config["particles"]:
                map_dir = os.path.join(
                    plenoirf_dir,
                    "response",
                    instrument_key,
                    site_key,
                    particle_key,
                    "map",
                )
                if has_filenames_of_certain_pattern_in_path(
                    path=map_dir, filename_wildcard="*.zip"
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
    instrument_site_particle_dir = os.path.join(
        job["plenoirf_dir"],
        "response",
        job["instrument_key"],
        job["site_key"],
        job["particle_key"],
    )
    with rnw.Path(os.path.join(par_dir, job["item_key"])) as out_path:
        reduce_item(
            instrument_site_particle_dir=instrument_site_particle_dir,
            item_key=job["item_key"],
            use_tmp_dir=job["use_tmp_dir"],
        )


class ZipFileBufferIO:
    def __init__(self, file):
        self.file = file

    def __enter__(self):
        self.zin = zipfile.ZipFile(file=self.file, mode="r")
        return self

    @property
    def filenames(self):
        return [i.filename for i in self.zin.filelist]

    def read(self, path, mode):
        with self.zin.open(path) as fin:
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

    def close(self):
        return self.zin.close()

    def read_event_table(self, path):
        with snt.open(file=self.read(path, "rb"), mode="r") as part:
            return part.query()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()


def _map_reduce_dirs(instrument_site_particle_dir):
    map_dir = opj(instrument_site_particle_dir, "map")
    reduce_dir = opj(instrument_site_particle_dir, "reduce")
    os.makedirs(reduce_dir, exist_ok=True)
    return map_dir, reduce_dir


def recude_event_table(
    instrument_site_particle_dir, run_ids, use_tmp_dir=True
):
    map_dir, reduce_dir = _map_reduce_dirs(instrument_site_particle_dir)
    out_path = opj(reduce_dir, "event_table.snt.zip")

    with rnw.Path(out_path, use_tmp_dir=use_tmp_dir) as tmp_path:
        with snt.open(
            tmp_path,
            mode="w",
            dtypes=event_table.structure.dtypes(),
            index_key=event_table.structure.UID_DTYPE[0],
            compress=True,
        ) as arc:
            for checkpoint_key in production.list_checkpoint_keys():
                for run_id in run_ids:
                    run_path = opj(
                        map_dir,
                        checkpoint_key,
                        f"{run_id:06d}.{checkpoint_key:s}.zip",
                    )
                    with ZipFileBufferIO(run_path) as zipbuff:
                        for ipath in zipbuff.filenames:
                            if ipath.endswith("event_table.snt.zip"):
                                buff = zipbuff.read(path=ipath, mode="rb")

                                with snt.open(file=buff, mode="r") as part:
                                    part_evttab = part.query()

                                arc.append_table(part_evttab)


def reduce_reconstructed_cherenkov(
    instrument_site_particle_dir, run_ids, use_tmp_dir=True
):
    map_dir, reduce_dir = _map_reduce_dirs(instrument_site_particle_dir)
    out_path = opj(reduce_dir, "reconstructed_cherenkov.loph.tar")

    with rnw.Path(out_path, use_tmp_dir=use_tmp_dir) as tmp_path:
        with plenopy.photon_stream.loph.LopfTarWriter(path=tmp_path) as lout:
            for run_id in run_ids:

                run_path = opj(
                    map_dir,
                    "cer2cls",
                    f"{run_id:06d}.cer2cls.zip",
                )

                with ZipFileBufferIO(run_path) as zipbuff:
                    buff = zipbuff.read(
                        path=opj(
                            f"{run_id:06d}",
                            "cer2cls",
                            "simulate_instrument_and_reconstruct_cherenkov",
                            "reconstructed_cherenkov.loph.tar",
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
    instrument_site_particle_dir,
    run_ids,
    roi=False,
    only_past_trigger=False,
    use_tmp_dir=True,
):
    _suff = "_roi" if roi else ""
    map_dir, reduce_dir = _map_reduce_dirs(instrument_site_particle_dir)
    filename_without_ext = f"ground_grid_intensity{_suff:s}"
    out_path = opj(reduce_dir, filename_without_ext + ".zip")

    with rnw.Path(out_path, use_tmp_dir=use_tmp_dir) as tmp_path:
        with zipfile.ZipFile(tmp_path, "w") as zout:
            for run_id in run_ids:
                if only_past_trigger:
                    run_cls2rec_path = opj(
                        map_dir,
                        "cer2cls",
                        f"{run_id:06d}.cer2cls.zip",
                    )

                    with ZipFileBufferIO(run_cls2rec_path) as zipbuff:
                        _evttab = zipbuff.read_event_table(
                            path=opj(
                                f"{run_id:06d}",
                                "cer2cls",
                                "simulate_instrument_and_reconstruct_cherenkov",
                                "event_table.snt.zip",
                            ),
                        )
                    past_trigger_uids = _evttab["pasttrigger"]["uid"]
                else:
                    past_trigger_uids = None

                run_prm2cer_path = opj(
                    map_dir,
                    "prm2cer",
                    f"{run_id:06d}.prm2cer.zip",
                )

                with ZipFileBufferIO(run_prm2cer_path) as zipbuff:
                    buff = zipbuff.read(
                        path=opj(
                            f"{run_id:06d}",
                            "prm2cer",
                            "simulate_shower_and_collect_cherenkov_light_in_grid",
                            filename_without_ext + ".tar",
                        ),
                        mode="rb",
                    )
                with sequential_tar.open(fileobj=buff, mode="r") as tarin:
                    for item in tarin:
                        event_uid = bookkeeping.uid.make_uid(
                            run_id=int(item.name[0:6]),
                            event_id=int(item.name[7 : 7 + 6]),
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


def reduce_benchmarks(instrument_site_particle_dir, run_ids, use_tmp_dir=True):
    map_dir, reduce_dir = _map_reduce_dirs(instrument_site_particle_dir)
    out_path = opj(reduce_dir, "benchmark.snt.zip")

    hostname_hashes = {}

    stats = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=_make_benchmarks_dtype()
    )

    for run_id in run_ids:
        run_path = opj(map_dir, "prm2cer", f"{run_id:06d}.prm2cer.zip")

        with ZipFileBufferIO(run_path) as zipbuff:
            _buff = zipbuff.read(
                path=opj(
                    f"{run_id:06d}",
                    "prm2cer",
                    "gather_and_export_provenance",
                    "provenance.json.gz",
                ),
                mode="rt|gz",
            )
            item = json_utils.loads(_buff.read())

            if item["hostname"] not in hostname_hashes:
                hostname_hashes[item["hostname"]] = hash(item["hostname"])

            _buff = zipbuff.read(
                path=opj(
                    f"{run_id:06d}",
                    "prm2cer",
                    "benchmark_compute_environment",
                    "benchmark.json.gz",
                ),
                mode="rt|gz",
            )
            bench = json_utils.loads(_buff.read())

        rec = {}
        rec["hostname_hash"] = hash(item["hostname"])
        rec["time_unix_s"] = item["time"]["unix"]
        rec["run_id"] = run_id
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


def reduce_event_uids_for_debugging(
    instrument_site_particle_dir, run_ids, use_tmp_dir=True
):
    map_dir, reduce_dir = _map_reduce_dirs(instrument_site_particle_dir)
    out_path = opj(reduce_dir, "event_uids_for_debugging.txt")

    with rnw.Path(out_path, use_tmp_dir=use_tmp_dir) as tmp_path:
        with open(
            tmp_path,
            "wt",
        ) as fout:
            for run_id in run_ids:
                run_path = opj(map_dir, "prm2cer", f"{run_id:06d}.prm2cer.zip")
                with ZipFileBufferIO(run_path) as zipbuff:
                    buff = zipbuff.read(
                        path=opj(
                            f"{run_id:06d}",
                            "prm2cer",
                            "draw_event_uids_for_debugging",
                            "event_uids_for_debugging.json",
                        ),
                        mode="rt",
                    )

                event_uids = json_utils.loads(buff.read())
                event_uids = sorted(event_uids)
                for event_uid in event_uids:
                    fout.write(f"{event_uid:012d}\n")
