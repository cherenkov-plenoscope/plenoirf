import os
import zipfile
from os.path import join as opj

import rename_after_writing as rnw
import sparse_numeric_table as snt
import json_utils
import plenopy
import dynamicsizerecarray
import sequential_tar

from .. import event_table
from .. import configuration
from .. import bookkeeping
from .. import production
from .. import utils

from . import memory
from . import logging
from ..utils import open_and_read_into_memory_when_small_enough
from .zipfilebufferio import ZipFileBufferIO


def reduce(
    plenoirf_dir,
    instrument_key,
    site_key,
    particle_key,
    run_ids,
    out_dir,
    memory_config=None,
    logger=None,
):
    """
    Reduce the many runs and their checkpoints (prm2cer, cer2cls, cls2rec) into
    output files which contain the statistics of many runs but only a narrow
    topic.

    Output topics and corresponding files:
        - 'event_table.snt.zip'
        - 'reconstructed_cherenkov.loph.tar'
        - 'ground_grid_intensity.zip'
        - 'ground_grid_intensity_roi.zip'
        - 'benchmark.snt.zip'
        - 'event_uids_for_debugging.txt'

    Parameters
    ----------
    plenoirf_dir : str
        Path to plenoirf's main work dir.
    instrument_key : str
        Name of the instrument.
    site_key : str
        Name of the site. ('chile', 'namibia', ... )
    particle_key : str
        Name of the particle. ('gamma', 'proton', ... )
    run_ids : list of int
        Run ids to be reduced into output files.
    out_dir : str
        Path to the output directory of this reduction.
    memory_config : dict (default None)
        How to handle memory, /tmp/ and buffer sizes to make life easier on
        the slow HPC nfs systems.
    """
    logger = logging.stdout_logger_if_logger_is_None(logger)
    logger.info("Start reduce ...")

    instrument_site_particle_dir = opj(
        plenoirf_dir,
        "response",
        instrument_key,
        site_key,
        particle_key,
    )

    before_out_dir, _ = os.path.split(out_dir)
    os.makedirs(before_out_dir, exist_ok=True)

    return _reduce_handle_output_tmp_dir(
        instrument_site_particle_dir=instrument_site_particle_dir,
        run_ids=run_ids,
        out_dir=out_dir,
        memory_config=memory_config,
        logger=logger,
    )


def _reduce_handle_output_tmp_dir(
    instrument_site_particle_dir,
    run_ids,
    out_dir,
    memory_config=None,
    logger=None,
):
    memory_config = memory.make_config_if_None(memory_config)
    logger = logging.stdout_logger_if_logger_is_None(logger)

    with rnw.Directory(
        path=out_dir, use_tmp_dir=memory_config["use_tmp_dir"]
    ) as tmp_dir:
        logger.info(f"tmp_dir: '{tmp_dir:s}'.")
        logger.info(f"out_dir: '{out_dir:s}'.")
        _reduce_handle_output_files(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            tmp_dir=tmp_dir,
            memory_config=memory_config,
            logger=logger,
        )


def _reduce_handle_output_files(
    instrument_site_particle_dir,
    run_ids,
    tmp_dir,
    memory_config=None,
    logger=None,
):
    memory_config = memory.make_config_if_None(memory_config)
    logger = logging.stdout_logger_if_logger_is_None(logger)

    output_files = {}

    logger.info(f"opening output files ...")
    with snt.open(
        opj(tmp_dir, "event_table.snt.zip"),
        mode="w",
        dtypes=event_table.structure.dtypes(),
        index_key=event_table.structure.UID_DTYPE[0],
        compress=True,
    ) as event_table_snt_zip, plenopy.photon_stream.loph.LopfTarWriter(
        path=opj(tmp_dir, "reconstructed_cherenkov.loph.tar")
    ) as reconstructed_cherenkov_loph_tar, zipfile.ZipFile(
        file=opj(tmp_dir, "ground_grid_intensity.zip"), mode="w"
    ) as ground_grid_intensity_zip, zipfile.ZipFile(
        file=opj(tmp_dir, "ground_grid_intensity_roi.zip"), mode="w"
    ) as ground_grid_intensity_roi_zip, snt.open(
        file=opj(tmp_dir, "benchmark.snt.zip"),
        mode="w",
        dtypes={"benchmark": _reduce__benchmark_snt_zip__make_dtype()},
        index_key="run_id",
        compress=True,
    ) as benchmark_snt_zip, open(
        opj(tmp_dir, "event_uids_for_debugging.txt"),
        mode="wt",
    ) as event_uids_for_debugging_txt:
        logger.info(f"opening output files ... done.")

        output_files["event_table.snt.zip"] = event_table_snt_zip
        output_files["reconstructed_cherenkov.loph.tar"] = (
            reconstructed_cherenkov_loph_tar
        )
        output_files["ground_grid_intensity.zip"] = ground_grid_intensity_zip
        output_files["ground_grid_intensity_roi.zip"] = (
            ground_grid_intensity_roi_zip
        )
        output_files["benchmark.snt.zip"] = benchmark_snt_zip
        output_files["event_uids_for_debugging.txt"] = (
            event_uids_for_debugging_txt
        )

        _reduce_loop_over_input_runs(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_ids=run_ids,
            output_files=output_files,
            memory_config=memory_config,
            logger=logger,
        )


def _reduce_loop_over_input_runs(
    instrument_site_particle_dir,
    run_ids,
    output_files,
    memory_config=None,
    logger=None,
):
    memory_config = memory.make_config_if_None(memory_config)
    logger = logging.stdout_logger_if_logger_is_None(logger)

    for i in range(len(run_ids)):
        run_id = run_ids[i]
        logger.info(f"run_id {run_id:06d} ({i+1:d} of {len(run_ids):d}) ...")
        _reduce_handle_single_run_files(
            instrument_site_particle_dir=instrument_site_particle_dir,
            run_id=run_id,
            output_files=output_files,
            memory_config=memory_config,
            logger=logger,
        )


def _reduce_handle_single_run_files(
    instrument_site_particle_dir,
    run_id,
    output_files,
    memory_config=None,
    logger=None,
):
    memory_config = memory.make_config_if_None(memory_config)
    logger = logging.stdout_logger_if_logger_is_None(logger)

    open_mem = open_and_read_into_memory_when_small_enough
    map_dir = opj(instrument_site_particle_dir, "map")

    run_zip_buffers = {}

    logger.info(f"opening and reading run zipfiles ...")
    with open_mem(
        opj(map_dir, "prm2cer", f"{run_id:06d}.prm2cer.zip"),
        size=memory_config["read_buffer_size"],
    ) as f_prm2cer, ZipFileBufferIO(file=f_prm2cer) as z_prm2cer, open_mem(
        opj(map_dir, "cer2cls", f"{run_id:06d}.cer2cls.zip"),
        size=memory_config["read_buffer_size"],
    ) as f_cer2cls, ZipFileBufferIO(
        file=f_cer2cls
    ) as z_cer2cls, open_mem(
        opj(map_dir, "cls2rec", f"{run_id:06d}.cls2rec.zip"),
        size=memory_config["read_buffer_size"],
    ) as f_cls2rec, ZipFileBufferIO(
        file=f_cls2rec
    ) as z_cls2rec:
        logger.info(f"opening and reading run zipfiles ... done.")

        run_zip_buffers["prm2cer"] = z_prm2cer
        run_zip_buffers["cer2cls"] = z_cer2cls
        run_zip_buffers["cls2rec"] = z_cls2rec

        _reduce_append_single_run(
            run_id=run_id,
            run_zip_buffers=run_zip_buffers,
            output_files=output_files,
            logger=logger,
        )


def _reduce_append_single_run(
    run_id,
    run_zip_buffers,
    output_files,
    logger=None,
):
    logger = logging.stdout_logger_if_logger_is_None(logger)

    with logging.TimeDelta(logger, "event_table.snt.zip"):
        _reduce__event_table_snt_zip(
            run_zip_buffers=run_zip_buffers,
            event_table_snt_zip=output_files["event_table.snt.zip"],
        )

    with logging.TimeDelta(logger, "reconstructed_cherenkov.loph.tar"):
        _reduce__reconstructed_cherenkov_loph_tar(
            run_cer2cls_zip_buffer=run_zip_buffers["cer2cls"],
            run_id=run_id,
            reconstructed_cherenkov_loph_tar=output_files[
                "reconstructed_cherenkov.loph.tar"
            ],
        )

    with logging.TimeDelta(logger, "ground_grid_intensity.zip"):
        _reduce__ground_grid_intensity_zip(
            run_prm2cer_zip_buffer=run_zip_buffers["prm2cer"],
            run_cer2cls_zip_buffer=run_zip_buffers["cer2cls"],
            run_id=run_id,
            ground_grid_intensity_zip=output_files[
                "ground_grid_intensity.zip"
            ],
            roi=False,
        )

    with logging.TimeDelta(logger, "ground_grid_intensity_roi.zip"):
        _reduce__ground_grid_intensity_zip(
            run_prm2cer_zip_buffer=run_zip_buffers["prm2cer"],
            run_cer2cls_zip_buffer=run_zip_buffers["cer2cls"],
            run_id=run_id,
            ground_grid_intensity_zip=output_files[
                "ground_grid_intensity_roi.zip"
            ],
            roi=True,
        )

    with logging.TimeDelta(logger, "benchmark.snt.zip"):
        _reduce__benchmark_snt_zip(
            run_prm2cer_zip_buffer=run_zip_buffers["prm2cer"],
            run_id=run_id,
            benchmark_snt_zip=output_files["benchmark.snt.zip"],
        )

    with logging.TimeDelta(logger, "event_uids_for_debugging.txt"):
        _reduce__event_uids_for_debugging_txt(
            run_prm2cer_zip_buffer=run_zip_buffers["prm2cer"],
            run_id=run_id,
            event_uids_for_debugging_txt=output_files[
                "event_uids_for_debugging.txt"
            ],
        )


def _reduce__event_table_snt_zip(
    run_zip_buffers,
    event_table_snt_zip,
):
    for checkpoint_key in ["prm2cer", "cer2cls", "cls2rec"]:
        checkpoint_zip_buffer = run_zip_buffers[checkpoint_key]

        for filename in checkpoint_zip_buffer.filenames:
            if filename.endswith("event_table.snt.zip"):
                buff = checkpoint_zip_buffer.read(path=filename, mode="rb")

                with snt.open(file=buff, mode="r") as qpart:
                    part_evttab = qpart.query()
                    event_table_snt_zip.append_table(part_evttab)


def _reduce__reconstructed_cherenkov_loph_tar(
    run_cer2cls_zip_buffer,
    run_id,
    reconstructed_cherenkov_loph_tar,
):
    filename = opj(
        f"{run_id:06d}",
        "cer2cls",
        "simulate_instrument_and_reconstruct_cherenkov",
        "reconstructed_cherenkov.loph.tar",
    )

    buff = run_cer2cls_zip_buffer.read(path=filename, mode="rb")

    with plenopy.photon_stream.loph.LopfTarReader(fileobj=buff) as lin:
        for event in lin:
            uid, phs = event
            reconstructed_cherenkov_loph_tar.add(uid=uid, phs=phs)


def _reduce__ground_grid_intensity_zip(
    run_prm2cer_zip_buffer,
    run_cer2cls_zip_buffer,
    run_id,
    ground_grid_intensity_zip,
    roi=False,
    only_past_trigger=True,
):
    _suffix = "_roi" if roi else ""
    filename = opj(
        f"{run_id:06d}",
        "prm2cer",
        "simulate_shower_and_collect_cherenkov_light_in_grid",
        f"ground_grid_intensity{_suffix:s}.tar",
    )
    buff = run_prm2cer_zip_buffer.read(filename, mode="rb")

    if only_past_trigger:
        _evttab = run_cer2cls_zip_buffer.read_event_table(
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
                with ground_grid_intensity_zip.open(item.name, "w") as fout:
                    fout.write(payload)


def _reduce__benchmark_snt_zip(
    run_prm2cer_zip_buffer,
    run_id,
    benchmark_snt_zip,
):
    hostname, time_unix_s = (
        _reduce__benchmark_snt_zip__get_hostname_and_time_unix(
            run_prm2cer_zip_buffer=run_prm2cer_zip_buffer,
            run_id=run_id,
        )
    )

    benchmark_results = _reduce__benchmark_snt_zip__get_benchmark_results(
        run_prm2cer_zip_buffer=run_prm2cer_zip_buffer,
        run_id=run_id,
    )

    benchmark_record = _reduce__benchmark_snt_zip__make_record(
        run_id=run_id,
        hostname=hostname,
        time_unix_s=time_unix_s,
        benchmark_results=benchmark_results,
    )

    benchmark_level = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=_reduce__benchmark_snt_zip__make_dtype()
    )
    benchmark_level.append(benchmark_record)
    benchmark_snt_zip.append_table({"benchmark": benchmark_level})


def _reduce__benchmark_snt_zip__get_hostname_and_time_unix(
    run_prm2cer_zip_buffer,
    run_id,
):
    _buff = run_prm2cer_zip_buffer.read(
        path=opj(
            f"{run_id:06d}",
            "prm2cer",
            "gather_and_export_provenance",
            "provenance.json.gz",
        ),
        mode="rt|gz",
    )
    item = json_utils.loads(_buff.read())
    return item["hostname"], item["time"]["unix"]


def _reduce__benchmark_snt_zip__get_benchmark_results(
    run_prm2cer_zip_buffer,
    run_id,
):
    _buff = run_prm2cer_zip_buffer.read(
        path=opj(
            f"{run_id:06d}",
            "prm2cer",
            "benchmark_compute_environment",
            "benchmark.json.gz",
        ),
        mode="rt|gz",
    )
    return json_utils.loads(_buff.read())


def _reduce__benchmark_snt_zip__make_record(
    run_id,
    hostname,
    time_unix_s,
    benchmark_results,
):
    bench = benchmark_results

    rec = {}
    rec["hostname_hash"] = hash(hostname)
    rec["time_unix_s"] = time_unix_s
    rec["run_id"] = run_id
    ccc = bench["corsika"]
    rec["corsika_total_s"] = ccc["total"]
    rec["corsika_initializing_s"] = ccc["initializing"]
    rec["corsika_energy_rate_GeV_per_s_avg"] = ccc["energy_rate_GeV_per_s"][
        "avg"
    ]
    rec["corsika_energy_rate_GeV_per_s_std"] = ccc["energy_rate_GeV_per_s"][
        "std"
    ]
    rec["corsika_cherenkov_bunch_rate_per_s_avg"] = ccc[
        "cherenkov_bunch_rate_per_s"
    ]["avg"]
    rec["corsika_cherenkov_bunch_rate_per_s_std"] = ccc[
        "cherenkov_bunch_rate_per_s"
    ]["avg"]
    ddd = bench["disk_write_rate"]
    rec["disk_write_rate_1k_rate_MB_per_s_avg"] = ddd["1k"]["rate_MB_per_s"][
        "avg"
    ]
    rec["disk_write_rate_1k_rate_MB_per_s_std"] = ddd["1k"]["rate_MB_per_s"][
        "std"
    ]
    rec["disk_write_rate_1M_rate_MB_per_s_avg"] = ddd["1M"]["rate_MB_per_s"][
        "avg"
    ]
    rec["disk_write_rate_1M_rate_MB_per_s_std"] = ddd["1M"]["rate_MB_per_s"][
        "std"
    ]
    rec["disk_write_rate_100M_rate_MB_per_s_avg"] = ddd["100M"][
        "rate_MB_per_s"
    ]["avg"]
    rec["disk_write_rate_100M_rate_MB_per_s_std"] = ddd["100M"][
        "rate_MB_per_s"
    ]["std"]
    ddd = bench["disk_create_write_close_open_read_remove_latency"]
    rec["disk_create_write_close_open_read_remove_latency_avg"] = ddd["avg"]
    rec["disk_create_write_close_open_read_remove_latency_std"] = ddd["std"]
    return rec


def _reduce__benchmark_snt_zip__make_dtype():
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


def _reduce__event_uids_for_debugging_txt(
    run_prm2cer_zip_buffer,
    run_id,
    event_uids_for_debugging_txt,
):
    buff = run_prm2cer_zip_buffer.read(
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
        event_uids_for_debugging_txt.write(f"{event_uid:012d}\n")
