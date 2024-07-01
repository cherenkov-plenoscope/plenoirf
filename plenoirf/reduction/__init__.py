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

from .. import event_table
from .. import configuration


def list_items():
    return [
        "event_table.zip",
        "reconstructed_cherenkov.tar",
        "ground_grid_intensity.zip",
        "ground_grid_intensity_roi.zip",
    ]


def reduce_item(map_dir, out_path, item_key):
    run_paths = glob.glob(os.path.join(map_dir, "*.zip"))
    if item_key == "event_table.zip":
        recude_event_table(run_paths=run_paths, out_path=out_path)
    elif item_key == "reconstructed_cherenkov.tar":
        reduce_reconstructed_cherenkov(run_paths=run_paths, out_path=out_path)
    elif item_key == "ground_grid_intensity.zip":
        reduce_ground_grid_intensity(
            run_paths=run_paths, out_path=out_path, roi=False
        )
    elif item_key == "ground_grid_intensity_roi.zip":
        reduce_ground_grid_intensity(
            run_paths=run_paths, out_path=out_path, roi=True
        )
    else:
        raise KeyError(f"No such item_key '{item_key}'.")


def make_jobs(plenoirf_dir, config=None, lazy=False):
    if config is None:
        config = configuration.read(plenoirf_dir)

    jobs = []
    for instrument_key in config["instruments"]:
        for site_key in config["sites"]["instruemnt_response"]:
            for particle_key in config["particles"]:
                for item_key in list_items():
                    if lazy:
                        job_out_path = os.path.join(
                            plenoirf_dir,
                            "response",
                            instrument_key,
                            "site_key",
                            "particle_key",
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
                    }
                    jobs.append(job)
    return jobs


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
        )


def zip_read_BytesIo(file, internal_path, mode="r"):
    with zipfile.ZipFile(file=file, mode="r") as zin:
        with zin.open(internal_path) as fin:
            buff = io.BytesIO()
            if "|gz" in mode:
                buff.write(gzip.decompress(fin.read()))
            else:
                buff.write(fin.read())
            buff.seek(0)
    return buff


def recude_event_table(run_paths, out_path):
    with snt.archive.open(
        out_path, mode="w", dtypes=event_table.structure.dtypes()
    ) as arc:
        for run_path in run_paths:
            run_basename = os.path.basename(run_path)
            run_id_str = os.path.splitext(run_basename)[0]
            buff = zip_read_BytesIo(
                file=run_path,
                internal_path=os.path.join(run_id_str, "event_table.tar.gz"),
                mode="r|gz",
            )
            run_evttab = snt.read(fileobj=buff, dynamic=False)
            arc.append_table(run_evttab)


def reduce_reconstructed_cherenkov(run_paths, out_path):
    with plenopy.photon_stream.loph.LopfTarWriter(path=out_path) as lout:
        for run_path in run_paths:
            run_basename = os.path.basename(run_path)
            run_id_str = os.path.splitext(run_basename)[0]
            buff = zip_read_BytesIo(
                file=run_path,
                internal_path=os.path.join(
                    run_id_str, "reconstructed_cherenkov.tar"
                ),
                mode="r",
            )
            with plenopy.photon_stream.loph.LopfTarReader(fileobj=buff) as lin:
                for event in lin:
                    uid, phs = event
                    lout.add(uid=uid, phs=phs)


def reduce_ground_grid_intensity(run_paths, out_path, roi=False):
    with zipfile.ZipFile(out_path, "w") as zout:
        for run_path in run_paths:
            run_basename = os.path.basename(run_path)
            run_id_str = os.path.splitext(run_basename)[0]

            suff = "_roi" if roi else ""
            internal_path = os.path.join(
                run_id_str,
                "plenoirf.production.simulate_shower_and_collect_cherenkov_light_in_grid",
                f"ground_grid_intensity{suff:s}.tar",
            )
            buff = zip_read_BytesIo(
                file=run_path,
                internal_path=internal_path,
                mode="r",
            )
            with sequential_tar.open(fileobj=buff, mode="r") as tarin:
                for item in tarin:
                    with zout.open(item.name, "w") as fout:
                        fout.write(item.read(mode="rb"))
