import importlib.resources
import os
import glob
import subprocess
import multiprocessing
import copy
import time
import json_utils
import datetime
from .. import provenance


def get_scripts_dir():
    return os.path.join(
        importlib.resources.files("plenoirf"), "summary", "scripts"
    )


def list_script_names(scripts_dir=None):
    if scripts_dir is None:
        scripts_dir = get_scripts_dir()

    script_paths = glob.glob(os.path.join(scripts_dir, "*.py"))

    script_filenames = [os.path.basename(s) for s in script_paths]
    _script_names = [os.path.splitext(s)[0] for s in script_filenames]
    script_names = []
    for sn in _script_names:
        if str.isdigit(sn[0:4]):
            script_names.append(sn)
    script_names.sort()
    return script_names


def estimate_script_dependencies(scripts_dir=None, script_names=None):
    if scripts_dir is None:
        scripts_dir = get_scripts_dir()
    if script_names is None:
        script_names = list_script_names(scripts_dir=scripts_dir)

    job_dependencies = {}
    for script_name in script_names:
        job_dependencies[script_name] = []
        script_path = os.path.join(scripts_dir, script_name + ".py")
        with open(script_path, "rt") as fin:
            code = fin.read()
            for sn in script_names:
                if str.find(code, sn) >= 0:
                    job_dependencies[script_name].append(sn)
    return job_dependencies


def _make_script_out_paths(
    script_name, plenoirf_dir, instrument_key, site_key, scripts_dir=None
):
    if scripts_dir is None:
        scripts_dir = get_scripts_dir()

    out_dir = os.path.join(
        plenoirf_dir, "analysis", instrument_key, site_key, script_name
    )
    return {
        "out_dir": out_dir,
        "stdout_path": os.path.join(out_dir, "stdout.txt"),
        "stderr_path": os.path.join(out_dir, "stderr.txt"),
    }


def call_script_in_subprocess(
    script_name,
    plenoirf_dir,
    instrument_key,
    site_key,
    scripts_dir=None,
    skip_when_script_out_dir_exists=False,
):
    if scripts_dir is None:
        scripts_dir = get_scripts_dir()

    paths = _make_script_out_paths(
        script_name=script_name,
        plenoirf_dir=plenoirf_dir,
        instrument_key=instrument_key,
        site_key=site_key,
        scripts_dir=scripts_dir,
    )

    if skip_when_script_out_dir_exists:
        if os.path.exists(paths["out_dir"]):
            rc = -100
            if os.path.exists(paths["stdout_path"]):
                stdout_size = os.stat(paths["stdout_path"]).st_size
            else:
                stdout_size = -1
            if os.path.exists(paths["stderr_path"]):
                stderr_size = os.stat(paths["stderr_path"]).st_size
            else:
                stderr_size = -1
            return rc, stdout_size, stderr_size

    script_path = os.path.join(scripts_dir, script_name + ".py")
    os.makedirs(paths["out_dir"], exist_ok=True)

    call = ["python", script_path, plenoirf_dir, instrument_key, site_key]
    rc = -1
    with open(paths["stdout_path"] + ".part", "wt") as o, open(
        paths["stderr_path"] + ".part", "wt"
    ) as e:
        rc = subprocess.call(call, stdout=o, stderr=e)

    os.rename(src=paths["stdout_path"] + ".part", dst=paths["stdout_path"])
    os.rename(src=paths["stderr_path"] + ".part", dst=paths["stderr_path"])

    stdout_size = os.stat(paths["stdout_path"]).st_size
    stderr_size = os.stat(paths["stderr_path"]).st_size

    return rc, stdout_size, stderr_size


def script_output_is_complete(
    script_name, plenoirf_dir, instrument_key, site_key, scripts_dir=None
):
    if scripts_dir is None:
        scripts_dir = get_scripts_dir()
    paths = _make_script_out_paths(
        script_name=script_name,
        plenoirf_dir=plenoirf_dir,
        instrument_key=instrument_key,
        site_key=site_key,
        scripts_dir=scripts_dir,
    )
    exists = os.path.exists
    return exists(paths["stdout_path"]) and exists(paths["stderr_path"])


def list_dependencies_of_script(script_name, script_dependencies):
    keeper = [script_name]
    keeper = _add_dependencies(
        dependencies=script_dependencies, target=script_name, keeper=keeper
    )
    return keeper


def _add_dependencies(dependencies, target, keeper):
    for item in dependencies[target]:
        keeper.append(item)
        keeper = _add_dependencies(
            dependencies=dependencies,
            target=item,
            keeper=keeper,
        )
    return keeper


def strip_script_dependencies(script_name, script_dependencies):
    keepers = list_dependencies_of_script(
        script_name=script_name,
        script_dependencies=script_dependencies,
    )
    out = {}
    for keeper in keepers:
        out[keeper] = copy.copy(script_dependencies[keeper])
    return out


def _safe_cpu_count():
    return multiprocessing.cpu_count() // 2 or 1


def run(
    plenoirf_dir,
    instrument_key,
    site_key,
    script_name=None,
    pool=None,
    polling_interval=1.0,
    num_threads=None,
    skip_when_script_out_dir_exists=False,
):
    if num_threads is None:
        num_threads = _safe_cpu_count()

    if pool is None:
        pool = multiprocessing.Pool(num_threads)

    script_dependencies = estimate_script_dependencies()
    if script_name is not None:
        script_dependencies = strip_script_dependencies(
            script_name=script_name, script_dependencies=script_dependencies
        )

    out_dir = os.path.join(plenoirf_dir, "analysis", instrument_key, site_key)
    os.makedirs(out_dir, exist_ok=True)
    _dt = datetime.datetime.now()
    _now_str = _dt.isoformat().replace(":", "-").replace(".", "p")
    json_utils.write(
        os.path.join(out_dir, f"provenance.{_now_str:s}.json"),
        provenance.make_provenance(),
    )

    script_names = list(script_dependencies.keys())
    scripts_dir = get_scripts_dir()

    job_statii = {}
    job_asyncs = {}
    job_stderr_len = {}
    for name in script_names:
        job_statii[name] = "pending"
        job_asyncs[name] = None
        job_stderr_len[name] = None

    num_polls = 0
    while True:
        if _num_job_statii(job_statii, "error"):
            break

        if _num_job_statii(job_statii, "complete") == len(job_statii):
            break

        num_free_threads = num_threads - _num_job_statii(job_statii, "running")
        jobs_ready_to_run = _find_jobs_ready_to_run(
            job_statii=job_statii, script_dependencies=script_dependencies
        )
        num_jobs_to_submit = min([len(jobs_ready_to_run), num_free_threads])

        for ii in range(num_jobs_to_submit):
            name = jobs_ready_to_run[ii]
            job_statii[name] = "running"
            job_asyncs[name] = pool.apply_async(
                call_script_in_subprocess,
                (
                    name,
                    plenoirf_dir,
                    instrument_key,
                    site_key,
                    scripts_dir,
                    skip_when_script_out_dir_exists,
                ),
            )

        for name in script_names:
            if job_asyncs[name] is not None:
                if job_asyncs[name].ready():
                    if job_asyncs[name].successful():
                        _script_rc, _stdout_size, _stderr_size = job_asyncs[
                            name
                        ].get()

                        job_stderr_len[name] = _stderr_size
                        job_statii[name] = "complete"
                    else:
                        job_statii[name] = "error"

                    job_asyncs[name] = None
                else:
                    assert job_statii[name] == "running"

        print("\n\n")
        print("[P]ending [R]unning [C]omplete len(stderr)")
        print("------------------------------------------ Polls:", num_polls)
        for name in script_names:
            status = job_statii[name]
            if status == "pending":
                print("{:<80s}    [P]. .  -".format(name))
            elif status == "running":
                print("{:<80s}     .[R].  -".format(name))
            elif status == "complete":
                elen = job_stderr_len[name]
                print("{:<80s}     . .[C] {:d}".format(name, elen))
            elif status == "error":
                print("{:<80s}     [ERROR]".format(name))
            else:
                print("{:<80s}     ? ? ? <{:s}>".format(name, status))

        time.sleep(polling_interval)
        num_polls += 1


def _num_job_statii(jobs, status):
    num = 0
    for name in jobs:
        if jobs[name] == status:
            num += 1
    return num


def _find_jobs_ready_to_run(job_statii, script_dependencies):
    jobs_ready_to_run = []
    for name in job_statii:
        if job_statii[name] == "pending":
            num_complete = 0
            for dep_name in script_dependencies[name]:
                if job_statii[dep_name] == "complete":
                    num_complete += 1
            if num_complete == len(script_dependencies[name]):
                jobs_ready_to_run.append(name)
    jobs_ready_to_run.sort()
    return jobs_ready_to_run
