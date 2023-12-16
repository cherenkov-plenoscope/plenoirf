import pkg_resources
import os
import glob
import subprocess
import time
import json_utils
from .. import provenance


def find_script_names(script_dir):
    script_paths = glob.glob(os.path.join(script_dir, "*.py"))

    script_filenames = [os.path.basename(s) for s in script_paths]
    _script_names = [os.path.splitext(s)[0] for s in script_filenames]
    script_names = []
    for sn in _script_names:
        if str.isdigit(sn[0:4]):
            script_names.append(sn)
    script_names.sort()
    return script_names


def find_job_dependencies(script_dir, script_names):
    job_dependencies = {}
    for script_name in script_names:
        job_dependencies[script_name] = []
        script_path = os.path.join(script_dir, script_name + ".py")
        with open(script_path, "rt") as fin:
            code = fin.read()
            for sn in script_names:
                if str.find(code, sn) >= 0:
                    job_dependencies[script_name].append(sn)
    return job_dependencies


def num_jobs(job_statii, status):
    num = 0
    for name in job_statii:
        if job_statii[name] == status:
            num += 1
    return num


def find_jobs_ready_to_run(job_statii, job_dependencies):
    jobs_ready_to_run = []
    for name in job_statii:
        if job_statii[name] == "pending":
            num_complete = 0
            for dep_name in job_dependencies[name]:
                if job_statii[dep_name] == "complete":
                    num_complete += 1
            if num_complete == len(job_dependencies[name]):
                jobs_ready_to_run.append(name)
    jobs_ready_to_run.sort()
    return jobs_ready_to_run


def find_script_names_not_yet_complete(run_dir, script_names):
    out = []
    for script_name in script_names:
        expected_outdir = os.path.join(run_dir, "summary", script_name)
        if not os.path.exists(expected_outdir):
            out.append(script_name)
    return out


def run_parallel(run_dir, num_threads=6, polling_interval=1):
    json_utils.write(
        path=os.path.join(run_dir, "summary", "provenance.json"),
        out_dict=provenance.make_provenance(),
    )

    script_dir = pkg_resources.resource_filename(
        "plenoirf", os.path.join("summary", "scripts")
    )

    script_names = find_script_names(script_dir=script_dir)
    script_names = find_script_names_not_yet_complete(
        run_dir=run_dir, script_names=script_names
    )
    job_dependencies = find_job_dependencies(
        script_dir=script_dir,
        script_names=script_names,
    )
    job_statii = {}
    job_handles = {}
    job_stderr_len = {}
    for name in script_names:
        job_statii[name] = "pending"
        job_handles[name] = None

    num_polls = 0
    while True:
        if num_jobs(job_statii, "error"):
            break

        if num_jobs(job_statii, "complete") == len(job_statii):
            break

        num_free_threads = num_threads - num_jobs(job_statii, "running")
        jobs_ready_to_run = find_jobs_ready_to_run(
            job_statii, job_dependencies
        )
        num_jobs_to_submit = min([len(jobs_ready_to_run), num_free_threads])

        for ii in range(num_jobs_to_submit):
            name = jobs_ready_to_run[ii]
            script_path = os.path.join(script_dir, name + ".py")
            pii = subprocess.Popen(
                args=["python", script_path, run_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            job_statii[name] = "running"
            job_handles[name] = pii

        for name in script_names:
            job_popen = job_handles[name]
            if job_popen is not None:
                rc = job_popen.poll()

                if rc is None:
                    assert job_statii[name] == "running"
                else:
                    job_stdout, job_stderr = job_handles[name].communicate()
                    job_stderr_len[name] = len(job_stderr)
                    script_out_dir = os.path.join(run_dir, "summary", name)
                    os.makedirs(script_out_dir, exist_ok=True)
                    opath = os.path.join(script_out_dir, "stdout.md")
                    epath = os.path.join(script_out_dir, "stderr.md")
                    with open(opath, "wb") as fo, open(epath, "wb") as fe:
                        fo.write(job_stdout)
                        fe.write(job_stderr)

                    if rc >= 0:
                        job_statii[name] = "complete"
                        job_handles[name] = None
                    else:
                        job_statii[name] = "error"
                        job_handles[name] = None

        print("\n\n")
        print("[P]ending [R]unning [C]omplete len(stderr)")
        print("------------------------------------------ Polls:", num_polls)
        for name in script_names:
            sta = job_statii[name]
            if sta == "pending":
                print("{:<70s}    [P]. .  -".format(name))
            elif sta == "running":
                print("{:<70s}     .[R].  -".format(name))
            elif sta == "complete":
                elen = job_stderr_len[name]
                print("{:<70s}     . .[C] {:d}".format(name, elen))
            else:
                print("{:<70s}     ? ? ?".format(name))

        time.sleep(polling_interval)
        num_polls += 1
