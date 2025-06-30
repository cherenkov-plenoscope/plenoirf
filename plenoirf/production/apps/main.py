import os
import pypoolparty
import argparse
import plenoirf
import time
import subprocess


parser = argparse.ArgumentParser(
    prog="keep_queue_busy.py",
    description=("Production on compute clusters"),
)
parser.add_argument(
    "plenoirf_dir",
    metavar="PLENOIRF",
    type=str,
    help="root directory of plenoscope irf",
)
parser.add_argument(
    "--queue",
    metavar="QUEUE",
    default="slurm",
    type=str,
    help="Name of the queue system.",
)

NUM_PER_SUBMISSION = 10
NUM_PENDING = 50
polling_interval_s = 30
args = parser.parse_args()
queue = args.queue
assert queue in ["sun_grid_engine", "slurm"]
plenoirf_dir = args.plenoirf_dir
assert os.path.exists(plenoirf_dir)

time_stamp = pypoolparty.utils.time_now_iso8601().replace(":", "-")
work_dir = f".keep_queue_busy.{time_stamp:s}"
os.makedirs(work_dir)


def log(self, msg):
    print(
        "[keep_queue_busy]",
        pypoolparty.utils.time_now_iso8601().replace(":", "-"),
        msg,
    )


def make_submission_script(queue, plenoirf_dir, num_jobs):
    script = ""
    script += "import plenoirf\n"
    script += "import pypoolparty\n"
    script += "\n"
    if queue == "slurm":
        script += "pool = pypoolparty.slurm.Pool(\n"
        script += "    max_num_resubmissions=100,\n"
        script += "    verbose=True,\n"
        script += ")\n"
    elif queue == "sun_grid_engine":
        script += "pool = pypoolparty.sun_grid_engine.Pool(\n"
        script += "    verbose=True,\n"
        script += ")\n"
    else:
        raise ValueError(f"Unknown queue.")
    script += "\n"
    script += "plenoirf.run(\n"
    script += "    plenoirf_dir='{:s}',\n".format(plenoirf_dir)
    script += "    pool=pool,\n"
    script += "    max_num_runs={:d},\n".format(num_jobs)
    script += ")\n"
    return script


def query_number_of_pending_jobs(queue):
    if queue == "slurm":
        jobs = pypoolparty.slurm.calling.squeue()
        _, pending, _ = (
            pypoolparty.slurm.organizing_jobs.split_jobs_in_running_pending_error(
                jobs=jobs
            )
        )
        return len(pending)
    elif queue == "sun_grid_engine":
        jobs = pypoolparty.sun_grid_engine.calling.qstat()
        _, pending, _ = (
            pypoolparty.sun_grid_engine.organizing_jobs.split_jobs_in_running_pending_error(
                jobs=jobs
            )
        )
        return len(pending)


script_filename = "submission_script.py"
script_path = os.path.join(work_dir, script_filename)
with open(script_path, "wt") as f:
    f.write(
        make_submission_script(
            queue=queue, plenoirf_dir=plenoirf_dir, num_jobs=NUM_PER_SUBMISSION
        )
    )


count = 0
while True:
    count += 1
    num_jobs_pending = query_number_of_pending_jobs(queue=queue)
    log(f"jobs pending: {num_jobs_pending:d}.")

    if num_jobs_pending < NUM_PENDING:
        log(f"submitting {NUM_PER_SUBMISSION:d} more jobs.")
        # fire and forget
        _ = subprocess.Popen(
            args=[
                "pyton",
                script_filename,
                ">>",
                f"{count:06d}.txt",
                "2>&1",
            ],
            cwd=work_dir,
            shell=True,
        )
    else:
        log(f"queue is busy.")

    time.sleep(polling_interval_s)
