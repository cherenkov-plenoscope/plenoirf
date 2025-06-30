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
parser.add_argument("--debug", action="store_true")

NUM_PER_SUBMISSION = 192
MAX_NUM_BLOCKS = 24
NUM_PENDING = 96
POLLING_INTERVAL_S = 60


def time_stamp():
    return pypoolparty.utils.time_now_iso8601().replace(":", "-")


args = parser.parse_args()
queue = args.queue
assert queue in ["sun_grid_engine", "slurm"]
plenoirf_dir = args.plenoirf_dir
assert os.path.exists(plenoirf_dir)

work_dir = f".keep_queue_busy.{time_stamp():s}"
os.makedirs(work_dir)


def log(msg):
    print(
        "[keep_queue_busy]",
        time_stamp(),
        msg,
    )


def process_open(path, args, cwd=None):
    pro = {}
    pro["path"] = path
    pro["o"] = open(path + ".o.part", "wt")
    pro["e"] = open(path + ".e.part", "wt")
    pro["p"] = subprocess.Popen(
        args=args, cwd=cwd, stdout=pro["o"], stderr=pro["e"]
    )
    return pro


def process_close(pro):
    assert pro["p"] is not None
    pro["o"].close()
    os.rename(pro["path"] + ".o.part", pro["path"] + ".o")
    pro["e"].close()
    os.rename(pro["path"] + ".e.part", pro["path"] + ".e")


def process_poll(pro):
    v = pro["p"].poll()
    if v is None:
        return False
    else:
        return True


def _make_pool_script(queue):
    script = ""
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
    return script


def make_plenoirf_submission_script(queue, plenoirf_dir, num_jobs):
    script = ""
    script += "import plenoirf\n"
    script += "import pypoolparty\n"
    script += "\n"
    script += _make_pool_script(queue=queue)
    script += "\n"
    script += "plenoirf.run(\n"
    script += "    plenoirf_dir='{:s}',\n".format(plenoirf_dir)
    script += "    pool=pool,\n"
    script += "    max_num_runs={:d},\n".format(num_jobs)
    script += ")\n"
    return script


def make_debug_submission_script(queue, num_jobs):
    script = ""
    script += "import numpy as np\n"
    script += "import pypoolparty\n"
    script += "\n"
    script += _make_pool_script(queue=queue)
    script += "\n"
    script += "jobs = np.arange(0, {:d})\n".format(num_jobs)
    script += "pool.map(np.sum, jobs)\n"
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


if args.debug:
    script = make_debug_submission_script(
        queue=queue, num_jobs=NUM_PER_SUBMISSION
    )
else:
    script = make_submission_script(
        queue=queue, plenoirf_dir=plenoirf_dir, num_jobs=NUM_PER_SUBMISSION
    )

script_filename = "submission_script.py"
script_path = os.path.join(work_dir, script_filename)
with open(script_path + ".part", "wt") as f:
    f.write(script)
os.rename(script_path + ".part", script_path)

i = 0
blocks = {}
while True:
    i += 1
    num_jobs_pending = query_number_of_pending_jobs(queue=queue)
    log(
        f"jobs pending: {num_jobs_pending:d}, "
        f"blocks running: {len(blocks):d}/{MAX_NUM_BLOCKS:d}."
    )

    have_returned = []
    for j in blocks:
        if process_poll(blocks[j]):
            log(f"Block {j:d} has returned.")
            have_returned.append(j)

    for j in have_returned:
        log(f"Closing block {j:d}.")
        process_close(blocks[j])
        _ = blocks.pop(j)

    if num_jobs_pending < NUM_PENDING:
        log(f"Queue has free slots.")
        if len(blocks) < MAX_NUM_BLOCKS:
            log(f"Opening block {i:d}.")
            log(f"submitting {NUM_PER_SUBMISSION:d} more jobs.")

            blocks[i] = process_open(
                path=os.path.join(work_dir, f"{i:06d}"),
                args=["python", script_filename],
                cwd=work_dir,
            )
        else:
            log(
                f"Can not open new block. "
                f"Already {len(blocks):d}/{MAX_NUM_BLOCKS:d} blocks running."
            )
    else:
        log(f"Queue is full.")

    if args.debug:
        input("press enter to continue.")
    else:
        time.sleep(POLLING_INTERVAL_S)
