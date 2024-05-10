import pandas as pd
import shutil
import os


def reduce(list_of_log_paths, out_path):
    log_records = reduce_into_records(list_of_log_paths=list_of_log_paths)
    log_df = pd.DataFrame(log_records)
    log_df = log_df.sort_values(by=["run_id"])
    log_df.to_csv(out_path + ".tmp", index=False, na_rep="nan")
    shutil.move(out_path + ".tmp", out_path)


def reduce_into_records(list_of_log_paths):
    list_of_log_records = []
    for log_path in list_of_log_paths:
        run_id = int(os.path.basename(log_path)[0:6])
        run = {"run_id": run_id}

        key = ":delta:"
        with open(log_path, "rt") as fin:
            for line in fin:
                logline = json.loads(line)
                if "m" in logline:
                    msg = logline["m"]
                    if key in msg:
                        iname = str.find(msg, key)
                        name = msg[:(iname)]
                        deltastr = msg[(iname + len(key)) :]
                        run[name] = float(deltastr)
            list_of_log_records.append(run)

    return list_of_log_records
