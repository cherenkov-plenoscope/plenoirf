#!/usr/bin/python
import sys
from os.path import join as opj
import os
import io
import pandas as pd
import numpy as np
import json_utils
import plenoirf as irf
import sparse_numeric_table as spt
import sebastians_matplotlib_addons as seb

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]


def read_csv_records(path):
    return pd.read_csv(path).to_records(index=False)


def write_csv_records(path, table):
    df = pd.DataFrame(table)
    with open(path + ".tmp", "wt") as f:
        f.write(df.to_csv(index=False))
    os.rename(path + ".tmp", path)


def _num_events_in_runs(event_table, level_key, run_ids, key):
    num_events_in_run = {}
    for run_id in run_ids:
        num_events_in_run[run_id] = 0
    event_run_ids, _ = irf.unique.split_uid(
        uid=event_table[level_key][spt.IDX]
    )
    for event_run_id in event_run_ids:
        try:
            num_events_in_run[event_run_id] += 1
        except KeyError:
            pass
    _out = []
    for run_id in num_events_in_run:
        _out.append({"run_id": run_id, key: int(num_events_in_run[run_id])})
    out = pd.DataFrame(_out).to_records(index=False)
    return out


def merge_event_table(runtime_table, event_table):
    runtime = runtime_table
    num_events_corsika = _num_events_in_runs(
        event_table=event_table,
        level_key="primary",
        run_ids=runtime["run_id"],
        key="num_events_corsika",
    )
    num_events_merlict = _num_events_in_runs(
        event_table=event_table,
        level_key="trigger",
        run_ids=runtime["run_id"],
        key="num_events_merlict",
    )
    num_events_past_trigger = _num_events_in_runs(
        event_table=event_table,
        level_key="pasttrigger",
        run_ids=runtime["run_id"],
        key="num_events_pasttrigger",
    )
    rta = pd.DataFrame(runtime)
    rta = pd.merge(rta, pd.DataFrame(num_events_corsika), on=["run_id"])
    rta = pd.merge(rta, pd.DataFrame(num_events_merlict), on=["run_id"])
    rta = pd.merge(rta, pd.DataFrame(num_events_past_trigger), on=["run_id"])
    return rta.to_records(index=False)


def write_relative_runtime(table, out_path, figure_style):
    ert = table
    total_times = {}
    total_time = 0

    KEYS = []
    for key in ert.dtype.names:
        if "num_" not in key and "run_id" not in key:
            KEYS.append(key)

    for key in KEYS:
        total_times[key] = np.sum(ert[key])
        total_time += total_times[key]

    relative_times = {}
    for key in KEYS:
        relative_times[key] = float(total_times[key] / total_time)

    fig = seb.figure(figure_style)
    ax = seb.add_axes(fig=fig, span=[0.5, 0.15, 0.45, 0.8])
    labels = []
    sizes = []
    _y = np.arange(len(KEYS))
    for ikey, key in enumerate(relative_times):
        labels.append(key)
        sizes.append(relative_times[key])
        x = relative_times[key]
        ax.plot(
            [0, x, x, 0],
            [_y[ikey] - 0.5, _y[ikey] - 0.5, _y[ikey] + 0.5, _y[ikey] + 0.5],
            "k",
        )
    ax.set_xlabel("relative runtime / 1")
    ax.set_yticks(_y)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlim([0, 1])
    out_path_jpg = out_path + ".jpg"
    fig.savefig(out_path_jpg + ".tmp.jpg")
    os.rename(out_path_jpg + ".tmp.jpg", out_path_jpg)
    seb.close(fig)
    out_path_json = out_path + ".json"
    with open(out_path_json + ".tmp", "wt") as fout:
        fout.write(json_utils.dumps(relative_times))
    os.rename(out_path_json + ".tmp", out_path_json)


def write_speed(table, out_path, figure_style):
    ert = table
    speed_keys = {
        "corsika_and_grid": "num_events_corsika",
        "merlict": "num_events_merlict",
        "pass_loose_trigger": "num_events_merlict",
        "classify_cherenkov": "num_events_pasttrigger",
        "extract_features": "num_events_pasttrigger",
        "estimate_primary_trajectory": "num_events_pasttrigger",
    }
    speeds = {}
    for key in speed_keys:
        num_events = ert[speed_keys[key]]
        mask = num_events > 0
        if np.sum(mask) == 0:
            speeds[key] = 0.0
        else:
            speeds[key] = float(np.mean(num_events[mask] / ert[key][mask]))

    fig = seb.figure(figure_style)
    ax = seb.add_axes(fig=fig, span=[0.5, 0.15, 0.45, 0.8])
    labels = []
    sizes = []
    _y = np.arange(len(speeds))
    for ikey, key in enumerate(speeds):
        labels.append(key)
        sizes.append(speeds[key])
        x = speeds[key]
        ax.plot(
            [0, x, x, 0],
            [_y[ikey] - 0.5, _y[ikey] - 0.5, _y[ikey] + 0.5, _y[ikey] + 0.5],
            "k",
        )
    sizes = np.array(sizes)
    valid = np.logical_not(np.logical_or(np.isinf(sizes), np.isnan(sizes)))
    ax.set_xlabel("processing-rate / events s$^{-1}$")
    ax.set_yticks(_y)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlim([0, np.max(sizes[valid]) * 1.1])
    fig.savefig(out_path + ".tmp" + ".jpg")
    os.rename(out_path + ".tmp" + ".jpg", out_path + ".jpg")
    seb.close(fig)
    with open(out_path + ".json" + ".tmp", "wt") as fout:
        fout.write(json_utils.dumps(speeds))
    os.rename(out_path + ".json" + ".tmp", out_path + ".json")


os.makedirs(pa["out_dir"], exist_ok=True)

for sk in SITES:
    for pk in PARTICLES:
        prefix_str = "{:s}_{:s}".format(sk, pk)

        extended_runtime_path = opj(pa["out_dir"], prefix_str + "_runtime.csv")
        if os.path.exists(extended_runtime_path):
            extended_runtime_table = read_csv_records(extended_runtime_path)
        else:
            event_table = spt.read(
                path=os.path.join(
                    pa["run_dir"],
                    "event_table",
                    sk,
                    pk,
                    "event_table.tar",
                ),
                structure=irf.table.STRUCTURE,
            )
            runtime_table = read_csv_records(
                opj(
                    pa["run_dir"],
                    "event_table",
                    sk,
                    pk,
                    "runtime.csv",
                )
            )
            extended_runtime_table = merge_event_table(
                runtime_table=runtime_table, event_table=event_table
            )
            write_csv_records(
                path=extended_runtime_path, table=extended_runtime_table
            )

        write_relative_runtime(
            table=extended_runtime_table,
            out_path=opj(pa["out_dir"], prefix_str + "_relative_runtime"),
            figure_style=seb.FIGURE_1_1,
        )

        write_speed(
            table=extended_runtime_table,
            out_path=opj(pa["out_dir"], prefix_str + "_speed_runtime"),
            figure_style=seb.FIGURE_1_1,
        )

        ertt = extended_runtime_table

        with np.errstate(divide="ignore"):
            t_corsika = np.median(
                ertt["corsika_and_grid"] / ertt["num_events_corsika"]
            )
            f_corsika = 1.0

            t_merlict = np.median(ertt["merlict"] / ertt["num_events_merlict"])
            f_merlict = np.median(
                ertt["num_events_merlict"] / ertt["num_events_corsika"]
            )

            t_pltrg = np.median(
                ertt["pass_loose_trigger"] / ertt["num_events_merlict"]
            )
            f_pltrg = f_merlict

            t_clscer = np.median(
                ertt["classify_cherenkov"] / ertt["num_events_pasttrigger"]
            )
            f_clscer = np.median(
                ertt["num_events_pasttrigger"] / ertt["num_events_corsika"]
            )

            t_extft = np.median(
                ertt["extract_features"] / ertt["num_events_pasttrigger"]
            )
            f_extft = f_clscer

            t_traje = np.median(
                ertt["estimate_primary_trajectory"]
                / ertt["num_events_pasttrigger"]
            )
            f_traje = f_clscer

        ostr = io.StringIO()
        ostr.write(
            "corsika and grid            {: 6.1f}s{: 4.0f}%\n".format(
                t_corsika, 100 * f_corsika
            )
        )
        ostr.write(
            "ray tracing and electronics {: 6.1f}s{: 4.0f}%\n".format(
                t_merlict, 100 * f_merlict
            )
        )
        ostr.write(
            "simulating trigger          {: 6.1f}s{: 4.0f}%\n".format(
                t_pltrg, 100 * f_pltrg
            )
        )
        ostr.write(
            "Cherenkov cleaning          {: 6.1f}s{: 4.0f}%\n".format(
                t_clscer, 100 * f_clscer
            )
        )
        ostr.write(
            "extracting features         {: 6.1f}s{: 4.0f}%\n".format(
                t_extft, 100 * f_extft
            )
        )
        ostr.write(
            "reconstruct trajectory      {: 6.1f}s{: 4.0f}%\n".format(
                t_traje, 100 * f_traje
            )
        )

        ostr.seek(0)
        o = ostr.read()
        print(o)
