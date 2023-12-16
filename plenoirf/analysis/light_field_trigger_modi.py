import numpy as np
import sparse_numeric_table as spt


def make_mask(
    trigger_table,
    threshold,
    modus,
):
    """
    The plenoscope's trigger has written its response into the trigger-table.
    The response includes the max. photon-equivalent seen in each
    trigger-image. The trigger-images are focused to different
    object-distances.
    Based on this response, different modi for the final trigger are possible.
    """
    KEY = "focus_{:02d}_response_pe"

    assert threshold >= 0
    assert modus["accepting_focus"] >= 0
    assert modus["rejecting_focus"] >= 0

    accepting_response_pe = trigger_table[KEY.format(modus["accepting_focus"])]
    rejecting_response_pe = trigger_table[KEY.format(modus["rejecting_focus"])]

    threshold_accepting_over_rejecting = np.interp(
        x=accepting_response_pe,
        xp=modus["accepting"]["response_pe"],
        fp=modus["accepting"]["threshold_accepting_over_rejecting"],
        left=None,
        right=None,
        period=None,
    )

    accepting_over_rejecting = accepting_response_pe / rejecting_response_pe

    size_over_threshold = accepting_response_pe >= threshold
    ratio_over_threshold = (
        accepting_over_rejecting >= threshold_accepting_over_rejecting
    )

    return np.logical_and(size_over_threshold, ratio_over_threshold)


def make_indices(
    trigger_table,
    threshold,
    modus,
):
    mask = make_mask(
        trigger_table,
        threshold,
        modus,
    )
    return trigger_table[spt.IDX][mask]


def make_trigger_modus_str(analysis_trigger, production_trigger):
    pro = production_trigger
    ana = analysis_trigger

    acc_foc = ana["modus"]["accepting_focus"]
    acc_obj = pro["object_distances_m"][acc_foc]
    rej_foc = ana["modus"]["rejecting_focus"]
    rej_obj = pro["object_distances_m"][rej_foc]

    modus = ana["modus"]

    s = ""
    s += "Modus\n"
    s += "    Accepting object-distance "
    s += "{:.1f}km, focus {:02d}\n".format(1e-3 * acc_obj, acc_foc)
    s += "    Rejecting object-distance "
    s += "{:.1f}km, focus {:02d}\n".format(1e-3 * rej_obj, rej_foc)
    s += "    Intensity-ratio between foci:\n"
    s += "        response / pe    ratio / 1\n"
    for i in range(len(modus["accepting"]["response_pe"])):
        xp = modus["accepting"]["response_pe"][i]
        fp = modus["accepting"]["threshold_accepting_over_rejecting"][i]
        s += "        {:1.2e}          {:.2f}\n".format(xp, fp)
    s += "Threshold\n"
    s += "    {:d}p.e. ".format(ana["threshold_pe"])
    s += "({:d}p.e. in production)\n".format(pro["threshold_pe"])
    return s
