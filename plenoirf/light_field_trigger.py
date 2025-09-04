import numpy as np
import sparse_numeric_table as snt
import warnings


def height_to_depth(height_m, zenith_rad):
    # an = height
    # hyp = depth
    # cos(z) = an / hyp
    # cos(z) = height / depth
    #
    #           O---------O
    #           |_|     /
    #           |     /
    # an/heihgt | z / hyp/depth
    #           | /
    #           O
    depth_m = height_m / np.cos(zenith_rad)
    return depth_m


def depth_to_height(depth_m, zenith_rad):
    height_m = np.cos(zenith_rad) * depth_m
    return height_m


def assign_to_bin(x, bin_edges):
    return np.digitize(x, bins=bin_edges) - 1


def assign_accepting_and_rejecting_focus_based_on_pointing_zenith(
    pointing_zenith_rad,
    accepting_height_above_observation_level_m,
    rejecting_height_above_observation_level_m,
    trigger_foci_bin_edges_m,
):
    accepting_depth_m = height_to_depth(
        height_m=accepting_height_above_observation_level_m,
        zenith_rad=pointing_zenith_rad,
    )

    rejecting_depth_m = height_to_depth(
        height_m=rejecting_height_above_observation_level_m,
        zenith_rad=pointing_zenith_rad,
    )

    accepting_focus = assign_to_bin(
        x=accepting_depth_m, bin_edges=trigger_foci_bin_edges_m
    )
    rejecting_focus = assign_to_bin(
        x=rejecting_depth_m, bin_edges=trigger_foci_bin_edges_m
    )

    return accepting_focus, rejecting_focus


def copy_focus_response_into_matrix(trigger_table):
    num_events = trigger_table.shape[0]
    num_foci = 0
    for key in trigger_table.dtype.names:
        if "focus_" in key and "_response_pe" in key:
            num_foci += 1

    focus_response_pe = np.zeros(
        shape=(num_events, num_foci),
        dtype=trigger_table[f"focus_00_response_pe"].dtype,
    )

    for f in range(num_foci):
        focus_response_pe[:, f] = trigger_table[f"focus_{f:02d}_response_pe"]

    return focus_response_pe


def find_accepting_and_rejecting_response(
    accepting_focus,
    rejecting_focus,
    focus_response_pe,
):
    num_events = focus_response_pe.shape[0]
    assert accepting_focus.shape[0] == num_events
    assert rejecting_focus.shape[0] == num_events
    num_foci = focus_response_pe.shape[1]

    accepting_response_pe = np.zeros(shape=num_events, dtype=int)
    rejecting_response_pe = np.zeros(shape=num_events, dtype=int)

    for focus in range(num_foci):
        accepting_mask = accepting_focus == focus
        accepting_response_pe[accepting_mask] = focus_response_pe[
            accepting_mask, focus
        ]

        rejecting_mask = rejecting_focus == focus
        rejecting_response_pe[rejecting_mask] = focus_response_pe[
            rejecting_mask, focus
        ]

    return accepting_response_pe, rejecting_response_pe


def get_trigger_threshold_corrected_for_pointing_zenith(
    pointing_zenith_rad, trigger, nominal_threshold_pe
):
    zenith_factor = np.interp(
        x=pointing_zenith_rad,
        xp=trigger["threshold_factor_vs_pointing_zenith"]["zenith_rad"],
        fp=trigger["threshold_factor_vs_pointing_zenith"]["factor"],
    )
    assert np.all(zenith_factor > 0)
    return np.round(zenith_factor * nominal_threshold_pe).astype(int)


def get_accepting_over_rejecting(
    pointing_zenith_rad,
    trigger,
    accepting_response_pe,
):
    pointing_zenith_rad = np.asarray(pointing_zenith_rad)
    accepting_response_pe = np.asarray(accepting_response_pe)
    acc = trigger["modus"]["accepting"]
    num_events = pointing_zenith_rad.shape[0]

    threshold_accepting_over_rejecting = np.interp(
        x=accepting_response_pe,
        xp=acc["response_pe"],
        fp=acc["threshold_accepting_over_rejecting"],
        left=None,
        right=None,
        period=None,
    )
    assert threshold_accepting_over_rejecting.shape[0] == num_events

    zenith_factor = np.interp(
        x=pointing_zenith_rad,
        xp=acc["correction_for_pointing_zenith"]["zenith_rad"],
        fp=acc["correction_for_pointing_zenith"]["factor"],
        left=None,
        right=None,
        period=None,
    )
    assert zenith_factor.shape[0] == num_events

    return threshold_accepting_over_rejecting * zenith_factor


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
    return trigger_table["uid"][mask]


def make_trigger_modus_str(analysis_trigger, production_trigger):
    pro = production_trigger
    ana = analysis_trigger

    acc_alt = ana["modus"]["accepting_altitude_asl_m"]
    rej_alt = ana["modus"]["rejecting_altitude_asl_m"]

    modus = ana["modus"]

    s = ""
    s += "    Accepting focus altitude: "
    s += "{:.2f}km\n".format(1e-3 * acc_alt)
    s += "    Rejecting focus altitude: "
    s += "{:.2f}km\n".format(1e-3 * rej_alt)
    s += "    Intensity ratio between foci:\n"
    s += "       accepting focus   accepting over\n"
    s += "        response / pe     rejecting / 1\n"
    for i in range(len(modus["accepting"]["response_pe"])):
        xp = modus["accepting"]["response_pe"][i]
        fp = modus["accepting"]["threshold_accepting_over_rejecting"][i]
        s += "        {:1.2e}          {:.2f}\n".format(xp, fp)
    s += "Threshold\n"
    s += "    {:d}p.e. ".format(ana["threshold_pe"])
    s += "({:d}p.e. in production)\n".format(pro["threshold_pe"])
    return s
