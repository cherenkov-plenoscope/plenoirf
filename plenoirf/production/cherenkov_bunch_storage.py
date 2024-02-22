import corsika_primary as cpw
import numpy as np
import spherical_coordinates
from . import un_bound_histogram


def _make_fake_runh():
    runh = np.zeros(273, dtype=np.float32)
    runh[cpw.I.RUNH.MARKER] = cpw.I.RUNH.MARKER_FLOAT32
    runh[cpw.I.RUNH.RUN_NUMBER] = 1
    runh[cpw.I.RUNH.NUM_EVENTS] = 1
    return runh


def _make_fake_evth():
    runh = np.zeros(273, dtype=np.float32)
    runh[cpw.I.EVTH.MARKER] = cpw.I.EVTH.MARKER_FLOAT32
    runh[cpw.I.EVTH.RUN_NUMBER] = 1
    runh[cpw.I.EVTH.EVENT_NUMBER] = 1
    return runh


def write(path, event_tape_cherenkov_reader):
    with cpw.cherenkov.CherenkovEventTapeWriter(path=path) as f:
        f.write_runh(_make_fake_runh())
        for event_number in [1]:
            f.write_evth(_make_fake_evth())
            for cherenkov_block in event_tape_cherenkov_reader:
                f.write_payload(cherenkov_block)


def read_with_mask(path, bunch_indices):
    mask_idxs = set(bunch_indices)
    ii = 0
    oo = 0
    outblock = float("nan") * np.ones(
        shape=(len(bunch_indices), cpw.I.BUNCH.NUM_FLOAT32), dtype=np.float32
    )
    with cpw.cherenkov.CherenkovEventTapeReader(path=path) as tr:
        for event in tr:
            evth, cherenkov_reader = event
            for cherenkov_block in cherenkov_reader:
                block_idxs = set(np.arange(ii, ii + cherenkov_block.shape[0]))

                match_idxs = np.array(
                    list(mask_idxs.intersection(block_idxs)), dtype=int
                )
                match_idxs = np.sort(match_idxs)

                block_match_idxs = match_idxs - ii

                cherenkov_sub_block = cherenkov_block[block_match_idxs]

                outblock[
                    oo : oo + cherenkov_sub_block.shape[0]
                ] = cherenkov_sub_block

                oo += cherenkov_sub_block.shape[0]
                ii += cherenkov_block.shape[0]

    return outblock


def make_cherenkovsize_record(path=None, cherenkov_bunches=None):
    sizerecord = {"num_bunches": 0, "num_photons": 0}

    if path is not None:
        assert cherenkov_bunches is None
        with cpw.cherenkov.CherenkovEventTapeReader(path=path) as tr:
            for event in tr:
                evth, cherenkov_reader = event
                for cherenkov_block in cherenkov_reader:
                    sizerecord["num_bunches"] += cherenkov_block.shape[0]
                    sizerecord["num_photons"] += np.sum(
                        cherenkov_block[:, cpw.I.BUNCH.BUNCH_SIZE_1]
                    )
    else:
        assert cherenkov_bunches is not None
        sizerecord["num_bunches"] += cherenkov_bunches.shape[0]
        sizerecord["num_photons"] += np.sum(
            cherenkov_bunches[:, cpw.I.BUNCH.BUNCH_SIZE_1]
        )
    return sizerecord


def inti_stats():
    ubh = un_bound_histogram.UnBoundHistogram
    MOMENTUM_TO_INCIDENT = -1.0
    s = {}
    s["x"] = {
        "hist": ubh(bin_width=25e2),
        "column": cpw.I.BUNCH.X_CM,
        "unit": "m",
        "factor": 1e-2,
    }
    s["y"] = {
        "hist": ubh(bin_width=25e2),
        "column": cpw.I.BUNCH.Y_CM,
        "unit": "m",
        "factor": 1e-2,
    }
    s["cx"] = {
        "hist": ubh(bin_width=np.deg2rad(0.05)),
        "column": cpw.I.BUNCH.UX_1,
        "unit": "1",
        "factor": MOMENTUM_TO_INCIDENT,
    }
    s["cy"] = {
        "hist": ubh(bin_width=np.deg2rad(0.05)),
        "column": cpw.I.BUNCH.VY_1,
        "unit": "1",
        "factor": MOMENTUM_TO_INCIDENT,
    }
    s["z_emission"] = {
        "hist": ubh(bin_width=10e2),
        "column": cpw.I.BUNCH.EMISSOION_ALTITUDE_ASL_CM,
        "unit": "m",
        "factor": 1e-2,
    }
    s["wavelength"] = {
        "hist": ubh(bin_width=1.0),
        "column": cpw.I.BUNCH.WAVELENGTH_NM,
        "unit": "m",
        "factor": 1e-9,
    }
    s["bunch_size"] = {
        "hist": ubh(bin_width=1e-2),
        "column": cpw.I.BUNCH.BUNCH_SIZE_1,
        "unit": "1",
        "factor": 1,
    }
    return s


def make_cherenkovpool_record(path=None, cherenkov_bunches=None):
    sts = inti_stats()

    if path is not None:
        assert cherenkov_bunches is None
        with cpw.cherenkov.CherenkovEventTapeReader(path=path) as tr:
            for event in tr:
                evth, cherenkov_reader = event
                for cherenkov_block in cherenkov_reader:
                    for key in sts:
                        sts[key]["hist"].assign(
                            cherenkov_block[:, sts[key]["column"]]
                        )
    else:
        assert cherenkov_bunches is not None
        for key in sts:
            sts[key]["hist"].assign(cherenkov_bunches[:, sts[key]["column"]])

    percentiles = [16, 50, 84]
    out = {}
    for key in sts:
        for pp in percentiles:
            stskey = "{:s}_p{:02d}_{:s}".format(key, pp, sts[key]["unit"])
            out[stskey] = sts[key]["factor"] * sts[key]["hist"].percentile(pp)
    return out


def cut_in_field_of_view(
    in_path, out_path, pointing, field_of_view_half_angle_rad
):
    with cpw.cherenkov.CherenkovEventTapeReader(
        path=in_path
    ) as tin, cpw.cherenkov.CherenkovEventTapeWriter(path=out_path) as tout:
        tout.write_runh(tin.runh)
        for event in tin:
            evth, cherenkov_reader = event
            tout.write_evth(evth)
            for cherenkov_block in cherenkov_reader:
                fov_mask = mask_cherenkov_bunches_in_instruments_field_of_view(
                    cherenkov_bunches=cherenkov_block,
                    pointing=pointing,
                    field_of_view_half_angle_rad=field_of_view_half_angle_rad,
                )
                cherenkov_block_in_fov = cherenkov_block[fov_mask]
                tout.write_payload(cherenkov_block_in_fov)


def mask_cherenkov_bunches_in_instruments_field_of_view(
    cherenkov_bunches,
    pointing,
    field_of_view_half_angle_rad,
):
    OVERHEAD = 2.0
    MOMENTUM_TO_INCIDENT = -1.0
    return mask_cherenkov_bunches_in_cone(
        cherenkov_bunches_cx=MOMENTUM_TO_INCIDENT
        * cherenkov_bunches[:, cpw.I.BUNCH.UX_1],
        cherenkov_bunches_cy=MOMENTUM_TO_INCIDENT
        * cherenkov_bunches[:, cpw.I.BUNCH.VY_1],
        cone_azimuth_rad=pointing["azimuth_rad"],
        cone_zenith_rad=pointing["zenith_rad"],
        cone_half_angle_rad=OVERHEAD * field_of_view_half_angle_rad,
    )


def mask_cherenkov_bunches_in_cone(
    cherenkov_bunches_cx,
    cherenkov_bunches_cy,
    cone_half_angle_rad,
    cone_azimuth_rad,
    cone_zenith_rad,
):
    cone_cx, cone_cy = spherical_coordinates.az_zd_to_cx_cy(
        azimuth_rad=cone_azimuth_rad,
        zenith_rad=cone_zenith_rad,
    )
    delta_rad = spherical_coordinates.angle_between_cx_cy(
        cx1=cherenkov_bunches_cx,
        cy1=cherenkov_bunches_cy,
        cx2=cone_cx,
        cy2=cone_cy,
    )
    return delta_rad < cone_half_angle_rad
