#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import plenopy as pl
import sebastians_matplotlib_addons as sebplt
import json_utils
import binning_utils
import confusion_matrix
import warnings


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

lfg = pl.LightFieldGeometry(
    opj(
        res.paths["plenoirf_dir"],
        "plenoptics",
        "instruments",
        res.instrument_key,
        "light_field_geometry",
    )
)

energy_bin = res.energy_binning(key="trigger_acceptance")

NUM_PLOT = 25
NUM_SHOWER = 100

def feature_pairwise_product(pax_hist):
    NUM = pax_hist.shape[0]
    vals = pax_hist / np.mean(pax_hist)
    corr = 0
    num_ij = (NUM * (NUM - 1)) / 2
    for i in range(NUM):
        for j in range(i + 1, NUM):
            ppp = vals[i] * vals[j]
            corr += ppp
    return corr / num_ij


def feature_mean_over_std(pax_hist):
    return np.mean(pax_hist) / np.std(pax_hist)


def pp_to_qq(pp):
    return -np.log10(1 - pp)


cache_dir = opj(res.paths["out_dir"], "__cache__")
os.makedirs(cache_dir, exist_ok=True)

flatness = {}
for pk in res.PARTICLES:
    pk_plot_counter = 0
    pk_counter = 0
    cache_path = opj(cache_dir, f"{pk:s}.snt.zip")

    if not os.path.exists(cache_path):
        print(pk)
        os.makedirs(opj(res.paths["out_dir"], pk), exist_ok=True)

        tab = snt.SparseNumericTable(
            index_key="uid",
            dtypes={
                "aperture_flatness": [
                    ("uid", "<u8"),
                    ("paxel_pairwise_product", "<f8"),
                    ("mean_over_std", "<f8"),
                ]
            },
        )

        loph_path = opj(
            res.response_path(particle_key=pk),
            "reconstructed_cherenkov.loph.tar",
        )

        with pl.photon_stream.loph.LopfTarReader(loph_path) as lophreader:
            for uid, loph in lophreader:
                size_pe = len(loph["photons"]["channels"])

                if size_pe == 0:
                    continue

                pix, pax = lfg.pixel_and_paxel_of_lixel(
                    loph["photons"]["channels"]
                )
                pax_hist = np.histogram(pax, np.arange(lfg.number_paxel + 1))[
                    0
                ]
                assert pax_hist.shape[0] == lfg.number_paxel

                ft_pp = feature_pairwise_product(pax_hist=pax_hist)
                ft_mos = feature_mean_over_std(pax_hist=pax_hist)

                tab["aperture_flatness"].append(
                    {
                        "uid": uid,
                        "paxel_pairwise_product": ft_pp,
                        "mean_over_std": ft_mos,
                    }
                )

                if pk_counter >= NUM_SHOWER:
                    break

                if pk_plot_counter < NUM_PLOT:

                    if 0.45 < ft_mos <= 0.65:
                        category = "hadron_like"
                    elif 0.75 < ft_mos < 1.0:
                        category = "gamma_like"
                    else:
                        category = None

                    if category is not None:
                        opath = os.path.join(
                            res.paths["out_dir"],
                            pk,
                            category,
                            f"{uid:012d}.jpg",
                        )
                        if not os.path.exists(opath):

                            pk_plot_counter += 1
                            os.makedirs(
                                opj(res.paths["out_dir"], pk, category),
                                exist_ok=True,
                            )

                            fig = sebplt.figure(style=sebplt.FIGURE_1_1)
                            ax = sebplt.add_axes(
                                fig=fig, span=[0.175, 0.15, 0.75, 0.8]
                            )

                            ax.set_title(f"mean over std: {ft_mos:.3f}")
                            pl.plot.image.add2ax(
                                ax=ax,
                                I=pax_hist,
                                px=lfg.x_mean,
                                py=lfg.y_mean,
                                colormap="viridis",
                                hexrotation=0,
                                vmin=0,
                                vmax=10,
                            )
                            ax.set_aspect("equal")
                            ax.set_xlim([-36, 36])
                            ax.set_ylim([-36, 36])
                            fig.savefig(opath)
                            sebplt.close(fig)

        with snt.open(
            file=cache_path, mode="w", dtypes_and_index_key_from=tab
        ) as tout:
            tout.append_table(tab)

    with snt.open(file=cache_path, mode="r") as tin:
        flatness[pk] = tin.query(
            levels_and_columns={"aperture_flatness": "__all__"}
        )


# prepare
# -------
for pk in flatness:
    for method in [
        "paxel_pairwise_product",
    ]:
        pp = flatness[pk]["aperture_flatness"][method]

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in log10"
            )
            flatness[pk]["aperture_flatness"][method] = pp_to_qq(pp)


METHODS = {
    "paxel_pairwise_product": {
        "bin": binning_utils.Binning(np.linspace(0.0, 4, 101)),
        "x_label": "pairwise product / 1",
    },
    "mean_over_std": {
        "bin": binning_utils.Binning(np.linspace(0.0, 4, 101)),
        "x_label": "mean over std / 1",
    },
}


for method in METHODS:
    fraction_pass = {}
    qq_bin = METHODS[method]["bin"]

    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in flatness:
        qq = flatness[pk]["aperture_flatness"][method]

        fraction_pass[pk] = {"rel": [], "abs": []}
        for iqq in range(qq_bin["num"]):
            qq_cut = qq_bin["centers"][iqq]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="invalid value encountered in divide"
                )
                fraction_pass[pk]["rel"].append(np.sum(qq >= qq_cut) / len(qq))
            fraction_pass[pk]["abs"].append(np.sum(qq >= qq_cut))
        fraction_pass[pk]["rel"] = np.array(fraction_pass[pk]["rel"])
        fraction_pass[pk]["abs"] = np.array(fraction_pass[pk]["abs"])

        qq_hist = np.histogram(qq, bins=qq_bin["edges"])[0]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in divide"
            )
            qq_rel_unc = np.sqrt(qq_hist) / qq_hist
            qq_norm = qq_hist / np.sum(qq_hist)

        sebplt.ax_add_histogram(
            ax=ax,
            bin_edges=qq_bin["edges"],
            bincounts=qq_norm,
            linestyle="-",
            linecolor=res.PARTICLE_COLORS[pk],
            linealpha=1.0,
            bincounts_upper=qq_norm * (1 + qq_rel_unc),
            bincounts_lower=qq_norm * (1 - qq_rel_unc),
            face_color=res.PARTICLE_COLORS[pk],
            face_alpha=0.3,
        )
    ax.set_xlim(qq_bin["limits"])
    ax.set_xlabel(METHODS[method]["x_label"])
    ax.set_ylabel("rel. intensity / 1")
    fig.savefig(opj(res.paths["out_dir"], f"hist_{method:s}.jpg"))
    sebplt.close(fig)

    frac_hadrons = (
        fraction_pass["proton"]["rel"] + fraction_pass["helium"]["rel"]
    ) / 2
    frac_gamma = fraction_pass["gamma"]["rel"]
    frac_valid = np.logical_and(
        fraction_pass["proton"]["abs"] > 10, fraction_pass["gamma"]["abs"] > 10
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in divide"
        )
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered in divide"
        )
        snr = frac_gamma / frac_hadrons
    snr_times_gamma = frac_gamma * snr
    arg_qq_cut = np.argmax(snr_times_gamma[frac_valid])
    good_qq_cut = qq_bin["centers"][frac_valid][arg_qq_cut]

    print(f"===== {method:s} =====")
    print("cut, SNR, frac. gamma, SNR*frac gamma")
    for i in range(qq_bin["num"]):
        print(
            f"{qq_bin['edges'][i]: 6.2f}, "
            f"{snr[i]: 6.2f}, "
            f"{frac_gamma[i]: 6.2f}, "
            f"{snr_times_gamma[i]: 6.2f}"
        )

    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in flatness:
        ax.plot(
            qq_bin["centers"],
            fraction_pass[pk]["rel"],
            color=res.PARTICLE_COLORS[pk],
        )
    ax.axvline(x=good_qq_cut, color="black", alpha=0.5, linestyle="-.")
    ax.text(s=f" Cut = {good_qq_cut:.3f}", x=good_qq_cut, y=0.9)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim(qq_bin["limits"])
    ax.set_xlabel(METHODS[method]["x_label"])
    ax.set_ylabel("fraction passing cut / 1")
    fig.savefig(opj(res.paths["out_dir"], f"cut_{method:s}.jpg"))
    sebplt.close(fig)

res.stop()
