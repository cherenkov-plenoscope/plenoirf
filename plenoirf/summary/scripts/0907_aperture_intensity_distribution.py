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
import dynamicsizerecarray

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

lfg = pl.LightFieldGeometry(
    opj(
        res.paths["plenoirf_dir"],
        "plenoptics",
        "instruments",
        res.instrument_key,
        "light_field_geometry",
    )
)


def pairwise_products(paxels):
    vals = paxels / np.mean(paxels)
    corr = 0
    num_ij = 0
    for i in range(paxels.shape[0]):
        for j in range(paxels.shape[0]):
            if i != j:
                ppp = vals[i] * vals[j]
                corr += ppp
                num_ij += 1
    return corr / num_ij


PLOT = False

flatness = {}
for pk in res.PARTICLES:
    print(pk)
    os.makedirs(opj(res.paths["out_dir"], pk), exist_ok=True)

    flatness[pk] = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=[("uid", "u8"), ("flatness", "f8")]
    )

    loph_path = opj(
        res.response_path(particle_key=pk), "reconstructed_cherenkov.loph.tar"
    )

    assign = {}
    with pl.photon_stream.loph.LopfTarReader(loph_path) as lophreader:
        for uid, loph in lophreader:
            if size_pe == 0:
                continue

            pix, pax = lfg.pixel_and_paxel_of_lixel(
                loph["photons"]["channels"]
            )
            pax_hist = np.histogram(pax, np.arange(lfg.number_paxel + 1))[0]
            assert pax_hist.shape[0] == lfg.number_paxel

            q = pairwise_products(pax_hist)
            flatness[pk].append({"uid": uid, "flatness": q})

            if PLOT:
                p = np.abs(np.log10(1 - q))

                if p >= 2.0:
                    key = "flat"
                elif p < 1.6:
                    key = "spike"
                else:
                    key = None

                if key is not None:
                    opath = os.path.join(
                        res.paths["out_dir"], pk, key, f"{uid:012d}.jpg"
                    )
                    if not os.path.exists(opath):
                        os.makedirs(
                            opj(res.paths["out_dir"], pk, key), exist_ok=True
                        )

                        fig = sebplt.figure(style=sebplt.FIGURE_1_1)
                        ax = sebplt.add_axes(
                            fig=fig, span=[0.175, 0.15, 0.75, 0.8]
                        )

                        ax.set_title(f"p: {p:.2f}")
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

res.stop()
