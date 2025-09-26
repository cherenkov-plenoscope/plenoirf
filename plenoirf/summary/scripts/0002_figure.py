#!/usr/bin/python
import sys
import plenoirf as irf
import os
from os.path import join as opj
import json_utils
import confusion_matrix
import sebastians_matplotlib_addons as sebplt

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt)
energy_bin = res.energy_binning("trigger_acceptance_onregion")
zenith_bin = res.zenith_binning(key="once")

for key in ["16:9", "1:1", "16:6", "4:3"]:

    fstyle, axspan = irf.summary.figure.style(key=key)
    fig = sebplt.figure(fstyle)
    ax = sebplt.add_axes(fig=fig, span=axspan)
    ax.set_xlabel(r"$x$ / 1")
    ax.set_ylabel(r"$y$ / 1")
    fig.savefig(opj(res.paths["out_dir"], f"{key:s}.jpg"))
    sebplt.close(fig)


min_number_samples = 3
cm = confusion_matrix.init(
    ax0_key="true_energy",
    ax0_values=[],
    ax0_bin_edges=energy_bin["edges"],
    ax1_key="reco_energy",
    ax1_values=[],
    ax1_bin_edges=energy_bin["edges"],
    min_exposure_ax0=min_number_samples,
    default_low_exposure=0.0,
)
fig = sebplt.figure(irf.summary.figure.style(key="6:7")[0])
ax_c = sebplt.add_axes(fig=fig, span=[0.2, 0.25, 0.75, 0.65])
ax_h = sebplt.add_axes(fig=fig, span=[0.2, 0.13, 0.75, 0.10])
ax_cb = sebplt.add_axes(fig=fig, span=[0.25, 0.96, 0.65, 0.015])
sebplt.add_axes_zenith_range_indicator(
    fig=fig,
    span=[0.02, 0.02, 0.12, 0.12],
    zenith_bin_edges_rad=zenith_bin["edges"],
    zenith_bin=1,
    fontsize=6,
)
_pcm_confusion = ax_c.pcolormesh(
    cm["ax0_bin_edges"],
    cm["ax1_bin_edges"],
    np.transpose(cm["counts_normalized_on_ax0"]),
    cmap="Greys",
    norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
)
ax_c.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
sebplt.plt.colorbar(
    _pcm_confusion, cax=ax_cb, extend="max", orientation="horizontal"
)
ax_c.set_ylabel("reco. energy / GeV")
ax_c.loglog()
ax_c.set_xticklabels([])
ax_h.semilogx()
ax_h.set_xlim([np.min(cm["ax0_bin_edges"]), np.max(cm["ax1_bin_edges"])])
ax_h.set_xlabel("true energy / GeV")
ax_h.set_ylabel("count / 1")
ax_h.axhline(min_number_samples, linestyle=":", color="k")
sebplt.ax_add_histogram(
    ax=ax_h,
    bin_edges=cm["ax0_bin_edges"],
    bincounts=cm["exposure_ax0"],
    linestyle="-",
    linecolor="black",
)
fig.savefig(opj(res.paths["out_dir"], f"confusion.jpg"))
sebplt.close(fig)


label_dir = opj(res.paths["out_dir"], "labels")
os.makedirs(label_dir, exist_ok=True)


def make_line_label(path, **kwargs):
    fig = sebplt.figure(style={"rows": 40, "cols": 120, "fontsize": 1})
    ax = sebplt.add_axes(fig=fig, span=[0, 0, 1, 1], style=sebplt.AXES_BLANK)
    ax.plot([-0.9, 0.9], [0, 0], **kwargs)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    fig.savefig(path)
    sebplt.close(fig)


colors = [
    "grey",
    "lightgrey",
    "limegreen",
    irf.other_instruments.fermi_lat.COLOR,
    irf.other_instruments.cherenkov_telescope_array_south.COLOR,
    irf.other_instruments.portal.COLOR,
]
for pk in irf.summary.figure.PARTICLE_COLORS:
    colors.append(irf.summary.figure.PARTICLE_COLORS[pk])

linestyles = ["-", "--", ":", "-."]

for color in colors:
    for linestyle in linestyles:
        fname = f"{color:s}_{linestyle:s}.jpg"
        make_line_label(
            path=opj(label_dir, fname),
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )


res.stop()
