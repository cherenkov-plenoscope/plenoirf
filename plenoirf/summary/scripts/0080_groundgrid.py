#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import atmospheric_cherenkov_response
import sparse_numeric_table as snt
import os
import json_utils
import magnetic_deflection as mdfl
import spherical_coordinates
import solid_angle_utils
import binning_utils
import sebastians_matplotlib_addons as seb
import confusion_matrix

paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)
seb.matplotlib.rcParams.update(res.analysis["plot"]["matplotlib"])

energy_bin = json_utils.read(
    os.path.join(paths["analysis_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

passing_trigger = json_utils.tree.read(
    os.path.join(paths["analysis_dir"], "0055_passing_trigger")
)

nat_bin = binning_utils.Binning(bin_edges=np.geomspace(1, 100_000, 31))

POINTNIG_ZENITH_BIN = res.ZenithBinning("once")

bbb = {}
for pk in res.PARTICLES:
    bbb[pk] = {"avg": [], "au": []}

    with res.open_event_table(particle_key=pk) as arc:
        table = arc.read_table(
            levels_and_columns={
                "primary": [
                    snt.IDX,
                    "energy_GeV",
                    "azimuth_rad",
                    "zenith_rad",
                ],
                "instrument_pointing": "__all__",
                "groundgrid": "__all__",
            }
        )

        _gg_res = arc.read_table(
            levels_and_columns={
                "groundgrid_result": [snt.IDX, "num_bins_above_threshold"],
            }
        )

    table["groundgrid"] = irf.event_table.structure.patch_groundgrid(
        groundgrid=table["groundgrid"],
        groundgrid_result=_gg_res["groundgrid_result"],
    )

    table = snt.sort_table_on_common_indices(
        table=table,
        common_indices=table["primary"][snt.IDX],
    )

    min_number_samples = 10

    for zdbin in range(POINTNIG_ZENITH_BIN.num):
        zd_start = POINTNIG_ZENITH_BIN.edges[zdbin]
        zd_stop = POINTNIG_ZENITH_BIN.edges[zdbin + 1]

        mask = np.logical_and(
            table["instrument_pointing"]["zenith_rad"] >= zd_start,
            table["instrument_pointing"]["zenith_rad"] < zd_stop,
        )

        # 1D
        # --
        hh_exposure = np.histogram(
            table["primary"]["energy_GeV"][mask],
            bins=energy_bin["edges"],
        )[0]
        hh_num_bins = np.histogram(
            table["primary"]["energy_GeV"][mask],
            bins=energy_bin["edges"],
            weights=table["groundgrid"]["num_bins_above_threshold"][mask],
        )[0]

        avg_num_bins = irf.utils._divide_silent(
            hh_num_bins, hh_exposure, default=float("nan")
        )
        avg_num_bins_ru = irf.utils._divide_silent(
            np.sqrt(hh_exposure), hh_exposure, default=float("nan")
        )
        avg_num_bins_au = avg_num_bins * avg_num_bins_ru

        bbb[pk]["avg"].append(avg_num_bins)
        bbb[pk]["au"].append(avg_num_bins_au)

        # 2D
        # --
        cm = confusion_matrix.init(
            ax0_key="primary/energy_GeV",
            ax0_values=table["primary"]["energy_GeV"][mask],
            ax0_bin_edges=energy_bin["edges"],
            ax1_key="groundgrid/num_bins_above_threshold",
            ax1_values=table["groundgrid"]["num_bins_above_threshold"][mask],
            ax1_bin_edges=nat_bin["edges"],
            weights=None,
            min_exposure_ax0=min_number_samples,
            default_low_exposure=0.0,
        )

        fig = seb.figure(style=seb.FIGURE_1_1)
        ax_c = seb.add_axes(fig=fig, span=[0.25, 0.27, 0.55, 0.65])
        ax_h = seb.add_axes(fig=fig, span=[0.25, 0.11, 0.55, 0.1])
        ax_cb = seb.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
        _pcm_confusion = ax_c.pcolormesh(
            cm["ax0_bin_edges"],
            cm["ax1_bin_edges"],
            np.transpose(cm["counts_normalized_on_ax0"]),
            cmap="Greys",
            norm=seb.plt_colors.PowerNorm(gamma=0.5),
        )
        seb.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
        circ_str = r"$^\circ{}$"
        zenith_range_str = (
            f"zenith range: {np.rad2deg(zd_start):0.1f}"
            + circ_str
            + " to "
            + f"{np.rad2deg(zd_stop):0.1f}"
            + circ_str
        )
        ax_c.set_title(zenith_range_str)
        ax_c.set_ylabel("num. bins above threshod / 1")
        ax_c.loglog()
        res.ax_add_site_marker(ax_c, x=0.2, y=0.1)
        seb.ax_add_grid(ax_c)
        ax_c.text(
            x=0.2,
            y=0.05,
            s="normalized for each column",
            fontsize=8,
            horizontalalignment="center",
            # verticalalignment="center",
            transform=ax_c.transAxes,
        )

        ax_h.semilogx()
        ax_h.set_xlim(
            [np.min(cm["ax0_bin_edges"]), np.max(cm["ax0_bin_edges"])]
        )
        ax_h.set_xlabel("energy / GeV")
        ax_h.set_ylabel("num. events / 1")
        ax_h.axhline(min_number_samples, linestyle=":", color="k")
        seb.ax_add_histogram(
            ax=ax_h,
            bin_edges=cm["ax0_bin_edges"],
            bincounts=cm["exposure_ax0"],
            linestyle="-",
            linecolor="k",
        )
        fig.savefig(
            os.path.join(
                paths["out_dir"], f"{pk:s}_zenith{zdbin:03d}_grid.jpg"
            )
        )
        seb.close(fig)


particle_colors = res.analysis["plot"]["particle_colors"]

fig = seb.figure(style=seb.FIGURE_1_1)
ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])

linestyles = ["-", "--", ":"]
for pk in res.PARTICLES:
    for zdbin in range(POINTNIG_ZENITH_BIN.num):
        seb.ax_add_histogram(
            ax=ax,
            bin_edges=energy_bin["edges"],
            bincounts=bbb[pk]["avg"][zdbin],
            linestyle=linestyles[zdbin],
            linecolor=particle_colors[pk],
            linealpha=1.0,
            bincounts_upper=None,
            bincounts_lower=None,
            face_color=particle_colors[pk],
            face_alpha=0.3,
        )
res.ax_add_site_marker(ax)
ax.set_ylim([1e3, 1e4])
ax.loglog()
ax.set_xlabel("energy / GeV")
ax.set_ylabel("num. bins above threshold / 1")
fig.savefig(os.path.join(paths["out_dir"], "num_bins_above_threshold.jpg"))
seb.close(fig)
