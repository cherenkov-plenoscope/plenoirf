import corsika_primary as cpw
import numpy as np
import os
import sparse_numeric_table as spt
import binning_utils
import sebastians_matplotlib_addons as sebplt
import spherical_coordinates

from .. import bookkeeping


def inspect_cherenkov_pools(
    cherenkov_pools_path,
    aperture_bin_edges,
    image_bin_edges_rad,
    time_bin_edges,
    out_dir,
    threshold_num_photons,
    field_of_view_center_rad,
    field_of_view_half_angle_rad,
    mirror_center,
    mirror_radius,
):
    CM_TO_M = 1e-2
    NS_TO_S = 1e-9
    aperture_bin = binning_utils.Binning(aperture_bin_edges)
    image_bin = binning_utils.Binning(image_bin_edges_rad)
    time_bin = binning_utils.Binning(time_bin_edges)

    os.makedirs(out_dir, exist_ok=True)

    events_visible_num_photons = {}

    with cpw.cherenkov.CherenkovEventTapeReader(
        path=cherenkov_pools_path
    ) as tr:
        for event in tr:
            evth, cherenkov_reader = event

            aperture = np.zeros(
                shape=(aperture_bin["num"], aperture_bin["num"]), dtype=float
            )
            image = np.zeros(
                shape=(image_bin["num"], image_bin["num"]), dtype=float
            )
            timeseries = np.zeros(time_bin["num"], dtype=float)

            total_visible_size = 0

            for cherenkov_block in cherenkov_reader:
                aperture += np.histogram2d(
                    CM_TO_M * cherenkov_block[:, cpw.I.BUNCH.X_CM],
                    CM_TO_M * cherenkov_block[:, cpw.I.BUNCH.Y_CM],
                    weights=cherenkov_block[:, cpw.I.BUNCH.BUNCH_SIZE_1],
                    bins=(aperture_bin["edges"], aperture_bin["edges"]),
                )[0]

                in_mirror = (
                    np.hypot(
                        CM_TO_M * cherenkov_block[:, cpw.I.BUNCH.X_CM],
                        CM_TO_M * cherenkov_block[:, cpw.I.BUNCH.Y_CM],
                    )
                    <= mirror_radius
                )

                image += np.histogram2d(
                    cherenkov_block[in_mirror, cpw.I.BUNCH.CX_RAD],
                    cherenkov_block[in_mirror, cpw.I.BUNCH.CY_RAD],
                    weights=cherenkov_block[
                        in_mirror, cpw.I.BUNCH.BUNCH_SIZE_1
                    ],
                    bins=(image_bin["edges"], image_bin["edges"]),
                )[0]

                in_fov = (
                    spherical_coordinates.angle_between_cx_cy(
                        cx1=cherenkov_block[:, cpw.I.BUNCH.CX_RAD],
                        cy1=cherenkov_block[:, cpw.I.BUNCH.CY_RAD],
                        cx2=0.0,
                        cy2=0.0,
                    )
                    <= field_of_view_half_angle_rad
                )

                is_visible = np.logical_and(in_mirror, in_fov)

                timeseries += np.histogram(
                    NS_TO_S * cherenkov_block[is_visible, cpw.I.BUNCH.TIME_NS],
                    weights=cherenkov_block[
                        is_visible, cpw.I.BUNCH.BUNCH_SIZE_1
                    ],
                    bins=time_bin["edges"],
                )[0]

                total_visible_size += np.sum(
                    cherenkov_block[is_visible, cpw.I.BUNCH.BUNCH_SIZE_1]
                )

            uid = bookkeeping.uid.make_uid_from_corsika_evth(evth=evth)
            uid_str = bookkeeping.uid.make_uid_str(uid=uid)

            events_visible_num_photons[uid_str] = total_visible_size

            if total_visible_size >= threshold_num_photons:
                fig = sebplt.figure(
                    style={"rows": 1280, "cols": 2560, "fontsize": 1}, dpi=240
                )
                ax_ape = sebplt.add_axes(fig=fig, span=[0.1, 0.1, 0.33, 0.33])
                ax_ape_cm = sebplt.add_axes(
                    fig=fig, span=[0.35, 0.1, 0.02, 0.33]
                )

                ax_img = sebplt.add_axes(fig=fig, span=[0.4, 0.1, 0.33, 0.33])
                ax_img_cm = sebplt.add_axes(
                    fig=fig, span=[0.65, 0.1, 0.02, 0.33]
                )

                ax_tim = sebplt.add_axes(fig=fig, span=[0.75, 0.1, 0.2, 0.33])

                # image
                _pcm_img = ax_img.pcolormesh(
                    image_bin["edges"],
                    image_bin["edges"],
                    np.transpose(image),
                    cmap="viridis",
                    norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
                )
                sebplt.plt.colorbar(_pcm_img, cax=ax_img_cm, extend="max")
                ax_img.set_aspect("equal")
                ax_img.set_title("image")
                ax_img.set_xlabel("cx/rad")
                ax_img.set_ylabel("cy/rad")
                sebplt.ax_add_grid(ax_img)

                if field_of_view_center_rad is not None:
                    sebplt.ax_add_circle(
                        ax=ax_img,
                        x=field_of_view_center_rad[0],
                        y=field_of_view_center_rad[1],
                        r=field_of_view_half_angle_rad,
                        linewidth=1.0,
                        linestyle="-",
                        color="white",
                        alpha=1,
                    )

                # aperture
                _pcm_ape = ax_ape.pcolormesh(
                    aperture_bin["edges"],
                    aperture_bin["edges"],
                    np.transpose(aperture),
                    cmap="viridis",
                    norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
                )
                sebplt.plt.colorbar(_pcm_ape, cax=ax_ape_cm, extend="max")
                ax_ape.set_aspect("equal")
                ax_ape.set_title("aperture")
                ax_ape.set_xlabel("x/m")
                ax_ape.set_ylabel("y/m")
                sebplt.ax_add_grid(ax_ape)

                if mirror_center is not None:
                    sebplt.ax_add_circle(
                        ax=ax_ape,
                        x=mirror_center[0],
                        y=mirror_center[1],
                        r=mirror_radius,
                        linewidth=1.0,
                        linestyle="-",
                        color="white",
                        alpha=1,
                    )

                # time
                ax_tim.set_title("size: {:f}".format(total_visible_size))
                ax_tim.set_xlabel("time / s")
                ax_tim.set_ylabel("intensity / 1")
                sebplt.ax_add_histogram(
                    ax=ax_tim,
                    bin_edges=time_bin["edges"],
                    bincounts=timeseries,
                    linestyle="-",
                    linecolor="k",
                )
                fig.savefig(os.path.join(out_dir, "{:s}.jpg".format(uid_str)))
                sebplt.close(fig)

    return events_visible_num_photons
