import numpy as np


FIGURE_STYLE = {"rows": 720, "cols": 1280, "fontsize": 1.0}
AX_SPAN = [0.2, 0.2, 0.75, 0.75]
AX_SPAN_WITH_COLORBAR_PAYLOAD = [0.2, 0.2, 0.55, 0.75]
AX_SPAN_WITH_COLORBAR_COLORBAR = [0.8, 0.2, 0.025, 0.75]

SOURCES = {
    "diffuse": {
        "label": "area $\\times$ solid angle",
        "unit": "m$^{2}$ sr",
        "limits": {
            "passed_trigger": [1e-1, 1e5],
            "passed_all_cuts": [1e-1, 1e5],
        },
    },
    "point": {
        "label": "area",
        "unit": "m$^{2}$",
        "limits": {
            "passed_trigger": [1e1, 1e6],
            "passed_all_cuts": [1e1, 1e6],
        },
    },
}

PARTICLE_COLORS = {
    "gamma": "black",
    "electron": "blue",
    "proton": "red",
    "helium": "orange",
}

MATPLOTLIB_RCPARAMS_LATEX = {
    "mathtext.fontset": "cm",
    "font.family": "STIXGeneral",
}

COLOR_photosensor_RGBA = np.array([192, 0, 0]) / 255
COLOR_BEAM_RGBA = np.array([0, 128, 255]) / 255
COLOR_EYE_WALLS_RGBA = np.array([0, 192, 0]) / 255


def mark_ax_airshower_spectrum(ax, x=0.93, y=0.93, fontsize=42):
    ax.text(
        x=x,
        y=y,
        s=r"$\star$",
        color="k",
        transform=ax.transAxes,
        fontsize=fontsize,
    )


def mark_ax_thrown_spectrum(ax, x=0.93, y=0.93, fontsize=42):
    ax.text(
        x=x,
        y=y,
        s=r"$\bullet$",
        color="k",
        transform=ax.transAxes,
        fontsize=fontsize,
    )


def add_aperture_plane_to_ax(ax, **kwargs):
    ax.plot([-1, 1], [0, 0], **kwargs)


def add_rays_to_ax(ax, object_distance, N=4, **kwargs):
    x_starts = np.linspace(-0.9, 0.9, N)
    y_starts = np.zeros(N)

    x_ends = -x_starts * 100
    y_ends = 2 * object_distance * np.ones(N) * 100

    for i in range(N):
        ax.plot(
            [x_starts[i], x_ends[i]],
            [y_starts[i], y_ends[i]],
            **kwargs,
        )


def is_in_roi(x, y, roi, margin=0.1):
    x0 = roi["x"][0] - margin
    x1 = roi["x"][1] + margin
    y0 = roi["y"][0] - margin
    y1 = roi["y"][1] + margin
    if x0 <= x < x1 and y0 <= y < y1:
        return True
    else:
        return False


def position_of_eye(light_field_geometry, eye_id):
    num_pax = light_field_geometry.number_paxel
    start = eye_id * num_pax
    stop = (eye_id + 1) * num_pax
    poly = light_field_geometry.lixel_polygons[start:stop]
    poly = np.array(poly)
    pp = []
    for pol in poly:
        x = np.mean(pol[:, 0])
        y = np.mean(pol[:, 1])
        pp.append([x, y])
    pp = np.array(pp)
    x_mean = np.mean(pp[:, 0])
    y_mean = np.mean(pp[:, 1])
    return np.array([x_mean, y_mean])


def positions_of_eyes_in_roi(light_field_geometry, roi, margin=0.1):
    positions_of_eyes = {}
    for eye_id in range(light_field_geometry.number_pixel):
        pos = position_of_eye(
            light_field_geometry=light_field_geometry, eye_id=eye_id
        )
        if is_in_roi(x=pos[0], y=pos[1], roi=roi, margin=margin):
            positions_of_eyes[eye_id] = pos
    return positions_of_eyes
