import copy
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.figure as mpl_f
import numpy as np


def visualize_2d_variogram_map(varmap: np.ndarray, res: Tuple[float, float], clims: Optional[Tuple[float, float]] = None):
    if varmap.ndim != 2:
        raise Exception("Variogram map must have 2 dimensions, not {}.".format(varmap.ndim))

    n_x, n_y = varmap.shape
    dx, dy = res
    h_ext = (
        -0.5 * n_x * dx, 0.5 * n_x * dx,
        -0.5 * n_y * dy, 0.5 * n_y * dy
    )

    fig = plt.figure()
    if clims is not None:
        plt.imshow(varmap, origin="lower left", vmin=clims[0], vmax=clims[1])
    else:
        plt.imshow(varmap, origin="lower left")
    plt.xlabel("h_x")
    plt.ylabel("h_y")
    plt.colorbar()
    return fig


def visualize_3d_variogram_map(ax,
                               varmap: np.ndarray,
                               res: Tuple[float, float, float],
                               clims: Optional[Tuple[float, float]] = None,
                               red_threshold: Optional[float] = None):
    if varmap.ndim != 3:
        raise Exception("Variogram map must have 3 dimensions, not {}.".format(varmap.ndim))

    n_x, n_y, n_z = varmap.shape
    dx, dy, dz = res

    # Plot xy-slice through center of volume
    k_mid = int(varmap.shape[2] / 2)
    slice = varmap[:, :, k_mid]
    h_ext = (
        -0.5 * n_x * dx, 0.5 * n_x * dx,
        -0.5 * n_y * dy, 0.5 * n_y * dy
    )

    ax.set_facecolor('gray')
    kwargs = dict(
        extent=h_ext,
        origin='lower'
    )
    if clims is not None:
        kwargs['vmin'] = clims[0]
        kwargs['vmax'] = clims[1] * (1.0 if red_threshold is None else red_threshold)
    img = ax.imshow(slice.T, **kwargs)
    if red_threshold is not None:
        cmap = copy.copy(img.cmap)
        cmap.set_over((0.8, 0, 0))
        img.set_cmap(cmap)
    ax.set_xlabel("h_x")
    ax.set_ylabel("h_y")
    return img


def visualize_crop(image_data: np.ndarray,
                   image_ext: Tuple[float, float, float, float],
                   polygon_x: List[float],
                   polygon_y: List[float]):
    fig = mpl_f.Figure()
    ax = fig.subplots()
    ax.imshow(image_data.T, extent=image_ext, origin='lower')
    ax.plot(polygon_x, polygon_y, "y-")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Crop box")

    return fig

