import copy
from typing import Optional, Tuple

import matplotlib.figure as mpl_f
import numpy as np

from vargrest.auxiliary.visualization import visualize_3d_variogram_map
from vargrest.variogramestimation.variogramestimation import NonparametricVariogramEstimate, ParametricVariogramEstimate


class VariogramPlot:
    def __init__(self,
                 ne: NonparametricVariogramEstimate,
                 pe: ParametricVariogramEstimate,
                 clims: Optional[Tuple[float, float]],
                 red_threshold: Optional[float]) -> None:
        self._fig = mpl_f.Figure((15, 10))
        self._axes = self._fig.subplots(2, 2)
        self._clims = clims
        self._red_threshold = red_threshold

        self._set_empirical_variogram_plot(ne)
        self._set_parametric_variogram_plot(ne, pe)

    @property
    def fig(self):
        return self._fig

    def _set_empirical_variogram_plot(self, ne: NonparametricVariogramEstimate):
        res_3d = ne.grid_resolution()
        vm = ne.variogram_map_values()
        ax1, ax2 = self._axes[0, 0], self._axes[0, 1]
        # Raw empirical variogram
        img1 = visualize_3d_variogram_map(ax1, vm, res_3d, self._clims)
        self._fig.colorbar(img1, ax=ax1)
        ax1.set_title('Empirical variogram')
        # Scaled empirical variogram
        img2 = visualize_3d_variogram_map(
            ax2, vm / np.nanmax(vm), res_3d, (0.0, self._red_threshold), red_threshold=self._red_threshold
        )
        self._fig.colorbar(img2, ax=ax2)
        ax2.set_title('Empirical variogram (scaled)')

    def _set_parametric_variogram_plot(self,
                                       ne: NonparametricVariogramEstimate,
                                       pe: ParametricVariogramEstimate):
        # Visualize estimated variogram function and display figure on screen
        nx, ny, nz = ne.variogram_map_values().shape
        dx, dy, dz = ne.grid_resolution()

        xmin = -int(nx / 2)
        xmax = -xmin
        ymin = -int(ny / 2)
        ymax = -ymin
        zmin = -int(nz / 2)
        zmax = -zmin

        xv = np.linspace(xmin, xmax, nx)
        yv = np.linspace(ymin, ymax, ny)
        zv = np.linspace(zmin, zmax, nz)
        xm, ym, zm = np.meshgrid(xv, yv, zv, indexing='ij')

        z_plot = int(np.floor(nz * 0.5))

        vfunc = pe.variogram_function()
        indep_data = np.vstack((xm.ravel(), ym.ravel(), zm.ravel()))
        params = list(pe.raw_parameters().values())

        if not np.any(np.isnan(params)):
            varest = vfunc(indep_data, *params)
            varest = np.reshape(varest, (nx, ny, nz))
        else:
            varest = np.full(shape=(nx, ny, nz), fill_value=np.nan)

        ax1, ax2 = self._axes[1, 0], self._axes[1, 1]
        # Setup axis 1
        if self._clims is not None:
            img = ax1.imshow(varest[:, :, z_plot].T,
                             extent=(dx * xmin, dx * xmax, dy * ymin, dy * ymax),
                             vmin=self._clims[0], vmax=self._clims[1],
                             origin='lower')
        else:
            img = ax1.imshow(varest[:, :, z_plot].T,
                             extent=(dx * xmin, dx * xmax, dy * ymin, dy * ymax),
                             origin='lower')
        ax1.set_xlabel("h_x")
        ax1.set_ylabel("h_y")
        ax1.set_title("Fitted variogram model")
        self._fig.colorbar(img, ax=ax1)
        # Setup axis 2
        sill = vfunc(np.full((3, 1), 1e16), *params)
        img2 = ax2.imshow(varest[:, :, z_plot].T / sill, vmax=self._red_threshold, vmin=0.0,
                          extent=(dx * xmin, dx * xmax, dy * ymin, dy * ymax), origin='lower')
        cm = copy.copy(img2.cmap)
        cm.set_over((0.8, 0.0, 0.0))
        img2.set_cmap(cm)
        ax2.set_xlabel("h_x")
        ax2.set_ylabel("h_y")
        ax2.set_title("Fitted variogram model (scaled)")
        self._fig.colorbar(img2, ax=ax2)
