import inspect
import json
import os
from typing import Tuple, Optional, Dict, Callable, Union

import numpy as np
import matplotlib.figure as mpl_f

from vargrest.auxiliary.curvefit import fit_3d_field, find_dominant_direction
from vargrest.variogramdata.variogramdata import VariogramDataInterface
from vargrest.variogramestimation.parametricvariogram import AnisotropicVariogram, VariogramType
from vargrest.variogramestimation.empiricalvariogram import _estimate_variogram_np_3d_dense,\
    _estimate_variogram_np_3d_random, _estimate_variogram_np_3d_sparse


class NonparametricVariogramEstimate:
    def __init__(self, varmap: np.ndarray, counts: np.ndarray,
                 ndims: int, gridres: Tuple[float, float, float]):
        super().__init__()
        self._varmap = varmap
        self._counts = counts
        self._ndims = ndims
        self._description = "{}D variogram map".format(ndims)
        self._gridres = gridres

    def variogram_map_values(self, normalized=False):
        if not normalized:
            return self._varmap
        else:
            return self._varmap / np.nanmean(self._varmap)

    def variogram_map_counts(self):
        return self._counts

    def grid_resolution(self) -> Tuple[float, float, float]:
        return self._gridres



class ParametricVariogramEstimate:
    def __init__(self, family: VariogramType,
                 nugget: bool,
                 params: dict,
                 quality: float,
                 ndims: int,
                 nonparest: NonparametricVariogramEstimate):
        super().__init__()
        self._ndims = ndims
        self._family = family
        self._quality = quality
        self._nugget = nugget
        self._parameters = params
        self._description = "{}D {} variogram estimate ({} parameters)".format(ndims, family, len(params))
        self._nonparest = nonparest

    def raw_parameters(self) -> Dict:
        return self._parameters

    def variogram_function(self) -> Callable:
        return AnisotropicVariogram(family=self._family, nug=self._nugget).get_variogram_function()

    def plot_estimated_variogram(self, clims: Optional[Tuple[float, float]] = None):
        # Visualize estimated variogram function and display figure on screen
        nx, ny, nz = self._nonparest.variogram_map_values().shape
        dx, dy, dz = self._nonparest.grid_resolution()

        xmin = -int(nx / 2)
        xmax = -xmin
        ymin = -int(ny / 2)
        ymax = -ymin
        zmin = -int(nz / 2)
        zmax = -zmin

        xv = np.linspace(xmin, xmax, nx)
        yv = np.linspace(ymin, ymax, ny)
        zv = np.linspace(zmin, zmax, nz)
        xm, ym, zm = np.meshgrid(xv, yv, zv)

        z_plot = int(np.floor(nz * 0.5))
        vfunc = AnisotropicVariogram(family=self._family, nug=self._nugget).get_variogram_function()
        indep_data = np.vstack((xm.ravel(), ym.ravel(), zm.ravel()))
        params = list(self._parameters.values())

        if not np.any(np.isnan(params)):
            varest = vfunc(indep_data, *params)
            varest = np.reshape(varest, (ny, nx, nz))
        else:
            varest = np.full(shape=(ny, nx, nz), fill_value=np.nan)

        fig = mpl_f.Figure()
        ax1, ax2 = fig.subplots(1, 2)
        # Setup axis 1
        if clims is not None:
            img = ax1.imshow(varest[:, :, z_plot],
                      extent=(dx * xmin, dx * xmax, dy * ymin, dy * ymax),
                      vmin=clims[0], vmax=clims[1])
        else:
            img = ax1.imshow(varest[:, :, z_plot],
                      extent=(dx * xmin, dx * xmax, dy * ymin, dy * ymax))
        ax1.set_xlabel("h_x")
        ax1.set_ylabel("h_y")
        ax1.set_title("Fitted variogram model")
        fig.colorbar(img, ax=ax1, orientation='horizontal')
        # Setup axis 2
        sill = vfunc(np.full((3, 1), np.inf), *params)
        img2 = ax2.imshow(varest[:, :, z_plot] / sill, vmax=1.0, vmin=0.0,
                          extent=(dx * xmin, dx * xmax, dy * ymin, dy * ymax))
        ax2.set_xlabel("h_x")
        ax2.set_ylabel("h_y")
        ax2.set_title("Fitted variogram model (scaled)")
        fig.colorbar(img2, ax=ax2, orientation='horizontal')
        return fig

    @property
    def family(self) -> VariogramType:
        return self._family

    @property
    def quality(self) -> float:
        return self._quality

    def description(self) -> str:
        return self._description

    def dump_to_screen(self):
        # Display parameter estimates and variogram family on screen
        pp = self.polished_parameters()
        pp['description'] = self.description()
        print(json.dumps(pp, indent=4))

    def dump_to_json(self, dir_name: str, file_name: str):
        # Save parameter estimates and variogram family to JSON file
        filename_suffix = "json"
        full_path = os.path.join(dir_name, file_name + "." + filename_suffix)
        with open(full_path, 'w') as fp:
            pp = self.polished_parameters()
            pp['description'] = self.description()
            json.dump(pp, fp, indent=4)

    def polished_parameters(self):
        # Convert lengths and angles to appropriate output formats
        dx, dy, dz = self._nonparest.grid_resolution()
        polished_parameters = {}
        npar = len(self._parameters)
        for key in self._parameters.keys():
            value = self._parameters[key]
            if key in ["rx", "r_x"]:
                new_key = "r_major"
                polished_value = dx * value
                unit = "m"
            elif key in ["ry", "r_y"]:
                new_key = "r_minor"
                polished_value = dy * value
                unit = "m"
            elif key in ["rz", "r_z"]:
                new_key = "r_vertical"
                polished_value = dz * value
                unit = "m"
            elif key in ["pwr", "pow", "power"]:
                new_key = "power"
                polished_value = value
                unit = "N/A"
            elif key in ["azi", "azimuth"]:
                new_key = "azimuth"
                polished_value = 90.0 - np.rad2deg(np.arctan(np.tan(value) * dy / dx))
                unit = "deg"
            elif key in ["dip"]:
                new_key = "dip"
                polished_value = np.rad2deg(np.arctan(np.tan(value) * dz / dx))
                unit = "deg"
            elif key in ["std", "sig", "sigma"]:
                new_key = "sigma"
                polished_value = value
                unit = "N/A"
            elif key in ["sng", "nugget", "nug", "tau"]:
                new_key = "tau"
                polished_value = value
                unit = "N/A"
            else:
                continue

            polished_parameters[new_key] = {"value": polished_value, "unit": unit}

        # Make sure major range is >= minor range, flip if necessary
        if polished_parameters["r_major"]["value"] < polished_parameters["r_minor"]["value"]:
            r_major = polished_parameters["r_minor"]["value"]
            r_minor = polished_parameters["r_major"]["value"]
            polished_parameters["r_major"]["value"] = r_major
            polished_parameters["r_minor"]["value"] = r_minor
            azimuth_rot = polished_parameters["azimuth"]["value"] + 90.0
            polished_parameters["azimuth"]["value"] = azimuth_rot

        # Choose output value of azimuth between -90.0 and 90.0 degrees
        azimuth = polished_parameters["azimuth"]["value"]
        if azimuth < -90.0:
            azimuth_polished = azimuth + 180.0
        elif azimuth > 90.0:
            azimuth_polished = azimuth - 180.0
        else:
            azimuth_polished = azimuth
        polished_parameters["azimuth"]["value"] = azimuth_polished

        return polished_parameters


class VariogramEstimator:
    def __init__(self, variogram_data: VariogramDataInterface, dz=0.25, **kwargs):
        self._grid_resolution = (variogram_data.dx, variogram_data.dy, dz)

        poro_reg = variogram_data.property_grid(dz, **kwargs)
        if 'log_transform' in kwargs and kwargs['log_transform']:
            logporo = np.log(poro_reg)
            sd_logporo = np.nanstd(logporo)
            mean_logporo = np.nanmean(logporo)
            normlogporo = (logporo - mean_logporo) / sd_logporo
            self._data = normlogporo
            self._transform_params = {"mean": mean_logporo, "sd": sd_logporo}
        else:
            self._data = poro_reg

    def make_variogram_map_xyz(self,
                               sampling: Optional[str] = "dense",
                               sub_sampling: Optional[int] = None,
                               stride: Optional[Tuple[int, int, int]] = None,
                               sampling_factor: Optional[float] = None,
                               max_samples: Optional[int] = None,
                               lag_x: Optional[int] = None,
                               lag_y: Optional[int] = None,
                               lag_z: Optional[int] = None) -> NonparametricVariogramEstimate:
        if lag_x is None:
            lag_x = int(0.5 * self.data().shape[0])
        if lag_y is None:
            lag_y = int(0.5 * self.data().shape[1])
        if lag_z is None:
            lag_z = int(0.5 * self.data().shape[2])

        if sampling == "dense":
            varmap, counts = _estimate_variogram_np_3d_dense(self._data, lag_x, lag_y, lag_z, sub_sampling)
        elif sampling == "random":
            varmap, counts = _estimate_variogram_np_3d_random(self._data, lag_x, lag_y, lag_z,
                                                              sampling_factor=sampling_factor,
                                                              max_samples=max_samples)
        elif sampling == "sparse":
            if stride is not None:
                step_x, step_y, step_z = stride
            else:
                step_x, step_y, step_z = 2, 2, 2
            lag_step_x = (-lag_x, lag_x + 1, step_x)
            lag_step_y = (-lag_y, lag_y + 1, step_y)
            lag_step_z = (-lag_z, lag_z + 1, step_z)
            varmap, counts = _estimate_variogram_np_3d_sparse(self._data, lag_step_x, lag_step_y, lag_step_z)
        else:
            assert False
        return NonparametricVariogramEstimate(varmap, counts, ndims=3, gridres=self._grid_resolution)

    def estimate_parametric_variogram_xyz(self,
                                          varmap: NonparametricVariogramEstimate,
                                          family: Union[VariogramType, str],
                                          nugget: bool,
                                          sigma_wt: Optional[float] = None) -> ParametricVariogramEstimate:
        # Get variogram function
        av = AnisotropicVariogram(family=family, nug=nugget)
        vfunc = av.get_variogram_function()
        vfunc_args = inspect.getfullargspec(vfunc)[0]

        # Prepare lower and upper bounds and initial guess
        parameter_names = vfunc_args[1:]
        popt, quality = self._find_optimum(varmap, vfunc, av.get_variogram_jacobian(), parameter_names, sigma_wt, normalize=True)

        # Form dictionary of names and values of estimated parameters
        par_est_dict = {par_name: par_value for (par_name, par_value) in zip(vfunc_args[1:], popt)}

        # Return estimation result as an object
        return ParametricVariogramEstimate(family=family, params=par_est_dict, quality=quality, ndims=3,
                                           nonparest=varmap, nugget=nugget)

    def data(self) -> np.ndarray:
        return self._data

    def generate_3d_slice_image(self, dir_name: str, file_name: str):
        try:
            import plotly.graph_objects as go
        except ImportError:
            print('Failed to find plotly, skipping 3d sliced image')
            return
        fig = go.Figure()
        xx, yy = np.meshgrid(np.arange(0, self._data.shape[0]) * self._grid_resolution[0],
                             np.arange(0, self._data.shape[1]) * self._grid_resolution[1],)
        x = xx.flatten()
        y = yy.flatten()
        cmedian = np.nanmedian(self._data)
        cmin, cmax = 0.8 * cmedian, 1.2 * cmedian
        for i in range(self._data.shape[2] - 1):
            d = self._data[:, :, i].T.flatten()
            d_valid = ~np.isnan(d)
            if not np.any(d_valid):
                continue
            fig.add_scatter(
                x=x[d_valid],
                y=y[d_valid],
                mode='markers',
                marker=dict(
                    cmin=cmin,
                    cmax=cmax,
                    color=d[d_valid],
                    colorscale='Earth',
                    showscale=True
                ),
                visible=False
            )
        if len(fig.data) > 0:
            fig.data[0].visible = True

        steps = []
        for i in range(len(fig.data)):
            s = dict(
                method='restyle',
                args=['visible', [False] * len(fig.data)]
            )
            s['args'][1][i] = True
            steps.append(s)
        sliders = [dict(
            active=0,
            steps=steps
        )]
        fig.layout.yaxis.scaleanchor = 'x'
        fig.layout.yaxis.range = (np.min(y), np.max(y))
        fig.layout.xaxis.range = (np.min(x), np.max(x))
        fig.update_layout(sliders=sliders)
        fig.write_html(os.path.join(dir_name, file_name) + '.html', include_plotlyjs='cdn')

    def _find_optimum(self, var_map, var_func, jac_func, parameter_names, sigma_wt, normalize=False):
        if np.all(np.isnan(var_map.variogram_map_values())):
            return np.full(len(parameter_names), fill_value=np.nan), np.nan

        par_guess = np.empty(shape=(len(parameter_names),))
        par_lbounds = np.empty(shape=(len(parameter_names),))
        par_ubounds = np.empty(shape=(len(parameter_names),))

        varmap_shape = var_map.variogram_map_values(normalize).shape
        varmap_mean = np.nanmean(var_map.variogram_map_values(normalize))
        varmap_max = np.nanmax(var_map.variogram_map_values(normalize))

        for ind, pn in enumerate(parameter_names):
            par_guess[ind] = self._parameter_guess(pn, varmap_shape, varmap_mean, varmap_max)
            lb, ub = self._parameter_bounds(pn, varmap_shape, varmap_mean, varmap_max)
            par_lbounds[ind] = lb
            par_ubounds[ind] = ub
        par_ubounds = np.maximum(par_ubounds, par_lbounds + 1e-12)  # Make sure bounds are well-defined

        # Identify azimuth by 'angle sweep' method
        azimuth_opt = find_dominant_direction(var_map.variogram_map_values(normalize), self._grid_resolution)
        ind_azimuth = parameter_names.index("azi")
        par_guess[ind_azimuth] = azimuth_opt
        par_lbounds[ind_azimuth] = azimuth_opt - 1e-12  # Clamp value of azimuth
        par_ubounds[ind_azimuth] = azimuth_opt + 1e-12

        # Fit function to map
        _popt, _quality = fit_3d_field(var_func,
                                       jac_func,
                                       var_map.variogram_map_values(normalize),
                                       self._grid_resolution,
                                       var_map.variogram_map_counts(),
                                       par_guess,
                                       (par_lbounds, par_ubounds),
                                       sigma_wt)

        if normalize:
            for ind, pn in enumerate(parameter_names):
                if pn in ["sig", "std", "sigma"]:
                    _popt[ind] *= np.sqrt(np.nanmean(var_map.variogram_map_values(normalized=False)))  # Un-normalize standard deviation estimate

        return _popt, _quality

    def _parameter_guess(self, parname: str, varmap_shape: Tuple[int, int, int],
                         varmap_mean: float, varmap_max: float) -> float:
        if parname in ["r_x", "rx"]:
            return 0.33 * min(varmap_shape[0], varmap_shape[1])
        elif parname in ["r_y", "ry"]:
            return 0.33 * min(varmap_shape[0], varmap_shape[1])
        elif parname in ["r_z", "rz"]:
            return 0.33 * varmap_shape[2]
        elif parname in ["pwr", "power", "nu"]:
            return 1.5
        elif parname in ["azi", "azimuth"]:
            return 0.0
        elif parname in ["dip"]:
            return 0.0
        elif parname in ["std", "sig", "sigma"]:
            return 1.0 * np.nanstd(self._data)
        elif parname in ["sng", "nug", "nugget", "tau"]:
            return 0.1 * np.nanstd(self._data)

    def _parameter_bounds(self, parname: str, varmap_shape: Tuple[int, int, int],
                          varmap_mean: float, varmap_max: float) -> Tuple[float, float]:
        if parname in ["r_x", "rx"]:
            return 0.0, 5 * varmap_shape[0]
        elif parname in ["r_y", "ry"]:
            return 0.0, 5 * varmap_shape[1]
        elif parname in ["r_z", "rz"]:
            return 0.0, 5 * varmap_shape[2]
        elif parname in ["pwr", "power", "nu"]:
            return 0.0, 2.0
        elif parname in ["azi", "azimuth"]:
            return -np.pi, np.pi
        elif parname in ["dip"]:
            return -1e-12, 1e-12
        elif parname in ["std", "sig", "sigma"]:
            return 0.0, 10.0 * np.sqrt(varmap_max)
        elif parname in ["sng", "nug", "nugget", "tau"]:
            return 0.0, 10.0 * np.sqrt(varmap_max)
