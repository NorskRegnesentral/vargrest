import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.integrate import trapz


def fit_3d_field(func, jac, array, resolution, counts, par_guess, bounds, sigma_wt):
    """
    Fits a grid-evaluated function to observations.

    :param func: Callable
        The grid-evaluated function. The first argument should be a 3xN array as input, representing the x, y, and z
        coordinates on where to evaluate the function. The returned valued should be an N-length array representing the
        values of the function at the given coordinates
    :param jac: Optional[Callable]
        Jacobian of the above func. Can be None if scipy should estimate the Jacobian instead.
    :param array: np.ndarray {shape (nx, ny, nz)}
        An array of observations (evaluations of func). array[i, j, k] represent the evaluation of func at
            [(-nx / 2 + i) * dx, (-ny / 2 + j) * dy, (-nz / 2 + k) * dz]
        np.nan may be provided for coordinates without data.
    :param resolution: Tuple[float, float, float]
        Length-3 tuple giving the spatial resolution (currently not in use)
    :param counts: np.ndarray {shape (nx, ny, nz)}
        The number of observations behind array, element-wise (currently not in use)
    :param par_guess: np.ndarray {shape (M,)}
        The initial parameter guess for the parameters of func (past the first argument)
    :param bounds: Tuple[np.ndarray {shape (M,)}, np.ndarray {shape (M,)}]
        The bounds for the parameters of func (past the first argument)
    :param sigma_wt: float
        A weighting parameter to adjust the weight of observations away from the center xyz = (0, 0, 0). A smaller value
        means more weight is put on observations close to the center. The scale should be considered as a number of grid
        cells.
    :return: Tuple[np.ndarray {shape (M,)}, float]
        First, an array of the optimal parameter set for func, fit to the data in array. Second, a scalar value
        representing the accuracy of the fit. This 'quality' value is 0.0 if the fit is equal (in terms of L2) to the
        fit of using baseline_parameters. It is 1.0 if the fit is 100%.

    """
    nx, ny, nz = array.shape
    dx, dy, dz = resolution

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

    indep_data = np.vstack((xm.ravel(), ym.ravel(), zm.ravel()))
    dep_data = array.ravel()
    counts = counts.ravel()

    # Filter out nan entries
    not_nan = np.ones_like(dep_data, dtype=np.bool)
    if np.all(np.isnan(dep_data)):
        # Not possible to calculate a proper estimate
        return np.full_like(par_guess, fill_value=np.nan), np.nan
    elif np.any(np.isnan(dep_data)):
        not_nan = np.squeeze(np.nonzero(~np.isnan(dep_data)))
        indep_data = indep_data[:, not_nan]
        dep_data = dep_data[not_nan]
        counts = counts[not_nan]
    xmv = xm.ravel()[not_nan]
    ymv = ym.ravel()[not_nan]
    zmv = zm.ravel()[not_nan]

    # Prepare weights, w_b ~ 1/\sigma_b^2
    if sigma_wt is not None:
        wt_coef = 1.0 / (2.0 * sigma_wt**2)
        dsq = xmv**2 + ymv**2 + zmv**2
        wts = np.exp(-wt_coef * dsq)
        wts = _transform_array(wts, 2)
        #wts = wts / wts.sum()
        sig = np.divide(1.0, wts, out=np.full_like(wts, np.inf), where=wts != 0.0)
    else:
        sig = None

    # Least-squares fit
    popt = curve_fit(func, indep_data, dep_data, sigma=sig, p0=par_guess, bounds=bounds, jac=jac)[0]

    # Calculate quality of solution
    quality = _calculate_quality_1(func, indep_data, dep_data, popt, not_nan, array)

    return popt, quality


def _calculate_quality(func, indep_data, dep_data, parameters, not_nan, array):
    # Calculate quality as the fraction of data points that differ by more than 15% of the empirical max value. Only
    # consider the three lines through the origin. This gives a more, in a sense, "linear" quality measure than using
    # the entire error volume. The disadvantage is that errors in azimuth may not always be picked up properly.
    err = np.full_like(array, fill_value=np.nan)
    err[not_nan] = np.abs(func(indep_data, *parameters) - dep_data)
    err_3d = err.reshape(array.shape)
    x_err, y_err, z_err = _center_slice(err_3d)
    x_arr, y_arr, z_arr = _center_slice(array)
    max_val = max(np.nanmax(x_arr), np.nanmax(y_arr), np.nanmax(z_arr))
    x_err /= max_val
    y_err /= max_val
    z_err /= max_val
    qth = 0.15
    quality = (x_err < qth).sum() + (y_err < qth).sum() + (z_err < qth).sum()
    quality /= (~np.isnan(x_err)).sum() + (~np.isnan(y_err)).sum() + (~np.isnan(z_err)).sum()
    return quality


def _calculate_quality_1(func, indep_data, dep_data, parameters, not_nan, array):
    def _to_3d(a):
        f1 = np.full(array.size, fill_value=np.nan)
        f1[not_nan] = a
        return f1.reshape(array.shape)

    dep_data_3d = _to_3d(dep_data)
    par_est_3d = _to_3d(func(indep_data, *parameters))

    # eval_volume = (par_est_3d < 0.8 * np.max(par_est_3d)) | (dep_data_3d < 0.8 * np.max(dep_data))
    eval_volume = ~np.isnan(dep_data_3d)

    # B
    pc = 0.5
    dep_eval = dep_data_3d[eval_volume]
    par_eval = par_est_3d[eval_volume]

    sigma_est = np.median(dep_data)
    top_v = dep_eval < pc * sigma_est
    top_p = par_eval < pc * sigma_est

    center = np.zeros_like(eval_volume, dtype=np.bool)
    cx, cy, cz = center.shape
    px, py, pz = cx // 4, cy // 4, cz // 4
    center[px:-px, py:-py, pz:-pz] = True

    top_volume = (top_v | top_p) & center[eval_volume]
    top_dev = dep_eval[top_volume] - par_eval[top_volume]
    worst_case = dep_eval[top_volume] - sigma_est
    top_err = np.abs(top_dev).sum() / np.abs(worst_case).sum()

    sub_err = np.median(np.abs(dep_eval - par_eval)) / sigma_est

    sub_err_contrib = min(sub_err, 0.25)
    top_err_contrib = top_err * 0.75
    return 1.0 - (top_err_contrib + sub_err_contrib)


def _center_slice(array):
    nx, ny, nz = array.shape
    x_slice = array[:, ny // 2, nz // 2]
    y_slice = array[nx // 2, :, nz // 2]
    z_slice = array[nx // 2, ny // 2, :]
    return x_slice, y_slice, z_slice


def _transform_array(x, n):
    x_below = 2**(n-1) * np.power(x, n)
    x_above = 1.0 - 2**(n-1) * np.power(1 - x, n)
    return np.where(x < 0.5, x_below, x_above)


def find_dominant_direction(variogram_map, grid_resolution):
    n_angles = 24   # Number of angles to check
    n_dists = 21    # Number of lag distance points to evaluate integrand at

    dx = grid_resolution[0]
    dy = grid_resolution[1]
    nx = variogram_map.shape[0]
    ny = variogram_map.shape[1]

    if variogram_map.ndim == 3:     # If 3D, extract middle horizontal slice
        nz = variogram_map.shape[2]
        if nz % 2 == 1:
            g = variogram_map[:, :, nz // 2]
        else:
            v0 = variogram_map[:, :, nz // 2 - 1]
            v1 = variogram_map[:, :, nz // 2]
            v01 = np.where(np.isnan(v0), 0.0, v0) + np.where(np.isnan(v1), 0.0, v1)
            w = ~np.isnan(v0) + ~np.isnan(v1)
            g = np.where(w > 0, w * v01, np.nan)
    elif variogram_map.ndim == 2:
        g = variogram_map
    else:
        raise NotImplementedError('Dominant direction not implemented for other than 2D and 3D')

    if np.all(np.isnan(g)):
        return 0.0

    xlength = (nx - 1) * dx
    ylength = (ny - 1) * dy
    h_max = 0.5 * min(xlength, ylength)    # Limiting distance (upper limit of integration)
    hh = np.linspace(0.0, h_max, n_dists)
    hxx = np.linspace(-0.5 * xlength, 0.5 * xlength, nx)
    hyy = np.linspace(-0.5 * ylength, 0.5 * ylength, ny)

    hxx, hyy = np.meshgrid(hxx, hyy, indexing='ij')

    hxx = hxx.ravel()  # Flatten input
    hyy = hyy.ravel()
    gg = g.ravel()

    ggnans = np.isnan(gg.astype(float))
    hxx = hxx[~ggnans]   # Filter out nans
    hyy = hyy[~ggnans]
    gg = gg[~ggnans]
    hxxyy = np.vstack((hxx, hyy)).transpose()

    azimuths = np.linspace(0.0, np.pi, n_angles)
    integrals = np.empty_like(azimuths)

    interpolator = LinearNDInterpolator(hxxyy, gg)
    for i, azi in enumerate(azimuths):      # Loop over angles
        hxx_i = np.cos(azi) * hh
        hyy_i = np.sin(azi) * hh
        gg_i = interpolator(hxx_i, hyy_i)
        integrals[i] = trapz(y=gg_i, x=hh)   # Compute approximate integral from 0 to h_max

    if np.all(np.isnan(integrals)):     # Get index of smallest integral
        return 0.0
    elif np.any(np.isnan(integrals)):
        i_min = np.nanargmin(integrals)
    else:
        i_min = np.argmin(integrals)

    return azimuths[i_min]              # Return corresponding azimuth
