from typing import Tuple, Optional
from warnings import warn

import numpy as np

from vargrest.auxiliary.progress import progress


def _nanvar(arr: np.ndarray) -> float:
    """
    Serves the same purpose as np.nanvar and should always return the same as np.nanvar. However, nanvar can be slow
    due to fill-ins and this implementation is intended to alleviate that.
    """
    if np.all(np.isnan(arr)) or arr.size == 0:
        return np.nan
    nonnan_diffs = arr[~np.isnan(arr)]
    nonnan_diffs -= np.mean(nonnan_diffs)
    if nonnan_diffs.size > 0:
        return np.sum(nonnan_diffs ** 2) / nonnan_diffs.size
    else:
        return np.nan


def _estimate_variogram_3d_single_lag(rf: np.ndarray,
                                      rf_nan: np.ndarray,
                                      h_x: int,
                                      h_y: int,
                                      h_z: int,
                                      sub_sampling: Optional[int] = None) -> Tuple[float, int]:
    """
    Faster estimation of variance at a single lag distance combination (h_x, h_y, h_z)
    :param rf: Random field
    :param rf_nan: boolean array that maps the nan-values of rf. Should have same shape as rf. Technically not required,
        but significantly improves performance
    :param h_x: Lag distance in x-direction
    :param h_y: Lag distance in y-direction
    :param h_z: Lag distance in z-direction
    :return: gamma: variogram value for given combination of lags
    :return n: number of cell differences used to compute gamma
    """
    def _slice(_h):
        if _h < 0:
            return slice(-_h, None)
        elif _h == 0:
            return slice(None, None)
        else:
            return slice(None, -_h)

    sx1 = _slice(h_x)
    sx0 = _slice(-h_x)
    sy1 = _slice(h_y)
    sy0 = _slice(-h_y)
    sz1 = _slice(h_z)
    sz0 = _slice(-h_z)

    d1 = rf[sx1, sy1, sz1]
    d0 = rf[sx0, sy0, sz0]

    if sub_sampling is not None:
        # TODO: This part of the code works and can have significant impact on run time. However, setting sub_sampling
        #  to a reasonable value is not straightforward, nor is it guaranteed to give any meaningful output, even if it
        #  is high. An initial test on model_10_poro_perm, based on the quality measures (with its caveats), shows that
        #  the sub_sampling value is difficult. Quality seem to improve up to sub_sampling=600, but at 1200, quality is
        #  worse. 2400 seems to get close, but obviously at the cost of run time.
        #  Therefore, this code segment mainly illustrates the potential speed-up that can be achieved by sub-sampling
        #  in a clever way, but the particular method used to extract a sub-sample should be improved. The challenge is
        #  avoiding ANY moderately expensive operations on d1 or d0.
        # Sample only a fraction of the values
        step = rf.size // sub_sampling
        # Randomly add 1 to step and choose a random starting location. This is to avoid possible biases introduced
        # by using step as a sub-selector instead of drawing randomly
        if np.random.uniform() > 0.5:
            step += 1
        start = np.random.randint(0, step)
        d1 = d1.flat[start::step]
        d0 = d0.flat[start::step]
        d_nnn = ~(np.isnan(d1) | np.isnan(d0))
        d1 = d1[d_nnn]
        d0 = d0[d_nnn]
    else:
        # This is slightly faster than using _nanvar on d1 - d0 directly
        is_nnan = ~(rf_nan[sx1, sy1, sz1] | rf_nan[sx0, sy0, sz0])
        d1 = d1[is_nnan]
        d0 = d0[is_nnan]
    diff = d1 - d0
    if diff.size == 0:
        return np.nan, 0
    else:
        return np.sum(diff ** 2) / diff.size, diff.size


def _estimate_variogram_np_3d_dense(rf: np.ndarray,
                                    x_lag: int,
                                    y_lag: int,
                                    z_lag: int,
                                    sub_sampling: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a nonparametric variogram estimate over a 3D mesh
    of x, y and z lags
    :param rf: Random field realization. 3D numpy array
    :param x_lag: Number of lags in x direction. Number of cells
    :param y_lag: Number of lags in y direction. Number of cells
    :param z_lag: Number of lags in z direction. Number of cells
    :return varmap: Variogram estimate at specified lags. 2D numpy array
    :return counts: Number of data points used to compute each element of varmap
    """

    x_lags_sequence = range(-x_lag, x_lag + 1)
    y_lags_sequence = range(-y_lag, y_lag + 1)
    z_lags_sequence = range(0, z_lag + 1)

    varmap = np.full(shape=(2 * x_lag + 1, 2 * y_lag + 1, 2 * z_lag + 1), fill_value=np.nan)
    counts = np.zeros_like(varmap, dtype=int)

    if rf.ndim != 3:
        raise Exception("rf must be a 3D array")

    # rf_t = np.transpose(rf, (1, 0, 2))

    rf_nan = np.isnan(rf)
    for i, h_x in progress(enumerate(x_lags_sequence), 'Estimating empirical variogram', len(x_lags_sequence)):
        for j, h_y in enumerate(y_lags_sequence):
            # Calculate half of k-index and mirror the array afterwards
            for k, h_z in enumerate(z_lags_sequence, start=z_lag):
                varmap_ijk, counts_ijk = _estimate_variogram_3d_single_lag(rf, rf_nan, h_x, h_y, h_z, sub_sampling)
                varmap[i, j, k] = varmap_ijk
                counts[i, j, k] = counts_ijk

    varmap[:, :, :z_lag] = varmap[::-1, ::-1, :z_lag:-1]

    return varmap, counts


def _estimate_variogram_np_3d_sparse(rf: np.ndarray,
                                     x_lags_min_max_step: Tuple[int, int, int],
                                     y_lags_min_max_step: Tuple[int, int, int],
                                     z_lags_min_max_step: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a nonparametric variogram estimate over a 3D mesh
    of x, y and z lags, skipping some cells for speed
    :param rf: Random field realization. 3D numpy array
    :param x_lags_min_max_step: Range of lags and step in x direction. Number of cells
    :param y_lags_min_max_step: Range of lags and step in y direction. Number of cells
    :param z_lags_min_max_step: Range of lags and step in z direction. Number of cells
    :return varmap: Variogram estimate at specified lags. 2D numpy array
    :return counts: Number of data points used to compute each element of varmap
    """
    def _lag_iter(start, end, step):
        max_lag = end + step - (end % step)
        return -max_lag, max_lag + 1, step

    # Adjust lags so that the center is included
    x_lag_iter = _lag_iter(*x_lags_min_max_step)
    y_lag_iter = _lag_iter(*y_lags_min_max_step)
    z_lag_iter = _lag_iter(*z_lags_min_max_step)

    varmap = np.full(shape=(x_lag_iter[1] - x_lag_iter[0],
                            y_lag_iter[1] - y_lag_iter[0],
                            z_lag_iter[1] - z_lag_iter[0]), fill_value=np.nan)
    counts = np.zeros_like(varmap, dtype=int)

    if rf.ndim != 3:
        raise Exception("rf must be a 3D array")

    rf_t = np.transpose(rf, (1, 0, 2))

    x_step = x_lag_iter[2]
    y_step = y_lag_iter[2]
    z_step = z_lag_iter[2]
    rf_nan = np.isnan(rf)
    for i, h_x in enumerate(range(*x_lag_iter)):
        for j, h_y in enumerate(range(*y_lag_iter)):
            for k, h_z in enumerate(range(*z_lag_iter)):
                varmap_si_sj_sk, counts_si_sj_sk = _estimate_variogram_3d_single_lag(rf_t, rf_nan, h_x, h_y, h_z)
                varmap[i * x_step, j * y_step, k * z_step] = varmap_si_sj_sk
                counts[i * x_step, j * y_step, k * z_step] = counts_si_sj_sk

    return varmap, counts

def _estimate_variogram_np_3d_random(rf: np.ndarray,
                                     x_lag: int,
                                     y_lag: int,
                                     z_lag: int,
                                     sampling_factor: Optional[float] = 0.10,
                                     max_samples: Optional[int] = 50000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a nonparametric variogram estimate over a 3D mesh
    of x, y and z lags, sampling only some cells for speed
    :param rf: Random field realization. 3D numpy array
    :param x_lag: Number of lags in x direction. Number of cells
    :param y_lag: Number of lags in y direction. Number of cells
    :param z_lag: Number of lags in z direction. Number of cells
    :param n_samples: Number of cells to sample
    :return varmap: Variogram estimate at specified lags. 2D numpy array
    :return counts: Number of data points used to compute each element of varmap
    """

    x_lags_sequence = range(-x_lag, x_lag + 1)
    y_lags_sequence = range(-y_lag, y_lag + 1)
    z_lags_sequence = range(-z_lag, z_lag + 1)

    n_lags_x = len(x_lags_sequence)
    n_lags_y = len(y_lags_sequence)
    n_lags_z = len(z_lags_sequence)

    varmap = np.full(shape=(n_lags_x, n_lags_y, n_lags_z), fill_value=np.nan)
    counts = np.zeros_like(varmap, dtype=int)

    if rf.ndim != 3:
        raise Exception("rf must be a 3D array")

    rf_t = np.transpose(rf, (1, 0, 2))

    n_lag_combos = n_lags_x * n_lags_y * n_lags_z
    n_samples = min(max_samples, int(sampling_factor * n_lag_combos))

    n_samples_actual = 0

    rf_nan = np.isnan(rf)
    while n_samples_actual < n_samples:
        i = np.random.randint(0, n_lags_x)
        j = np.random.randint(0, n_lags_y)
        k = np.random.randint(0, n_lags_z)
        h_x = x_lags_sequence[i]
        h_y = y_lags_sequence[j]
        h_z = z_lags_sequence[k]
        varmap_ijk, counts_ijk = _estimate_variogram_3d_single_lag(rf_t, rf_nan, h_x, h_y, h_z)

        if not np.isnan(varmap_ijk):
            varmap[i, j, k] = varmap_ijk
            counts[i, j, k] = counts_ijk
            n_samples_actual += 1

    return varmap, counts
