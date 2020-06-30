from typing import List, Callable, Optional

import numpy as np


def approximate_porosity(grain_sizes: List[np.ndarray], q_values: List[float]) -> np.ndarray:
    # Get quantiles most closely surrounding 0.25 and 0.75
    q_a = max(q for q in q_values if q <= 0.25)
    q_b = min(q for q in q_values if q >= 0.25)
    q_c = max(q for q in q_values if q <= 0.75)
    q_d = min(q for q in q_values if q >= 0.75)

    d_a = grain_sizes[q_values.index(q_a)]
    d_b = grain_sizes[q_values.index(q_b)]
    d_c = grain_sizes[q_values.index(q_c)]
    d_d = grain_sizes[q_values.index(q_d)]

    # Interpolate linearly to approximate the 0.25 and 0.75 quantiles
    d_25 = ((0.25 - q_a) / (q_b - q_a)) * d_a + ((q_b - 0.25) / (q_b - q_a)) * d_b
    d_75 = ((0.75 - q_c) / (q_d - q_c)) * d_c + ((q_d - 0.75) / (q_d - q_c)) * d_d

    return 0.2091 + 0.2290 / np.sqrt(d_75 / d_25)


def _resample_trace(elev_ij, prop_ij, n_grid, v_res) -> Optional[np.ndarray]:
    """
    Re-samples properties of a trace onto a regular grid. The method works as follows:
      The goal is to assign values to the cells of the ordered concatenated grid:
      Input grid (elev_ij):                  |aaaaaa|bbb|ccc|d|eeeeeeee|fffffffffffffff|g|hhh|
      Resampled grid (grid_ij):              |----|----|----|----|----|----|----|----|----|----|
      Concatenated grid: (sorted_elev_ij)    |----|-|--||---|-|--|----||---|----|----|-|-||--|-|
      Concatenated values: (full_prop_ij)    |aaaa|a|bb||ccc|d|ee|eeee||fff|ffff|ffff|f|g||hh|?|

    :param elev_ij: Elevations (z) of the trace. Shape: (n,)
    :param prop_ij: Properties of the trace. Shape: (n-1,)
    :param n_grid:  Number of elevations in the regular grid (= number of cells + 1)
    :param v_res:   Vertical resolution (float)

    :return: None, if the values cannot be resampled properly, an ndarray of resampled values otherwise
    """
    # If all properties are nan, there is no need to continue
    if np.all(np.isnan(prop_ij)):
        return None
    # Filter out negative and zero-volume cells
    vols_ij = np.diff(elev_ij, append=np.inf)  # p (artificial cell appended to do proper filtering)
    zero_vols = vols_ij <= 0.0
    elev_ij = elev_ij[~zero_vols]  # n
    prop_ij = prop_ij[~zero_vols[:-1]]  # n - 1
    if vols_ij.size == 0:
        return None
    # If all properties of non-zero volume cells are non, there is no need to continue
    if np.all(np.isnan(prop_ij)):
        return None
    del vols_ij, zero_vols  # Just to be explicit that we won't be needing these anymore

    # Define the grid we are re-sampling to (up to the necessary height)
    grid_ij = np.arange(n_grid) * v_res  # (m,)

    # Create an elevation grid consisting of input elevations and re-sampling elevations
    full_elev_ij = np.hstack((elev_ij, grid_ij))  # (n + m,)

    # Determine the sort-order of the scrambled grid
    order_ij = np.argsort(full_elev_ij)  # n + m

    # Determine the grid indexes in the input grid from which each cell in the concatenated shall get its values
    left_cell_0 = np.zeros(order_ij.size, dtype=int)
    left_cell_0[order_ij < elev_ij.size] = np.arange(elev_ij.size)
    left_cell = np.maximum.accumulate(left_cell_0)  # n + m
    # left_cell[i] refers to the index of input grid that cell "i" in the concatenated grid should get its
    # value from

    # Find if any of the cells in the concatenated grid are "out-of-bounds", that is, getting values from beyond the
    # top-most cell in the input grid. Such values are not well-defined and will be excluded.
    oob = left_cell == prop_ij.size  # (n + m,)
    last_valid = np.argmax(oob) if np.any(oob) else oob.size
    left_cell = left_cell[:last_valid]

    # Define the full concatenated grid, the corresponding values and volumes
    full_prop_ij = prop_ij[left_cell]  # (nf,)
    sorted_elev_ij = full_elev_ij[order_ij][:full_prop_ij.size + 1]  # (nf + 1,)
    full_vols_ij = np.diff(sorted_elev_ij)  # nf

    # Remove cells in the concatenated grid that have nan-values
    nan_props = np.isnan(full_prop_ij)
    valid_props = full_prop_ij[~nan_props]
    valid_elevs = sorted_elev_ij[:-1][~nan_props]
    valid_volus = full_vols_ij[~nan_props]
    # TODO: histogram does not take advantage of equispaced binning. Speed-up might be gained from that. One may provide
    #  number of bins and range as arguments, but that is not faster for some reason. We should instead implement our
    #  own tailored method for this. Another area for improvement is that we are using the same bin resolution for each
    #  trace. Some of the calculation are therefore the same for each trace, something we may take advantage of.
    p_sum = np.histogram(valid_elevs, grid_ij, weights=valid_props * valid_volus)[0]
    w_sum = np.histogram(valid_elevs, grid_ij, weights=valid_volus)[0]
    # Normalize based on volume used within each cell. Ignore if volume used is below 1e-8
    prop_reg_ij = p_sum / np.maximum(w_sum, 1e-8)
    prop_reg_ij[w_sum < 1e-8] = np.nan

    # QC:
    # import matplotlib.pyplot as plt
    # def _sw(_elev, _prop):
    #     _el = []
    #     _pr = []
    #     for k in range(_elev.size - 1):
    #         _el.append(_elev[k])
    #         _el.append(_elev[k + 1])
    #         _pr.append(_prop[k])
    #         _pr.append(_prop[k])
    #     return _el, _pr
    #
    # plt.plot(*_sw(elev_ij, prop_ij), 'o-')
    # plt.plot(*_sw(grid_ij[:-1], prop_reg_ij[:-1]))
    #
    # plt.legend(['Input', 'No zero-vol', 'Regular (2)', 'Regular (1)'])
    # ---
    return prop_reg_ij


# TODO: Enable output grid to differ from input grid
# If dx and dy are different in the input grid, sample onto a grid with dx = dy
# Allow output grid to be rotated relative to input grid (for local estimation in a specified rectangle)
def resample_onto_regular_grid(elev: np.ndarray,
                               prop: np.ndarray,
                               vres: float) -> np.ndarray:
    n_t, n_x, n_y = elev.shape
    top = np.max(elev)
    if top > 30.0:
        print(f'NB! The vertical thickness of the estimation grid is high ({top}m). Consider inspecting the pillars for'
              f' unintended high values')
    n_z = int(np.round(top / vres))

    prop_reg = np.empty(shape=(n_x, n_y, n_z))
    prop_reg[:] = np.nan

    for i in range(n_x):
        for j in range(n_y):
            k_max_ij = int(np.floor(elev[-1, i, j] / vres))
            k_max_ij = min(k_max_ij, n_z - 1)
            if k_max_ij == 0:
                continue
            elev_ij = elev[:, i, j]  # p
            prop_ij = prop[:-1, i, j]  # p - 1
            prop_reg_ij = _resample_trace(elev_ij, prop_ij, k_max_ij + 2, vres)
            if prop_reg_ij is None:
                continue
            prop_reg[i, j, :prop_reg_ij.size] = prop_reg_ij

    return prop_reg


def mask_array(a: np.ndarray, b: np.ndarray, v: float) -> np.ndarray:
    c = np.full_like(a, np.nan)
    c[b == v] = a[b == v]
    return c


def mask_array_complement(a: np.ndarray, b: np.ndarray, v: float) -> np.ndarray:
    c = np.full_like(a, np.nan)
    c[b != v] = a[b != v]
    return c
