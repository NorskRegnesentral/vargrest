import numpy as np
import h5py
from typing import List, Optional, Tuple, Union

from vargrest.auxiliary.box import Box
from vargrest.auxiliary.visualization import visualize_crop
from vargrest.variogramdata import _utilities

from nrresqml.derivatives.ijkgridcreator import IjkGridCreator
from nrresqml.derivatives.dataextraction import extract_property, crop_array, extract_geometry
from nrresqml.resqml import ResQml
from nrresqml.structures.resqml.representations import IjkGridRepresentation


class VariogramDataInterface:
    # TODO:
    #  Update _utilities.mask_array to support multiple subenvs/archels
    def __init__(self,
                 x0: float,
                 y0: float,
                 dx: float,
                 dy: float,
                 box: Optional[Box],
                 z: np.ndarray,
                 property_grid: np.ndarray,
                 archel: np.ndarray) -> None:
        nx, ny = z.shape[1:]
        if box is None:
            box = Box(x0, y0, x0 + nx * dx, y0 + ny * dy)
        # Save original lowest property-level for later
        self._p_base = property_grid[-1, :, :]
        self._crop_path = np.array([
            [box.x0 - x0, box.y0 - y0],
            [box.x0 - x0, box.y1 - y0],
            [box.x1 - x0, box.y1 - y0],
            [box.x1 - x0, box.y0 - y0],
            [box.x0 - x0, box.y0 - y0],
        ])
        crop_args = x0, y0, dx, dy, box.x0, box.y0, box.x1, box.y1
        z = crop_array(z, *crop_args)
        property_grid = crop_array(property_grid, *crop_args)
        archel = crop_array(archel, *crop_args)
        # Save crop paths for later
        # Crop all grids
        self._dx = dx
        self._dy = dy
        self._z = z - z[0, :, :]  # Make the grid bottom conforming
        self._property_grid = property_grid

        assert archel.dtype == np.int
        self._archel = archel

    def property_grid(self,
                      dz: float,
                      archels: Optional[List[int]] = None
                      ) -> np.ndarray:

        grid = self._property_grid
        if archels is None:
            ae_inactive = 0  # Inactive cells archel
            masked_grid = _utilities.mask_array_complement(grid, self._archel, ae_inactive)
            return _utilities.resample_onto_regular_grid(self._z, masked_grid, dz)
        else:
            masked_grid = _utilities.mask_array(grid, self._archel, archels[0])
            return _utilities.resample_onto_regular_grid(self._z, masked_grid, dz)

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy
    
    @property
    def archel_set(self) -> np.ndarray:
        return np.unique(self._archel)

    def plot_crop_box(self,
                      save_figure: Optional[bool] = False,
                      dir_name: Optional[str] = None,
                      file_name: Optional[str] = None) -> None:
        # Extract bottom layer from uncropped data array
        image_data = self._p_base
        n_x, n_y = image_data.shape
        image_ext = (0.0, n_x * self.dx, 0.0, n_y * self.dy)

        # polygon_x = self._crop_paths["xy"][:, 1]
        # polygon_y = n_x * dx - self._crop_paths["xy"][:, 0]

        polygon_x = self._crop_path[:, 0]
        polygon_y = self._crop_path[:, 1]

        fig = visualize_crop(image_data, image_ext, polygon_x, polygon_y)

        if save_figure:
            import os
            filename_suffix = "png"
            full_path = os.path.join(dir_name, file_name + "." + filename_suffix)
            fig.savefig(full_path)

    @staticmethod
    def create_from_delft3d(input_file: str,
                            archel_file: str,
                            grid_resolution: Tuple[float, float] = (50.0, 50.0),
                            box: Optional[Box] = None):
        f = h5py.File(input_file, 'r')
        # Import temporal layer depths
        dps = np.array(f['DPS'], dtype='float')
        # Import grain diameter quantiles
        d_10 = np.array(f['DXX01'], dtype='float')
        d_16 = np.array(f['DXX02'], dtype='float')
        d_50 = np.array(f['DXX03'], dtype='float')
        d_84 = np.array(f['DXX04'], dtype='float')
        d_90 = np.array(f['DXX05'], dtype='float')
        f.close()

        z_botconf = IjkGridCreator.mono_elevation(-dps)
        poro = _utilities.approximate_porosity([d_10, d_16, d_50, d_84, d_90],
                                               [0.10, 0.16, 0.50, 0.84, 0.90])

        with h5py.File(archel_file) as hf:
            if 'archel' in hf.keys():
                archel = np.array(hf['archel'], dtype=np.int)
            else:
                archel = np.zeros_like(poro, dtype=np.int)
            return VariogramDataInterface(0.0, 0.0, grid_resolution[0], grid_resolution[1], box, z_botconf,
                                          poro, archel)

    @staticmethod
    def create_from_resqml(rq: ResQml, box: Optional[Box], an: Optional[str], indicator: Union[None, int, str]):
        assert not (an is None and indicator is None)
        assert not (an is not None and indicator is not None)
        ijk, xx, yy, pillars = extract_geometry(rq, True, 'kij')
        dx = xx[1, 0] - xx[0, 0]
        dy = yy[0, 1] - yy[0, 0]

        # Extract categorical parameters for archel and subenvironment
        archel = extract_property(rq, ijk, 'archel', True)

        # Copy remaining arrays from HDF5 file to memory for easier post-processing
        archel = np.array(archel, dtype=np.int)
        pillars = np.array(pillars)

        # Extract estimation parameter
        if an is not None:
            param = _extract_parameter(rq, ijk, an)
        else:
            if isinstance(indicator, int):
                param = np.logical_and(archel == indicator, ~np.isnan(archel)).astype(np.float)
            else:
                assert isinstance(indicator, str)
                # Currently, the following syntax is enforced:
                # parameter<value
                p, v = indicator.split('<')
                pg = extract_property(rq, ijk, p, False)
                arr = np.array(pg, dtype=np.float)
                param = (np.where(np.isnan(arr), -np.inf, arr) > float(v)).astype(np.float)

        return VariogramDataInterface(xx[0, 0], yy[0, 0], dx, dy, box, pillars, param, archel)


""" Utility functions for ResQml data extraction """


def _extract_parameter(rq: ResQml, ijk: IjkGridRepresentation, an: str):
    try:
        # Try to extract an attribute whose name matches an. If it does not exist, use rough porosity approximation
        return extract_property(rq, ijk, an, False)
    except AssertionError:
        pass
    # Names below should match what is used in Delft3DResQmlAdaptor
    # Extract continuous properties
    try:
        dxx0x = [
            np.array(extract_property(rq, ijk, f'DXX0{i}', False))
            for i in range(1, 6)
        ]
        return _utilities.approximate_porosity(dxx0x, [0.10, 0.16, 0.50, 0.84, 0.90])
    except AssertionError:
        d50s = np.array(extract_property(rq, ijk, f'd50_per_sedclass', False))  # TODO: technically not related to ijk
        fracs = [
            np.array(extract_property(rq, ijk, f'Sed{i}_volfrac', False))
            for i in range(1, 7)
        ]
        acc_fracs = np.cumsum([f.flatten() for f in fracs], axis=0).T

        def _quantile(q):
            second = np.argmax(acc_fracs > q, axis=1)
            first = second - 1
            af0 = acc_fracs[np.arange(first.size), first]
            af1 = acc_fracs[np.arange(second.size), second]
            t = (q - af0) / (af1 - af0)
            d_q = d50s[first] + t * (d50s[second] - d50s[first])
            d_q[second == 0] = d50s[0]
            d_q[acc_fracs[:, -1] == 0.0] = d50s[0]
            return d_q.reshape(fracs[0].shape)

        poro = 0.2091 + 0.2290 / np.sqrt(_quantile(0.75) / _quantile(0.25))
        return poro
