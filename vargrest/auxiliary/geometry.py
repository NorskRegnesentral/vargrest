from typing import Tuple

import numpy as np


class CoordinateTransformation:
    def __init__(self, rx, ry=1.0, rz=1.0, azi=0.0, dip=0.0):
        self._rx, self._ry, self._rz = rx, ry, rz
        self._azi = azi
        self._dip = dip

        # Factors
        cos_rot = np.cos(azi)
        sin_rot = np.sin(azi)
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)

        # Rotation matrix
        self._rot_mat = np.array([
            [cos_dip * cos_rot, -sin_rot, sin_dip * cos_rot],
            [cos_dip * sin_rot, cos_rot, sin_dip * sin_rot],
            [-sin_dip, 0.0, cos_dip]
        ])

        # Scaling matrix
        self._scale_mat = np.diag([1 / rx, 1 / ry, 1 / rz])

        # Coefficients used to calculate distance directly
        f1 = 1 / rx ** 2
        f2 = 1 / ry ** 2
        f3 = 1 / rz ** 2
        self._txx = cos_rot * cos_rot * cos_dip * cos_dip * f1 + sin_rot * sin_rot * f2 + cos_rot * cos_rot * sin_dip * sin_dip * f3
        self._tyy = sin_rot * sin_rot * cos_dip * cos_dip * f1 + cos_rot * cos_rot * f2 + sin_rot * sin_rot * sin_dip * sin_dip * f3
        self._tzz = sin_dip * sin_dip * f1 + cos_dip * cos_dip * f3
        self._txy = 2 * (
                cos_dip * cos_dip * cos_rot * sin_rot * f1 - sin_rot * cos_rot * f2 + sin_dip * sin_dip * cos_rot * sin_rot * f3)
        self._txz = 2 * (cos_rot * cos_dip * sin_dip * f1 - cos_rot * cos_dip * sin_dip * f3)
        self._tyz = 2 * (sin_rot * cos_dip * sin_dip * f1 - sin_rot * cos_dip * sin_dip * f3)

    def distance(self, dx, dy=0.0, dz=0.0):
        return np.sqrt(self._txx * dx ** 2 + self._tyy * dy ** 2 + self._tzz * dz ** 2 +
                       self._txy * dx * dy + self._txz * dx * dz + self._tyz * dy * dz)

    def forward_transform(self, s: np.ndarray) -> np.ndarray:
        if 0 < s.ndim < 3:
            s_3d = np.vstack((s, np.zeros(shape=(3 - s.ndim, s.shape[0]))))
        r = np.matmul(self._rot_mat, s)
        return np.matmul(self._scale_mat, r)

    def distance_field(self,
                       x_lags: Tuple[int, int],
                       y_lags: Tuple[int, int],
                       n_x: int,
                       n_y: int) -> Tuple[np.ndarray, np.ndarray]:
        h_x = np.linspace(x_lags[0], x_lags[1], n_x)
        h_y = np.linspace(y_lags[0], y_lags[1], n_y)
        hx_mesh, hy_mesh = np.meshgrid(h_x, h_y)
        hh = self.distance(hx_mesh.flatten(), hy_mesh.flatten())
        return hh

    def axis_aligned_bounding_box(self, factor: float) -> np.ndarray:
        m = (1.0/factor) * np.matmul(self._scale_mat, self._rot_mat)
        omega_inv = np.linalg.inv(np.matmul(m.transpose(), m))
        return np.sqrt(np.diag(omega_inv))
