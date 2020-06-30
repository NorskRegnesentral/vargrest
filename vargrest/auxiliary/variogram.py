import numpy as np


class Variogram:
    def __init__(self, rx, ry=1.0, rz=1.0, azi=0.0, dip=0.0, std=1.0):
        self._rx, self._ry, self._rz = rx, ry, rz
        self._azi = azi
        self._dip = dip
        self._var = std * std  # Not currently used?

        # Factors
        cos_rot = np.cos(azi)
        sin_rot = np.sin(azi)
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
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

    def _distance(self, dx, dy=0.0, dz=0.0):
        return np.sqrt(self._txx * dx ** 2 + self._tyy * dy ** 2 + self._tzz * dz ** 2 +
                       self._txy * dx * dy + self._txz * dx * dz + self._tyz * dy * dz)

    def _corr(self, dx, dy=0.0, dz=0.0):
        d = self._distance(dx, dy, dz)
        return self._corr_1d(d)

    def _corr_1d(self, d):
        raise NotImplementedError()

    def create_corr_array(self, nx, dx, ny=1, dy=1.0, nz=1, dz=1.0, centered=True):
        xx, yy, zz = np.meshgrid(np.arange(-nx, nx + 1), np.arange(-ny, ny + 1), np.arange(-nz, nz + 1), indexing='ij')
        c = self._corr(xx * dx, yy * dy, zz * dz)
        if not centered:
            c = np.roll(c, shift=tuple(-(s // 2) for s in c.shape), axis=(0, 1, 2))
        return c


class ExponentialVariogram(Variogram):
    def _corr_1d(self, d):
        return np.exp(-3 * d)


class GaussianVariogram(Variogram):
    def _corr_1d(self, d):
        return np.exp(-3 * d ** 2)


class GeneralExponentialVariogram(Variogram):
    def __init__(self, rx, ry=1.0, rz=1.0, azi=0.0, dip=0.0, std=1.0, power=1.5):
        super().__init__(rx, ry, rz, azi, dip, std)
        self._power = power

    def _corr_1d(self, d):
        return np.exp(-3 * d ** self._power)


class SphericalVariogram(Variogram):
    def _corr_1d(self, d):
        return np.where(d < 1.0, 1.0 - d * (1.5 - 0.5 * (d ** 2)), 0.0)


class Matern32Variogram(Variogram):
    def _corr_1d(self, d):
        sd = 4.744 * d
        return np.exp(-sd) * (1.0 + sd)


class Matern52Variogram(Variogram):
    def _corr_1d(self, d):
        sd = 5.918 * d
        return np.exp(-sd) * (1.0 + sd + sd ** 2 / 3.0)


class Matern72Variogram(Variogram):
    def _corr_1d(self, d):
        sd = 6.877 * d
        return np.exp(-sd) * (1.0 + sd + 0.4 * sd ** 2 + sd ** 3 / 15)
