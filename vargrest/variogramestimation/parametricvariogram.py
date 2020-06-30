import warnings
from enum import Enum

import numpy as np
from typing import Callable, Union, Optional


class VariogramType(Enum):
    Exponential = 'exponential'
    Gaussian = 'gaussian'
    GeneralExponential = 'general_exponential'
    Spherical = 'spherical'


# Correlation functions
def _exponential_corr(d, r):
    return np.exp(-3 * d / r)


def _d_exponential_corr(d, r):
    return -3 / r * _exponential_corr(d, r)


def _gaussian_corr(d, r):
    return np.exp(-3 * (d / r) ** 2)


def _d_gaussian_corr(d, r):
    return -6 * (d / r ** 2) * _gaussian_corr(d, r)


def _general_exponential_corr(d, r, p):
    return np.exp(-3 * np.power(d / r, p))


def _d_general_exponential_corr(d, r, p):
    return -3 * p * (d / r) ** (p - 1) / r * _general_exponential_corr(d, r, p)


def _spherical_corr(d, r):
    return np.where(d / r < 1.0, 1.0 - (d / r) * (1.5 - 0.5 * ((d / r) ** 2)), 0.0)


def _d_spherical_corr(d, r):
    return np.where(d / r < 1.0, 1.5 / r * (-1.0 + (d / r) ** 2), 0.0)


class AnisotropicVariogram:
    GEN_EXP_PWR = 1.5  # We are currently not parameterizing the exponent. May want to look into that later

    def __init__(self, family: Union[VariogramType, str], nug: bool = False):
        # Setup variogram function
        if isinstance(family, str):
            family = VariogramType(family)
        if family == VariogramType.Exponential:
            self._func = self._get_exponential_variogram_function(nug=nug)
        elif family == VariogramType.Gaussian:
            self._func = self._get_gaussian_variogram_function(nug=nug)
        elif family == VariogramType.GeneralExponential:
            self._func = self._get_general_exponential_variogram_function(nug=nug)
        elif family == VariogramType.Spherical:
            self._func = self._get_spherical_variogram_function(nug=nug)
        else:
            raise Exception('Invalid variogram family name: {}'.format(family))

        # Setup (optional) variogram function Jacobian
        if nug is True:
            # Currently not implemented
            self._g_func = None
            self._dg_func = None
        else:
            self._g_func = {
                VariogramType.Exponential: _exponential_corr,
                VariogramType.Gaussian: _gaussian_corr,
                VariogramType.GeneralExponential: lambda d, r: _general_exponential_corr(d, r, self.GEN_EXP_PWR),
                VariogramType.Spherical: _spherical_corr,
            }[family]
            self._dg_func = {
                VariogramType.Exponential: _d_exponential_corr,
                VariogramType.Gaussian: _d_gaussian_corr,
                VariogramType.GeneralExponential: lambda d, r: _d_general_exponential_corr(d, r, self.GEN_EXP_PWR),
                VariogramType.Spherical: _d_spherical_corr,
            }[family]

    def get_variogram_function(self) -> Callable:
        return self._func

    def get_variogram_jacobian(self) -> Optional[Callable]:
        if self._dg_func is None or self._g_func is None:
            return None
        return self._jac

    def _distance(self, dx, dy, dz, rx, ry=1.0, rz=1.0, azi=0.0, dip=0.0):
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
        self._txy = 2 * (cos_dip * cos_dip * cos_rot * sin_rot * f1 - sin_rot * cos_rot * f2 + sin_dip * sin_dip * cos_rot * sin_rot * f3)
        self._txz = 2 * (cos_rot * cos_dip * sin_dip * f1 - cos_rot * cos_dip * sin_dip * f3)
        self._tyz = 2 * (sin_rot * cos_dip * sin_dip * f1 - sin_rot * cos_dip * sin_dip * f3)

        return np.sqrt(self._txx * dx ** 2 + self._tyy * dy ** 2 + self._tzz * dz ** 2 +
                       self._txy * dx * dy + self._txz * dx * dz + self._tyz * dy * dz)

    def _d_distance2(self, dx, dy, dz, rx, ry, rz, azi, dip):
        cos = np.cos
        sin = np.sin
        # See symdiff.py for calculation of dt
        # dt: 5 x 6
        #   rows: derivatives wrt rx, ry, rz, azi, dip
        #   cols: derivatives of txx, tyy, tzz, txy, txz, tyz
        dt = np.array([
            # d rx
            [
                -2*cos(azi)**2*cos(dip)**2/rx**3,
                -2*sin(azi)**2*cos(dip)**2/rx**3,
                -2*sin(dip)**2/rx**3,
                -4*sin(azi)*cos(azi)*cos(dip)**2/rx**3,
                -4*sin(dip)*cos(azi)*cos(dip)/rx**3,
                -4*sin(azi)*sin(dip)*cos(dip)/rx**3,
            ],
            # d ry
            [
                -2*sin(azi)**2/ry**3,
                -2*cos(azi)**2/ry**3,
                0,
                4*sin(azi)*cos(azi)/ry**3,
                0,
                0,
            ],
            # d rz
            [
                -2*sin(dip)**2*cos(azi)**2/rz**3,
                -2*sin(azi)**2*sin(dip)**2/rz**3,
                -2*cos(dip)**2/rz**3,
                -4*sin(azi)*sin(dip)**2*cos(azi)/rz**3,
                -4*sin(dip)*cos(azi)*cos(dip)/rz**3,
                -4*sin(azi)*sin(dip)*cos(dip)/rz**3,
            ],
            # d azi
            [
                -2*sin(azi)*sin(dip)**2*cos(azi)/rz**2 + 2*sin(azi)*cos(azi)/ry**2 - 2*sin(azi)*cos(azi)*cos(dip)**2/rx**2,
                2*sin(azi)*sin(dip)**2*cos(azi)/rz**2 - 2*sin(azi)*cos(azi)/ry**2 + 2*sin(azi)*cos(azi)*cos(dip)**2/rx**2,
                0,
                -(2*sin(dip)**2/rz**2 - 2/ry**2 + 2*cos(dip)**2/rx**2)*sin(azi)**2 + (2*sin(dip)**2/rz**2 - 2/ry**2 + 2*cos(dip)**2/rx**2)*cos(azi)**2,
                -(2/rz**2 + 2/rx**2)*sin(azi)*sin(dip)*cos(dip),
                (2/rz**2 + 2/rx**2)*sin(dip)*cos(azi)*cos(dip),
            ],
            # d dip
            [
                2*sin(dip)*cos(azi)**2*cos(dip)/rz**2 - 2*sin(dip)*cos(azi)**2*cos(dip)/rx**2,
                2*sin(azi)**2*sin(dip)*cos(dip)/rz**2 - 2*sin(azi)**2*sin(dip)*cos(dip)/rx**2,
                -2*sin(dip)*cos(dip)/rz**2 + 2*sin(dip)*cos(dip)/rx**2,
                (4*sin(dip)*cos(dip)/rz**2 - 4*sin(dip)*cos(dip)/rx**2)*sin(azi)*cos(azi),
                -(2/rz**2 + 2/rx**2)*sin(dip)**2*cos(azi) + (2/rz**2 + 2/rx**2)*cos(azi)*cos(dip)**2,
                -(2/rz**2 + 2/rx**2)*sin(azi)*sin(dip)**2 + (2/rz**2 + 2/rx**2)*sin(azi)*cos(dip)**2,
            ],
        ])
        h = np.stack([
            dx ** 2,
            dy ** 2,
            dz ** 2,
            dx * dy,
            dx * dz,
            dy * dz,
        ])
        return dt @ h

    def _jac(self, dxyz, rx, ry, rz, azi, dip, std):
        dx, dy, dz = dxyz[0, :], dxyz[1, :], dxyz[2, :]
        d = self._distance(dx, dy, dz, rx, ry, rz, azi, dip)
        d_std = 2 * std * (1 - self._g_func(d, 1.0))
        with np.errstate(divide='ignore', invalid='ignore'):
            # Division by zero will occur, but we ignore it and clean up afterwards instead
            d_other = -std ** 2 * self._dg_func(d, 1.0) * 0.5 * self._d_distance2(dx, dy, dz, rx, ry, rz, azi, dip) / d
        d_other[np.isnan(d_other)] = 0.0  # divide-by-zero may occur
        jac = np.zeros((6, dx.size))
        jac[:-1, :] = d_other
        jac[-1, :] = d_std
        return jac.T

    """ Variogram functions """

    def _get_exponential_variogram_function(self, nug: bool = False) -> Callable:
        if nug:
            return lambda dxyz, rx, ry, rz, azi, dip, std, sng: (sng**2) + (std**2) * (
                    1.0 - _exponential_corr(self._distance(dxyz[0, :], dxyz[1, :], dxyz[2, :], rx, ry, rz, azi, dip), 1.0))
        elif not nug:
            return lambda dxyz, rx, ry, rz, azi, dip, std: (std ** 2) * (
                    1.0 - _exponential_corr(self._distance(dxyz[0, :], dxyz[1, :], dxyz[2, :], rx, ry, rz, azi, dip), 1.0))

    def _get_gaussian_variogram_function(self, nug: bool = False) -> Callable:
        if nug:
            return lambda dxyz, rx, ry, rz, azi, dip, std, sng: (sng ** 2) + (std ** 2) * (
                    1.0 - _gaussian_corr(self._distance(dxyz[0, :], dxyz[1, :], dxyz[2, :], rx, ry, rz, azi, dip), 1.0))
        elif not nug:
            return lambda dxyz, rx, ry, rz, azi, dip, std: (std ** 2) * (
                    1.0 - _gaussian_corr(self._distance(dxyz[0, :], dxyz[1, :], dxyz[2, :], rx, ry, rz, azi, dip), 1.0))

    def _get_general_exponential_variogram_function(self, nug: bool = False) -> Callable:
        pwr = self.GEN_EXP_PWR
        if nug:
            return lambda dxyz, rx, ry, rz, azi, dip, std, sng: (sng ** 2) + (std ** 2) * (
                    1.0 - _general_exponential_corr(self._distance(dxyz[0, :], dxyz[1, :], dxyz[2, :], rx, ry, rz, azi, dip), 1.0, pwr))
        elif not nug:
            return lambda dxyz, rx, ry, rz, azi, dip, std: (std ** 2) * (
                    1.0 - _general_exponential_corr(self._distance(dxyz[0, :], dxyz[1, :], dxyz[2, :], rx, ry, rz, azi, dip), 1.0, pwr))

    def _get_spherical_variogram_function(self, nug: bool = False) -> Callable:
        if nug:
            return lambda dxyz, rx, ry, rz, azi, dip, std, sng: (sng ** 2) + (std ** 2) * (
                    1.0 - _spherical_corr(self._distance(dxyz[0, :], dxyz[1, :], dxyz[2, :], rx, ry, rz, azi, dip), 1.0))
        elif not nug:
            return lambda dxyz, rx, ry, rz, azi, dip, std: (std ** 2) * (
                    1.0 - _spherical_corr(self._distance(dxyz[0, :], dxyz[1, :], dxyz[2, :], rx, ry, rz, azi, dip), 1.0))
