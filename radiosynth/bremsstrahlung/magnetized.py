# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Free-free absorption of the two magneto-ionic modes in a magnetic field.

In a magnetised plasma the free-free absorption coefficient differs for the
ordinary (o) and extraordinary (x) modes.  For Y = nu_B/nu < 1 and away from
quasi-transverse propagation the standard approximation (e.g. Zheleznyakov
1970; Dulk 1985; Gelfreikh 2004) is

    kappa_sigma = kappa_ff / (1 + sigma Y |cos(theta)|)^2,
    sigma = +1 (o-mode), -1 (x-mode)

with kappa_ff the unmagnetised coefficient (bremsstrahlung/emission.py) and
source function S_sigma = B_nu/2 = nu^2 k_B T_e / c^2 for each mode
(Kirchhoff's law per mode, so that I_o + I_x = B_nu in the thick limit).
In an optically thin source this gives the circular polarisation degree
P = (I_x - I_o)/(I_x + I_o) = 2 Y |cos(theta)| (for Y << 1); in an optically
thick isothermal source P -> 0.  Where X >= 1 - Y the x-mode is cut off.
"""
import numpy as np

from ..common.constants import C_LIGHT, K_B, R_SUN
from ..common.magnetoionic import magnetoionic_params
from .emission import bremsstrahlung_coefficients

__all__ = ["magnetized_ff_coefficients", "magnetoionic_params"]


def magnetized_ff_coefficients(ne, te, b_gauss, cos_theta, freq, gaunt=10.0,
                               length_unit=R_SUN, y_max=0.9):
    """Return (kappa_o, kappa_x, S_mode) [1/m, 1/m, W m^-2 Hz^-1 sr^-1].

    b_gauss   : |B| [G];  cos_theta : cosine of the angle between B and the
                line of sight; y_max caps Y |cos| in the denominator (the
                approximation fails as Y -> 1, i.e. near the s=1 layer).
    """
    kappa, S = bremsstrahlung_coefficients(ne, te, freq, length_unit=length_unit, gaunt=gaunt)
    _, Y = magnetoionic_params(ne, b_gauss, freq)
    q = np.minimum(Y * np.abs(np.asarray(cos_theta, dtype=float)), y_max)
    kappa_o = kappa / (1.0 + q) ** 2
    kappa_x = kappa / (1.0 - q) ** 2
    S_mode = 0.5 * S
    return kappa_o, kappa_x, S_mode


def rayleigh_jeans_mode(te, freq):
    """Source function of one mode, nu^2 k T / c^2 [W m^-2 Hz^-1 sr^-1]."""
    return np.asarray(freq, dtype=float) ** 2 * K_B * np.asarray(te, dtype=float) / C_LIGHT**2
