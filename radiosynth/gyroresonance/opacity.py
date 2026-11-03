# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Thermal gyroresonance opacity of a harmonic layer.

A thermal electron gyrating in a field B radiates and absorbs at the harmonics
s nu_B of the gyrofrequency (nu_B = 2.80 MHz B[G]).  For a Maxwellian plasma
the absorption at frequency nu is confined to thin layers where nu = s nu_B,
broadened only by the Doppler width  ~ nu beta_T |cos(theta)|.  Integrating
the absorption coefficient across such a layer gives the optical depth
(Zheleznyakov 1970; Melrose 1980; White & Kundu 1997; Fleishman & Kuznetsov
2014, ApJ 781, 77, Eq. for tau_s^M):

    tau_s^sigma = (pi e^2 n_e / (m_e c nu)) L_B (s^2 / s!)
                  * (s^2 beta_T^2 sin^2(theta) / 2)^(s-1) * F_sigma(theta)

    beta_T^2 = k_B T_e / (m_e c^2),     L_B = |B / (dB/dl)|  along the ray,
    F_sigma  = (1 + T_sigma cos(theta))^2 / (1 + T_sigma^2)

with T_sigma the axial ratio of mode sigma (common/magnetoionic.py), signed so
that T_x cos(theta) > 0 and T_o = -1/T_x.  This is the small-Larmor-radius
(lowest order in beta_T) result: the x-mode, which co-rotates with the
electrons, has F_x -> 2 along B, and the o-mode F_o -> 0; at perpendicular
propagation F_x = 1 and F_o vanishes at this order (the o-mode opacity is then
of higher order in beta_T and is underestimated).  The refractive index is
set to 1 (valid for X = (nu_p/nu)^2 << 1, normally true where gyroresonance
emission forms) and the longitudinal part of the mode polarisation is
neglected.

Kirchhoff's law holds per mode: each layer emits the source function
S = nu^2 k_B T_e / c^2 (half the Rayleigh-Jeans intensity), so an optically
thick layer has T_b = T_e in that mode.

In SI the prefactor pi e^2/(m_e c) is pi (e^2/4 pi eps0)/(m_e c) = 2.654e-6 m^2/s.
"""
import math

import numpy as np

from ..common.constants import C_LIGHT, E2_JM, K_B, M_E, PI
from ..common.magnetoionic import axial_ratio_x

#: pi e^2 / (m_e c) in SI [m^2 s^-1]
GR_PREFACTOR = PI * E2_JM / (M_E * C_LIGHT)


def mode_factor(X, Y, cos_theta, mode):
    """F_sigma = (1 + T_sigma cos)^2/(1 + T_sigma^2) for mode 'x' or 'o'."""
    c = np.abs(np.asarray(cos_theta, dtype=float))
    tx = axial_ratio_x(X, Y, c)
    if mode == "x":
        return (1.0 + tx * c) ** 2 / (1.0 + tx * tx)
    if mode == "o":
        return (tx - c) ** 2 / (1.0 + tx * tx)
    raise ValueError("mode must be 'x' or 'o'")


def layer_optical_depth(s, ne, te, cos_theta, l_b, freq, mode, X=None):
    """Optical depth of the s-th harmonic layer.

    s         : harmonic number (int >= 1)
    ne        : electron density at the layer [m^-3]
    te        : electron temperature [K]
    cos_theta : cosine of the angle between B and the line of sight
    l_b       : magnetic scale length |B/(dB/dl)| along the ray [m]
    freq      : observing frequency [Hz]
    mode      : 'x' or 'o'
    X         : (nu_p/nu)^2 at the layer (for the mode polarisation; default 0)
    """
    s = int(s)
    ne = np.asarray(ne, dtype=float)
    te = np.asarray(te, dtype=float)
    c = np.asarray(cos_theta, dtype=float)
    beta2 = K_B * te / (M_E * C_LIGHT**2)
    sin2 = np.maximum(1.0 - c * c, 0.0)
    X = np.zeros_like(ne) if X is None else np.asarray(X, dtype=float)
    F = mode_factor(X, 1.0 / s, c, mode)
    return (GR_PREFACTOR * ne * np.asarray(l_b, dtype=float) / freq
            * (s * s / math.factorial(s)) * (0.5 * s * s * beta2 * sin2) ** (s - 1) * F)


def harmonic_field(s, freq):
    """|B| [G] of the s-th harmonic layer at frequency freq [Hz]."""
    from ..common.magnetoionic import NU_B_PER_GAUSS
    return freq / (s * NU_B_PER_GAUSS)
