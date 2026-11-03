# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cold-plasma magneto-ionic theory shared by the polarized mechanisms.

Notation (all dimensionless):

    X = (nu_p / nu)^2 = n_e / n_cr          (n_cr: critical density)
    Y = nu_B / nu,      nu_B = e B / (2 pi m_e c) = 2.7992 MHz * B[G]
    cos(theta) = B . k / (|B| |k|),   k pointing towards the observer

The two natural (magneto-ionic) modes are the extraordinary (x) mode, whose
electric vector rotates in the same sense as the gyrating electrons, and the
ordinary (o) mode.  Appleton-Hartree refractive indices:

    n^2 = 1 - X (1-X) / [ (1-X) - Y^2 sin^2/2  -/+ sqrt(Y^4 sin^4/4 + (1-X)^2 Y^2 cos^2) ]

(upper sign: o-mode, lower sign: x-mode).  Cutoffs: o-mode X = 1, x-mode
X = 1 - Y (for Y < 1).

Mode polarisation: in the frame with B along z and k in the x-z plane the
transverse electric vector of mode sigma is e ~ (T_sigma cos(theta), i, -T_sigma
sin(theta)) / sqrt(1 + T_sigma^2) (longitudinal part neglected), with axial
ratio T_x T_o = -1 and

    |T_x| = 2 (1-X) |cos| / [ Y sin^2 + sqrt(Y^2 sin^4 + 4 (1-X)^2 cos^2) ]

so |T_x| = 1 (circular) for parallel propagation and 0 (linear, E perp B) for
perpendicular propagation.  Handedness: for B . k > 0 the x-mode is
right-hand circularly polarised in the IAU/IEEE radio convention (Stokes V > 0).
"""
import numpy as np

from .constants import E_CHARGE, M_E, PI, critical_density

#: electron gyrofrequency per gauss [Hz/G]:  e B / (2 pi m_e)  with B = 1e-4 T
NU_B_PER_GAUSS = E_CHARGE * 1e-4 / (2.0 * PI * M_E)        # 2.799249e6 Hz/G


def gyrofrequency(b_gauss):
    """Electron gyrofrequency [Hz] for |B| in gauss."""
    return NU_B_PER_GAUSS * np.asarray(b_gauss, dtype=float)


def magnetoionic_params(ne, b_gauss, freq):
    """X = n_e/n_cr and Y = nu_B/nu for density [m^-3], |B| [G] and frequency [Hz]."""
    X = np.asarray(ne, dtype=float) / critical_density(freq)
    Y = gyrofrequency(b_gauss) / freq
    return X, Y


def axial_ratio_x(X, Y, cos_theta):
    """|T_x|, the axial ratio of the x-mode (see module docstring);
    |T_o| = 1/|T_x|."""
    c = np.abs(np.asarray(cos_theta, dtype=float))
    s2 = np.maximum(1.0 - c * c, 0.0)
    a = np.maximum(1.0 - np.asarray(X, dtype=float), 1e-12)
    Y = np.asarray(Y, dtype=float)
    den = Y * s2 + np.sqrt(Y * Y * s2 * s2 + 4.0 * a * a * c * c)
    return np.where(den > 0, 2.0 * a * c / np.where(den > 0, den, 1.0), 1.0)


def refractive_index2(X, Y, cos_theta, mode):
    """Appleton-Hartree n^2 for mode 'o' or 'x' (cold, collisionless plasma)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    c2 = np.asarray(cos_theta, dtype=float) ** 2
    s2 = 1.0 - c2
    a = 1.0 - X
    root = np.sqrt(0.25 * Y**4 * s2 * s2 + a * a * Y * Y * c2)
    sgn = 1.0 if mode == "o" else -1.0
    den = a - 0.5 * Y * Y * s2 + sgn * root
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - X * a / den


def cutoff(X, Y, mode):
    """True where mode 'o' (X >= 1) or 'x' (X >= 1 - Y) cannot propagate."""
    X = np.asarray(X, dtype=float)
    if mode == "o":
        return X >= 1.0
    return X >= 1.0 - np.asarray(Y, dtype=float)
