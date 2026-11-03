# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Thermal free-free (bremsstrahlung) emission coefficients.

Exactly as in ModUserAwsom.f90 (Moschou et al. 2018, Eqs. 15-16), with a
constant Gaunt factor (default <g_ff> = 10, as in the paper):

    S     = 2 nu^2 k_B T_e / c^2                       (Rayleigh-Jeans)
    kappa = 4/3 sqrt(2 pi/3) [e^2/sqrt(k_B T_e m_e)]^3 (n_e/nu)^2 g_ff / c

``gaunt='dulk'`` instead uses the Coulomb-logarithm form of Dulk (1985,
ARA&A 23, 169, Eq. 21), more accurate than g = 10.
"""
import numpy as np

from ..common.constants import PI, K_B, M_E, C_LIGHT, E2_JM, R_SUN

_BREMS_PREF = 4.0 / 3.0 * np.sqrt(2.0 * PI / 3.0) * E2_JM**3 / C_LIGHT


def bremsstrahlung_coefficients(ne, te, freq, length_unit=R_SUN, gaunt=10.0):
    """(kappa [1/m], S [W m^-2 Hz^-1 sr^-1]) of thermal free-free emission.

    length_unit is accepted for a uniform interface and not used (SI units)."""
    ne = np.asarray(ne, dtype=float)
    te = np.maximum(np.asarray(te, dtype=float), 1.0)
    S = 2.0 * freq**2 * K_B * te / C_LIGHT**2
    if isinstance(gaunt, str):
        if gaunt.lower() != "dulk":
            raise ValueError("gaunt must be a number or 'dulk'")
        # Dulk (1985) Eq. 21, cgs:  kappa = 9.78e-3 n^2 / (nu^2 T^1.5) * G  [cm^-1]
        n_cc = ne * 1e-6
        G = np.where(te < 2e5,
                     18.2 + np.log(te**1.5) - np.log(freq),
                     24.5 + np.log(te) - np.log(freq))
        G = np.maximum(G, 1.0)
        kappa = 9.78e-3 * n_cc**2 / (freq**2 * te**1.5) * G * 100.0  # -> 1/m
    else:
        kappa = (_BREMS_PREF / (K_B * te * M_E)**1.5
                 * (ne / freq)**2 * float(gaunt))
    return kappa, S
