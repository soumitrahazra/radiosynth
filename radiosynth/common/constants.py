# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical constants (SI) used by radiosynth.

Values follow the BATS-R-US ModConst conventions where it matters
(solar radius 6.96e8 m) so that results can be compared one-to-one.
"""
import numpy as np
from scipy import constants as _c

PI = np.pi
K_B = _c.k                      # Boltzmann constant [J/K]
M_E = _c.m_e                    # electron mass [kg]
M_P = _c.m_p                    # proton mass [kg]
E_CHARGE = _c.e                 # elementary charge [C]
C_LIGHT = _c.c                  # speed of light [m/s]
EPS0 = _c.epsilon_0
# e^2/(4 pi eps0)  [J m]  == cElectronChargeSquaredJm in BATS-R-US
E2_JM = E_CHARGE**2 / (4.0 * PI * EPS0)
R_SUN = 6.96e8                  # BATS-R-US rSun [m]
AU = 1.495978707e11             # [m]
AU_RSUN = AU / R_SUN            # ~215 R_sun
JANSKY = 1.0e-26                # W m^-2 Hz^-1
SFU = 1.0e-22                   # solar flux unit


def critical_density(freq_hz):
    """Electron density [m^-3] at which the plasma frequency equals freq_hz.

    n_cr = pi m_e f^2 / (e^2/(4 pi eps0))   (identical to BATS-R-US NElectronSiCr)
    """
    return PI * M_E * np.asarray(freq_hz, dtype=float)**2 / E2_JM


def plasma_frequency(ne_si):
    """Electron plasma frequency [Hz] for density [m^-3]."""
    return np.sqrt(np.asarray(ne_si) * E2_JM / (PI * M_E))
