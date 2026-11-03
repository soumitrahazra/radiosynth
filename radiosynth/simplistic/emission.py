# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""The 'simplistic' radio emission mechanism of Benkevitch et al. (2010).

An ad-hoc emissivity kept in BATS-R-US (ModRadioWaveImage, 'simplistic'):

    S = 1,   kappa * L = (n/n_cr)^2 (0.5 - n/n_cr)^2

with L the normalisation length (R_sun, or R_star for stars).  The intensity
is dimensionless; the emissivity peaks where n_e = n_cr/4 and vanishes at
n_e = n_cr/2, so the image shows the layer close to the critical surface.
It needs neither T_e nor B and is useful to visualise refraction.
"""
import numpy as np

from ..common.constants import R_SUN, critical_density


def simplistic_coefficients(ne, te, freq, length_unit=R_SUN):
    """(kappa [1/length], S=1) of the simplistic mechanism at n_e [m^-3]."""
    ne = np.asarray(ne, dtype=float)
    r = ne / critical_density(freq)
    # BATS-R-US defines it per normalised length (R_sun or R_star)
    kappa = r**2 * (0.5 - r)**2 / length_unit
    return kappa, np.ones_like(ne)
