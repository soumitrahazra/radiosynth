# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analytic reference for thermal free-free emission from a stellar wind.

Isothermal, fully ionised spherical wind  n = n0 (r0/r)^2  with a constant
Gaunt factor (Wright & Barlow 1975; Panagia & Felli 1975).  Used to validate
the ray tracer (tests/test_stellar.py) and as a quick estimate of the radio
flux of a wind.
"""
import math

import numpy as np

from ..common.constants import C_LIGHT, K_B
from ..common.emission import EmissionModel


# ----------------------------------------------------------------------------
# analytic reference: isothermal, fully ionised spherical wind n = n0 (r0/r)^2
# ----------------------------------------------------------------------------
def wind_optical_depth(p, n0, r0_m, te, freq, gaunt=10.0, r_body_m=None):
    """Free-free optical depth along a straight line at impact parameter p [m]
    through n = n0 (r0/r)^2 (constant T, constant Gaunt factor):
        tau(p) = kappa1 n0^2 r0^4 * pi / (2 p^3)       (infinite line, p > r_body)
    kappa1 = kappa / n_e^2.  For p < r_body the line stops at the stellar
    surface (front half only), computed numerically."""
    k1 = EmissionModel("bremsstrahlung", gaunt=gaunt)(np.array([1.0]), np.array([te]), freq)[0][0]
    A = k1 * n0**2 * r0_m**4
    p = np.atleast_1d(np.asarray(p, float))
    tau = A * math.pi / (2.0 * p**3)
    if r_body_m is not None:
        inside = p < r_body_m
        if np.any(inside):
            # integral of dl/(p^2+l^2)^2 from l0=sqrt(rb^2-p^2) to infinity
            pp = p[inside]
            l0 = np.sqrt(r_body_m**2 - pp**2)
            F = lambda l, q: l / (2 * q**2 * (q**2 + l**2)) + np.arctan(l / q) / (2 * q**3)  # noqa: E731
            tau[inside] = A * (math.pi / (4 * pp**3) - F(l0, pp))
    return tau


def wind_flux_density(n0, r0_m, te, freq, distance_m, gaunt=10.0):
    """Analytic flux density [W m^-2 Hz^-1] of an infinite isothermal wind
    n = n0 (r0/r)^2 with constant Gaunt factor, neglecting the stellar disk
    (valid when the tau = 1 radius is >> r0):
        S = pi Gamma(1/3) B_nu a^(2/3) / D^2,   tau(p) = a / p^3
    which gives the classic spectrum S ~ nu^(2/3) (nu^0.6 when the Gaunt factor's
    weak frequency dependence is included; Wright & Barlow 1975, Panagia & Felli 1975)."""
    k1 = EmissionModel("bremsstrahlung", gaunt=gaunt)(np.array([1.0]), np.array([te]), freq)[0][0]
    a = k1 * n0**2 * r0_m**4 * math.pi / 2.0
    B = 2.0 * freq**2 * K_B * te / C_LIGHT**2
    return math.pi * math.gamma(1.0 / 3.0) * B * a**(2.0 / 3.0) / distance_m**2
