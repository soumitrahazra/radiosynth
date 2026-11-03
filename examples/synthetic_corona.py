# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analytic test corona used for validation and as a usage example.

Newkirk (1961) density  n_e(r) = f * 4.2e4 * 10^(4.32/r) cm^-3, isothermal
T_e, optionally with a dense 'CME' blob and an equatorial streamer.
"""
import numpy as np

from radiosynth import MHDGrid


def newkirk(r, fold=1.0):
    """Electron density [m^-3]."""
    return fold * 4.2e4 * 10.0**(4.32 / r) * 1e6


def model_fields(X, Y, Z, te=1.5e6, fold=1.0, blob=None, streamer=False, r_body=1.0):
    """n_e [m^-3] and T_e [K] of the analytic corona at Cartesian points [R_sun]."""
    R = np.sqrt(X**2 + Y**2 + Z**2)
    ne = newkirk(np.maximum(R, r_body * 0.98), fold)
    if streamer:
        lat = np.arcsin(np.clip(Z / np.maximum(R, 1e-6), -1, 1))
        ne = ne * (1.0 + 2.0 * np.exp(-(lat / 0.15)**2))
    if blob is not None:
        c, w, amp = blob
        d2 = (X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2
        ne = ne * (1.0 + amp * np.exp(-d2 / w**2))
    return ne, np.full_like(ne, te)


def make_model(half=3.0, n=161, te=1.5e6, fold=1.0, blob=None, streamer=False, r_body=1.0):
    x = np.linspace(-half, half, n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    ne, T = model_fields(X, Y, Z, te, fold, blob, streamer, r_body)
    return MHDGrid(x, x, x, ne, T, r_body=r_body)


def radial_model(fold=1.0, te=1.5e6):
    """Analytic n_e(r), T_e for exact reference integrals."""
    return (lambda r: newkirk(r, fold)), (lambda r: te)


# active regions on the visible (x<0) hemisphere: (lat, lon offset from disk centre) deg
ARS = [(18.0, -25.0, 6.0), (-12.0, 30.0, 4.0), (25.0, 55.0, 5.0)]


def _unit(lat, dlon):
    """Unit vector at heliographic lat and longitude offset dlon (west +) from
    the disk centre seen by an observer on the -x axis (west = -y, north = +z)."""
    la, lo = np.radians(lat), np.radians(dlon)
    return np.array([-np.cos(la) * np.cos(lo), -np.cos(la) * np.sin(lo), np.sin(la)])


def structured_model(t_min=None, half=3.0, n=201, te0=1.4e6, ar_shift=0.0, ar_amp=1.0):
    """Electron density / temperature grid at time t_min [min] after CME onset
    (t_min=None: no CME).  ar_shift/ar_amp perturb the active regions (used to
    make a mock 'observation' that is similar but not identical)."""
    x = np.linspace(-half, half, n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    Rc = np.maximum(R, 0.98)
    ne = newkirk(Rc)
    lat = np.arcsin(np.clip(Z / np.maximum(R, 1e-6), -1, 1))
    ne *= 1.0 + 2.0 * np.exp(-(lat / 0.15)**2)            # equatorial streamer
    te = np.full_like(ne, te0)
    for la, lo, amp in ARS:                                # hot dense active regions
        c = 1.06 * _unit(la + ar_shift, lo + ar_shift)
        d2 = (X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2
        w = np.exp(-d2 / 0.07**2)
        ne *= 1.0 + ar_amp * amp * w
        te += 1.6e6 * w
    if t_min is not None:                                  # three-part CME
        u = _unit(25.0, 75.0)                              # NW, close to the limb
        rc = 1.45 + 0.03 * t_min                           # centre, ~350 km/s
        a_ = 0.30 + 0.012 * t_min                          # radius of the bubble
        c = rc * u
        dist = np.sqrt((X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2)
        dil = (1.45 / rc)**2                               # dilution as it expands
        front = 6.0 * dil * np.exp(-((dist - a_) / (0.07 + 0.15 * a_))**2)
        cavity = -0.6 * np.exp(-(dist / (0.75 * a_))**4)   # depleted cavity
        core = 10.0 * dil * np.exp(-(dist / (0.30 * a_))**2)
        ne *= np.maximum(1.0 + front + cavity + core, 0.05)
        # dense legs (sheath) trailing back towards the Sun
        s = X * u[0] + Y * u[1] + Z * u[2]
        perp2 = np.maximum(R**2 - s**2, 0.0)
        leg = np.exp(-perp2 / (0.10 * rc)**2) * (s > 1.0) * (s < rc - 0.5 * a_)
        ne *= 1.0 + 3.0 * dil * leg
    return MHDGrid(x, x, x, ne, te, r_body=1.0)
