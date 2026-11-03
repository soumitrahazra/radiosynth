# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analytic active-region model with a magnetic field, for gyroresonance.

A sub-photospheric dipole (horizontal by default, giving a bipolar region
with a positive and a negative spot) at disk centre as seen from an observer
on the -x axis, inside a gravitationally stratified isothermal corona:

    n_e(r) = n0 exp(-(r - 1)/H) + Newkirk background,   T_e = const,
    B      = dipole of moment m at depth d below the surface.

Used by tests/test_gyroresonance.py and examples/gyroresonance_example.py.
"""
import numpy as np

from radiosynth import MHDGrid, SphericalGrid
from synthetic_corona import newkirk


def dipole_field(X, Y, Z, b_max=2500.0, depth=0.05, moment=(0.0, 1.0, 0.0),
                 centre=(-1.0, 0.0, 0.0)):
    """Field [G] of a point dipole at distance ``depth`` below the surface point
    ``centre`` (unit vector times R_sun), scaled so that |B| on the surface
    straight above the dipole is b_max (for a horizontal moment: b_max/1)."""
    c = np.asarray(centre, float)
    c = c / np.linalg.norm(c)
    p0 = c * (1.0 - depth)
    m = np.asarray(moment, float)
    m = m / np.linalg.norm(m)
    rx, ry, rz = X - p0[0], Y - p0[1], Z - p0[2]
    R = np.sqrt(rx * rx + ry * ry + rz * rz)
    R = np.maximum(R, 1e-3)
    mr = (m[0] * rx + m[1] * ry + m[2] * rz) / R
    # |B| straight above the dipole: horizontal m -> k/d^3, radial m -> 2k/d^3
    k = b_max * depth**3 / (1.0 if abs(np.dot(m, c)) < 0.5 else 2.0)
    f = k / R**3
    return (f * (3 * mr * rx / R - m[0]), f * (3 * mr * ry / R - m[1]),
            f * (3 * mr * rz / R - m[2]))


def ar_fields(X, Y, Z, n0=2e15, H=0.05, te=2.5e6, **dip):
    r = np.sqrt(X * X + Y * Y + Z * Z)
    ne = n0 * np.exp(-np.maximum(r - 1.0, 0.0) / H) + newkirk(np.maximum(r, 1.0))
    T = np.full_like(ne, te)
    bx, by, bz = dipole_field(X, Y, Z, **dip)
    return ne, T, (bx, by, bz)


def make_ar_model(n=100, half=0.25, x_range=(-1.25, -0.75), **kw):
    """Cartesian box above disk centre (observer on the -x axis)."""
    x = np.linspace(*x_range, n)
    y = np.linspace(-half, half, n)
    z = np.linspace(-half, half, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    ne, te, b = ar_fields(X, Y, Z, **kw)
    g = MHDGrid(x, y, z, ne, te, r_body=1.0)
    g.add_field(*b)
    return g


def make_ar_spherical(nr=120, nth=90, nph=90, r_max=1.3, half_deg=16.0, **kw):
    """The same model on a native spherical (r, theta, phi) patch around the
    -x axis (theta = 90 deg, phi = 180 deg), field given in (B_r, B_theta, B_phi)."""
    r_e = np.linspace(1.0, r_max, nr + 1)
    t_e = np.radians(np.linspace(90 - half_deg, 90 + half_deg, nth + 1))
    p_e = np.radians(np.linspace(180 - half_deg, 180 + half_deg, nph + 1))
    rc, tc, pc = (0.5 * (e[1:] + e[:-1]) for e in (r_e, t_e, p_e))
    R, TH, PH = np.meshgrid(rc, tc, pc, indexing="ij")
    X, Y, Z = R * np.sin(TH) * np.cos(PH), R * np.sin(TH) * np.sin(PH), R * np.cos(TH)
    ne, te, (bx, by, bz) = ar_fields(X, Y, Z, **kw)
    st, ct, sp, cp = np.sin(TH), np.cos(TH), np.sin(PH), np.cos(PH)
    br = bx * st * cp + by * st * sp + bz * ct
    bt = bx * ct * cp + by * ct * sp - bz * st
    bp = -bx * sp + by * cp
    g = SphericalGrid(rc, tc, pc, ne, te, r_edges=r_e, theta_edges=t_e, phi_edges=p_e,
                      r_body=1.0)
    g.add_field(br, bt, bp, components="spherical")
    return g, (r_e, t_e, p_e), (ne, te, br, bt, bp)
