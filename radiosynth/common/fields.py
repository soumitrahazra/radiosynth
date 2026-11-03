# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Derived fields on the model grid.

current_density(grid) computes |J| = |curl B| / mu0 on the grid from the
magnetic field attached with add_field(), and stores it as the scalar field
'j' [A m^-2] (grid.sample_scalar('j', points)).  It is a common proxy for the
sites of magnetic energy release (current sheets, reconnection regions), e.g.
to place fast electrons for gyrosynchrotron emission.
"""
import numpy as np

MU0 = 4e-7 * np.pi


def _grad(a, x, axis):
    if a.shape[axis] < 2:
        return np.zeros_like(a)
    return np.gradient(a, x, axis=axis, edge_order=2 if a.shape[axis] > 2 else 1)


def curl_b(grid):
    """curl B on the grid nodes, in gauss per coordinate unit.

    Cartesian grids: (curl B)_x,y,z.  Spherical grids: components (r, theta,
    phi) of curl B computed with the spherical formulas from the stored
    (B_r, B_theta, B_phi) (or from Cartesian components converted to them)."""
    if not getattr(grid, "has_field", False):
        raise ValueError("the grid has no magnetic field")
    if hasattr(grid, "x"):                                   # MHDGrid
        b = grid._bflat.reshape(grid.shape + (3,)).astype(float)
        bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
        x, y, z = grid.x, grid.y, grid.z
        cx = _grad(bz, y, 1) - _grad(by, z, 2)
        cy = _grad(bx, z, 2) - _grad(bz, x, 0)
        cz = _grad(by, x, 0) - _grad(bx, y, 1)
        return cx, cy, cz
    # SphericalGrid
    b = grid._bflat.reshape(grid.shape + (3,)).astype(float)
    R = grid.r[:, None, None]
    TH = grid.theta[None, :, None]
    PH = grid.phi[None, None, :]
    if grid._bcomp == "cartesian":
        st, ct, sp, cp = np.sin(TH), np.cos(TH), np.sin(PH), np.cos(PH)
        bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
        br = bx * st * cp + by * st * sp + bz * ct
        bt = bx * ct * cp + by * ct * sp - bz * st
        bp = -bx * sp + by * cp
    else:
        br, bt, bp = b[..., 0], b[..., 1], b[..., 2]
    S = np.maximum(np.sin(TH), 1e-6)
    phi = grid.phi
    c_r = (_grad(S * bp, grid.theta, 1) - _grad(bt, phi, 2)) / (R * S)
    c_t = (_grad(br, phi, 2) / S - _grad(R * bp, grid.r, 0)) / R
    c_p = (_grad(R * bt, grid.r, 0) - _grad(br, grid.theta, 1)) / R
    return c_r, c_t, c_p


def current_density(grid, store=True):
    """|J| [A m^-2] on the grid; stored as grid scalar 'j' when store=True."""
    c = curl_b(grid)
    mag = np.sqrt(sum(ci * ci for ci in c))                 # gauss per coordinate unit
    j = mag * 1e-4 / (grid.length_unit * MU0)                # T/m / mu0 -> A/m^2
    if store:
        grid.add_scalar("j", j)
    return j
