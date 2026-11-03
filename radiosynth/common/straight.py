# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Straight-ray sampler for mechanisms that do not need refraction.

Gyroresonance emission comes from thin layers where B is large and the
frequency is far above the plasma frequency, so refraction is negligible and
the rays are straight lines (the standard approach of gyroresonance codes).
This module cuts every line of sight into equal steps and samples n_e, T_e and
B at the step nodes, chunk by chunk, so that a mechanism can do its radiative
transfer with vectorised numpy along the ray axis.

Each line starts where it enters both the integration sphere and the grid
(common/image.py start_points, as for the refractive tracer) and ends where it
leaves the domain or the integration sphere, or hits the body r = r_body.
Node 0 is the end nearest to the observer (front-to-back ordering).
"""
import numpy as np

from .raytrace import ACTIVE, ESCAPED, HIT_BODY, MISSED


def segment_ends(grid, pos, dirs, status0, r_integration):
    """Length of each straight line inside the domain (coordinate units) and
    whether it ends on the body.  pos: start points, dirs: unit directions."""
    # rays that miss the domain have non-finite start points; they get length 0
    pos = np.where(np.isfinite(pos), pos, 0.0)
    _, t_out = grid.domain_intersection(pos, dirs)
    b = np.einsum("ij,ij->i", pos, dirs)
    c = np.einsum("ij,ij->i", pos, pos)
    # far exit of the integration sphere
    disc = b * b - (c - r_integration**2)
    t_sph = -b + np.sqrt(np.maximum(disc, 0.0))
    t_end = np.minimum(t_out, t_sph)
    # first hit of the body sphere
    disc_b = b * b - (c - grid.r_body**2)
    t_body = -b - np.sqrt(np.maximum(disc_b, 0.0))
    hit = (disc_b > 0) & (t_body > 0) & (t_body < t_end)
    t_end = np.where(hit, t_body, t_end)
    t_end = np.where(status0 == ACTIVE, np.maximum(t_end, 0.0), 0.0)
    status = np.where(status0 == ACTIVE, np.where(hit, HIT_BODY, ESCAPED), MISSED).astype(np.int8)
    return t_end, status


def default_step(grid, length, ds=None, max_nodes=20000):
    """Step length: ds if given, else half the smallest cell, but at least
    length/max_nodes."""
    if ds is None:
        ds = 0.5 * float(grid.min_cell)
    return max(float(ds), float(np.max(length, initial=0.0)) / max_nodes)


def iter_samples(grid, pos, dirs, t_end, ds, max_points=400000, with_field=True):
    """Yield dict(idx, t, p, ne, te, inside, b, dl) for chunks of rays.

    t    : (nr, nk+1) node parameters along each line (coordinate units)
    p    : (nr, nk+1, 3) node positions
    ne, te, inside : (nr, nk+1) sampled density [m^-3], temperature [K], flag
    b    : (nr, nk+1, 3) magnetic field [G] (if with_field)
    dl   : (nr, nk) step lengths (coordinate units; 0 past the end of the line)
    """
    nray = dirs.shape[0]
    pos = np.where(np.isfinite(pos), pos, 0.0)
    nk_all = np.ceil(t_end / ds).astype(np.int64)
    order = np.argsort(nk_all)               # group rays of similar length
    start = 0
    while start < nray:
        nk = max(int(nk_all[order[start]]), 1)
        stop = min(nray, start + max(1, max_points // (nk + 1)))
        nk = max(int(nk_all[order[stop - 1]]), 1)       # longest ray of the chunk
        stop = start + max(1, min(stop - start, max_points // (nk + 1)))
        nr = stop - start
        idx = order[start:stop]
        k = np.arange(nk + 1)
        t = np.minimum(k[None, :] * ds, t_end[idx][:, None])
        p = pos[idx][:, None, :] + t[..., None] * dirs[idx][:, None, :]
        flat = p.reshape(-1, 3)
        ne, _, te, _, ins = grid.sample(flat)
        out = dict(idx=idx, t=t, p=p,
                   ne=np.where(ins, ne, 0.0).reshape(nr, nk + 1),
                   te=te.reshape(nr, nk + 1), inside=ins.reshape(nr, nk + 1),
                   dl=np.diff(t, axis=1))
        if with_field:
            b = grid.sample_field(flat)
            b[~ins] = 0.0
            out["b"] = b.reshape(nr, nk + 1, 3)
        yield out
        start = stop
