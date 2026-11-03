# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Spherically symmetric (1D) models.

A 1D model gives n_e(r) and T_e(r).  Assuming spherical symmetry it is turned
into an axisymmetric SphericalGrid with a single colatitude cell spanning 0..pi
(grid_from_profile, read_profile, and load_pluto for 1D spherical output).

Because the image of a symmetric model depends only on the distance p from
the disk centre, make_symmetric_image traces a single row of rays
(0 <= p <= p_max) and rotates the result into a 2D image; it also returns the
radial brightness profile and the flux density integrated exactly over the
profile, S = (2 pi / D^2) int I(p) p dp.  This is much cheaper than the full
2D calculation and exact for a symmetric model (also with refraction and for
an observer at a finite distance, since the problem is symmetric about the
observer-Sun axis).

1D support is provided for the mechanisms of the refractive ray tracer
(bremsstrahlung, simplistic).  Gyroresonance and gyrosynchrotron are not
offered for 1D models: a spherically symmetric field is a magnetic monopole and
the fast electrons of flares are not spherically symmetric, so such images
would not be meaningful.
"""
import math

import numpy as np

from .constants import K_B, M_P, R_SUN
from .grid import SphericalGrid


def grid_from_profile(r, ne, te=None, r_edges=None, r_body=None, length_unit=R_SUN,
                      dtype=np.float32):
    """SphericalGrid of a spherically symmetric model.

    r      : radii [coordinate units: R_sun, or R_star with length_unit]
    ne     : electron density [m^-3];  te : temperature [K] (default 1 K, enough
             for the simplistic mechanism only)
    r_edges: cell edges (default: midpoints between the radii)
    r_body : inner boundary (default: first edge)
    """
    r = np.asarray(r, float)
    if r.ndim != 1 or r.size < 2 or np.any(np.diff(r) <= 0):
        raise ValueError("r must be 1D and strictly increasing")
    ne = np.asarray(ne, float).reshape(-1, 1)
    te_a = None if te is None else np.asarray(te, float).reshape(-1, 1)
    g = SphericalGrid(r, np.array([0.5 * np.pi]), None, ne, te_a, r_edges=r_edges,
                      theta_edges=np.array([0.0, np.pi]), r_body=r_body, dtype=dtype,
                      length_unit=length_unit)
    g.symmetric = True
    return g


def read_profile(path, r_col=0, density_col=1, te_col=2, *, r_unit=1.0,
                 density="ne", density_unit="cm^-3", mu_e=1.0, te_unit=1.0, skiprows=0,
                 delimiter=None, **kw):
    """Read a 1D model from a text table (one radius per line) and return
    grid_from_profile(...).

    r_unit       : coordinate units per file unit (e.g. 1/6.96e10 for cm -> R_sun)
    density      : 'ne' (electron density) or 'rho' (mass density)
    density_unit : for 'ne': 'cm^-3' or 'm^-3'; for 'rho': 'g/cm^3' or 'kg/m^3'
    mu_e         : mass per electron in m_p for 'rho'
    te_unit      : kelvin per file unit;  te_col : None if absent
    """
    d = np.loadtxt(path, skiprows=skiprows, delimiter=delimiter, ndmin=2)
    r = d[:, r_col] * r_unit
    q = d[:, density_col]
    if density == "ne":
        ne = q * (1e6 if density_unit == "cm^-3" else 1.0)
    elif density == "rho":
        rho = q * (1e3 if density_unit == "g/cm^3" else 1.0)          # kg/m^3
        ne = rho / (mu_e * M_P)
    else:
        raise ValueError("density must be 'ne' or 'rho'")
    te = None if te_col is None else d[:, te_col] * te_unit
    order = np.argsort(r)
    return grid_from_profile(r[order], ne[order], None if te is None else te[order], **kw)


def is_symmetric(grid):
    """True for a grid built from a 1D profile (or any spherical grid with a
    single theta and phi cell)."""
    return bool(getattr(grid, "symmetric", False)) or (
        isinstance(grid, SphericalGrid) and grid.shape[1] == 1 and grid.shape[2] == 1)


# ----------------------------------------------------------------------------
def _rotate(row_p, row_vals, X, Y):
    """Values of a radial profile at the pixels of a 2D image (linear in p)."""
    rr = np.sqrt(X**2 + Y**2)
    out = np.interp(rr.ravel(), row_p, row_vals, right=0.0)
    return out.reshape(rr.shape)


def make_symmetric_image(grid, frequencies, mechanism="bremsstrahlung",
                         observer=(-215.0, 0.0, 0.0), x_size=None, y_size=None, nx=256,
                         ny=256, n_p=1000, p_max=None, verbose=True, **kw):
    """Image of a spherically symmetric model from one row of rays.

    mechanism : 'bremsstrahlung' or 'simplistic' (or an emission function)
    x_size, y_size, nx, ny : the 2D image produced by rotation (default size:
                2 p_max square)
    n_p, p_max : number of rays (default 1000; the limb, where the brightness
                 changes abruptly, needs fine sampling) and the largest distance
                 from the centre [coordinate units] (default: half of the image
                 diagonal, or the outer radius of the grid)
    Other keywords go to make_radio_image (transfer, gaunt, no_refraction,
    distance_pc, step_factor, ...), so refraction can be switched on or off.

    Returns a RadioImage on the (ny, nx) pixel grid, with
    img.profile = dict(p [coordinate units],
    intensity (nf, n_p+1), tb (nf, n_p+1), flux (nf,) [W m^-2 Hz^-1] within the
    circle p <= p_max (the image's own flux_density covers its square field of
    view)).
    """
    from .image import make_radio_image
    if not is_symmetric(grid):
        print("radiosynth: WARNING - make_symmetric_image assumes a spherically symmetric "
              "model; this grid is not built from a 1D profile")
    if p_max is None:
        if x_size is not None or y_size is not None:
            p_max = 0.5 * math.hypot(x_size or y_size, y_size or x_size)
        else:
            p_max = float(getattr(grid, "r_outer", 5.0))
    x_size = x_size if x_size is not None else 2.0 * p_max
    y_size = y_size if y_size is not None else x_size
    # one row through the centre: rays at p = k dp, k = -n_p..n_p (one exactly at the
    # centre), 1 pixel high
    dp = p_max / n_p
    row = dict(x_size=(2 * n_p + 1) * dp, y_size=dp, nx=2 * n_p + 1, ny=1, observer=observer)
    mech = mechanism.lower() if isinstance(mechanism, str) else mechanism
    # integrate along the whole line of sight through the model (the default of the
    # image functions is a sphere sized to the field of view)
    kw.setdefault("r_integration", max(float(getattr(grid, "r_outer", p_max)), p_max) + 1.0)
    if verbose:
        print(f"radiosynth: spherically symmetric model - {n_p + 1} rays (0 <= p <= {p_max:g}), "
              f"mechanism {mechanism if isinstance(mechanism, str) else 'user function'}")
    if callable(mech) or mech in ("bremsstrahlung", "simplistic"):
        im = make_radio_image(grid, frequencies, emission=mech, verbose=False, **row, **kw)
    else:
        raise ValueError("1D (spherically symmetric) images are available for 'bremsstrahlung' "
                         "and 'simplistic'; gyroresonance and gyrosynchrotron need a 2.5D or 3D "
                         "magnetic field")

    nf = len(im.freqs)
    half = slice(n_p, 2 * n_p + 1)                   # pixels with p >= 0
    p = np.abs(im.x[half])
    p[0] = 0.0
    # rotate every per-pixel array (nf, 1, 2 n_p) into (nf, ny, nx)
    xc = -0.5 * x_size + (np.arange(nx) + 0.5) * x_size / nx
    yc = -0.5 * y_size + (np.arange(ny) + 0.5) * y_size / ny
    X, Y = np.meshgrid(xc, yc)
    profile = {}
    for name, val in list(vars(im).items()):
        if not (isinstance(val, np.ndarray) and val.shape == (nf, 1, 2 * n_p + 1)):
            continue
        v = val[:, 0, half]
        if np.issubdtype(v.dtype, np.floating):
            new = np.stack([_rotate(p, v[k], X, Y) for k in range(nf)])
        else:                                         # status codes etc.: nearest ray
            idx = np.clip(np.rint(np.sqrt(X**2 + Y**2) / dp).astype(np.int64), 0, n_p)
            new = v[:, idx].astype(v.dtype)
        setattr(im, name, new)
        if name == "intensity":
            profile[name] = v
    im.x, im.y = xc, yc
    im.meta = dict(im.meta, x_size=x_size, y_size=y_size, nx=nx, ny=ny, symmetric=True,
                   n_p=n_p, p_max=p_max)
    # flux from the profile: S = 2 pi / D^2 int_0^p_max I(p) p dp  (all azimuths, i.e.
    # the whole disk out to p_max, not only the square field of view of the image)
    trap = getattr(np, "trapezoid", None) or np.trapz
    flux = 2.0 * math.pi * trap(profile["intensity"] * p, p, axis=1) / im.distance**2
    f2 = np.asarray(im.freqs, float)[:, None] ** 2
    from .constants import C_LIGHT
    profile.update(p=p, tb=profile["intensity"] * C_LIGHT**2 / (2.0 * K_B * f2), flux=flux)
    im.profile = profile
    if verbose:
        for k, f in enumerate(im.freqs):
            print(f"   {f / 1e6:10.2f} MHz: flux {flux[k]:.4e} W/m^2/Hz "
                  f"({flux[k] / 1e-22:.4g} sfu), T_b(centre) {profile['tb'][k, 0]:.4g} K")
    return im


def plot_profile(img, path=None, ax=None, units="rsun"):
    """T_b against the distance from the disk centre for every frequency
    (img from make_symmetric_image)."""
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .image import freq_label
    pr = img.profile
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
    else:
        fig = ax.figure
    for k, f in enumerate(img.freqs):
        ax.semilogy(pr["p"], np.maximum(pr["tb"][k], 1e-30), label=freq_label(f))
    ax.axvline(1.0, color="gray", ls="--", lw=0.7)
    ax.set_xlabel(r"distance from disk centre [$R$]")
    ax.set_ylabel(r"$T_b$ [K]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig
