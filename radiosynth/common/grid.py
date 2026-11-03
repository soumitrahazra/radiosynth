# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cartesian (rectilinear) MHD grid with fast vectorised trilinear sampling.

This replaces the BATS-R-US machinery used by ModRadioWaveImage
(``interpolate_grid_amr_gc``, ``calc_gradient_ghost``, ``CellSize_DB``):

* electron density n_e [m^-3] and its gradient [m^-3 per R_sun] are stored
  on the grid and trilinearly interpolated to the ray position, exactly as
  BATS-R-US interpolates State_VGB and GradVar_DGB;
* the recommended step ``DeltaSNew`` is the local minimum cell size.

Coordinates are in units of R_sun (as in BATS-R-US).  The axes may be
non-uniform (stretched) but must be monotonically increasing.
"""
import numpy as np

from .constants import R_SUN


class MHDGrid:
    """Electron density and temperature on a rectilinear 3D grid.

    Parameters
    ----------
    x, y, z : 1D arrays, strictly increasing, in R_sun
    ne      : (nx, ny, nz) electron density [m^-3]
    te      : (nx, ny, nz) electron temperature [K]  (may be None for the
              'simplistic' emission model)
    r_body  : radius of the inner boundary (photosphere) in R_sun; rays that
              reach it are stopped (IsBody in BATS-R-US).
    dtype   : storage precision of the interpolated fields (float32 halves
              memory; arithmetic is always done in float64).
    length_unit : metres per coordinate unit (default R_sun; for a star with
              coordinates in stellar radii use r_star * R_SUN)
    """

    NFIELD = 5   # ne, dne/dx, dne/dy, dne/dz, Te

    def __init__(self, x, y, z, ne, te=None, r_body=1.0, dtype=np.float32, length_unit=R_SUN):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        for a, name in ((self.x, "x"), (self.y, "y"), (self.z, "z")):
            if a.ndim != 1 or a.size < 2 or np.any(np.diff(a) <= 0):
                raise ValueError(f"axis {name} must be 1D, strictly increasing, size>=2")
        shape = (self.x.size, self.y.size, self.z.size)
        ne = np.asarray(ne, dtype=float)
        if ne.shape != shape:
            raise ValueError(f"ne has shape {ne.shape}, expected {shape}")
        ne = np.where(np.isfinite(ne), np.maximum(ne, 0.0), 0.0)
        if te is None:
            te = np.ones(shape)
        te = np.asarray(te, dtype=float)
        if te.shape != shape:
            raise ValueError(f"te has shape {te.shape}, expected {shape}")
        te = np.where(np.isfinite(te), te, 1.0)

        self.shape = shape
        self.r_body = float(r_body)
        self.length_unit = float(length_unit)
        # density gradient per R_sun (2nd order central, like BATS-R-US)
        gx, gy, gz = np.gradient(ne, self.x, self.y, self.z, edge_order=2)
        f = np.empty(shape + (self.NFIELD,), dtype=dtype)
        f[..., 0] = ne
        f[..., 1] = gx
        f[..., 2] = gy
        f[..., 3] = gz
        f[..., 4] = te
        self._flat = f.reshape(-1, self.NFIELD)
        del f, gx, gy, gz
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.dz = np.diff(self.z)
        self.lo = np.array([self.x[0], self.y[0], self.z[0]])
        self.hi = np.array([self.x[-1], self.y[-1], self.z[-1]])
        self._uniform = all(np.allclose(d, d[0], rtol=1e-6) for d in (self.dx, self.dy, self.dz))
        self._strides = np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)
        offs = []
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    offs.append(di * self._strides[0] + dj * self._strides[1] + dk)
        self._corner_off = np.array(offs, dtype=np.int64)

    # ------------------------------------------------------------------
    @property
    def nbytes(self):
        return self._flat.nbytes

    def __repr__(self):
        return (f"MHDGrid(shape={self.shape}, x=[{self.x[0]:.3g},{self.x[-1]:.3g}], "
                f"y=[{self.y[0]:.3g},{self.y[-1]:.3g}], z=[{self.z[0]:.3g},{self.z[-1]:.3g}], "
                f"min cell={self.min_cell:.3g} R_sun, r_body={self.r_body})")

    @property
    def min_cell(self):
        return float(min(self.dx.min(), self.dy.min(), self.dz.min()))

    def inside(self, p):
        """Boolean mask: points (N,3) lie within the grid box."""
        return np.all((p >= self.lo) & (p <= self.hi), axis=1)

    def _locate(self, axis, d, u):
        if self._uniform:
            i = np.floor((u - axis[0]) / d[0]).astype(np.int64)
        else:
            i = np.searchsorted(axis, u, side="right") - 1
        i = np.clip(i, 0, axis.size - 2)
        t = (u - axis[i]) / d[i]
        return i, np.clip(t, 0.0, 1.0)

    def sample(self, p):
        """Trilinear interpolation at points p (N,3).

        Returns
        -------
        ne   : (N,)   electron density [m^-3]
        grad : (N,3)  density gradient [m^-3 / R_sun]
        te   : (N,)   electron temperature [K]
        cell : (N,)   local minimum cell size [R_sun]  (BATS-R-US DeltaSNew)
        inside : (N,) bool, point within the grid box
        """
        p = np.asarray(p, dtype=float)
        ins = self.inside(p)
        i, tx = self._locate(self.x, self.dx, p[:, 0])
        j, ty = self._locate(self.y, self.dy, p[:, 1])
        k, tz = self._locate(self.z, self.dz, p[:, 2])
        base = i * self._strides[0] + j * self._strides[1] + k
        out = np.zeros((p.shape[0], self.NFIELD))
        wx = (1 - tx, tx)
        wy = (1 - ty, ty)
        wz = (1 - tz, tz)
        c = 0
        for di in (0, 1):
            for dj in (0, 1):
                wxy = wx[di] * wy[dj]
                for dk in (0, 1):
                    w = wxy * wz[dk]
                    out += w[:, None] * self._flat[base + self._corner_off[c]]
                    c += 1
        cell = np.minimum(np.minimum(self.dx[i], self.dy[j]), self.dz[k])
        return np.maximum(out[:, 0], 0.0), out[:, 1:4], out[:, 4], cell, ins

    # ---- magnetic field (gyroresonance, polarisation) -------------------
    @property
    def has_field(self):
        return getattr(self, "_bflat", None) is not None

    def add_field(self, bx, by, bz, dtype=np.float32):
        """Attach the magnetic field (Cartesian components, gauss) on the grid nodes."""
        b = np.stack([np.asarray(c, dtype=float) for c in (bx, by, bz)], axis=-1)
        if b.shape[:3] != self.shape:
            raise ValueError(f"B has shape {b.shape[:3]}, expected {self.shape}")
        b = np.where(np.isfinite(b), b, 0.0)
        self._bflat = b.astype(dtype).reshape(-1, 3)
        return self

    def _interp_flat(self, flat, p):
        """Trilinear interpolation of a (N_nodes, k) array at points p (N,3)."""
        p = np.asarray(p, dtype=float)
        i, tx = self._locate(self.x, self.dx, p[:, 0])
        j, ty = self._locate(self.y, self.dy, p[:, 1])
        k, tz = self._locate(self.z, self.dz, p[:, 2])
        base = i * self._strides[0] + j * self._strides[1] + k
        out = np.zeros((p.shape[0], flat.shape[1]))
        wx, wy, wz = (1 - tx, tx), (1 - ty, ty), (1 - tz, tz)
        c = 0
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    out += (wx[di] * wy[dj] * wz[dk])[:, None] * flat[base + self._corner_off[c]]
                    c += 1
        return out

    def sample_field(self, p):
        """Trilinear interpolation of B [G] at points p (N,3) -> (N,3) Cartesian."""
        if not self.has_field:
            raise ValueError("this grid has no magnetic field: use add_field() or "
                             "load_mhd(..., with_b=True)")
        return self._interp_flat(self._bflat, p)

    # ---- extra scalar fields (e.g. current density, fast-electron density) --
    def add_scalar(self, name, values, dtype=np.float32):
        """Attach a named scalar field on the grid nodes (any quantity, e.g. 'j')."""
        v = np.asarray(values, dtype=float)
        if v.shape != self.shape:
            raise ValueError(f"{name} has shape {v.shape}, expected {self.shape}")
        if not hasattr(self, "scalars"):
            self.scalars = {}
        self.scalars[name] = np.where(np.isfinite(v), v, 0.0).astype(dtype).reshape(-1, 1)
        return self

    def sample_scalar(self, name, p):
        """Trilinear interpolation of the scalar field ``name`` at points p (N,3)."""
        return self._interp_flat(self.scalars[name], p)[:, 0]

    def box_intersection(self, origin, direction):
        """Slab-method ray/box intersection.  Returns (t_enter, t_exit) arrays;
        a ray misses the box when t_exit < max(t_enter, 0)."""
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        o = np.broadcast_to(o, d.shape)
        par = np.abs(d) < 1e-300
        dsafe = np.where(par, 1.0, d)
        t1 = (self.lo - o) / dsafe
        t2 = (self.hi - o) / dsafe
        tmin = np.minimum(t1, t2)
        tmax = np.maximum(t1, t2)
        # ray parallel to a slab: either always inside it or never
        in_slab = (o >= self.lo) & (o <= self.hi)
        tmin = np.where(par, np.where(in_slab, -np.inf, np.inf), tmin)
        tmax = np.where(par, np.where(in_slab, np.inf, -np.inf), tmax)
        return tmin.max(axis=1), tmax.min(axis=1)

    # common API -------------------------------------------------------
    domain_intersection = box_intersection

    def to_npz(self, path):
        """Save n_e, T_e (and B) (reload with radiosynth.load_mhd(path))."""
        extra = {}
        if self.has_field:
            extra["b"] = self._bflat.reshape(self.shape + (3,))
        np.savez(path, geometry="cartesian", x=self.x, y=self.y, z=self.z,
                 ne=self._flat[:, 0].reshape(self.shape),
                 te=self._flat[:, 4].reshape(self.shape), r_body=self.r_body,
                 length_unit=self.length_unit, **extra)

    @property
    def r_outer(self):
        """Largest distance from the origin inside the domain (corner of the box)."""
        return float(np.linalg.norm(np.maximum(np.abs(self.lo), np.abs(self.hi))))


# ============================================================================
class SphericalGrid:
    """Electron density and temperature on a spherical (r, theta, phi) grid.

    Native sampler for spherical MHD output (PLUTO, BATS-R-US 'sph', ...):
    no resampling to a Cartesian box, so the fine radial resolution near the
    Sun is preserved.

    Parameters
    ----------
    r      : 1D cell-centre radii [R_sun], increasing
    theta  : 1D cell-centre colatitudes [rad] in [0, pi], increasing
    phi    : 1D cell-centre longitudes [rad], increasing; a single value
             (or None) means an axisymmetric 2.5D model (fields independent of phi)
    ne, te : (nr, ntheta, nphi) electron density [m^-3] and temperature [K]
    r_edges, theta_edges, phi_edges : optional cell-edge arrays used for the
             domain extent (default: half a cell beyond the centres)
    r_body : inner boundary [R_sun] where rays stop (default inner r edge)
    length_unit : metres per coordinate unit (default R_sun)

    Cartesian convention: x = r sin(theta) cos(phi), y = r sin(theta) sin(phi),
    z = r cos(theta)   (PLUTO / BATS-R-US convention).
    """

    NFIELD = 5   # ne, dn/dr, (1/r) dn/dtheta, 1/(r sin) dn/dphi, Te

    def __init__(self, r, theta, phi, ne, te=None, r_edges=None, theta_edges=None,
                 phi_edges=None, r_body=None, dtype=np.float32, length_unit=R_SUN):
        self.r = np.asarray(r, float)
        self.theta = np.asarray(theta, float)
        self.axisym = phi is None or np.size(phi) == 1
        self.phi = np.zeros(1) if phi is None else np.atleast_1d(np.asarray(phi, float))
        nr, nt, nph = self.r.size, self.theta.size, self.phi.size
        ne = np.asarray(ne, float).reshape(nr, nt, nph)
        ne = np.where(np.isfinite(ne), np.maximum(ne, 0.0), 0.0)
        te = np.ones_like(ne) if te is None else np.asarray(te, float).reshape(nr, nt, nph)
        te = np.where(np.isfinite(te), te, 1.0)
        self.shape = (nr, nt, nph)

        def edges(c, e):
            if e is not None:
                return np.asarray(e, float)
            if c.size == 1:
                return np.array([c[0], c[0]])
            m = 0.5 * (c[1:] + c[:-1])
            return np.concatenate([[c[0] - (m[0] - c[0])], m, [c[-1] + (c[-1] - m[-1])]])

        self.r_edges = edges(self.r, r_edges)
        self.theta_edges = np.clip(edges(self.theta, theta_edges), 0.0, np.pi)
        self.phi_edges = edges(self.phi, phi_edges)
        self.r_lo, self.r_hi = self.r_edges[0], self.r_edges[-1]
        self.full_theta = self.theta_edges[0] <= 1e-6 and self.theta_edges[-1] >= np.pi - 1e-6
        span = self.phi_edges[-1] - self.phi_edges[0]
        self.full_phi = self.axisym or span >= 2 * np.pi * (1 - 1e-6)
        self.r_body = float(self.r_lo if r_body is None else r_body)
        self.length_unit = float(length_unit)

        # gradient in spherical components
        R = self.r[:, None, None]
        S = np.maximum(np.sin(self.theta), 1e-6)[None, :, None]
        g_r = np.gradient(ne, self.r, axis=0, edge_order=2 if nr > 2 else 1)
        g_t = (np.gradient(ne, self.theta, axis=1, edge_order=2 if nt > 2 else 1) / R
               if nt > 1 else np.zeros_like(ne))
        if nph > 1:
            if self.full_phi:
                pad = np.concatenate([ne[:, :, -1:], ne, ne[:, :, :1]], axis=2)
                ph = np.concatenate([[self.phi[-1] - 2 * np.pi], self.phi, [self.phi[0] + 2 * np.pi]])
                g_p = np.gradient(pad, ph, axis=2)[:, :, 1:-1]
            else:
                g_p = np.gradient(ne, self.phi, axis=2, edge_order=2 if nph > 2 else 1)
            g_p = g_p / (R * S)
        else:
            g_p = np.zeros_like(ne)
        f = np.empty(self.shape + (self.NFIELD,), dtype=dtype)
        f[..., 0], f[..., 1], f[..., 2], f[..., 3], f[..., 4] = ne, g_r, g_t, g_p, te
        self._flat = f.reshape(-1, self.NFIELD)
        del f, g_r, g_t, g_p
        self.dr = np.diff(self.r_edges)
        self.dth = np.diff(self.theta_edges)
        self.dph = np.diff(self.phi_edges) if nph > 1 else np.array([np.inf])
        self._strides = np.array([nt * nph, nph, 1], dtype=np.int64)

    def __repr__(self):
        kind = "axisymmetric (r,theta)" if self.axisym else "3D (r,theta,phi)"
        return (f"SphericalGrid({kind}, shape={self.shape}, r=[{self.r_lo:.3g},{self.r_hi:.3g}] R_sun, "
                f"theta=[{np.degrees(self.theta_edges[0]):.1f},{np.degrees(self.theta_edges[-1]):.1f}] deg"
                + ("" if self.axisym else
                   f", phi=[{np.degrees(self.phi_edges[0]):.1f},{np.degrees(self.phi_edges[-1]):.1f}] deg")
                + f", min dr={self.dr.min():.3g} R_sun, r_body={self.r_body})")

    @property
    def nbytes(self):
        return self._flat.nbytes

    @property
    def min_cell(self):
        return float(min(self.dr.min(), self.r_lo * self.dth.min()))

    @staticmethod
    def _sph(p):
        r = np.sqrt(np.einsum("ij,ij->i", p, p))
        th = np.arccos(np.clip(p[:, 2] / np.maximum(r, 1e-300), -1.0, 1.0))
        ph = np.mod(np.arctan2(p[:, 1], p[:, 0]), 2 * np.pi)
        return r, th, ph

    def _phi_rel(self, ph):
        """phi relative to the first edge, wrapped to [0, 2pi)."""
        return np.mod(ph - self.phi_edges[0], 2 * np.pi) + self.phi_edges[0]

    def inside(self, p):
        r, th, ph = self._sph(np.asarray(p, float))
        ok = (r >= self.r_lo) & (r <= self.r_hi)
        if not self.full_theta:
            ok &= (th >= self.theta_edges[0]) & (th <= self.theta_edges[-1])
        if not self.full_phi:
            phr = self._phi_rel(ph)
            ok &= phr <= self.phi_edges[-1]
        return ok

    @staticmethod
    def _locate(c, u):
        n = c.size
        if n == 1:
            return np.zeros(u.size, np.int64), np.zeros(u.size)
        i = np.clip(np.searchsorted(c, u, side="right") - 1, 0, n - 2)
        t = np.clip((u - c[i]) / (c[i + 1] - c[i]), 0.0, 1.0)
        return i, t

    def sample(self, p):
        p = np.asarray(p, float)
        r, th, ph = self._sph(p)
        ins = (r >= self.r_lo) & (r <= self.r_hi)
        if not self.full_theta:
            ins &= (th >= self.theta_edges[0]) & (th <= self.theta_edges[-1])
        i, ti = self._locate(self.r, r)
        j, tj = self._locate(self.theta, th)
        nph = self.phi.size
        if nph == 1:
            k0 = np.zeros(r.size, np.int64)
            k1 = k0
            tk = np.zeros(r.size)
        elif self.full_phi:
            # periodic: cells between phi[-1] and phi[0]+2pi wrap around
            phr = np.mod(ph - self.phi[0], 2 * np.pi)
            rel = self.phi - self.phi[0]
            k0 = np.searchsorted(rel, phr, side="right") - 1
            k0 = np.clip(k0, 0, nph - 1)
            k1 = (k0 + 1) % nph
            nxt = np.where(k0 == nph - 1, 2 * np.pi, rel[np.minimum(k0 + 1, nph - 1)])
            tk = np.clip((phr - rel[k0]) / (nxt - rel[k0]), 0.0, 1.0)
        else:
            phr = self._phi_rel(ph)
            ins &= phr <= self.phi_edges[-1]
            k0, tk = self._locate(self.phi, phr)
            k1 = k0 + 1
        st = self._strides
        out = np.zeros((r.size, self.NFIELD))
        for di, wi in ((0, 1 - ti), (1, ti)):
            ii = np.minimum(i + di, self.r.size - 1)
            for dj, wj in ((0, 1 - tj), (1, tj)):
                jj = np.minimum(j + dj, self.theta.size - 1)
                for kk, wk in ((k0, 1 - tk), (k1, tk)):
                    w = wi * wj * wk
                    out += w[:, None] * self._flat[ii * st[0] + jj * st[1] + kk]
        ne, gr, gt, gp, te = out.T
        sth, cth = np.sin(th), np.cos(th)
        sph, cph = np.sin(ph), np.cos(ph)
        grad = np.empty((r.size, 3))
        grad[:, 0] = gr * sth * cph + gt * cth * cph - gp * sph
        grad[:, 1] = gr * sth * sph + gt * cth * sph + gp * cph
        grad[:, 2] = gr * cth - gt * sth
        dr = self.dr[np.clip(np.searchsorted(self.r_edges, r) - 1, 0, self.dr.size - 1)]
        rdt = r * self.dth[np.clip(np.searchsorted(self.theta_edges, th) - 1, 0, self.dth.size - 1)]
        cell = np.minimum(dr, rdt)
        if nph > 1:
            dph = self.dph[np.clip(k0, 0, self.dph.size - 1)]
            cell = np.minimum(cell, np.maximum(r * np.abs(sth) * dph, 0.25 * rdt))
        return np.maximum(ne, 0.0), grad, te, cell, ins

    # ---- magnetic field (gyroresonance, polarisation) -------------------
    @property
    def has_field(self):
        return getattr(self, "_bflat", None) is not None

    def add_field(self, b1, b2, b3, components="spherical", dtype=np.float32):
        """Attach the magnetic field [G] at the cell centres.

        components='spherical': (B_r, B_theta, B_phi) (PLUTO bx1, bx2, bx3);
        'cartesian': (B_x, B_y, B_z)."""
        b = np.stack([np.asarray(c, dtype=float).reshape(self.shape) for c in (b1, b2, b3)],
                     axis=-1)
        b = np.where(np.isfinite(b), b, 0.0)
        if components not in ("spherical", "cartesian"):
            raise ValueError("components must be 'spherical' or 'cartesian'")
        self._bcomp = components
        self._bflat = b.astype(dtype).reshape(-1, 3)
        return self

    def _interp_flat(self, flat, p):
        """Trilinear interpolation (in r, theta, phi index space) of a
        (N_cells, k) array at points p (N,3).  Returns (values, theta, phi)."""
        p = np.asarray(p, float)
        r, th, ph = self._sph(p)
        i, ti = self._locate(self.r, r)
        j, tj = self._locate(self.theta, th)
        nph = self.phi.size
        if nph == 1:
            k0 = np.zeros(r.size, np.int64)
            k1 = k0
            tk = np.zeros(r.size)
        elif self.full_phi:
            phr = np.mod(ph - self.phi[0], 2 * np.pi)
            rel = self.phi - self.phi[0]
            k0 = np.clip(np.searchsorted(rel, phr, side="right") - 1, 0, nph - 1)
            k1 = (k0 + 1) % nph
            nxt = np.where(k0 == nph - 1, 2 * np.pi, rel[np.minimum(k0 + 1, nph - 1)])
            tk = np.clip((phr - rel[k0]) / (nxt - rel[k0]), 0.0, 1.0)
        else:
            k0, tk = self._locate(self.phi, self._phi_rel(ph))
            k1 = k0 + 1
        st = self._strides
        out = np.zeros((r.size, flat.shape[1]))
        for di, wi in ((0, 1 - ti), (1, ti)):
            ii = np.minimum(i + di, self.r.size - 1)
            for dj, wj in ((0, 1 - tj), (1, tj)):
                jj = np.minimum(j + dj, self.theta.size - 1)
                for kk, wk in ((k0, 1 - tk), (k1, tk)):
                    out += (wi * wj * wk)[:, None] * flat[ii * st[0] + jj * st[1] + kk]
        return out, th, ph

    def sample_field(self, p):
        """Trilinear interpolation of B [G] at points p (N,3) -> (N,3) Cartesian."""
        if not self.has_field:
            raise ValueError("this grid has no magnetic field: use add_field() or "
                             "load_mhd(..., with_b=True)")
        out, th, ph = self._interp_flat(self._bflat, p)
        if self._bcomp == "cartesian":
            return out
        br, bt, bp = out.T
        sth, cth = np.sin(th), np.cos(th)
        sph, cph = np.sin(ph), np.cos(ph)
        b = np.empty((out.shape[0], 3))
        b[:, 0] = br * sth * cph + bt * cth * cph - bp * sph
        b[:, 1] = br * sth * sph + bt * cth * sph + bp * cph
        b[:, 2] = br * cth - bt * sth
        return b

    def add_scalar(self, name, values, dtype=np.float32):
        """Attach a named scalar field at the cell centres (e.g. 'j')."""
        v = np.asarray(values, dtype=float).reshape(self.shape)
        if not hasattr(self, "scalars"):
            self.scalars = {}
        self.scalars[name] = np.where(np.isfinite(v), v, 0.0).astype(dtype).reshape(-1, 1)
        return self

    def sample_scalar(self, name, p):
        """Trilinear interpolation of the scalar field ``name`` at points p (N,3)."""
        return self._interp_flat(self.scalars[name], p)[0][:, 0]

    def domain_intersection(self, origin, direction):
        """Entry/exit parameters of straight rays with the outer sphere r_hi."""
        o = np.broadcast_to(np.asarray(origin, float), np.shape(direction))
        d = np.asarray(direction, float)
        b = np.einsum("ij,ij->i", o, d)
        c = np.einsum("ij,ij->i", o, o) - self.r_hi**2
        disc = b * b - c
        sq = np.sqrt(np.maximum(disc, 0.0))
        t0 = np.where(disc > 0, -b - sq, np.inf)
        t1 = np.where(disc > 0, -b + sq, -np.inf)
        return t0, t1

    box_intersection = domain_intersection

    def to_npz(self, path):
        extra = {}
        if self.has_field:
            extra = dict(b=self._bflat.reshape(self.shape + (3,)), b_components=self._bcomp)
        np.savez(path, geometry="spherical", r=self.r, theta=self.theta,
                 phi=self.phi if not self.axisym else np.array([0.0]),
                 r_edges=self.r_edges, theta_edges=self.theta_edges,
                 phi_edges=self.phi_edges, axisym=self.axisym,
                 ne=self._flat[:, 0].reshape(self.shape),
                 te=self._flat[:, 4].reshape(self.shape), r_body=self.r_body,
                 length_unit=self.length_unit, **extra)

    @property
    def r_outer(self):
        return float(self.r_hi)
