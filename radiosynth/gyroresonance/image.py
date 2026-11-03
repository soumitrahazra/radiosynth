# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gyroresonance + free-free images with circular polarisation.

make_gr_image uses the same image-plane geometry, observers (Earth-based,
dated, stellar/parallel) and output container as the refractive free-free
images (common/image.py), but straight lines of sight (common/straight.py)
and the polarised transfer of transfer.py.
"""
import math
import time

import numpy as np

from ..common.constants import R_SUN, critical_density
from ..common.image import (RadioImage, freq_label, image_geometry, parse_frequencies,
                            start_points)
from ..common.raytrace import HIT_BODY, MISSED, OVERDENSE
from ..common.straight import default_step, iter_samples, segment_ends
from .transfer import transfer_chunk

PC = 3.0856775814913673e16


class GRImage(RadioImage):
    """RadioImage with Stokes V and gyroresonance diagnostics.

    intensity   : Stokes I = I_1 + I_2      (nfreq, ny, nx) [W m^-2 Hz^-1 sr^-1]
    stokes_v    : Stokes V (IAU/IEEE: V > 0 right-hand circular)
    pol1, pol2  : intensities of the two polarisations: R, L (coupling='weak')
                  or x, o (coupling='strong')
    tau         : optical depth of polarisation 1;  tau2 that of polarisation 2
    dominant_s  : harmonic carrying most of the intensity (0 = free-free,
                  -1 = no emission)
    radius      : intensity-weighted emission radius [coordinate units]
    gr_fraction : fraction of I from gyroresonance layers
    x_flag      : True where a significant layer had (nu_p/nu)^2 > 0.25 (the
                  n = 1 approximation of the layer opacity is then rough)
    """

    def __init__(self, *a, stokes_v=None, pol1=None, pol2=None, tau2=None, dominant_s=None,
                 radius=None, gr_fraction=None, x_flag=None, pol_names=("R", "L"), **kw):
        super().__init__(*a, **kw)
        self.stokes_v = stokes_v
        self.pol1, self.pol2, self.tau2 = pol1, pol2, tau2
        self.dominant_s = dominant_s
        self.radius = radius
        self.gr_fraction = gr_fraction
        self.x_flag = x_flag
        self.pol_names = tuple(pol_names)

    def _tb(self, a):
        return a * self.brightness_temperature_factor

    @property
    def brightness_temperature_factor(self):
        from ..common.constants import C_LIGHT, K_B
        return C_LIGHT**2 / (2.0 * K_B * self.freqs[:, None, None] ** 2)

    @property
    def tb_v(self):
        """Stokes V as a brightness temperature [K]."""
        return self._tb(self.stokes_v)

    @property
    def tb_pol(self):
        """Brightness temperatures of the two polarisations, T_p = I_p c^2/(k nu^2)
        (each polarisation carries half of the Rayleigh-Jeans intensity, so
        T_p = T_e for an optically thick isothermal source and
        T_b(I) = (T_1 + T_2)/2)."""
        return 2.0 * self._tb(self.pol1), 2.0 * self._tb(self.pol2)

    @property
    def polarization(self):
        """Degree of circular polarisation V/I (NaN where I = 0)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.intensity > 0, self.stokes_v / self.intensity, np.nan)

    @property
    def flux_density_v(self):
        return self.stokes_v.sum(axis=(1, 2)) * self.pixel_solid_angle

    def summary(self):
        lines = []
        for k, f in enumerate(self.freqs):
            I = self.flux_density[k]
            V = self.flux_density_v[k]
            tb = self.brightness_temperature[k]
            ds = self.dominant_s[k]
            harm = ", ".join(f"s={s}:{int((ds == s).sum())}" for s in np.unique(ds) if s > 0)
            lines.append(f"{freq_label(f):>12s}: max Tb={tb.max():.3e} K, flux={I / 1e-22:.3e} sfu"
                         f" ({I / 1e-26:.3e} Jy), V/I={V / I if I > 0 else 0:+.3f}, "
                         f"GR fraction of flux={np.nansum(self.gr_fraction[k] * self.intensity[k]) / max(self.intensity[k].sum(), 1e-300):.2f}"
                         f"; dominant harmonic pixels: {harm or 'none'}")
        return "\n".join(lines)

    def save_npz(self, path):
        np.savez_compressed(path, intensity=self.intensity, stokes_v=self.stokes_v,
                            pol1=self.pol1, pol2=self.pol2, pol_names=np.array(self.pol_names),
                            tau=self.tau, tau2=self.tau2, status=self.status, x=self.x, y=self.y,
                            freqs=self.freqs, observer=self.observer, distance=self.distance,
                            length_unit=self.length_unit, flux_density=self.flux_density,
                            flux_density_v=self.flux_density_v, tb=self.brightness_temperature,
                            tb_v=self.tb_v, dominant_s=self.dominant_s, radius=self.radius,
                            gr_fraction=self.gr_fraction, x_flag=self.x_flag, meta=repr(self.meta))

    def save_fits(self, path, quantity="intensity"):
        """quantity: 'intensity', 'tb', 'v' (Stokes V intensity), 'tb_v', 'pol'."""
        from ..common.fitsio import write_fits
        if quantity in ("intensity", "tb"):
            return super().save_fits(path, quantity)
        data = {"v": self.stokes_v, "tb_v": self.tb_v, "pol": self.polarization}[quantity]
        unit = {"v": "W m-2 Hz-1 sr-1", "tb_v": "K", "pol": ""}[quantity]
        h = {"BUNIT": unit, "STOKES": "V" if quantity != "pol" else "V/I",
             "CTYPE1": "SOLX", "CUNIT1": "Rsun", "CRPIX1": 1.0,
             "CRVAL1": float(self.x[0]), "CDELT1": float(self.x[1] - self.x[0]),
             "CTYPE2": "SOLY", "CUNIT2": "Rsun", "CRPIX2": 1.0,
             "CRVAL2": float(self.y[0]), "CDELT2": float(self.y[1] - self.y[0])}
        for k, f in enumerate(self.freqs):
            h[f"FREQ{k + 1}"] = float(f)
        write_fits(path, np.nan_to_num(data), h)

    def plot(self, quantity="tb", path=None, log=None, cmap=None, ncols=None,
             vmin=None, vmax=None, show_limb=True):
        """quantity: 'tb', 'intensity', 'v' (T_b of Stokes V), 'pol' (V/I),
        'harmonic' (dominant s), 'radius', 'gr_fraction'."""
        if quantity in ("tb", "intensity"):
            return super().plot(quantity, path=path, log=True if log is None else log,
                                cmap=cmap or "inferno", ncols=ncols, vmin=vmin, vmax=vmax,
                                show_limb=show_limb)
        import matplotlib
        if path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        data, label, cm, lo, hi = {
            "v": (self.tb_v, r"$T_b(V)$ [K]", "RdBu_r", None, None),
            "pol": (self.polarization, "V/I", "RdBu_r", -1.0, 1.0),
            "harmonic": (np.where(self.dominant_s >= 0, self.dominant_s, np.nan).astype(float),
                         "dominant harmonic s (0 = free-free)", "viridis", None, None),
            "radius": (self.radius, r"emission radius [$R$]", "viridis", None, None),
            "gr_fraction": (self.gr_fraction, getattr(self, "fraction_label", "gyroresonance fraction"),
                            "magma", 0.0, 1.0),
        }[quantity]
        nf = len(self.freqs)
        ncols = ncols or min(nf, 3)
        nrows = math.ceil(nf / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows), squeeze=False)
        dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        ext = [self.x[0] - dx / 2, self.x[-1] + dx / 2, self.y[0] - dy / 2, self.y[-1] + dy / 2]
        for k in range(nrows * ncols):
            ax = axes.flat[k]
            if k >= nf:
                ax.axis("off")
                continue
            d = data[k]
            a, b = (vmin if vmin is not None else lo), (vmax if vmax is not None else hi)
            if quantity == "v" and a is None:
                m = np.nanmax(np.abs(d)) or 1.0
                a, b = -m, m
            im = ax.imshow(d, origin="lower", extent=ext, cmap=cmap or cm, vmin=a, vmax=b)
            if show_limb:
                t = np.linspace(0, 2 * np.pi, 361)
                ax.plot(np.cos(t), np.sin(t), "k--", lw=0.7, alpha=0.6)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_title(freq_label(self.freqs[k]))
            ax.set_xlabel(r"X [$R_\odot$]")
            ax.set_ylabel(r"Y [$R_\odot$]")
            fig.colorbar(im, ax=ax, label=label, shrink=0.85)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=130)
            plt.close(fig)
        return fig


# ----------------------------------------------------------------------------
def make_gr_image(grid, frequencies, observer=(-215.0, 0.0, 0.0), x_size=2.0, y_size=2.0,
                  nx=128, ny=128, *, mechanisms="gr+ff", gaunt="dulk", s_max=10,
                  coupling="weak", ds=None, t_body=None, r_integration=None,
                  up=(0.0, 0.0, 1.0), distance_pc=None, max_points=400000, verbose=True):
    """Gyroresonance (+ free-free) image with Stokes I and V along straight rays.

    grid        : MHDGrid / SphericalGrid with a magnetic field (add_field or
                  load_mhd(..., with_b=True)); coordinates and B in gauss
    frequencies : '5GHz, 17GHz' or numbers in Hz
    observer, x_size, y_size, nx, ny, up, distance_pc, r_integration : as in
                  make_radio_image (the default field of view is 2 x 2 R_sun)
    mechanisms  : 'gr+ff' (default: both, absorbing each other), 'gr' (layers
                  only) or 'ff' (magnetised free-free only, straight rays)
    gaunt       : free-free Gaunt factor (number or 'dulk', default 'dulk')
    s_max       : highest harmonic considered (default 10)
    coupling    : 'weak' (polarisation sense conserved through quasi-transverse
                  regions; default) or 'strong' (mode conserved)
    ds          : step along the rays [coordinate units] (default half the
                  smallest cell); the layers are found between the nodes, so
                  ds only has to resolve the variation of n_e, T_e and B
    t_body      : brightness temperature of the body (e.g. 1e4 K for the
                  chromosphere) seen by rays that reach r_body; default None
                  (no emission, as in the refractive tracer)
    """
    if not getattr(grid, "has_field", False):
        raise ValueError("gyroresonance needs the magnetic field: grid.add_field(bx, by, bz) "
                         "or load_mhd(..., with_b=True)")
    mechanisms = mechanisms.lower().replace(" ", "")
    if mechanisms not in ("gr+ff", "ff+gr", "gr", "ff"):
        raise ValueError("mechanisms must be 'gr+ff', 'gr' or 'ff'")
    freqs = parse_frequencies(frequencies)
    L = float(getattr(grid, "length_unit", R_SUN))
    parallel = distance_pc is not None
    xc, yc, dirs, tau_v, xi_v, normal, pix = image_geometry(observer, x_size, y_size, nx, ny,
                                                            up, parallel=parallel)
    diag = math.sqrt((0.5 * x_size) ** 2 + (0.5 * y_size) ** 2)
    if r_integration is None:
        r_integration = float(math.ceil(max(diag + 1.0, 5.0)))
    if parallel:
        distance = distance_pc * PC / L
        far = 4.0 * max(r_integration, getattr(grid, "r_outer", r_integration), diag)
        pos, status0 = start_points(grid, pix + far * normal, dirs, r_integration)
    else:
        distance = float(np.linalg.norm(observer))
        pos, status0 = start_points(grid, observer, dirs, r_integration)
    t_end, status = segment_ends(grid, pos, dirs, status0, r_integration)
    step = default_step(grid, t_end, ds)
    hit_body = status == HIT_BODY
    nray = dirs.shape[0]
    if verbose:
        print(f"radiosynth.gyroresonance: {nx}x{ny} straight rays, mechanisms={mechanisms}, "
              f"coupling={coupling}, s_max={s_max}, step={step:.3g}, gaunt={gaunt}")
        print(f"            {grid}")

    nf = len(freqs)
    keys = ("I1", "I2", "tau1", "tau2", "V", "radius", "gr_fraction")
    res = {k: np.zeros((nf, nray)) for k in keys}
    res["radius"][:] = np.nan
    dom = np.full((nf, nray), -1, np.int16)
    cut = np.zeros((nf, nray), bool)
    xf = np.zeros((nf, nray), bool)
    for k, f in enumerate(freqs):
        t0 = time.time()
        for sm in iter_samples(grid, pos, dirs, t_end, step, max_points=max_points):
            o = transfer_chunk(sm, dirs, f, L, mechanisms=mechanisms, gaunt=gaunt,
                               s_max=s_max, coupling=coupling, t_body=t_body,
                               hit_body=hit_body)
            i = sm["idx"]
            for key in keys:
                res[key][k, i] = o[key]
            dom[k, i], cut[k, i], xf[k, i] = o["dominant_s"], o["cutoff"], o["x_flag"]
        if verbose:
            print(f"   {freq_label(f)}: {time.time() - t0:.1f} s  (n_cr = {critical_density(f):.3e} m^-3)")

    shp = (nf, ny, nx)
    st = np.broadcast_to(status, (nf, nray)).copy()
    st[cut & (st != MISSED)] = OVERDENSE
    meta = dict(x_size=x_size, y_size=y_size, nx=nx, ny=ny, mechanisms=mechanisms,
                gaunt=gaunt, s_max=s_max, coupling=coupling, ds=step, t_body=t_body,
                distance_pc=distance_pc, r_integration=r_integration,
                tau_vec=tau_v, xi_vec=xi_v, emission="gyroresonance", transfer="full")
    R = lambda a: a.reshape(shp)                                   # noqa: E731
    img = GRImage(R(res["I1"] + res["I2"]), R(res["tau1"]), R(st),
                  np.full(shp, int(np.ceil(np.max(t_end, initial=0) / step))), xc, yc, freqs,
                  observer, meta, distance=distance, length_unit=L,
                  stokes_v=R(res["V"]), pol1=R(res["I1"]), pol2=R(res["I2"]),
                  tau2=R(res["tau2"]), dominant_s=R(dom), radius=R(res["radius"]),
                  gr_fraction=R(res["gr_fraction"]), x_flag=R(xf),
                  pol_names=("R", "L") if coupling == "weak" else ("x", "o"))
    if verbose:
        print(img.summary())
    return img
