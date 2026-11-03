# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic radio image driver (port of write_plot_radiowave).

Image-plane geometry (identical to ModWritePlotRadioWave.f90):
  * the image plane passes through the Sun centre, perpendicular to the
    observer direction  Normal = Obs/|Obs|;
  * in-plane unit vectors  Tau = (z x Normal)/|...|  (image x, "right"),
                           Xi  = Normal x Tau        (image y, "up");
  * pixel centres  X = XLower + (i-1/2) dX,  Y = YLower + (j-1/2) dY;
  * each ray starts at the observer and points at its pixel centre; the
    straight segment observer -> integration sphere is emission-free.
    rIntegration = ceil(max(ImagePlaneDiagRadius + 1, 5)).
"""
import math
import multiprocessing as mp
import re
import time

import numpy as np

from .constants import C_LIGHT, K_B, R_SUN, JANSKY, critical_density
from .emission import EmissionModel
from .raytrace import trace_rays, ACTIVE, MISSED, STATUS_NAMES


# ----------------------------------------------------------------------------
def parse_frequencies(spec):
    """'1500kHz, 11MHz, 42.7MHz, 1.08GHz' -> array of Hz (like parse_freq_string)."""
    if np.isscalar(spec) and not isinstance(spec, str):
        return np.array([float(spec)])
    if not isinstance(spec, str):
        # list of numbers [Hz] or of strings such as ["80MHz", "1.2GHz"]
        return np.array([parse_frequencies(v)[0] if isinstance(v, str) else float(v)
                         for v in spec])
    scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12}
    out = []
    for num, unit in re.findall(r"([-+0-9.eEdD]+)\s*([a-zA-Z]+)?", spec):
        unit = (unit or "Hz").lower()
        if unit not in scale:
            raise ValueError(f"Unrecognized frequency unit '{unit}'. Use Hz, kHz, MHz, GHz or THz")
        out.append(abs(float(num.replace("d", "e").replace("D", "e"))) * scale[unit])
    return np.array(out)


def freq_label(f):
    for lim, u in ((1e12, "THz"), (1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"), (0, "Hz")):
        if f >= lim:
            return f"{f / max(lim, 1):.2f} {u}" if lim else f"{f:.2f} Hz"


def image_geometry(observer, x_size, y_size, nx, ny, up=(0.0, 0.0, 1.0), parallel=False):
    """Return pixel coordinate axes, unit ray directions (ny*nx, 3), the image
    unit vectors and the 3D pixel-centre positions.

    Ray index = j*nx + i (x fastest), as in the Fortran code.
    parallel=True: observer at infinity in the direction of ``observer`` (stars)."""
    obs = np.asarray(observer, dtype=float)
    dist = np.linalg.norm(obs)
    normal = obs / dist
    up = np.asarray(up, dtype=float)
    tau = np.cross(up, normal)
    if np.linalg.norm(tau) < 1e-12:          # observer on the 'up' axis
        tau = np.cross(np.array([0.0, 1.0, 0.0]), normal)
    tau /= np.linalg.norm(tau)
    xi = np.cross(normal, tau)
    dxp = x_size / nx
    dyp = y_size / ny
    xc = -0.5 * x_size + (np.arange(nx) + 0.5) * dxp
    yc = -0.5 * y_size + (np.arange(ny) + 0.5) * dyp
    X, Y = np.meshgrid(xc, yc)                # (ny, nx)
    pix = X.reshape(-1, 1) * tau + Y.reshape(-1, 1) * xi
    if parallel:                              # distant observer: parallel rays
        d = np.broadcast_to(-normal, pix.shape).copy()
    else:
        d = pix - obs
        d /= np.linalg.norm(d, axis=1)[:, None]
    return xc, yc, d, tau, xi, normal, pix


def start_points(grid, observer, dirs, r_integration):
    """Starting points: where the straight ray first enters both the
    integration sphere and the MHD grid box.  Returns (pos, status0)."""
    obs = np.asarray(observer, dtype=float)
    if obs.ndim == 1:
        D2 = obs @ obs
        od = dirs @ obs
    else:                                    # one origin per ray (parallel projection)
        D2 = np.einsum("ij,ij->i", obs, obs)
        od = np.einsum("ij,ij->i", dirs, obs)
    disc = r_integration**2 - D2 + od**2
    t_sph = -od - np.sqrt(np.maximum(disc, 0.0))
    t_in, t_out = grid.domain_intersection(obs, dirs)
    t0 = np.maximum(np.maximum(t_sph, t_in), 0.0)
    # far exit of the sphere
    t_sph_out = -od + np.sqrt(np.maximum(disc, 0.0))
    t1 = np.minimum(t_out, t_sph_out)
    status0 = np.full(dirs.shape[0], ACTIVE, np.int8)
    status0[(disc <= 0) | (t1 <= t0)] = MISSED
    # nudge inside the box to avoid round-off at the faces
    span = np.where(status0 == ACTIVE, t1 - t0, 0.0)
    t0 = t0 + np.minimum(1e-9 * np.maximum(1.0, np.sqrt(D2)), 1e-3 * span)
    t0 = np.where(np.isfinite(t0), t0, 0.0)          # missed rays: any finite point
    pos = obs + t0[:, None] * dirs
    return pos, status0


# ----------------------------------------------------------------------------
_G = {}


def _worker(args):
    idx, freq = args
    g = _G
    res = trace_rays(g["grid"], g["pos"][idx], g["dirs"][idx], freq, g["emission"],
                     status0=g["status0"][idx], **g["kw"])
    return idx, res["intensity"], res["tau"], res["status"], res["nstep"]


class RadioImage:
    """Container for a multi-frequency synthetic radio image.

    Attributes
    ----------
    intensity : (nfreq, ny, nx)  [W m^-2 Hz^-1 sr^-1] (dimensionless for 'simplistic')
    tau       : (nfreq, ny, nx)  optical depth along the ray (full transfer)
    status    : (nfreq, ny, nx)  termination code of each ray
    x, y      : pixel-centre coordinates in the image plane [R_sun]
    freqs     : [Hz]
    """

    def __init__(self, intensity, tau, status, nstep, x, y, freqs, observer, meta,
                 distance=None, length_unit=R_SUN):
        self.intensity = intensity
        self.tau = tau
        self.status = status
        self.nstep = nstep
        self.x = x
        self.y = y
        self.freqs = np.asarray(freqs)
        self.observer = np.asarray(observer, dtype=float)
        self.meta = meta
        # observer distance in coordinate units (sets the flux and angular scale)
        self.distance = float(distance if distance is not None else np.linalg.norm(self.observer))
        self.length_unit = float(length_unit)

    @property
    def brightness_temperature(self):
        """Rayleigh-Jeans brightness temperature  T_b = I c^2 / (2 k nu^2)  [K]."""
        f = self.freqs[:, None, None]
        return self.intensity * C_LIGHT**2 / (2.0 * K_B * f**2)

    @property
    def pixel_solid_angle(self):
        D = self.distance
        return (self.x[1] - self.x[0]) * (self.y[1] - self.y[0]) / D**2   # [sr]

    @property
    def flux_density(self):
        """Total flux density at the observer [W m^-2 Hz^-1] for each frequency."""
        return self.intensity.sum(axis=(1, 2)) * self.pixel_solid_angle

    @property
    def flux_density_sfu(self):
        return self.flux_density / 1e-22

    @property
    def flux_density_mjy(self):
        """Flux density in mJy (stellar sources)."""
        return self.flux_density / 1e-29

    @property
    def distance_pc(self):
        return self.distance * self.length_unit / 3.0856775814913673e16

    def summary(self):
        lines = []
        for k, f in enumerate(self.freqs):
            st = self.status[k]
            counts = ", ".join(f"{STATUS_NAMES[c]}={int((st == c).sum())}"
                               for c in np.unique(st))
            lines.append(f"{freq_label(f):>12s}: max I={self.intensity[k].max():.3e}, "
                         f"max Tb={self.brightness_temperature[k].max():.3e} K, "
                         f"flux={self.flux_density[k]:.3e} W/m2/Hz "
                         f"({self.flux_density[k] / JANSKY:.3e} Jy at {self.distance_pc:.3g} pc); "
                         f"rays: {counts}")
        return "\n".join(lines)

    # ----- output --------------------------------------------------------
    def save_npz(self, path):
        np.savez_compressed(path, intensity=self.intensity, tau=self.tau,
                            status=self.status, nstep=self.nstep, x=self.x, y=self.y,
                            freqs=self.freqs, observer=self.observer,
                            distance=self.distance, length_unit=self.length_unit,
                            flux_density=self.flux_density,
                            tb=self.brightness_temperature, meta=repr(self.meta))

    def save_tecplot(self, path):
        """ASCII Tecplot file in the same layout as BATS-R-US write_plot_radiowave."""
        names = " ".join(f'"f={freq_label(f).replace(" ", "_")}"' for f in self.freqs)
        nx, ny = self.x.size, self.y.size
        with open(path, "w") as fh:
            fh.write('TITLE="radiosynth: Radiotelescope Image"\n')
            fh.write(f'VARIABLES = "X" "Y" {names}\n')
            fh.write(f'ZONE T="radio", I={nx}, J={ny}, K=1, F=POINT\n')
            X, Y = np.meshgrid(self.x, self.y)
            data = np.column_stack([X.ravel(), Y.ravel()] +
                                   [self.intensity[k].ravel() for k in range(len(self.freqs))])
            np.savetxt(fh, data, fmt="%14.6E")

    def save_fits(self, path, quantity="intensity"):
        """One FITS file with a (nfreq, ny, nx) cube (astropy used if installed)."""
        from .fitsio import write_fits
        data = self.brightness_temperature if quantity == "tb" else self.intensity
        h = {"BUNIT": "K" if quantity == "tb" else "W m-2 Hz-1 sr-1",
             "CTYPE1": "SOLX", "CUNIT1": "Rsun", "CRPIX1": 1.0,
             "CRVAL1": float(self.x[0]), "CDELT1": float(self.x[1] - self.x[0]),
             "CTYPE2": "SOLY", "CUNIT2": "Rsun", "CRPIX2": 1.0,
             "CRVAL2": float(self.y[0]), "CDELT2": float(self.y[1] - self.y[0])}
        for k, f in enumerate(self.freqs):
            h[f"FREQ{k + 1}"] = float(f)
        h["OBSX"], h["OBSY"], h["OBSZ"] = (float(v) for v in self.observer)
        write_fits(path, data, h)

    def plot(self, quantity="tb", path=None, log=True, cmap="inferno", ncols=None,
             vmin=None, vmax=None, show_limb=True):
        import matplotlib
        if path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        data = self.brightness_temperature if quantity == "tb" else self.intensity
        label = r"$T_b$ [K]" if quantity == "tb" else r"$I_\nu$ [W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]"
        nf = len(self.freqs)
        ncols = ncols or min(nf, 3)
        nrows = math.ceil(nf / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows), squeeze=False)
        ext = [self.x[0] - 0.5 * (self.x[1] - self.x[0]), self.x[-1] + 0.5 * (self.x[1] - self.x[0]),
               self.y[0] - 0.5 * (self.y[1] - self.y[0]), self.y[-1] + 0.5 * (self.y[1] - self.y[0])]
        for k in range(nrows * ncols):
            ax = axes.flat[k]
            if k >= nf:
                ax.axis("off")
                continue
            d = data[k]
            pos = d[d > 0]
            if log and pos.size:
                lo = vmin if vmin is not None else max(pos.min(), pos.max() * 1e-4)
                hi = vmax if vmax is not None else pos.max()
                norm = LogNorm(vmin=lo, vmax=hi)
                im = ax.imshow(np.where(d > 0, d, np.nan), origin="lower", extent=ext,
                               cmap=cmap, norm=norm)
            else:
                im = ax.imshow(d, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax)
            if show_limb:
                t = np.linspace(0, 2 * np.pi, 361)
                ax.plot(np.cos(t), np.sin(t), "w--", lw=0.7, alpha=0.8)
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
def make_radio_image(grid, frequencies, observer=(-215.0, 0.0, 0.0),
                     x_size=6.0, y_size=6.0, nx=128, ny=128, *,
                     emission="bremsstrahlung", gaunt=10.0, transfer="thin",
                     no_refraction=False, r_integration=None, up=(0.0, 0.0, 1.0),
                     n_workers=1, verbose=True, distance_pc=None, **trace_kw):
    """Compute a synthetic radio image from an MHDGrid.

    Parameters mirror the BATS-R-US #SAVEPLOT 'rwi' (radiowave image) input:
    observer position ObsPos [R_sun], image size X_Size_Image/Y_Size_Image
    [R_sun], pixel numbers n_Pix_X/n_Pix_Y, the frequency list (string such as
    '42.3MHz, 125MHz, 17GHz' or numbers in Hz), and UseNoRefraction.

    distance_pc : if given, the observer is at this distance (parsec) in the
    direction of ``observer`` and the rays are parallel (distant star); the
    flux density is computed for that distance.  Otherwise the rays diverge from
    ``observer`` (Sun seen from the Earth, as in BATS-R-US).

    Extra keywords are passed to raytrace.trace_rays (tolerance, step_factor,
    tau_max, kappa_over_mu, ...).
    """
    freqs = parse_frequencies(frequencies)
    em = emission if callable(emission) else EmissionModel(emission, gaunt=gaunt)
    L = float(getattr(grid, "length_unit", R_SUN))
    parallel = distance_pc is not None
    xc, yc, dirs, tau_v, xi_v, normal, pix = image_geometry(observer, x_size, y_size, nx, ny,
                                                            up, parallel=parallel)
    diag = math.sqrt((0.5 * x_size)**2 + (0.5 * y_size)**2)
    if r_integration is None:
        r_integration = float(math.ceil(max(diag + 1.0, 5.0)))
    if parallel:
        distance = distance_pc * 3.0856775814913673e16 / L      # coordinate units
        far = 4.0 * max(r_integration, getattr(grid, "r_outer", r_integration), diag)
        origins = pix + far * normal
        pos, status0 = start_points(grid, origins, dirs, r_integration)
    else:
        distance = float(np.linalg.norm(observer))
        pos, status0 = start_points(grid, observer, dirs, r_integration)
    nray = dirs.shape[0]
    if verbose:
        where = (f"distance {distance_pc:g} pc (parallel rays), direction "
                 f"{tuple(round(float(v), 4) for v in normal)}" if parallel
                 else f"observer={tuple(float(v) for v in observer)}")
        print(f"radiosynth: {nx}x{ny} rays, {where}, "
              f"rIntegration={r_integration} R_sun, refraction={'off' if no_refraction else 'on'}, "
              f"emission={em}, transfer={transfer}")
        print(f"            {grid}")
        if (status0 == MISSED).any():
            print(f"            {(status0 == MISSED).sum()} rays never enter the grid box")

    kw = dict(transfer=transfer, no_refraction=no_refraction,
              r_integration=r_integration, **trace_kw)
    I = np.zeros((len(freqs), nray))
    T = np.zeros_like(I)
    S = np.zeros((len(freqs), nray), np.int8)
    NS = np.zeros((len(freqs), nray), np.int64)

    use_pool = n_workers and n_workers > 1 and "fork" in mp.get_all_start_methods()
    for k, f in enumerate(freqs):
        t0 = time.time()
        if verbose:
            print(f"RAYTRACE START: f = {freq_label(f)}  (n_cr = {critical_density(f):.3e} m^-3)")
        if use_pool:
            _G.update(grid=grid, pos=pos, dirs=dirs, status0=status0, emission=em, kw=kw)
            nchunk = 4 * n_workers
            chunks = [(np.arange(c, nray, nchunk), f) for c in range(nchunk)]
            with mp.get_context("fork").Pool(n_workers) as pool:
                for idx, i_, t_, s_, n_ in pool.imap_unordered(_worker, chunks):
                    I[k, idx], T[k, idx], S[k, idx], NS[k, idx] = i_, t_, s_, n_
            _G.clear()
        else:
            r = trace_rays(grid, pos, dirs, f, em, status0=status0, **kw)
            I[k], T[k], S[k], NS[k] = r["intensity"], r["tau"], r["status"], r["nstep"]
        if verbose:
            print(f"RAYTRACE END:   {time.time() - t0:.1f} s, "
                  f"mean steps/ray {NS[k][S[k] != MISSED].mean():.0f}")

    shp = (len(freqs), ny, nx)
    meta = dict(x_size=x_size, y_size=y_size, nx=nx, ny=ny, emission=repr(em),
                transfer=transfer, no_refraction=no_refraction, distance_pc=distance_pc,
                r_integration=r_integration, tau_vec=tau_v, xi_vec=xi_v, **trace_kw)
    img = RadioImage(I.reshape(shp), T.reshape(shp), S.reshape(shp), NS.reshape(shp),
                     xc, yc, freqs, observer, meta, distance=distance, length_unit=L)
    if verbose:
        print(img.summary())
    return img


def trace_paths(grid, frequency, observer, x_pix, y_pix, *, x_size=6.0, y_size=6.0,
                r_integration=None, up=(0.0, 0.0, 1.0), emission="bremsstrahlung",
                gaunt=10.0, **trace_kw):
    """Trace individual rays through given image-plane points (x_pix, y_pix)
    [R_sun] and return their paths (for plots like Benkevitch+2010 Fig. 7-9)."""
    obs = np.asarray(observer, dtype=float)
    _, _, _, tau_v, xi_v, _, _ = image_geometry(obs, x_size, y_size, 2, 2, up)
    xp = np.atleast_1d(x_pix).astype(float)
    yp = np.atleast_1d(y_pix).astype(float)
    pix = xp[:, None] * tau_v + yp[:, None] * xi_v
    d = pix - obs
    d /= np.linalg.norm(d, axis=1)[:, None]
    if r_integration is None:
        diag = math.sqrt((0.5 * x_size)**2 + (0.5 * y_size)**2)
        r_integration = float(math.ceil(max(diag + 1.0, 5.0)))
    pos, st0 = start_points(grid, obs, d, r_integration)
    em = emission if callable(emission) else EmissionModel(emission, gaunt=gaunt)
    f = parse_frequencies(frequency)[0]
    r = trace_rays(grid, pos, d, f, em, status0=st0, r_integration=r_integration,
                   record_ids=np.arange(len(xp)), **trace_kw)
    return r


# ----------------------------------------------------------------------------
REFRACTIVE_MECHANISMS = ("bremsstrahlung", "simplistic")


def compare_refraction(grid, frequencies, observer=(-215.0, 0.0, 0.0), x_size=6.0,
                       y_size=6.0, nx=128, ny=128, *, emission="bremsstrahlung",
                       quantity="tb", path=None, verbose=True, **kw):
    """Same image with refracted rays and with straight lines of sight.

    Available for the mechanisms that use the refractive ray tracer
    ('bremsstrahlung', 'simplistic', or a user emission function).  Both images
    use identical settings (all keywords of make_radio_image, e.g. transfer,
    gaunt, distance_pc, n_workers); only no_refraction differs.  Gyroresonance
    and gyrosynchrotron are computed along straight rays only and are not
    accepted here.

    quantity : 'tb' or 'intensity' for the figure
    path     : if given, write a figure with one row per frequency: refracted,
               straight, and the ratio refracted/straight

    Returns dict(refracted=RadioImage, straight=RadioImage,
                 flux_ratio=(nfreq,) refracted/straight flux density,
                 changed=(nfreq,) fraction of emitting pixels changed by > 10%,
                 figure=matplotlib figure or None)
    """
    if isinstance(emission, str) and emission.lower() not in REFRACTIVE_MECHANISMS:
        raise ValueError(f"compare_refraction works for {REFRACTIVE_MECHANISMS} (refractive "
                         f"ray tracer); '{emission}' is computed along straight rays only")
    kw.pop("no_refraction", None)
    common = dict(observer=observer, x_size=x_size, y_size=y_size, nx=nx, ny=ny,
                  emission=emission, verbose=False, **kw)
    ref = make_radio_image(grid, frequencies, no_refraction=False, **common)
    los = make_radio_image(grid, frequencies, no_refraction=True, **common)
    flux_ratio = ref.flux_density / np.where(los.flux_density > 0, los.flux_density, np.nan)
    a, b = ref.intensity, los.intensity
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(a - b) / np.maximum(np.maximum(a, b), 1e-300)
    emitting = np.maximum(a, b) > 1e-3 * np.maximum(a, b).max(axis=(1, 2), keepdims=True)
    changed = np.array([(rel[k][emitting[k]] > 0.1).mean() if emitting[k].any() else 0.0
                        for k in range(a.shape[0])])
    if verbose:
        for k, f in enumerate(ref.freqs):
            print(f"{freq_label(f):>12s}: flux refracted/straight = {flux_ratio[k]:.4f}, "
                  f"{100 * changed[k]:.1f}% of the emitting pixels change by more than 10%")
    fig = None
    if path is not None:
        fig = _figure_refraction(ref, los, quantity, path)
    return dict(refracted=ref, straight=los, flux_ratio=flux_ratio, changed=changed, figure=fig)


def _figure_refraction(ref, los, quantity, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    da = ref.brightness_temperature if quantity == "tb" else ref.intensity
    db = los.brightness_temperature if quantity == "tb" else los.intensity
    label = r"$T_b$ [K]" if quantity == "tb" else r"$I_\nu$"
    nf = len(ref.freqs)
    fig, axes = plt.subplots(nf, 3, figsize=(14, 4.0 * nf), squeeze=False)
    dx, dy = ref.x[1] - ref.x[0], ref.y[1] - ref.y[0]
    ext = [ref.x[0] - dx / 2, ref.x[-1] + dx / 2, ref.y[0] - dy / 2, ref.y[-1] + dy / 2]
    t = np.linspace(0, 2 * np.pi, 361)
    for k in range(nf):
        hi = max(da[k].max(), db[k].max())
        pos = np.concatenate([da[k][da[k] > 0], db[k][db[k] > 0]])
        lo = max(pos.min(), hi * 1e-4) if pos.size else 1e-30
        for j, (d, title) in enumerate(((da[k], "refracted rays"), (db[k], "straight lines of sight"))):
            ax = axes[k, j]
            im = ax.imshow(np.where(d > 0, d, np.nan), origin="lower", extent=ext, cmap="inferno",
                           norm=LogNorm(vmin=lo, vmax=hi))
            fig.colorbar(im, ax=ax, label=label, shrink=0.85)
            ax.set_title(f"{freq_label(ref.freqs[k])}: {title}")
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where((da[k] > lo) | (db[k] > lo), da[k] / db[k], np.nan)
        ax = axes[k, 2]
        im = ax.imshow(np.log10(ratio), origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, label="log10(refracted / straight)", shrink=0.85)
        ax.set_title(f"flux ratio {ref.flux_density[k] / max(los.flux_density[k], 1e-300):.3f}")
        for ax in axes[k]:
            ax.plot(np.cos(t), np.sin(t), "k--", lw=0.6, alpha=0.6)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_xlabel(r"X [$R_\odot$]")
            ax.set_ylabel(r"Y [$R_\odot$]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return fig
