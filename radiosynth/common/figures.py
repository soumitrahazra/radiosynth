# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Publication-style figures in the manner of Moschou et al. (2018).

* ``figure_compare_observation``  -- Fig. 2 style: synthetic map (log scale, white
  log contours) next to an observed map (e.g. Nobeyama 17 GHz FITS) with the
  synthetic contours and the limb overplotted, both in helioprojective arcsec.
* ``figure_single_map``            -- Fig. 3 style: one frequency, rainbow colour
  map, white log-spaced contours, limb circle, frequency label.
* ``figure_time_series``           -- Fig. 5 style: rows = times, columns =
  frequencies, log images with log-spaced (green) contours, time stamps and one
  shared colour bar.  ``animate_time_series`` writes the animation (GIF/MP4).
* ``make_time_series``             -- run make_radio_image on a list of MHD
  snapshots (files or grids) and keep the images.
* ``observer_for_date``, ``sun_ephemeris`` -- Earth's position (B0, L0, P,
  distance) for a date (sunpy if installed, otherwise Meeus' algorithm).
* ``fit_longitude_offset``         -- find the longitude offset of a model
  frame (e.g. a PLUTO run without solar coordinates) by correlating synthetic
  maps with an observed map.

Coordinates
-----------
radiosynth images are in R_sun in the image plane through the Sun centre.
For an observer at distance D [R_sun] the helioprojective angle of a point at
image coordinate X is  Tx = atan(X / D)  (``rsun_to_arcsec``).  If the model is
in a heliographic frame (z = solar rotation axis; BATS-R-US SC/IH use HGR,
i.e. Carrington) and the observer is placed with ``earth_observer(B0, L0)``,
the image x axis points to solar West and y to solar North, the same
orientation as standard solar images (e.g. Nobeyama, MWA, LOFAR).
"""
import math

import numpy as np

ARCSEC_PER_RAD = 180.0 / math.pi * 3600.0


# ----------------------------------------------------------------------------
# coordinates
# ----------------------------------------------------------------------------
def earth_observer(b0_deg=0.0, l0_deg=0.0, distance_rsun=215.03):
    """Observer position in a heliographic (Carrington/Stonyhurst) model frame.

    b0_deg : heliographic latitude of the observer (solar B0 angle)
    l0_deg : heliographic longitude of the observer (Carrington L0 for an
             HGR/Carrington model, 0 for a Stonyhurst model)
    With up=(0,0,1) the image then has solar North up and West to the right.
    (sunpy: ``sunpy.coordinates.sun.B0(t)``, ``sunpy.coordinates.sun.L0(t)``,
    ``sunpy.coordinates.get_earth(t).radius`` give the values for a date.)
    """
    b, l_ = math.radians(b0_deg), math.radians(l0_deg)
    return (distance_rsun * math.cos(b) * math.cos(l_),
            distance_rsun * math.cos(b) * math.sin(l_),
            distance_rsun * math.sin(b))


def _julian_day(time):
    """Julian day from a datetime, an ISO string ('2014-11-01T03:00') or a number
    (taken as a Julian day).  UTC is used for TT (error < 0.01 deg in L0)."""
    import datetime as _dt
    if isinstance(time, (int, float, np.floating)):
        return float(time)
    if isinstance(time, str):
        time = _dt.datetime.fromisoformat(time.replace("Z", "").replace(" ", "T"))
    if time.tzinfo is not None:
        time = time.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    j2000 = _dt.datetime(2000, 1, 1, 12, 0, 0)
    return 2451545.0 + (time - j2000).total_seconds() / 86400.0


def sun_ephemeris(time, use_sunpy=True):
    """Solar B0, Carrington L0, position angle P [deg] and the Sun-Earth
    distance [R_sun] at ``time``.

    Uses sunpy if it is installed; otherwise the algorithm of Meeus
    (Astronomical Algorithms, 2nd ed., ch. 25 and 29), accurate to about
    0.01 deg, which is far below the resolution of radio images."""
    from .constants import AU, R_SUN
    if use_sunpy:
        try:
            from sunpy.coordinates import sun
            import astropy.units as u
            from astropy.time import Time
            t = Time(time) if not isinstance(time, (int, float)) else Time(time, format="jd")
            return dict(B0=sun.B0(t).deg, L0=sun.L0(t).deg, P=sun.P(t).deg,
                        distance_rsun=float(sun.earth_distance(t).to(u.m).value / R_SUN),
                        source="sunpy")
        except ImportError:
            pass
    jd = _julian_day(time)
    T = (jd - 2451545.0) / 36525.0
    rad = math.radians
    # Sun's apparent longitude (Meeus ch. 25, low precision)
    Ls = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
    C = ((1.914602 - 0.004817 * T - 0.000014 * T**2) * math.sin(rad(M))
         + (0.019993 - 0.000101 * T) * math.sin(rad(2 * M)) + 0.000289 * math.sin(rad(3 * M)))
    true_long = Ls + C
    omega = 125.04 - 1934.136 * T
    lam = true_long - 0.00569 - 0.00478 * math.sin(rad(omega))      # apparent longitude
    e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T**2
    R_au = 1.000001018 * (1 - e**2) / (1 + e * math.cos(rad(M + C)))
    eps = (23.0 + 26.0 / 60 + 21.448 / 3600 - (46.815 * T) / 3600
           + 0.00256 * math.cos(rad(omega)))                         # true obliquity
    # physical ephemeris (Meeus ch. 29)
    theta = (jd - 2398220.0) * 360.0 / 25.38
    I = 7.25
    K = 73.6667 + 1.3958333 * (jd - 2396758.0) / 36525.0
    lamp = lam                                       # nutation in longitude neglected (<0.005 deg)
    x = math.degrees(math.atan(-math.cos(rad(lamp)) * math.tan(rad(eps))))
    y = math.degrees(math.atan(-math.cos(rad(lam - K)) * math.tan(rad(I))))
    P = x + y
    B0 = math.degrees(math.asin(math.sin(rad(lam - K)) * math.sin(rad(I))))
    eta = math.degrees(math.atan2(-math.sin(rad(lam - K)) * math.cos(rad(I)),
                                  -math.cos(rad(lam - K))))
    L0 = (eta - theta) % 360.0
    return dict(B0=B0, L0=L0, P=P, distance_rsun=R_au * AU / R_SUN, source="Meeus")


def observer_for_date(time, phi_offset_deg=0.0, frame="carrington", distance_rsun=None,
                      verbose=True):
    """Observer position (Earth) in the model frame for a given date.

    frame = 'carrington' : model longitude phi = Carrington longitude (data-driven
                           runs from synoptic maps; BATS-R-US HGR).  Observer at
                           longitude L0(time) + phi_offset_deg.
    frame = 'stonyhurst' : model longitude 0 faces the Earth.  Observer at
                           longitude phi_offset_deg.
    phi_offset_deg : extra longitude shift, e.g. when the setup maps the
                     magnetogram with an offset (find it with fit_longitude_offset).
    The latitude is B0(time); the distance is the true Sun-Earth distance
    unless distance_rsun is given.  Use with up=(0, 0, 1) (solar north = +z)."""
    eph = sun_ephemeris(time)
    lon = (eph["L0"] if frame.lower().startswith("carr") else 0.0) + phi_offset_deg
    D = distance_rsun or eph["distance_rsun"]
    if verbose:
        print(f"radiosynth: {time}: B0 = {eph['B0']:+.3f} deg, L0 = {eph['L0']:.3f} deg, "
              f"P = {eph['P']:+.2f} deg, D = {eph['distance_rsun']:.2f} R_sun ({eph['source']}); "
              f"observer longitude in model = {lon % 360:.2f} deg")
    return earth_observer(eph["B0"], lon, D)


def rsun_to_arcsec(x_rsun, distance_rsun):
    """Image-plane coordinate [R_sun] -> helioprojective angle [arcsec]."""
    return np.arctan(np.asarray(x_rsun, float) / distance_rsun) * ARCSEC_PER_RAD


def _dist(img):
    """Observer distance in coordinate units (true distance for stellar images)."""
    return float(getattr(img, "distance", np.linalg.norm(img.observer)))


def _unit_label(img, units):
    if units == "arcsec":
        return "[arcsec]"
    if units == "mas":
        return "[mas]"
    from .constants import R_SUN
    L = float(getattr(img, "length_unit", R_SUN))
    return r"[$R_\odot$]" if abs(L / R_SUN - 1) < 1e-6 else r"[$R_\ast$]"


def image_extent(img, units="rsun"):
    """[left, right, bottom, top] of the pixel edges in coordinate units ('rsun':
    R_sun or R_star), 'arcsec' or 'mas' (milliarcsec, for stars / VLBI)."""
    dx = img.x[1] - img.x[0]
    dy = img.y[1] - img.y[0]
    e = np.array([img.x[0] - dx / 2, img.x[-1] + dx / 2, img.y[0] - dy / 2, img.y[-1] + dy / 2])
    if units in ("arcsec", "mas"):
        e = rsun_to_arcsec(e, _dist(img)) * (1000.0 if units == "mas" else 1.0)
    return list(e)


def limb_radius(img, units="rsun"):
    """Apparent photospheric radius in the chosen units."""
    D = _dist(img)
    if units == "rsun":
        return 1.0
    return math.asin(1.0 / D) * ARCSEC_PER_RAD * (1000.0 if units == "mas" else 1.0)


def _freq_index(img, freq):
    if freq is None:
        return 0
    if isinstance(freq, (int, np.integer)) and freq < len(img.freqs):
        return int(freq)
    from .image import parse_frequencies
    f = parse_frequencies(freq)[0] if isinstance(freq, str) else float(freq)
    return int(np.argmin(np.abs(img.freqs - f)))


def _data(img, k, quantity):
    return (img.brightness_temperature if quantity == "tb" else img.intensity)[k]


def _label(quantity):
    return (r"$T_b$ [K]" if quantity == "tb"
            else r"$I_\nu$ [W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]")


def smooth_to_beam(data, pixel_arcsec, fwhm_arcsec):
    """Convolve a map with a circular Gaussian beam (FWHM in arcsec)."""
    from scipy.ndimage import gaussian_filter
    if not fwhm_arcsec:
        return data
    sig = fwhm_arcsec / (2.0 * math.sqrt(2.0 * math.log(2.0))) / pixel_arcsec
    return gaussian_filter(data, sig)


def log_levels(data, n=10, decades=None, vmin=None, vmax=None):
    """n logarithmically spaced contour levels between vmin and vmax
    (default: the positive range of data, limited to ``decades`` if given)."""
    pos = data[np.isfinite(data) & (data > 0)]
    if pos.size == 0:
        return np.array([])
    hi = vmax if vmax is not None else pos.max()
    lo = vmin if vmin is not None else (hi * 10.0**(-decades) if decades else pos.min())
    return np.geomspace(lo, hi, n + 1)[:-1] if hi > lo else np.array([lo])


def decade_levels(data, lo_exp=None, hi_exp=None):
    """Contours at every power of ten inside the data range (Moschou Fig. 2)."""
    pos = data[np.isfinite(data) & (data > 0)]
    if pos.size == 0:
        return np.array([])
    a = math.ceil(math.log10(pos.min())) if lo_exp is None else lo_exp
    b = math.floor(math.log10(pos.max())) if hi_exp is None else hi_exp
    return 10.0**np.arange(a, b + 1)


# ----------------------------------------------------------------------------
# observations
# ----------------------------------------------------------------------------
def read_fits_map(path, hdu=0):
    """Read a solar FITS image (e.g. Nobeyama NoRH, MWA, LOFAR) and return
    (data, extent_arcsec) assuming a helioprojective, solar-north-up image with a
    linear WCS (CRPIX/CRVAL/CDELT in arcsec; CROTA/rotation is applied if present).

    For maps that are not north-up, use sunpy:  ``m = sunpy.map.Map(path).rotate()``
    and pass ``(m.data, [Tx0, Tx1, Ty0, Ty1])`` instead."""
    from .fitsio import read_fits
    data, h = read_fits(path, hdu)
    data = np.squeeze(data)
    ny, nx = data.shape[-2:]
    scale = {"deg": 3600.0, "arcmin": 60.0, "arcsec": 1.0}

    def ax(i, n):
        cu = str(h.get(f"CUNIT{i}", "arcsec")).strip().lower()
        f = scale.get(cu, 1.0)
        crpix, crval = h.get(f"CRPIX{i}", (n + 1) / 2.0), h.get(f"CRVAL{i}", 0.0)
        cdelt = h.get(f"CDELT{i}", 1.0)
        return (np.array([0.5, n + 0.5]) - crpix) * cdelt * f + crval * f

    ex, ey = ax(1, nx), ax(2, ny)
    rot = float(h.get("CROTA2", h.get("CROTA", 0.0)) or 0.0)
    if abs(rot) > 0.5:
        from scipy.ndimage import rotate
        data = rotate(data, -rot, reshape=False, order=1, mode="nearest")
    return data, [ex[0], ex[1], ey[0], ey[1]]


# ----------------------------------------------------------------------------
# Fig. 3 style
# ----------------------------------------------------------------------------
def figure_single_map(img, freq=None, quantity="intensity", units="rsun", cmap="jet",
                      log=False, n_contours=8, contour_decades=4, contour_color="white",
                      limb_color="black", beam_fwhm_arcsec=None, ax=None, path=None,
                      vmin=None, vmax=None, title=None):
    """One-frequency map in the style of Moschou et al. (2018) Fig. 3: rainbow
    colour map (linear by default), white log-spaced intensity contours, the
    photospheric limb and an 'f = ... MHz' label."""
    import matplotlib
    if path is not None and ax is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from .image import freq_label
    k = _freq_index(img, freq)
    d = _data(img, k, quantity).copy()
    ext = image_extent(img, units)
    if beam_fwhm_arcsec:
        pix = rsun_to_arcsec(img.x[1] - img.x[0], _dist(img))
        d = smooth_to_beam(d, pix, beam_fwhm_arcsec)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
    else:
        fig = ax.figure
    pos = d[d > 0]
    if log and pos.size:
        hi = vmax or pos.max()
        norm = LogNorm(vmin=vmin or hi * 1e-4, vmax=hi)
        im = ax.imshow(np.where(d > 0, d, np.nan), origin="lower", extent=ext, cmap=cmap, norm=norm)
    else:
        im = ax.imshow(d, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax)
    lev = log_levels(d, n_contours, contour_decades)
    if lev.size:
        X = np.linspace(ext[0], ext[1], d.shape[1])
        Y = np.linspace(ext[2], ext[3], d.shape[0])
        ax.contour(X, Y, d, levels=lev, colors=contour_color, linewidths=0.8)
    t = np.linspace(0, 2 * np.pi, 400)
    R = limb_radius(img, units)
    ax.plot(R * np.cos(t), R * np.sin(t), color=limb_color, lw=1.4)
    ax.text(0.03, 0.97, title or f"f = {freq_label(img.freqs[k])}", transform=ax.transAxes,
            ha="left", va="top", color="white", fontsize=13, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none"))
    un = _unit_label(img, units)
    ax.set_xlabel(f"X {un}")
    ax.set_ylabel(f"Y {un}")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label=_label(quantity), shrink=0.9)
    if own:
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=150)
            plt.close(fig)
    return fig


# ----------------------------------------------------------------------------
# Fig. 2 style
# ----------------------------------------------------------------------------
def figure_compare_observation(img, obs, obs_extent=None, freq=None, quantity="intensity",
                               syn_cmap="YlOrRd_r", obs_cmap="hot", beam_fwhm_arcsec=None,
                               levels="decades", n_contours=6, contour_color="white",
                               syn_limb_color="black", obs_limb_color="lime",
                               fov_arcsec=None, obs_label="Observation", path=None,
                               obs_vmin=None, obs_vmax=None):
    """Synthetic map next to an observed map, with the synthetic contours drawn
    on the observation (Moschou et al. 2018, Fig. 2).

    obs        : path to a FITS file, or a 2D array (then give obs_extent =
                 [Tx_min, Tx_max, Ty_min, Ty_max] in arcsec, solar north up)
    levels     : 'decades' (one contour per power of ten), 'log' (n_contours
                 log-spaced levels) or an explicit list of levels
    beam_fwhm_arcsec : smooth the synthetic map to the instrument beam before
                 contouring (e.g. ~10 arcsec for NoRH at 17 GHz)
    """
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from .image import freq_label
    k = _freq_index(img, freq)
    syn = _data(img, k, quantity).copy()
    ext = image_extent(img, "arcsec")
    pix = rsun_to_arcsec(img.x[1] - img.x[0], _dist(img))
    syn_s = smooth_to_beam(syn, pix, beam_fwhm_arcsec)
    if isinstance(obs, str):
        odata, oext = read_fits_map(obs)
    else:
        odata, oext = np.asarray(obs, float), list(obs_extent)
    if isinstance(levels, str):
        lev = decade_levels(syn_s) if levels == "decades" else log_levels(syn_s, n_contours, 5)
    else:
        lev = np.asarray(levels)
    X = np.linspace(ext[0], ext[1], syn.shape[1])
    Y = np.linspace(ext[2], ext[3], syn.shape[0])
    R = limb_radius(img, "arcsec")
    t = np.linspace(0, 2 * np.pi, 400)

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.8))
    pos = syn_s[syn_s > 0]
    norm = LogNorm(vmin=max(pos.min(), pos.max() * 1e-5), vmax=pos.max()) if pos.size else None
    im0 = axs[0].imshow(np.where(syn_s > 0, syn_s, np.nan), origin="lower", extent=ext,
                        cmap=syn_cmap, norm=norm)
    axs[0].contour(X, Y, syn_s, levels=lev, colors=contour_color, linewidths=0.8)
    axs[0].plot(R * np.cos(t), R * np.sin(t), color=syn_limb_color, lw=1.4)
    axs[0].set_title(f"Synthetic, {freq_label(img.freqs[k])}")
    cb = fig.colorbar(im0, ax=axs[0], shrink=0.88, label=_label(quantity))
    for lv in lev:
        cb.ax.axhline(lv, color=contour_color, lw=0.8)

    im1 = axs[1].imshow(odata, origin="lower", extent=oext, cmap=obs_cmap,
                        vmin=obs_vmin, vmax=obs_vmax)
    axs[1].contour(X, Y, syn_s, levels=lev, colors=contour_color, linewidths=0.8)
    axs[1].plot(R * np.cos(t), R * np.sin(t), color=obs_limb_color, lw=1.4)
    axs[1].set_title(f"{obs_label} with synthetic contours")
    fig.colorbar(im1, ax=axs[1], shrink=0.88, label="observed (instrument units)")
    lim = fov_arcsec or max(abs(ext[0]), abs(ext[1]))
    for a in axs:
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect("equal")
        a.set_xlabel("Solar X [arcsec]")
        a.set_ylabel("Solar Y [arcsec]")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


# ----------------------------------------------------------------------------
# Fig. 5 style
# ----------------------------------------------------------------------------
def make_time_series(snapshots, frequencies, observer=(-215.0, 0.0, 0.0), x_size=6.0,
                     y_size=6.0, nx=128, ny=128, loader=None, save_prefix=None, **kw):
    """Make one RadioImage per MHD snapshot.

    snapshots : list (or iterable) of file names / PLUTO directories (read
                with radiosynth.load_mhd) or of grids; with ``loader`` given,
                any items, each turned into a grid by ``loader(item)`` (e.g.
                PLUTO output numbers: loader=lambda n: load_pluto(d, nout=n))
    save_prefix : if given, each image is saved as f'{save_prefix}_{i:03d}.npz'
    Remaining keywords go to make_radio_image (transfer, n_workers, ...)."""
    from .image import make_radio_image
    from .readers import load_mhd
    out = []
    for i, s in enumerate(snapshots):
        if loader is not None:
            g = loader(s)
        elif isinstance(s, str):
            g = load_mhd(s)
        else:
            g = s
        img = make_radio_image(g, frequencies, observer, x_size, y_size, nx, ny, **kw)
        if save_prefix:
            img.save_npz(f"{save_prefix}_{i:03d}.npz")
        out.append(img)
    return out


def figure_time_series(images, times=None, freqs=None, quantity="intensity", units="rsun",
                       cmap="gray", n_contours=10, contour_decades=6, contour_color="lime",
                       vmin=None, vmax=None, time_fmt="Time: {:.1f} min", fov=None,
                       limb=False, path=None, panel_size=3.2):
    """Grid of images, rows = times, columns = frequencies, common logarithmic
    colour scale, n log-spaced contours per panel (Moschou et al. 2018, Fig. 5).

    images : list of RadioImage (one per time), all with the same frequencies
    times  : list of times (labels) or None
    freqs  : frequencies to show (strings, Hz or indices); default all
    """
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from .image import freq_label
    ref = images[0]
    ks = list(range(len(ref.freqs))) if freqs is None else [_freq_index(ref, f) for f in
                                                              (freqs if isinstance(freqs, (list, tuple)) else [freqs])]
    nr, nc = len(images), len(ks)
    alld = np.concatenate([_data(im, k, quantity).ravel() for im in images for k in ks])
    pos = alld[alld > 0]
    hi = vmax or pos.max()
    lo = vmin or hi * 10.0**(-contour_decades)
    norm = LogNorm(vmin=lo, vmax=hi)
    lev = np.geomspace(lo, hi, n_contours + 1)[:-1]
    fig, axs = plt.subplots(nr, nc, figsize=(panel_size * nc + 1.2, panel_size * nr),
                            squeeze=False, gridspec_kw=dict(wspace=0.02, hspace=0.02))
    for r, im in enumerate(images):
        ext = image_extent(im, units)
        for c, k in enumerate(ks):
            a = axs[r, c]
            d = _data(im, k, quantity)
            h = a.imshow(np.where(d > 0, d, np.nan), origin="lower", extent=ext, cmap=cmap, norm=norm)
            a.set_facecolor("black")
            X = np.linspace(ext[0], ext[1], d.shape[1])
            Y = np.linspace(ext[2], ext[3], d.shape[0])
            a.contour(X, Y, d, levels=lev, colors=contour_color, linewidths=0.6)
            if limb:
                tt = np.linspace(0, 2 * np.pi, 300)
                R = limb_radius(im, units)
                a.plot(R * np.cos(tt), R * np.sin(tt), "w--", lw=0.6)
            a.text(0.5, 0.97, freq_label(im.freqs[k]), transform=a.transAxes, ha="center",
                   va="top", color="white", fontsize=10, fontweight="bold")
            if c == 0 and times is not None:
                a.text(0.04, 0.04, time_fmt.format(times[r]), transform=a.transAxes,
                       color="white", fontsize=10)
            if fov:
                a.set_xlim(-fov, fov)
                a.set_ylim(-fov, fov)
            a.set_xticks([])
            a.set_yticks([])
    cb = fig.colorbar(h, ax=axs, shrink=0.9, pad=0.01, label=_label(quantity))
    for lv in lev:
        cb.ax.axhline(lv, color=contour_color, lw=0.6)
    if path is not None:
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    return fig


def animate_time_series(images, path, times=None, freq=None, quantity="intensity",
                        units="rsun", cmap="gray", n_contours=10, contour_decades=6,
                        contour_color="lime", fps=4, time_fmt="Time: {:.1f} min"):
    """Write an animation (GIF via Pillow, or MP4 if path ends in .mp4 and ffmpeg
    is installed) of one frequency through a list of RadioImages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import LogNorm
    from .image import freq_label
    k = _freq_index(images[0], freq)
    alld = np.concatenate([_data(im, k, quantity).ravel() for im in images])
    hi = alld.max()
    lo = hi * 10.0**(-contour_decades)
    lev = np.geomspace(lo, hi, n_contours + 1)[:-1]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ext = image_extent(images[0], units)

    def draw(i):
        ax.clear()
        d = _data(images[i], k, quantity)
        ax.imshow(np.where(d > 0, d, np.nan), origin="lower", extent=ext, cmap=cmap,
                  norm=LogNorm(vmin=lo, vmax=hi))
        ax.set_facecolor("black")
        X = np.linspace(ext[0], ext[1], d.shape[1])
        Y = np.linspace(ext[2], ext[3], d.shape[0])
        ax.contour(X, Y, d, levels=lev, colors=contour_color, linewidths=0.6)
        ax.set_title(freq_label(images[i].freqs[k]))
        if times is not None:
            ax.text(0.04, 0.04, time_fmt.format(times[i]), transform=ax.transAxes, color="white")
        ax.set_xticks([])
        ax.set_yticks([])
        return []

    ani = animation.FuncAnimation(fig, draw, frames=len(images), blit=False)
    writer = animation.FFMpegWriter(fps=fps) if path.endswith(".mp4") else animation.PillowWriter(fps=fps)
    ani.save(path, writer=writer)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# orientation of models without solar coordinates
# ----------------------------------------------------------------------------
def _regrid_to_obs(img, k, quantity, beam_fwhm_arcsec, tx, ty):
    """Synthetic map (optionally beam-smoothed) interpolated to observed pixel
    centres tx, ty [arcsec]."""
    from scipy.interpolate import RegularGridInterpolator
    D = _dist(img)
    d = _data(img, k, quantity)
    xs = rsun_to_arcsec(img.x, D)
    ys = rsun_to_arcsec(img.y, D)
    if beam_fwhm_arcsec:
        d = smooth_to_beam(d, xs[1] - xs[0], beam_fwhm_arcsec)
    f = RegularGridInterpolator((ys, xs), d, bounds_error=False, fill_value=np.nan)
    TY, TX = np.meshgrid(ty, tx, indexing="ij")
    return f(np.column_stack([TY.ravel(), TX.ravel()])).reshape(TY.shape)


def fit_longitude_offset(grid, obs, frequency, b0_deg, l0_deg, offsets=None, obs_extent=None,
                         x_size=2.6, y_size=None, npix=128, distance_rsun=215.0,
                         up=(0.0, 0.0, 1.0), beam_fwhm_arcsec=None, quantity="tb",
                         use_log=True, r_mask=1.1, refine=True, n_workers=1,
                         plot_path=None, verbose=True, **image_kw):
    """Find the model longitude offset that best matches an observed map.

    For each trial offset the observer is placed at heliographic latitude b0_deg and
    longitude l0_deg + offset, a synthetic map is computed, smoothed to the beam,
    interpolated onto the observed pixels and correlated with the observation
    (Pearson correlation of log10 values inside r < r_mask apparent solar radii).

    obs        : FITS file or 2D array (then give obs_extent in arcsec), solar north up
    offsets    : trial offsets [deg]; default -60..60 in 5-deg steps
    refine     : refine the best offset with a parabola through the three best points
    Returns dict(offsets, corr, best, best_refined, image_best).
    Then use earth_observer(b0_deg, l0_deg + best_refined) (or observer_for_date(...,
    phi_offset_deg=best_refined)) for all other frequencies and times."""
    from .image import make_radio_image
    if offsets is None:
        offsets = np.arange(-60.0, 60.1, 5.0)
    offsets = np.asarray(offsets, float)
    if isinstance(obs, str):
        odata, oext = read_fits_map(obs)
    else:
        odata, oext = np.asarray(obs, float), list(obs_extent)
    ny, nx = odata.shape
    tx = oext[0] + (np.arange(nx) + 0.5) * (oext[1] - oext[0]) / nx
    ty = oext[2] + (np.arange(ny) + 0.5) * (oext[3] - oext[2]) / ny
    TY, TX = np.meshgrid(ty, tx, indexing="ij")
    rlimb = math.asin(1.0 / distance_rsun) * ARCSEC_PER_RAD
    mask = (np.hypot(TX, TY) < r_mask * rlimb) & np.isfinite(odata)
    corr = np.full(offsets.size, np.nan)
    best_img, best_c = None, -np.inf
    for i, off in enumerate(offsets):
        o = earth_observer(b0_deg, l0_deg + off, distance_rsun)
        img = make_radio_image(grid, frequency, o, x_size, y_size or x_size, npix, npix,
                               up=up, n_workers=n_workers, verbose=False, **image_kw)
        syn = _regrid_to_obs(img, 0, quantity, beam_fwhm_arcsec, tx, ty)
        m = mask & np.isfinite(syn)
        a, b = syn[m], odata[m]
        if use_log:
            good = (a > 0) & (b > 0)
            a, b = np.log10(a[good]), np.log10(b[good])
        if a.size > 10:
            corr[i] = np.corrcoef(a, b)[0, 1]
        if verbose:
            print(f"  offset {off:+7.2f} deg: correlation {corr[i]:.4f}")
        if np.isfinite(corr[i]) and corr[i] > best_c:
            best_c, best_img = corr[i], img
    ib = int(np.nanargmax(corr))
    best = float(offsets[ib])
    best_ref = best
    if refine and 0 < ib < offsets.size - 1 and np.all(np.isfinite(corr[ib - 1:ib + 2])):
        x3, y3 = offsets[ib - 1:ib + 2], corr[ib - 1:ib + 2]
        a2, a1, _ = np.polyfit(x3, y3, 2)
        if a2 < 0:
            best_ref = float(np.clip(-a1 / (2 * a2), x3[0], x3[-1]))
    if verbose:
        print(f"radiosynth: best longitude offset {best:+.2f} deg (refined {best_ref:+.2f} deg), "
              f"correlation {corr[ib]:.4f}")
    if plot_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(offsets, corr, "o-")
        ax.axvline(best_ref, color="C3", ls="--", label=f"best {best_ref:+.1f} deg")
        ax.set_xlabel("longitude offset [deg]")
        ax.set_ylabel("correlation with observation")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
    return dict(offsets=offsets, corr=corr, best=best, best_refined=best_ref,
                image_best=best_img)


def compare_with_observation(img, obs, freq=None, quantity="tb", beam_fwhm_arcsec=None,
                             obs_extent=None, r_max=3.0, n_bins=30, sector_halfwidth=30.0,
                             plot_path=None):
    """Quantitative comparison of a synthetic map with an observed map.

    The synthetic map is smoothed to the beam and interpolated onto the observed
    pixels (helioprojective arcsec, solar north up).  Returns a dict with
      syn, obs, tx, ty, r     : maps on the observed pixels; r in apparent solar radii
      mean_disk_obs/syn       : mean value for r < 1
      mean_limb_obs/syn       : mean value for 1 <= r < 1.3
      corr_log                : Pearson correlation of log10 values for r < 1.3
      ratio_disk              : mean_disk_syn / mean_disk_obs
      r_bins, prof_obs, prof_syn : azimuthally averaged radial profiles
      r_half_obs/syn          : outermost radius where the profile is >= half its
                                maximum (size of the radio Sun)
      prof_sector_obs/syn     : radial profiles in the N, E, S, W sectors
                                (+- sector_halfwidth deg of position angle)
      r_half_sector_obs/syn   : radio-Sun size in each sector
      elongation_obs/syn      : equatorial (E,W) / polar (N,S) size; the main
                                measure for axisymmetric (2.5D) models
    plot_path : optional figure with both radial profiles and the residual map.
    """
    k = _freq_index(img, freq)
    if isinstance(obs, str):
        odata, oext = read_fits_map(obs)
    else:
        odata, oext = np.asarray(obs, float), list(obs_extent)
    ny, nx = odata.shape
    tx = oext[0] + (np.arange(nx) + 0.5) * (oext[1] - oext[0]) / nx
    ty = oext[2] + (np.arange(ny) + 0.5) * (oext[3] - oext[2]) / ny
    syn = _regrid_to_obs(img, k, quantity, beam_fwhm_arcsec, tx, ty)
    TY, TX = np.meshgrid(ty, tx, indexing="ij")
    rlimb = limb_radius(img, "arcsec")
    r = np.hypot(TX, TY) / rlimb
    ok = np.isfinite(odata) & np.isfinite(syn)
    disk = ok & (r < 1.0)
    limb = ok & (r >= 1.0) & (r < 1.3)
    inner = ok & (r < 1.3) & (odata > 0) & (syn > 0)
    edges = np.linspace(0.0, r_max, n_bins + 1)

    def prof(m):
        return np.array([np.nanmean(m[ok & (r >= a) & (r < b)]) if np.any(ok & (r >= a) & (r < b))
                         else np.nan for a, b in zip(edges[:-1], edges[1:])])

    def r_half(p):
        """Outermost radius where the profile is still >= half its maximum
        (size of the radio Sun; robust to limb brightening and central dips)."""
        rc = 0.5 * (edges[1:] + edges[:-1])
        half = 0.5 * np.nanmax(p)
        above = np.where(p >= half)[0]
        if above.size == 0 or above[-1] + 1 >= p.size:
            return np.nan
        j = above[-1]
        return float(np.interp(half, [p[j + 1], p[j]], [rc[j + 1], rc[j]]))

    po, ps = prof(odata), prof(syn)
    # position angle from solar north, counter-clockwise (towards East = -X)
    pa = np.degrees(np.arctan2(-TX, TY)) % 360.0
    sectors = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}

    def prof_sector(m, c):
        dpa = np.abs((pa - c + 180.0) % 360.0 - 180.0)
        sel = ok & (dpa <= sector_halfwidth)
        return np.array([np.nanmean(m[sel & (r >= a) & (r < b)]) if np.any(sel & (r >= a) & (r < b))
                         else np.nan for a, b in zip(edges[:-1], edges[1:])])
    res = dict(syn=syn, obs=odata, tx=tx, ty=ty, r=r,
               mean_disk_obs=float(np.nanmean(odata[disk])), mean_disk_syn=float(np.nanmean(syn[disk])),
               mean_limb_obs=float(np.nanmean(odata[limb])), mean_limb_syn=float(np.nanmean(syn[limb])),
               corr_log=float(np.corrcoef(np.log10(odata[inner]), np.log10(syn[inner]))[0, 1])
               if inner.sum() > 10 else np.nan,
               r_bins=0.5 * (edges[1:] + edges[:-1]), prof_obs=po, prof_syn=ps,
               r_half_obs=r_half(po), r_half_syn=r_half(ps))
    res["ratio_disk"] = res["mean_disk_syn"] / res["mean_disk_obs"]
    for tag, m in (("obs", odata), ("syn", syn)):
        sp = {k_: prof_sector(m, c) for k_, c in sectors.items()}
        rh = {k_: r_half(v) for k_, v in sp.items()}
        res[f"prof_sector_{tag}"] = sp
        res[f"r_half_sector_{tag}"] = rh
        res[f"elongation_{tag}"] = (0.5 * (rh["E"] + rh["W"])) / (0.5 * (rh["N"] + rh["S"]))
    if plot_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
        rb = res["r_bins"]
        for tag, ls in (("obs", "-"), ("syn", "--")):
            sp = res[f"prof_sector_{tag}"]
            axs[2].plot(rb, 0.5 * (sp["E"] + sp["W"]), "C0" + ls, label=f"equator ({tag})")
            axs[2].plot(rb, 0.5 * (sp["N"] + sp["S"]), "C3" + ls, label=f"poles ({tag})")
        axs[2].set_xlabel(r"r [apparent $R_\odot$]")
        axs[2].set_ylabel(_label(quantity))
        axs[2].set_title(f"equator vs poles (elongation obs {res['elongation_obs']:.2f}, "
                         f"syn {res['elongation_syn']:.2f})", fontsize=10)
        axs[2].legend(fontsize=8)
        axs[2].grid(alpha=0.3)
        axs[0].plot(res["r_bins"], po, "o-", label="observation")
        axs[0].plot(res["r_bins"], ps, "s-", label="synthetic")
        axs[0].set_xlabel(r"r [apparent $R_\odot$]")
        axs[0].set_ylabel(_label(quantity))
        axs[0].legend()
        axs[0].grid(alpha=0.3)
        axs[0].set_title(f"radial profiles (r_half: obs {res['r_half_obs']:.2f}, "
                         f"syn {res['r_half_syn']:.2f})", fontsize=10)
        resid = syn - odata
        v = np.nanpercentile(np.abs(resid), 99)
        im = axs[1].imshow(resid, origin="lower", extent=oext, cmap="RdBu_r", vmin=-v, vmax=v)
        t = np.linspace(0, 2 * np.pi, 300)
        axs[1].plot(rlimb * np.cos(t), rlimb * np.sin(t), "k-", lw=0.8)
        axs[1].set_title(f"synthetic - observed (corr_log {res['corr_log']:.3f})", fontsize=10)
        axs[1].set_xlabel("Solar X [arcsec]")
        axs[1].set_ylabel("Solar Y [arcsec]")
        fig.colorbar(im, ax=axs[1], shrink=0.85)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
    return res
