# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stellar radio emission: distant observers, spectra, rotational modulation.

For other stars the model coordinates are in stellar radii (grid.length_unit =
R_star in metres, set with ``r_star=`` in the readers) and the star is seen from
parsecs away, so the rays are parallel and the result is mostly a flux density
(mJy), a spectrum and its spectral index, or a light curve over the rotation.
VLBI-resolved images can be shown in milliarcseconds with the figure routines
(units="mas").

Viewing geometry: the stellar rotation axis is the model z axis.  The observer
direction is given by the inclination i (angle between the rotation axis and the
line of sight; i = 90 deg is equator-on) and the rotational phase phi (model
longitude facing the observer):  n = (sin i cos phi, sin i sin phi, cos i).
"""
import math

import numpy as np

from .image import make_radio_image, parse_frequencies

PC = 3.0856775814913673e16          # parsec [m]


def stellar_observer(inclination_deg=90.0, phase_deg=0.0, distance=1.0e4):
    """Observer position vector for inclination i and rotational phase phi
    (direction matters; with distance_pc in make_radio_image the rays are
    parallel and |vector| is irrelevant)."""
    i, p = math.radians(inclination_deg), math.radians(phase_deg)
    return (distance * math.sin(i) * math.cos(p), distance * math.sin(i) * math.sin(p),
            distance * math.cos(i))


def _default_fov(grid, fov):
    if fov is not None:
        return float(fov)
    return 2.0 * float(getattr(grid, "r_outer", 10.0)) / math.sqrt(2.0)


def make_stellar_image(grid, frequencies, inclination_deg=90.0, phase_deg=0.0,
                       distance_pc=10.0, fov=None, npix=128, up=(0.0, 0.0, 1.0), **kw):
    """Synthetic image of a star at ``distance_pc`` seen at inclination and phase.

    fov : full width of the image in stellar radii (default: the largest square
          inside the model domain).  Remaining keywords go to make_radio_image
          (transfer, gaunt, n_workers, no_refraction, ...).  The image object has
          flux_density_mjy and can be plotted in 'mas' with radiosynth.figures."""
    fov = _default_fov(grid, fov)
    kw.setdefault("transfer", "full")
    kw.setdefault("r_integration", float(getattr(grid, "r_outer", fov)))
    return make_radio_image(grid, frequencies, stellar_observer(inclination_deg, phase_deg),
                            fov, fov, npix, npix, up=up, distance_pc=distance_pc, **kw)


def spectral_index(freqs, flux):
    """Local spectral index alpha = dlog S / dlog nu at the geometric mid-points."""
    f = np.asarray(freqs, float)
    s = np.asarray(flux, float)
    alpha = np.diff(np.log(s)) / np.diff(np.log(f))
    return np.sqrt(f[1:] * f[:-1]), alpha


def stellar_spectrum(grid, frequencies, inclination_deg=90.0, phase_deg=0.0, distance_pc=10.0,
                     fov=None, npix=128, plot_path=None, verbose=False, **kw):
    """Flux-density spectrum of a star: returns dict(freqs [Hz], flux_mjy, nu_mid,
    alpha, image).  ``frequencies`` as for make_radio_image (e.g. np.geomspace)."""
    img = make_stellar_image(grid, frequencies, inclination_deg, phase_deg, distance_pc, fov,
                             npix, verbose=verbose, **kw)
    f = img.freqs
    s = img.flux_density_mjy
    nu_mid, alpha = spectral_index(f, s)
    if plot_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        axs[0].loglog(f / 1e9, s, "o-")
        axs[0].set_xlabel("frequency [GHz]")
        axs[0].set_ylabel(f"flux density at {distance_pc:g} pc [mJy]")
        axs[0].grid(alpha=0.3, which="both")
        axs[1].semilogx(nu_mid / 1e9, alpha, "s-")
        axs[1].axhline(0.6, color="gray", ls=":", label="0.6 (ionised wind)")
        axs[1].axhline(2.0, color="gray", ls="--", label="2 (optically thick)")
        axs[1].axhline(-0.1, color="gray", ls="-.", label="-0.1 (optically thin)")
        axs[1].set_xlabel("frequency [GHz]")
        axs[1].set_ylabel(r"spectral index $\alpha$ ($S\propto\nu^\alpha$)")
        axs[1].legend(fontsize=8)
        axs[1].grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
    return dict(freqs=f, flux_mjy=s, nu_mid=nu_mid, alpha=alpha, image=img)


def rotational_modulation(grid, frequency, phases_deg, inclination_deg=90.0, distance_pc=10.0,
                          fov=None, npix=96, plot_path=None, verbose=False, **kw):
    """Flux density over a stellar rotation: returns dict(phases, flux_mjy, images)."""
    phases = np.asarray(list(phases_deg), float)
    imgs, flux = [], []
    for ph in phases:
        img = make_stellar_image(grid, frequency, inclination_deg, ph, distance_pc, fov, npix,
                                 verbose=verbose, **kw)
        imgs.append(img)
        flux.append(img.flux_density_mjy[0])
    flux = np.array(flux)
    if plot_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(phases, flux, "o-")
        ax.set_xlabel("rotational phase [deg]")
        ax.set_ylabel(f"flux density at {distance_pc:g} pc [mJy]")
        ax.set_title(f"{parse_frequencies(frequency)[0] / 1e9:.3g} GHz, i = {inclination_deg:g} deg")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
    return dict(phases=phases, flux_mjy=flux, images=imgs)


# ----------------------------------------------------------------------------
# The analytic free-free wind solution belongs to the bremsstrahlung mechanism;
# also available here (radiosynth.stellar.wind_*).
# ----------------------------------------------------------------------------
from ..bremsstrahlung.wind import wind_optical_depth, wind_flux_density  # noqa: E402,F401
