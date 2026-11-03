# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Demo: synthetic radio images of an analytic corona with a streamer and a
dense CME-like blob, reproducing the kinds of figures in Moschou+2018.

python examples/demo.py   -> figures in examples/output/
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import LogNorm                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import radiosynth as rs                               # noqa: E402
from synthetic_corona import make_model               # noqa: E402

OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
NW = os.cpu_count() or 1
OBS = (-215.0, 0.0, 0.0)          # Earth-like observer on the -x axis, z up
NPIX = 128

# CME blob west of the Sun (image right = -y for this observer), slightly north
grid = make_model(half=4.0, n=201, te=1.5e6, streamer=True,
                  blob=((0.3, -1.9, 0.6), 0.45, 4.0))
print(grid)

# 1. multi-frequency images, BATS-R-US mode (optically thin) and full transfer
t = time.time()
thin = rs.make_radio_image(grid, "42.3MHz, 80MHz, 150MHz, 327MHz, 1GHz, 17GHz", OBS,
                           8, 8, NPIX, NPIX, n_workers=NW)
full = rs.make_radio_image(grid, "42.3MHz, 80MHz, 150MHz, 327MHz, 1GHz, 17GHz", OBS,
                           8, 8, NPIX, NPIX, transfer="full", n_workers=NW, verbose=False)
print(f"images: {time.time() - t:.0f} s")
thin.plot("intensity", path=os.path.join(OUT, "images_thin_intensity.png"))
full.plot("tb", path=os.path.join(OUT, "images_full_Tb.png"), log=False, vmin=0)
thin.save_npz(os.path.join(OUT, "images_thin.npz"))
thin.save_tecplot(os.path.join(OUT, "images_thin.dat"))

# 2. refraction vs straight line of sight (Moschou+2018 Fig. 4 analogue)
f = "80MHz"
ref = rs.make_radio_image(grid, f, OBS, 8, 8, NPIX, NPIX, n_workers=NW, verbose=False)
los = rs.make_radio_image(grid, f, OBS, 8, 8, NPIX, NPIX, n_workers=NW, verbose=False,
                          no_refraction=True)
strict = rs.make_radio_image(grid, f, OBS, 8, 8, NPIX, NPIX, n_workers=NW, verbose=False,
                             strict_batsrus=True)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
ext = [-4, 4, -4, 4]
vmax = max(ref.intensity.max(), los.intensity.max())
for ax, im, title in zip(axs, (ref, los, strict),
                         ("refracted rays (default)", "straight line of sight",
                          "refracted, strict_batsrus=True")):
    h = ax.imshow(im.intensity[0], origin="lower", extent=ext, cmap="inferno",
                  norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax))
    tt = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(tt), np.sin(tt), "w--", lw=0.7)
    ax.set_title(f"80 MHz, {title}\nflux {im.flux_density[0] / 1e-22:.1f} sfu")
    ax.set_xlabel(r"X [$R_\odot$]")
    ax.set_ylabel(r"Y [$R_\odot$]")
    fig.colorbar(h, ax=ax, label=r"$I_\nu$ [W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]", shrink=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "refraction_vs_los_80MHz.png"), dpi=130)
plt.close(fig)

# 3. flux-density spectrum (Moschou+2018 Fig. 1 analogue), quiet corona only
quiet = make_model(half=4.0, n=161, te=1.5e6)
freqs = np.geomspace(20e6, 20e9, 16)
spec = {}
for tr in ("thin", "full"):
    im = rs.make_radio_image(quiet, freqs, OBS, 8, 8, 64, 64, transfer=tr,
                             n_workers=NW, verbose=False)
    spec[tr] = im.flux_density / 1e-22
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(freqs / 1e6, spec["thin"], "o-", label="optically thin  (BATS-R-US: I = ∫κB ds)")
ax.loglog(freqs / 1e6, spec["full"], "s-", label="full transfer  (I = ∫B e$^{-\\tau}$ dτ)")
ax.set_xlabel("frequency [MHz]")
ax.set_ylabel("flux density at 1 AU [sfu]")
ax.set_title("Newkirk corona, $T_e$=1.5 MK, no chromosphere")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "flux_spectrum.png"), dpi=130)
plt.close(fig)
for fq, a, b in zip(freqs, spec["thin"], spec["full"]):
    print(f"{fq / 1e6:10.1f} MHz  thin {a:10.3e} sfu   full {b:10.3e} sfu")
