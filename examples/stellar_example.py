# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stellar radio emission with radiosynth: an active star at 10 pc.

Model (analytic, in units of the stellar radius R_star = 0.8 R_sun): a hot,
dense corona (n ~ r^-6 near the surface), an equatorial streamer belt, a hot
active region at longitude 0, and an isothermal wind (n ~ r^-2) further out.
With a real model use rs.load_mhd(file, r_star=0.8) / rs.load_pluto(dir, r_star=0.8).

python examples/stellar_example.py  -> examples/output/stellar_*.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import radiosynth as rs                                    # noqa: E402
from radiosynth import stellar as st                       # noqa: E402
from radiosynth import figures as rf                       # noqa: E402
from radiosynth.constants import R_SUN                     # noqa: E402

OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
NW = os.cpu_count() or 1
R_STAR = 0.8                                               # R_sun
D_PC = 10.0

x = np.linspace(-8, 8, 201)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
R = np.maximum(np.sqrt(X**2 + Y**2 + Z**2), 0.98)
lat = np.arcsin(np.clip(Z / R, -1, 1))
ne = 3e16 * R**-6 + 2e14 * R**-2                           # corona + wind [m^-3]
ne *= 1 + 3 * np.exp(-(lat / 0.25)**2)                     # streamer belt
spot = np.exp(-((X - 1.2)**2 + Y**2 + (Z - 0.3)**2) / 0.25**2)
ne *= 1 + 8 * spot
te = 3e6 + 1.2e7 * spot                                    # 3 MK corona, 15 MK active region
grid = rs.MHDGrid(x, x, x, ne, te, length_unit=R_STAR * R_SUN)

# image at 5 GHz, in milliarcseconds, equator-on, active region facing us
img = st.make_stellar_image(grid, "1.4GHz, 5GHz, 15GHz", inclination_deg=70, phase_deg=0,
                            distance_pc=D_PC, fov=12, npix=160, gaunt="dulk", n_workers=NW)
for f in ("1.4GHz", "5GHz", "15GHz"):
    rf.figure_single_map(img, f, quantity="tb", units="mas", cmap="inferno", log=False,
                         n_contours=6, contour_decades=2, limb_color="white",
                         path=os.path.join(OUT, f"stellar_image_{f}.png"))
print("flux densities [mJy]:", dict(zip(["1.4GHz", "5GHz", "15GHz"], np.round(img.flux_density_mjy, 4))))

# spectrum and spectral index
sp = st.stellar_spectrum(grid, np.geomspace(0.5e9, 50e9, 12), inclination_deg=70,
                         distance_pc=D_PC, fov=15, npix=128, gaunt="dulk", n_workers=NW,
                         plot_path=os.path.join(OUT, "stellar_example_spectrum.png"))
print("spectral index:", np.round(sp["alpha"], 2))

# rotational modulation at 5 GHz
lc = st.rotational_modulation(grid, "5GHz", range(0, 360, 20), inclination_deg=70,
                              distance_pc=D_PC, fov=12, npix=96, gaunt="dulk", n_workers=NW,
                              plot_path=os.path.join(OUT, "stellar_example_rotation.png"))
print("rotational modulation max/min:", round(lc["flux_mjy"].max() / lc["flux_mjy"].min(), 3))
