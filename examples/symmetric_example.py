# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""1D (spherically symmetric) model: radial brightness profiles, image and spectrum.

The Newkirk density profile n_e(r) with T_e = 1.5 MK, given as 1D arrays (it could
equally be read from a text table with rs.read_profile, or from 1D PLUTO output).
Produces (in examples/output/symmetric/):
  sym_profiles.png     T_b against the distance from disk centre, 80 MHz - 17 GHz
  sym_refraction.png   80 MHz profile with and without refraction
  sym_spectrum.png     flux-density spectrum 30 MHz - 30 GHz
  sym_image.png        the rotated 2D image at 150 MHz
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402

import radiosynth as rs                                        # noqa: E402
from synthetic_corona import newkirk                           # noqa: E402

OUT = os.path.join(HERE, "output", "symmetric")
os.makedirs(OUT, exist_ok=True)

r = np.geomspace(1.0, 10.0, 600)                     # radii [R_sun]
grid = rs.grid_from_profile(r, newkirk(r), np.full(r.size, 1.5e6), r_body=1.0)

# 1. radial T_b profiles -------------------------------------------------------
t0 = time.time()
img = rs.make_symmetric_image(grid, "80MHz, 150MHz, 300MHz, 1GHz, 17GHz", transfer="full",
                              x_size=6, nx=200, ny=200)
print(f"symmetric image, 5 frequencies: {time.time() - t0:.1f} s")
rs.plot_profile(img, path=os.path.join(OUT, "sym_profiles.png"))
img.plot("tb", path=os.path.join(OUT, "sym_image.png"), ncols=5)

# 2. refraction ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for nr, lab in ((False, "refracted rays"), (True, "straight lines of sight")):
    s = rs.make_symmetric_image(grid, "80MHz", transfer="full", no_refraction=nr, x_size=6,
                                nx=64, ny=64, verbose=False)
    ax.plot(s.profile["p"], s.profile["tb"][0] / 1e6, label=f"{lab} ({s.profile['flux'][0] / 1e-22:.2f} sfu)")
ax.axvline(1.0, color="gray", ls="--", lw=0.7)
ax.set_xlabel(r"distance from disk centre [$R_\odot$]")
ax.set_ylabel(r"$T_b$ [MK]")
ax.set_title("80 MHz, full transfer")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sym_refraction.png"), dpi=130)
plt.close(fig)

# 3. spectrum -------------------------------------------------------------------
freqs = np.geomspace(3e7, 3e10, 16)
s = rs.make_symmetric_image(grid, freqs, transfer="full", x_size=12, nx=32, ny=32, verbose=False)
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.loglog(freqs / 1e6, s.profile["flux"] / 1e-22, "o-")
ax.set_xlabel("frequency [MHz]")
ax.set_ylabel("flux density [sfu]")
ax.set_title("1D Newkirk corona, 1.5 MK (whole disk out to 8.5 R$_\\odot$)")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sym_spectrum.png"), dpi=130)
plt.close(fig)
print("written to", OUT)
