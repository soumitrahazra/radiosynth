# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gyroresonance + free-free images of an analytic bipolar active region.

Produces (in examples/output/gyroresonance/):
  gr_tb.png        T_b (Stokes I) at 3, 5, 10, 17 GHz
  gr_pol.png       degree of circular polarisation V/I
  gr_harmonic.png  dominant harmonic (0 = free-free)
  gr_mech.png      combined vs gyroresonance-only vs free-free-only at 5 GHz
  gr_spectrum.png  flux density spectrum, I and V, 1-20 GHz
Run:  python examples/gyroresonance_example.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402

import radiosynth as rs                                        # noqa: E402
from active_region import make_ar_model                        # noqa: E402

OUT = os.path.join(HERE, "output", "gyroresonance")
os.makedirs(OUT, exist_ok=True)

grid = make_ar_model(n=120)                       # T_e = 2.5 MK, |B| = 2500 G above the dipole
view = dict(observer=(-215.0, 0.0, 0.0), x_size=0.36, y_size=0.36, nx=120, ny=120)

# 1. images at four frequencies -----------------------------------------------
img = rs.gyroresonance.make_image(grid, "3GHz, 5GHz, 10GHz, 17GHz", **view)
img.plot("tb", path=os.path.join(OUT, "gr_tb.png"), ncols=4)
img.plot("pol", path=os.path.join(OUT, "gr_pol.png"), ncols=4)
img.plot("harmonic", path=os.path.join(OUT, "gr_harmonic.png"), ncols=4)
img.save_npz(os.path.join(OUT, "gr_ar.npz"))

# 2. which mechanism? ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ext = [img.x[0], img.x[-1], img.y[0], img.y[-1]]
for ax, mech in zip(axes, ("gr+ff", "gr", "ff")):
    m = rs.make_gr_image(grid, "5GHz", mechanisms=mech, verbose=False, **view)
    im = ax.imshow(m.brightness_temperature[0] / 1e6, origin="lower", extent=ext,
                   cmap="inferno", vmin=0, vmax=2.6)
    ax.set_title(f"5 GHz, mechanisms='{mech}'  ({m.flux_density_sfu[0]:.1f} sfu)")
    ax.set_xlabel(r"X [$R_\odot$]")
    ax.set_ylabel(r"Y [$R_\odot$]")
    fig.colorbar(im, ax=ax, label=r"$T_b$ [MK]", shrink=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gr_mech.png"), dpi=130)
plt.close(fig)

# 3. spectrum -----------------------------------------------------------------
freqs = np.geomspace(1e9, 2e10, 14)
spec = rs.make_gr_image(grid, freqs, verbose=False, **dict(view, nx=80, ny=80))
ff = rs.make_gr_image(grid, freqs, mechanisms="ff", verbose=False, **dict(view, nx=80, ny=80))
# V of the positive (left) spot only - the bipolar region has V ~ 0 in total
vpos = spec.stokes_v[:, :, :40].sum(axis=(1, 2)) * spec.pixel_solid_angle
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(freqs / 1e9, spec.flux_density_sfu, "o-", label="I (gr+ff)")
ax.loglog(freqs / 1e9, ff.flux_density_sfu, "s--", label="I (free-free only)")
ax.loglog(freqs / 1e9, np.abs(vpos) / 1e-22, "^-", label="|V| of the positive spot")
ax.set_xlabel("frequency [GHz]")
ax.set_ylabel("flux density [sfu]")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gr_spectrum.png"), dpi=130)
plt.close(fig)

# 4. weak vs strong coupling (identical here: B.k keeps its sign along each ray
#    above the spots) ------------------------------------------------------------
strong = rs.make_gr_image(grid, "5GHz", coupling="strong", verbose=False, **view)
print("weak vs strong coupling, max |dV|/max|V| at 5 GHz:",
      float(np.abs(strong.stokes_v[0] - img.stokes_v[1]).max() / np.abs(img.stokes_v[1]).max()))
print("written to", OUT)
