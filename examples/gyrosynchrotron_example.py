# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gyrosynchrotron emission from fast electrons in the analytic active region.

Needs the fast GS codes (once):  python -m radiosynth.gyrosynchrotron.library --build

Produces (in examples/output/gyrosynchrotron/):
  gs_tb.png        T_b at 3, 5, 10, 17 GHz: fast electrons in a blob at the loop top
  gs_pol.png       V/I
  gs_fraction.png  fraction of the intensity due to the fast electrons
  gs_spectrum.png  flux spectrum 1-30 GHz: isotropic vs loss cone, delta = 3.5 and 4.5
  gs_trapped.png   fast electrons as a fixed fraction of the thermal density in the
                   arcade core, instead of a blob
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
from radiosynth.gyrosynchrotron import (FastElectrons, in_sphere, in_box, product,   # noqa: E402
                                        fraction_of_thermal)
from active_region import make_ar_model                        # noqa: E402

OUT = os.path.join(HERE, "output", "gyrosynchrotron")
os.makedirs(OUT, exist_ok=True)

grid = make_ar_model(n=100)
view = dict(observer=(-215.0, 0.0, 0.0), x_size=0.36, y_size=0.36, nx=60, ny=60)

# fast electrons: gaussian blob of radius 0.015 R_sun (10 Mm) at the top of the
# arcade (35 Mm above the dipole centre), n_b = 1e7 cm^-3 above 20 keV, delta = 3.5
blob = in_sphere((-1.04, 0.0, 0.0), 0.015, 1e7)
el = FastElectrons(nb=blob, delta=3.5, emin_mev=0.02, emax_mev=5.0)

img = rs.gyrosynchrotron.make_image(grid, "3GHz, 5GHz, 10GHz, 17GHz", el, diagnostics=True,
                                    **view)
img.plot("tb", path=os.path.join(OUT, "gs_tb.png"), ncols=4)
img.plot("pol", path=os.path.join(OUT, "gs_pol.png"), ncols=4)
img.plot("gs_fraction", path=os.path.join(OUT, "gs_fraction.png"), ncols=4)
img.save_npz(os.path.join(OUT, "gs_ar.npz"))

# spectrum of the blob for several prescriptions --------------------------------
freqs = np.geomspace(1e9, 3e10, 12)
small = dict(view, x_size=0.1, y_size=0.1, nx=16, ny=16)        # just the blob
cases = [("isotropic, $\\delta$=3.5", FastElectrons(nb=blob, delta=3.5, emin_mev=0.02, emax_mev=5.0)),
         ("isotropic, $\\delta$=4.5", FastElectrons(nb=blob, delta=4.5, emin_mev=0.02, emax_mev=5.0)),
         ("loss cone 45$^\\circ$, $\\delta$=3.5",
          FastElectrons(nb=blob, delta=3.5, emin_mev=0.02, emax_mev=5.0, pitch="gau",
                        alpha_c=45.0, dmu=0.2))]
fig, ax = plt.subplots(figsize=(6.2, 4.3))
for lab, e in cases:
    s = rs.make_gs_image(grid, freqs, e, verbose=False, **small)
    ax.loglog(freqs / 1e9, s.flux_density_sfu, "o-", label=lab)
ff = rs.make_gs_image(grid, freqs, el, gyrosynchrotron=False, verbose=False, **small)
ax.loglog(freqs / 1e9, ff.flux_density_sfu, "k--", label="no fast electrons (free-free)")
ax.set_xlabel("frequency [GHz]")
ax.set_ylabel("flux density [sfu]")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gs_spectrum.png"), dpi=130)
plt.close(fig)

# a trapped population: n_b = 1e-3 n_e in the core of the arcade ----------------
# (for an MHD model with currents, power_of("j", ...) after rs.current_density(grid)
#  places the electrons at current concentrations; this dipole field is current-free)
et = FastElectrons(nb=product(fraction_of_thermal(1e-3),
                              in_box((-1.08, -0.06, -0.03), (-1.01, 0.06, 0.03), 1.0)),
                   delta=3.5, emin_mev=0.02, emax_mev=5.0, pitch="gau", alpha_c=45.0, dmu=0.2)
imt = rs.make_gs_image(grid, "5GHz, 17GHz", et, verbose=False, **view)
imt.plot("tb", path=os.path.join(OUT, "gs_trapped.png"))
print("written to", OUT)
