# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Free-free + gyroresonance + gyrosynchrotron in one image.

The analytic bipolar active region (examples/active_region.py) with a blob of
fast electrons at the top of the arcade (as in gyrosynchrotron_example.py).
Needs the fast GS codes:  python -m radiosynth.gyrosynchrotron.library --build

Produces (in examples/output/combined/):
  cb_tb.png         T_b of the combined image at 2, 5, 10, 17 GHz
  cb_mechanism.png  dominant mechanism in every pixel
  cb_pol.png        V/I
  cb_spectrum.png   flux spectrum: combined vs each mechanism alone vs their sum
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
from radiosynth.gyrosynchrotron import FastElectrons, in_sphere   # noqa: E402
from active_region import make_ar_model                        # noqa: E402

OUT = os.path.join(HERE, "output", "combined")
os.makedirs(OUT, exist_ok=True)

grid = make_ar_model(n=100)
view = dict(observer=(-215.0, 0.0, 0.0), x_size=0.36, y_size=0.36, nx=60, ny=60)
# weaker fast-electron population than in gyrosynchrotron_example.py, so that
# all three mechanisms are visible
el = FastElectrons(nb=in_sphere((-1.04, 0.0, 0.0), 0.015, 3e5), delta=3.5,
                   emin_mev=0.02, emax_mev=5.0)

img = rs.combined.make_image(grid, "2GHz, 5GHz, 10GHz, 17GHz", el, **view)
img.plot("tb", path=os.path.join(OUT, "cb_tb.png"), ncols=4)
img.plot("mechanism", path=os.path.join(OUT, "cb_mechanism.png"), ncols=4)
img.plot("pol", path=os.path.join(OUT, "cb_pol.png"), ncols=4)
img.save_npz(os.path.join(OUT, "combined.npz"))

# spectrum: each mechanism alone and together --------------------------------
freqs = np.geomspace(1e9, 3e10, 12)
sv = dict(view, nx=40, ny=40, verbose=False)
spec = {}
for lab, inc in (("combined", ("ff", "gr", "gs")), ("free-free", ("ff",)),
                 ("gyroresonance", ("gr",)), ("gyrosynchrotron", ("gs",))):
    spec[lab] = rs.make_combined_image(grid, freqs, el, include=inc, **sv).flux_density_sfu
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for lab, st in (("combined", "ko-"), ("free-free", "s--"), ("gyroresonance", "^--"),
                ("gyrosynchrotron", "v--")):
    ax.loglog(freqs / 1e9, spec[lab], st, label=lab, ms=4)
ax.loglog(freqs / 1e9, spec["free-free"] + spec["gyroresonance"] + spec["gyrosynchrotron"],
          ":", color="gray", label="sum of the three alone")
ax.set_xlabel("frequency [GHz]")
ax.set_ylabel("flux density [sfu]")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "cb_spectrum.png"), dpi=130)
plt.close(fig)
print("written to", OUT)
