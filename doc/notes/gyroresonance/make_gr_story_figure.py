# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
# Makes gr_story_images.pdf (Fig. 2 of radiosynth_gyroresonance_story.tex).
# Usage: python make_gr_story_figure.py   (from anywhere; uses the radiosynth of this repository)
import sys, numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))      # repository root
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import radiosynth as rs
from active_region import make_ar_model

grid = make_ar_model(n=120)        # bipolar active region: T_e = 2.5 MK, |B| up to 2500 G
view = dict(observer=(-215.0, 0.0, 0.0), x_size=0.36, y_size=0.36, nx=120, ny=120, verbose=False)
img = rs.make_gr_image(grid, "5GHz, 17GHz", **view)
ff = rs.make_gr_image(grid, "5GHz", mechanisms="ff", **view)
for k, f in enumerate(img.freqs):
    print(f"{f/1e9:g} GHz: max Tb {img.brightness_temperature[k].max()/1e6:.2f} MK, flux {img.flux_density[k]/1e-22:.2f} sfu, V/I flux {img.flux_density_v[k]/img.flux_density[k]:+.3f}")
print("5 GHz ff only: max Tb %.3f MK flux %.3f sfu" % (ff.brightness_temperature[0].max()/1e6, ff.flux_density[0]/1e-22))
ds = img.dominant_s[0]; print("5GHz dominant s pixel counts", {int(s): int((ds == s).sum()) for s in np.unique(ds)})
ext = [img.x[0], img.x[-1], img.y[0], img.y[-1]]
fig, axs = plt.subplots(2, 2, figsize=(9.2, 7.6)); ax = axs.ravel()
for a, data, title, cm, lo, hi, lab in (
        (ax[0], img.brightness_temperature[0] / 1e6, "5 GHz: $T_b$", "inferno", 0, None, "$T_b$ [MK]"),
        (ax[1], img.polarization[0], "5 GHz: circular polarisation $V/I$", "RdBu_r", -1, 1, "$V/I$"),
        (ax[2], np.where(img.dominant_s[0] >= 0, img.dominant_s[0], np.nan), "5 GHz: harmonic we see", "viridis", 0, 4, "$s$ (0 = free--free)"),
        (ax[3], img.brightness_temperature[1] / 1e6, "17 GHz: $T_b$", "inferno", 0, None, "$T_b$ [MK]")):
    im = a.imshow(data, origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi)
    a.set_title(title, fontsize=10); a.set_xlabel(r"X [$R_\odot$]"); a.set_ylabel(r"Y [$R_\odot$]")
    fig.colorbar(im, ax=a, label=lab, shrink=0.85)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "gr_story_images.pdf"))
