# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Orientation helpers: solar ephemeris and longitude-offset fit.

1. sun_ephemeris against Meeus' worked example (Astronomical Algorithms, ex. 29.a).
2. fit_longitude_offset recovers a known longitude offset: a mock 'observation'
   is made from the active-region test corona with the observer rotated by a
   known angle (beam-smoothed, with noise); the fit must find that angle.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
import radiosynth as rs                                  # noqa: E402
from radiosynth import figures as rf                     # noqa: E402
from synthetic_corona import structured_model            # noqa: E402

OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
ok = []

e = rf.sun_ephemeris(2448908.5, use_sunpy=False)          # 1992 Oct 13.0 TD
d = max(abs(e["P"] - 26.27), abs(e["B0"] - 5.99), abs(e["L0"] - 238.63))
ok.append(d < 0.05)
print(f"[{'PASS' if ok[-1] else 'FAIL'}] ephemeris vs Meeus ex. 29.a: P={e['P']:.2f} B0={e['B0']:.2f} "
      f"L0={e['L0']:.2f} (max deviation {d:.3f} deg)")

grid = structured_model(None, half=3.0, n=161)
B0, L0, D = 0.0, 180.0, 215.0
rng = np.random.default_rng(1)
for true_off in (20.0, 17.0):
    mock = rs.make_radio_image(grid, "17GHz", rf.earth_observer(B0, L0 + true_off, D), 2.6, 2.6,
                               128, 128, n_workers=os.cpu_count(), verbose=False)
    tb = mock.brightness_temperature[0]
    pix = float(rf.rsun_to_arcsec(mock.x[1] - mock.x[0], D))
    tb = rf.smooth_to_beam(tb, pix, 10.0)
    tb = tb * (1 + 0.03 * rng.normal(size=tb.shape))
    ext = rf.image_extent(mock, "arcsec")
    res = rf.fit_longitude_offset(grid, tb, "17GHz", B0, L0, offsets=np.arange(-40, 41, 5),
                                  obs_extent=ext, npix=96, beam_fwhm_arcsec=10.0,
                                  n_workers=os.cpu_count(), verbose=False,
                                  plot_path=os.path.join(OUT, f"lonfit_{int(true_off)}.png"))
    err = abs(res["best_refined"] - true_off)
    ok.append(err < 2.5)
    print(f"[{'PASS' if ok[-1] else 'FAIL'}] longitude-offset fit: true {true_off:+.1f} deg, "
          f"found {res['best']:+.1f} (grid) / {res['best_refined']:+.2f} deg (refined), "
          f"max corr {np.nanmax(res['corr']):.3f}")

obs = rf.observer_for_date("2014-11-01T03:00", phi_offset_deg=10.0, verbose=True)
r = np.linalg.norm(obs)
ok.append(200 < r < 225)
print(f"[{'PASS' if ok[-1] else 'FAIL'}] observer_for_date: |O| = {r:.1f} R_sun")

# --- compare_with_observation: identical maps -> corr 1, ratio 1; scaled -> ratio
img = rs.make_radio_image(grid, "17GHz", rf.earth_observer(0.0, 180.0, 215.0), 2.6, 2.6, 96, 96,
                          n_workers=os.cpu_count(), verbose=False)
tb = img.brightness_temperature[0]
ext = rf.image_extent(img, "arcsec")
c = rf.compare_with_observation(img, 0.5 * tb, obs_extent=ext,
                                plot_path=os.path.join(OUT, "compare_stats.png"))
good = abs(c["ratio_disk"] - 2.0) < 0.05 and c["corr_log"] > 0.99 and abs(c["r_half_obs"] - c["r_half_syn"]) < 0.11
ok.append(good)
print(f"[{'PASS' if good else 'FAIL'}] compare_with_observation: ratio {c['ratio_disk']:.3f} (2), "
      f"corr_log {c['corr_log']:.4f}, r_half obs/syn {c['r_half_obs']:.2f}/{c['r_half_syn']:.2f}")
print("Summary:", sum(ok), "/", len(ok), "passed")
