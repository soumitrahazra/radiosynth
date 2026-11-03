# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Figures in the style of Moschou et al. (2018), Figs. 2, 3 and 5.

Uses an analytic test corona (Newkirk background + streamer + three hot, dense
active regions + an outward-moving CME blob).  With real data, replace
``structured_model(...)`` by ``rs.load_mhd(...)`` / ``rs.load_pluto(...)`` and the
mock observation by your Nobeyama / MWA FITS file.

python examples/moschou_figures.py      -> examples/output/moschou_fig{2,3,5}.png, cme_125MHz.gif
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import radiosynth as rs                                   # noqa: E402
from radiosynth import figures as rf                      # noqa: E402
from radiosynth.constants import R_SUN                    # noqa: E402
from synthetic_corona import structured_model             # noqa: E402

OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
NW = os.cpu_count() or 1
OBS = rf.earth_observer(b0_deg=0.0, l0_deg=180.0)   # observer on the -x axis, north up

t0 = time.time()
# ---------------------------------------------------------------------------
# Fig. 3 style: 125 MHz intensity map, rainbow colours, white contours
# ---------------------------------------------------------------------------
g = structured_model(None, half=3.0, n=201)
img125 = rs.make_radio_image(g, "125MHz", OBS, 4.4, 4.4, 160, 160, n_workers=NW, verbose=False)
rf.figure_single_map(img125, "125MHz", units="arcsec", cmap="jet", n_contours=8,
                     contour_decades=3, path=os.path.join(OUT, "moschou_fig3_125MHz.png"))
print(f"Fig. 3 done ({time.time() - t0:.0f} s)")

# ---------------------------------------------------------------------------
# Fig. 2 style: 17 GHz synthetic vs (mock) observation with contours
# ---------------------------------------------------------------------------
img17 = rs.make_radio_image(g, "17GHz", OBS, 2.6, 2.6, 200, 200, n_workers=NW, verbose=False)
# --- mock observation: a perturbed model, smoothed to a 10" beam, with noise,
#     written as a FITS file in helioprojective arcsec (the format of NoRH maps)
g_mock = structured_model(None, half=3.0, n=201, ar_shift=2.0, ar_amp=1.3)
mock = rs.make_radio_image(g_mock, "17GHz", OBS, 2.6, 2.6, 200, 200, n_workers=NW, verbose=False)
tb = mock.brightness_temperature[0]
pix = float(rf.rsun_to_arcsec(mock.x[1] - mock.x[0], np.linalg.norm(OBS)))
tb = rf.smooth_to_beam(tb, pix, 10.0)
tb = tb + np.random.default_rng(0).normal(0, 0.02 * tb.max(), tb.shape)
from radiosynth.fitsio import write_fits                    # noqa: E402
h = {"CTYPE1": "HPLN-TAN", "CUNIT1": "arcsec", "CDELT1": pix,
     "CTYPE2": "HPLT-TAN", "CUNIT2": "arcsec", "CDELT2": pix,
     "CRPIX1": (tb.shape[1] + 1) / 2.0, "CRPIX2": (tb.shape[0] + 1) / 2.0,
     "CRVAL1": 0.0, "CRVAL2": 0.0, "BUNIT": "K",
     "OBJECT": "MOCK OBSERVATION - synthetic stand-in, not real data"}
mock_fits = os.path.join(OUT, "mock_obs_17GHz.fits")
write_fits(mock_fits, tb, h)
rf.figure_compare_observation(img17, mock_fits, freq="17GHz", beam_fwhm_arcsec=10.0,
                              levels="decades", obs_label="Mock observation (synthetic)",
                              fov_arcsec=1250, path=os.path.join(OUT, "moschou_fig2_17GHz.png"))
print(f"Fig. 2 done ({time.time() - t0:.0f} s)")

# ---------------------------------------------------------------------------
# Fig. 5 style: CME at 125, 250, 300, 500 MHz, two times, + animation
# ---------------------------------------------------------------------------
times = [10.0, 38.0]
snaps = [structured_model(t, half=4.0, n=201) for t in times]
imgs = rf.make_time_series(snaps, "125MHz, 250MHz, 300MHz, 500MHz", OBS, 8.0, 8.0, 128, 128,
                           n_workers=NW, verbose=False)
rf.figure_time_series(imgs, times, n_contours=10, contour_decades=5, fov=3.5, cmap="afmhot",
                      path=os.path.join(OUT, "moschou_fig5_cme.png"))
print(f"Fig. 5 done ({time.time() - t0:.0f} s)")

anim_t = np.arange(0.0, 61.0, 6.0)
anim = rf.make_time_series((structured_model(t, half=4.0, n=161) for t in anim_t), "125MHz",
                           OBS, 8.0, 8.0, 96, 96, n_workers=NW, verbose=False)
rf.animate_time_series(anim, os.path.join(OUT, "cme_125MHz.gif"), times=anim_t, fps=3,
                       cmap="afmhot", contour_decades=5)
print(f"animation done ({time.time() - t0:.0f} s)")
