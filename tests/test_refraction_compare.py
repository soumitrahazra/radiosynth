# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of compare_refraction (refracted rays vs straight lines of sight).

 1. the two images are identical to separate make_radio_image calls
 2. refraction matters at 80 MHz and is negligible at 17 GHz
 3. simplistic mechanism via radiosynth.simplistic.compare_refraction
 4. gyroresonance / gyrosynchrotron are rejected (straight rays only)
 5. command line --compare-refraction
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))
import radiosynth as rs                                   # noqa: E402
from synthetic_corona import make_model                   # noqa: E402

OUT = os.path.join(HERE, "output", "refraction")
os.makedirs(OUT, exist_ok=True)
results = []


def check(name, cond, msg):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {msg}")


grid = make_model(half=3.0, n=81)
kw = dict(observer=(-215, 0, 0), x_size=5, y_size=5, nx=40, ny=40, transfer="full")
res = rs.compare_refraction(grid, "80MHz, 17GHz", path=os.path.join(OUT, "compare.png"),
                            verbose=False, **kw)
a = rs.make_radio_image(grid, "80MHz, 17GHz", verbose=False, **kw)
b = rs.make_radio_image(grid, "80MHz, 17GHz", no_refraction=True, verbose=False, **kw)
same = (np.array_equal(res["refracted"].intensity, a.intensity)
        and np.array_equal(res["straight"].intensity, b.intensity))
check("compare_refraction = separate make_radio_image calls", same,
      "refracted and straight images bit-for-bit identical")

fr = res["flux_ratio"]
ch = res["changed"]
check("refraction matters at 80 MHz, negligible at 17 GHz",
      abs(fr[1] - 1) < 0.01 and (abs(fr[0] - 1) > 0.02 or ch[0] > 0.05) and ch[1] < 0.01,
      f"flux refracted/straight = {np.round(fr, 4)}, pixels changed by >10%: "
      f"{np.round(100 * ch, 1)} %; figure {os.path.isfile(os.path.join(OUT, 'compare.png'))}")

rs_ = rs.simplistic.compare_refraction(grid, "80MHz", verbose=False, **dict(kw, transfer="thin"))
check("simplistic mechanism", rs_["refracted"].meta["emission"].startswith("EmissionModel(kind='simplistic'"),
      f"flux ratio {rs_['flux_ratio'][0]:.3f}")

ok = True
for m in ("gyroresonance", "gyrosynchrotron"):
    try:
        rs.compare_refraction(grid, "5GHz", emission=m, verbose=False, **kw)
        ok = False
    except ValueError:
        pass
check("gyroresonance / gyrosynchrotron rejected", ok, "straight rays only")

p = os.path.join(OUT, "grid.npz")
grid.to_npz(p)
out = os.path.join(OUT, "cli")
r = subprocess.run([sys.executable, "-m", "radiosynth", p, "--freq", "80MHz", "--size", "5", "5",
                    "--npix", "24", "24", "--transfer", "full", "--compare-refraction",
                    "--out", out], cwd=ROOT, capture_output=True, text=True)
r2 = subprocess.run([sys.executable, "-m", "radiosynth", p, "--freq", "5GHz",
                     "--mechanism", "gyroresonance", "--compare-refraction", "--out", out],
                    cwd=ROOT, capture_output=True, text=True)
okc = (r.returncode == 0 and all(os.path.isfile(out + s) for s in
                                 ("_refracted.npz", "_straight.npz", "_refraction.png"))
       and r2.returncode != 0)
check("CLI --compare-refraction", okc,
      " | ".join(l.strip() for l in r.stdout.splitlines() if "flux" in l)[:160])

print(f"Summary: {sum(results)} / {len(results)} passed")
sys.exit(0 if all(results) else 1)
