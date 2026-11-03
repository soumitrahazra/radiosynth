# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of the 1D (spherically symmetric) support.

 1. symmetric image (one row of rays, rotated) = full 2D image of the same grid
 2. 1D profile grid = 3D Cartesian grid of the same analytic corona
 3. optically thick stellar wind: profile flux = exact integral over p
 4. PLUTO 1D spherical output (.dbl) is read as a symmetric model
 5. simplistic mechanism; gyroresonance/gyrosynchrotron are not offered for 1D
 6. command line: text profile + --symmetric; PLUTO 1D directory
"""
import math
import os
import subprocess
import sys

import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))
import radiosynth as rs                                         # noqa: E402
from radiosynth.common.constants import R_SUN, K_B, C_LIGHT, M_P   # noqa: E402
from radiosynth.bremsstrahlung.wind import wind_optical_depth  # noqa: E402
from synthetic_corona import newkirk, make_model               # noqa: E402

OUT = os.path.join(HERE, "output", "symmetric")
os.makedirs(OUT, exist_ok=True)
results = []


def check(name, cond, msg):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {msg}")


r = np.geomspace(1.0, 8.0, 400)
g1 = rs.grid_from_profile(r, newkirk(r), np.full(r.size, 1.5e6), r_body=1.0)

# 1. ------------------------------------------------------------------------
fr = ["80MHz", "300MHz", "2GHz"]
s = rs.make_symmetric_image(g1, fr, transfer="full", x_size=6, nx=128, ny=128, verbose=False)
full = rs.make_radio_image(g1, fr, x_size=6, y_size=6, nx=128, ny=128, transfer="full",
                           verbose=False)
# the profile flux covers the circle p <= p_max (half diagonal), the images the square
e1 = np.abs(s.profile["flux"] / full.flux_density - 1).max()
e1b = np.abs(s.flux_density / full.flux_density - 1).max()
# T_b along the central row of the full image vs the profile
row = full.brightness_temperature[:, 64, 64:]
prof = np.stack([np.interp(full.x[64:], s.profile["p"], s.profile["tb"][k]) for k in range(3)])
m = row > 0.05 * row.max(axis=1, keepdims=True)
e1c = np.median(np.abs(prof[m] / row[m] - 1))
check("symmetric image (one row of rays) = full 2D image", e1 < 0.01 and e1b < 0.01 and e1c < 0.01,
      f"flux: profile {e1:.1e}, rotated image {e1b:.1e}; T_b along the row (median) {e1c:.1e}")
rs.plot_profile(s, path=os.path.join(OUT, "profile.png"))

# 2. ------------------------------------------------------------------------
cart = make_model(half=3.0, n=161)
c = rs.make_radio_image(cart, fr, x_size=6, y_size=6, nx=128, ny=128, transfer="full",
                        verbose=False)
e2 = np.abs(s.profile["flux"] / c.flux_density - 1).max()
check("1D profile grid = 3D Cartesian grid of the same corona", e2 < 0.03,
      f"flux ratio - 1 = {np.round(s.profile['flux'] / c.flux_density - 1, 4)}")

# 3. ------------------------------------------------------------------------
R_STAR, TE, GAUNT, D_PC = 0.5, 2.0e6, 10.0, 10.0
L = R_STAR * R_SUN
k1 = rs.EmissionModel("bremsstrahlung", gaunt=GAUNT)(np.array([1.0]), np.array([TE]), 5e9)[0][0]
n0 = math.sqrt(2.0 * (6.0 * L) ** 3 / (math.pi * k1 * L**4))
rw = np.geomspace(1.0, 200.0, 600)
tbl = os.path.join(OUT, "wind_profile.txt")
np.savetxt(tbl, np.column_stack([rw, n0 * rw**-2 * 1e-6, np.full(rw.size, TE)]),
           header="r [R_star]  n_e [cm^-3]  T_e [K]")
gw = rs.read_profile(tbl, r_body=1.0, length_unit=L)
pmax = 32.0
sw = rs.make_symmetric_image(gw, [2e9, 5e9, 10e9], observer=(1e4, 0, 0), distance_pc=D_PC,
                             n_p=800, p_max=pmax, gaunt=GAUNT, transfer="full",
                             no_refraction=True, tau_max=60.0, verbose=False)
D = D_PC * 3.0856775814913673e16
errs = []
for k, f in enumerate([2e9, 5e9, 10e9]):
    B = 2 * f**2 * K_B * TE / C_LIGHT**2
    fn = lambda p: -math.expm1(-wind_optical_depth(p * L, n0, L, TE, f, GAUNT, r_body_m=L)[0]) * p  # noqa: E731
    ex = 2 * math.pi * B * L**2 / D**2 * integrate.quad(fn, 1e-6, pmax, limit=400, points=[1.0])[0]
    errs.append(sw.profile["flux"][k] / ex - 1)
check("optically thick wind (text table): profile flux = exact integral",
      np.max(np.abs(errs)) < 0.01,
      f"flux / exact - 1 = {np.round(errs, 4)}; {sw.profile['flux'][1] / 1e-29:.4g} mJy at 5 GHz")

# 4. ------------------------------------------------------------------------
UD, UL, UV = M_P * 1e3 * 1e8, 6.96e10, 1e7
d = os.path.join(OUT, "pluto1d")
os.makedirs(d, exist_ok=True)
r_e = np.geomspace(1.0, 8.0, 301)
rc = 0.5 * (r_e[1:] + r_e[:-1])
ne_c = newkirk(rc)
rho = ne_c * M_P * 1e-3 / UD
prs = 2 * ne_c * 1e-6 * 1.380649e-16 * 1.5e6 / (UD * UV**2)
with open(os.path.join(d, "grid.out"), "w") as fh:
    fh.write("# PLUTO 4.4 Grid File\n# DIMENSIONS: 1\n# GEOMETRY:   SPHERICAL\n")
    for e in (r_e, np.array([0.0, np.pi]), np.array([0.0, 2 * np.pi])):
        fh.write(f"{e.size - 1}\n")
        for i in range(e.size - 1):
            fh.write(f" {i + 1}  {e[i]:.12e}  {e[i + 1]:.12e}\n")
with open(os.path.join(d, "dbl.out"), "w") as fh:
    fh.write("0 0.0 1e-4 0 single_file little rho prs\n")
with open(os.path.join(d, "data.0000.dbl"), "wb") as fh:
    for a_ in (rho, prs):
        fh.write(a_.astype("<f8").tobytes())
gp = rs.load_pluto(d, unit_density=UD, unit_length=UL, unit_velocity=UV, verbose=False)
pts = np.array([[1.3, 0.2, -0.4], [-2.0, 1.0, 0.5], [0.0, 0.0, 4.0]])
ne_p = gp.sample(pts)[0]
ne_x = newkirk(np.linalg.norm(pts, axis=1))
check("PLUTO 1D spherical output read as a symmetric model",
      rs.common.symmetric.is_symmetric(gp) and np.abs(ne_p / ne_x - 1).max() < 0.01,
      f"{gp}; n_e at 3 points / analytic - 1 = {np.round(ne_p / ne_x - 1, 4)}")

# 5. ------------------------------------------------------------------------
ss = rs.make_symmetric_image(g1, "80MHz", mechanism="simplistic", x_size=6, nx=96, ny=96,
                             verbose=False)
fs = rs.make_radio_image(g1, "80MHz", emission="simplistic", x_size=6, y_size=6, nx=96, ny=96,
                         verbose=False)
e5 = abs(ss.flux_density[0] / fs.flux_density[0] - 1)
rej = 0
for m_ in ("gyroresonance", "gyrosynchrotron", "combined"):
    try:
        rs.make_symmetric_image(g1, "5GHz", mechanism=m_, verbose=False)
    except ValueError:
        rej += 1
check("simplistic mechanism; magnetic mechanisms not offered for 1D", e5 < 0.01 and rej == 3,
      f"simplistic flux ratio - 1 = {e5:.1e}; gyroresonance/gyrosynchrotron/combined rejected")

# 6. ------------------------------------------------------------------------
tab = os.path.join(OUT, "corona.txt")
np.savetxt(tab, np.column_stack([r, newkirk(r) * 1e-6, np.full(r.size, 1.5e6)]))
out = os.path.join(OUT, "cli")
r1 = subprocess.run([sys.executable, "-m", "radiosynth", tab, "--profile", "--symmetric",
                     "--freq", "80MHz, 300MHz", "--size", "6", "6", "--npix", "64", "64",
                     "--n-p", "120", "--transfer", "full", "--r-body", "1", "--out", out],
                    cwd=ROOT, capture_output=True, text=True)
out2 = os.path.join(OUT, "cli_pluto")
r2 = subprocess.run([sys.executable, "-m", "radiosynth", d, "--symmetric", "--freq", "150MHz",
                     "--unit-density", str(UD), "--unit-length", str(UL), "--unit-velocity",
                     str(UV), "--size", "6", "6", "--npix", "48", "48", "--out", out2],
                    cwd=ROOT, capture_output=True, text=True)
okc = (r1.returncode == 0 and r2.returncode == 0
       and all(os.path.isfile(out + x) for x in ("_profile.txt", "_profile.png", ".npz"))
       and os.path.isfile(out2 + "_profile.txt"))
check("CLI: --profile text table and PLUTO 1D with --symmetric", okc,
      (r1.stderr.strip().splitlines() or r2.stderr.strip().splitlines() or ["ok"])[-1] + " | " +
      " ".join(l for l in r1.stdout.splitlines() if "flux density from" in l)[:120])

print(f"Summary: {sum(results)} / {len(results)} passed")
sys.exit(0 if all(results) else 1)
