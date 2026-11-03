# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stellar mode: units, distant observer, wind free-free spectrum, rotation.

1. Isothermal spherical wind n = n0 (r0/r)^2 around a star of 0.5 R_sun, seen
   from 10 pc with parallel rays:
   - optical depth of every pixel vs the analytic tau(p) (incl. rays that hit
     the star, which see only the front half of the wind);
   - flux density vs the exact sum of B (1 - exp(-tau(p))) over the same pixels;
   - local spectral index vs the analytic value from the same pixel sums.
2. Analytic wind formula (pi Gamma(1/3) B a^(2/3) / D^2) vs direct quadrature.
3. The same physical model in R_sun coordinates and in R_star coordinates
   (r_star = 0.5) gives the same flux.
4. Rotational modulation: a hot dense spot on the equator at longitude 0 is
   occulted by the star at phase 180 deg (minimum of the light curve).
"""
import math
import os
import subprocess
import sys

import numpy as np
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
import radiosynth as rs                                   # noqa: E402
from radiosynth import stellar as st                      # noqa: E402
from radiosynth.constants import R_SUN, K_B, C_LIGHT      # noqa: E402

OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
NW = os.cpu_count() or 1
ok = []


def report(name, good, detail):
    ok.append(bool(good))
    print(f"[{'PASS' if good else 'FAIL'}] {name}: {detail}")


R_STAR = 0.5                       # R_sun
L = R_STAR * R_SUN                 # metres per coordinate unit (R_star)
TE = 2.0e6
D_PC = 10.0
GAUNT = 10.0

# n0 such that the tau = 1 radius at 5 GHz is 6 R_star
k1 = rs.EmissionModel("bremsstrahlung", gaunt=GAUNT)(np.array([1.0]), np.array([TE]), 5e9)[0][0]
p1 = 6.0 * L
n0 = math.sqrt(2.0 * p1**3 / (math.pi * k1 * L**4))
print(f"wind: n0 = {n0:.3e} m^-3 at r0 = R_star = {R_STAR} R_sun, T = {TE:.1e} K")


def wind_grid(coord_unit_rsun, rmax_star=200.0, nr=420):
    """Spherically symmetric wind on a SphericalGrid; coordinates in units of
    coord_unit_rsun (R_sun or R_star)."""
    f = R_STAR / coord_unit_rsun                     # R_star expressed in coordinate units
    r_e = np.geomspace(1.0, rmax_star, nr + 1) * f
    rc = np.sqrt(r_e[1:] * r_e[:-1])
    ne = n0 * (f / rc)**2
    return rs.SphericalGrid(rc, np.array([np.pi / 2]), None, ne[:, None], np.full((nr, 1), TE),
                            r_edges=r_e, theta_edges=np.array([0.0, np.pi]),
                            r_body=f, length_unit=coord_unit_rsun * R_SUN)


g = wind_grid(R_STAR)                                # coordinates in R_star
FOV, NPIX = 64.0, 128
freqs = [2e9, 5e9, 10e9]
img = st.make_stellar_image(g, freqs, inclination_deg=90, distance_pc=D_PC, fov=FOV, npix=NPIX,
                            gaunt=GAUNT, no_refraction=True, tau_max=60.0, n_workers=NW,
                            verbose=False)
X, Y = np.meshgrid(img.x, img.y)
P = np.hypot(X, Y)                                   # impact parameter [R_star]
D_m = D_PC * st.PC
pix_sr = img.pixel_solid_angle
ref_flux = []
for k, f in enumerate(img.freqs):
    tau_ref = st.wind_optical_depth(P.ravel() * L, n0, L, TE, f, GAUNT, r_body_m=L).reshape(P.shape)
    m = (tau_ref < 20) & (P > 0.3)
    e_tau = np.median(np.abs(img.tau[k][m] / tau_ref[m] - 1))
    report(f"wind optical depth per pixel, {f / 1e9:g} GHz", e_tau < 0.02,
           f"median rel. error {e_tau:.2e} over {m.sum()} pixels")
    B = 2 * f**2 * K_B * TE / C_LIGHT**2
    ref = np.sum(B * -np.expm1(-tau_ref)) * pix_sr
    ref_flux.append(ref)
    e = img.flux_density[k] / ref - 1
    report(f"wind flux density, {f / 1e9:g} GHz", abs(e) < 0.02,
           f"{img.flux_density_mjy[k]:.4g} mJy vs exact pixel sum {ref / 1e-29:.4g} mJy ({e:+.2%})")
_, a_code = st.spectral_index(img.freqs, img.flux_density)
_, a_ref = st.spectral_index(img.freqs, ref_flux)
report("spectral index (code vs exact, same field of view)", np.max(np.abs(a_code - a_ref)) < 0.02,
       f"alpha code {np.round(a_code, 3)}, exact {np.round(a_ref, 3)}; infinite wind: 2/3")

# --- 2. analytic formula vs quadrature (no stellar disk)
for f in (2e9, 10e9):
    Bf = 2 * f**2 * K_B * TE / C_LIGHT**2

    def integrand(lnp):                      # integrate in ln p: smooth over decades
        p = math.exp(lnp)
        return -math.expm1(-st.wind_optical_depth(p, n0, L, TE, f, GAUNT)[0]) * p * p
    num = 2 * math.pi * Bf * quad(integrand, math.log(1e-4 * L), math.log(1e8 * L),
                                  limit=400, epsabs=0, epsrel=1e-10)[0] / D_m**2
    ana = st.wind_flux_density(n0, L, TE, f, D_m, GAUNT)
    report(f"analytic wind flux vs quadrature, {f / 1e9:g} GHz", abs(ana / num - 1) < 1e-4,
           f"{ana / 1e-29:.6g} vs {num / 1e-29:.6g} mJy ({ana / num - 1:+.1e})")

# --- 3. same model in R_sun and R_star coordinates
g_sun = wind_grid(1.0)
img_sun = st.make_stellar_image(g_sun, 5e9, 90, 0, D_PC, fov=FOV * R_STAR, npix=NPIX,
                                gaunt=GAUNT, no_refraction=True, tau_max=60.0, n_workers=NW,
                                verbose=False)
e = img_sun.flux_density_mjy[0] / img.flux_density_mjy[1] - 1
report("coordinates in R_sun vs R_star (r_star = 0.5)", abs(e) < 1e-6,
       f"{img_sun.flux_density_mjy[0]:.6g} vs {img.flux_density_mjy[1]:.6g} mJy")

# --- 4. rotational modulation with a hot spot at longitude 0
x = np.linspace(-4, 4, 121)
Xg, Yg, Zg = np.meshgrid(x, x, x, indexing="ij")
Rg = np.sqrt(Xg**2 + Yg**2 + Zg**2)
ne = 2e15 * (1.0 / np.maximum(Rg, 0.98))**8
spot = np.exp(-((Xg - 1.15)**2 + Yg**2 + Zg**2) / 0.12**2)
ne = ne * (1 + 30 * spot)
te = 1.5e6 + 8e6 * spot
gs = rs.MHDGrid(x, x, x, ne, te, length_unit=L)
lc = st.rotational_modulation(gs, "5GHz", range(0, 360, 30), inclination_deg=90, distance_pc=D_PC,
                              fov=6, npix=64, n_workers=NW,
                              plot_path=os.path.join(OUT, "stellar_rotation.png"))
fl = lc["flux_mjy"]
ph = lc["phases"]
low = ph[fl <= fl.min() * 1.005]                     # phases on the minimum plateau
centre = float(np.mean(low))
sym = np.max(np.abs(fl[1:6] - fl[11:6:-1])) / fl.max()   # f(phi) = f(360 - phi)
report("rotational modulation: spot occulted around phase 180",
       abs(centre - 180) <= 15 and sym < 0.01,
       f"minimum plateau centred at {centre:.0f} deg ({low.min():.0f}-{low.max():.0f}), "
       f"max/min = {fl.max() / fl.min():.2f}, asymmetry {sym:.1e}")

# spectrum figure
sp = st.stellar_spectrum(g, np.geomspace(0.5e9, 40e9, 10), 90, 0, D_PC, fov=FOV, npix=96,
                         gaunt="dulk", n_workers=NW, plot_path=os.path.join(OUT, "stellar_spectrum.png"))
print("    spectrum (Dulk Gaunt factor): alpha =", np.round(sp["alpha"], 2))

# --- CLI
g.to_npz(os.path.join(OUT, "wind_grid.npz"))         # saved grid, reloaded by the CLI
r = subprocess.run([sys.executable, "-m", "radiosynth", os.path.join(OUT, "wind_grid.npz"),
                    "--freq", "5GHz", "--distance-pc", "10", "--inclination", "90", "--size", "64", "64",
                    "--npix", "64", "64", "--transfer", "full", "--gaunt", "10",
                    "--out", os.path.join(OUT, "cli_star")], cwd=ROOT, capture_output=True, text=True)
line = [ln for ln in r.stdout.splitlines() if "mJy" in ln and "flux density at" in ln]
report("CLI with --distance-pc / --inclination", r.returncode == 0 and bool(line),
       line[0] if line else r.stderr[-300:])
print("Summary:", sum(ok), "/", len(ok), "passed")
