# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of the gyrosynchrotron folder (wrapper of the public fast GS codes).

Needs the compiled library (python -m radiosynth.gyrosynchrotron.library --build);
the library tests are skipped if it is not available.

 1. image of a uniform slab = direct GET_MW call on the same voxels (wrapper exact)
 2. thin gyrosynchrotron emissivity vs Dulk & Marsh (1982) approximation
 3. polarisation sign follows B; B -> -B gives V -> -V
 4. thin free-free from the library vs radiosynth (intensity; V/I = 2q/(1+q^2))
 5. body behind empty space: T_b = t_body
 6. array-defined power law = analytical power law
 7. fast-electron rules, energy check, current density of an analytic field
 8. command line --mechanism gyrosynchrotron
"""
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))
import radiosynth as rs                                                   # noqa: E402
from radiosynth.gyrosynchrotron import (FastElectrons, uniform, fraction_of_thermal,  # noqa: E402
                                        power_of, in_sphere, in_box, product)
from radiosynth.gyrosynchrotron.library import find_library, load_library          # noqa: E402
from radiosynth.common.magnetoionic import NU_B_PER_GAUSS                  # noqa: E402
from radiosynth.common.fields import current_density                       # noqa: E402

OUT = os.path.join(HERE, "output", "gyrosynchrotron")
os.makedirs(OUT, exist_ok=True)
results = []


def check(name, cond, msg):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {msg}")


x = np.linspace(-3.0, -2.0, 21)
y = np.linspace(-0.2, 0.2, 5)
z = y.copy()
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")


def slab(n, T, B, th_deg):
    """Uniform slab 1 R_sun thick along x; B at th_deg to the direction of an
    observer on the -x side (B . n_obs > 0)."""
    g = rs.MHDGrid(x, y, z, np.full(X.shape, n), np.full(X.shape, T))
    th = math.radians(th_deg)
    g.add_field(np.full(X.shape, -B * math.cos(th)), np.full(X.shape, -B * math.sin(th)), 0 * X)
    return g


kw = dict(observer=(-1.0, 0, 0), x_size=0.1, y_size=0.1, nx=3, ny=3, distance_pc=1e-6,
          verbose=False)

# --------------------------------------------------------------------------
# 7. (no library needed) rules, energy check, current density
# --------------------------------------------------------------------------
f = dict(pos=np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]]), ne=np.array([1e15, 2e15, 0.0]),
         te=np.full(3, 1e6), bmag=np.array([100.0, 200.0, 400.0]), r=np.array([0.0, 1, 3]))
ok = np.allclose(uniform(5.0)(f), [5, 5, 0])
ok &= np.allclose(fraction_of_thermal(1e-3)(f), [1e6, 2e6, 0])
ok &= np.allclose(power_of("bmag", 10.0, 100.0, 2.0)(f), [10, 40, 160])
ok &= np.allclose(in_sphere((0, 0, 0), 1.0, 7.0)(f), [7, 7 * math.exp(-1), 0])     # cut at 3 radii
ok &= np.allclose(in_box((-0.5, -1, -1), (1.5, 1, 1), 2.0)(f), [2, 2, 0])
ok &= np.allclose(product(uniform(2.0), 3.0)(f), [6, 6, 0])
e_hi = FastElectrons(nb=1e12, delta=3.0, emin_mev=0.01, emax_mev=10)
ratio = e_hi.check_energy(f, f["bmag"], verbose=False)
# mean energy of a delta=3 power law from 10 keV to 10 MeV ~ 2 Emin = 20 keV
w_exp = 1e12 * 0.02 * 1.602e-6 / (100.0**2 / (8 * math.pi))
ok_e = abs(ratio / w_exp - 1) < 0.01
g = rs.MHDGrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21), np.linspace(-1, 1, 5),
               np.ones((21, 21, 5)))
XX, YY, _ = np.meshgrid(g.x, g.y, g.z, indexing="ij")
g.add_field(-50.0 * YY, 50.0 * XX, 0 * XX)            # curl B = (0, 0, 100) G per R_sun
j = current_density(g)
j_exp = 100.0 * 1e-4 / (rs.R_SUN * 4e-7 * math.pi)
ok_j = np.allclose(j, j_exp, rtol=1e-6) and np.isclose(g.sample_scalar("j", np.array([[0.1, 0.2, 0.0]]))[0], j_exp, rtol=1e-5)
check("fast-electron rules, energy check, current density",
      ok and ok_e and ok_j,
      f"rules ok={bool(ok)}; W_fast/W_B = {ratio:.3e} (expected {w_exp:.3e}); "
      f"|J| = {j.mean():.4e} A/m^2 (expected {j_exp:.4e})")

# --------------------------------------------------------------------------
try:
    libpath = find_library()
except FileNotFoundError as exc:
    print("SKIP library tests:", str(exc).splitlines()[0])
    print(f"Summary: {sum(results)} / {len(results)} passed (library tests skipped)")
    sys.exit(0 if all(results) else 1)
lib = load_library()
print("GS codes library:", libpath)

# --------------------------------------------------------------------------
# 1. wrapper = direct call
# --------------------------------------------------------------------------
n0, T0, B0, th0 = 1e15, 2e7, 200.0, 50.0
gs = slab(n0, T0, B0, th0)
el = FastElectrons(nb=uniform(1e5), delta=3.5, emin_mev=0.02, emax_mev=5.0, pitch="gau",
                   alpha_c=60.0, dmu=0.2)
freqs = np.array([1e9, 3e9, 1e10, 3e10])
im = rs.make_gs_image(gs, freqs, el, ds=0.05, **kw)
Nz = 20
Lp = np.zeros(11, np.int32)
Lp[0], Lp[1] = Nz, freqs.size
Rp = np.zeros(5)
S_cm2 = (0.1 / 3 * rs.R_SUN * 100) ** 2
Rp[0], Rp[3], Rp[4] = S_cm2, 12, 12
P = np.zeros((24, Nz), order="F")
P[0] = rs.R_SUN * 100 / Nz
P[1], P[2], P[3], P[4], P[5] = T0, n0 * 1e-6, B0, th0, 4
P[6], P[7], P[9], P[10], P[12] = 3, 1e5, 0.02, 5.0, 3.5
P[14], P[15], P[16], P[17] = 3, 60.0, 0.2, 10.0
P[18], P[21] = n0 * 1e-6, 1
RL = np.zeros((7, freqs.size), order="F")
RL[0] = freqs / 1e9
d = np.zeros(1)
lib.get_mw(Lp, Rp, P, d, d, d, RL)
to_int = 1e-22 * rs.common.constants.AU**2 / (S_cm2 * 1e-4)
rel = max(np.abs(im.pol1[:, 1, 1] / (RL[6] * to_int) - 1).max(),
          np.abs(im.pol2[:, 1, 1] / (RL[5] * to_int) - 1).max())
check("wrapper = direct GET_MW call (same voxels)", rel < 1e-6,
      f"max relative difference {rel:.1e}; T_b = {np.round(im.brightness_temperature[:, 1, 1] / 1e6, 2)} MK")

# --------------------------------------------------------------------------
# 2. Dulk & Marsh (1982)
# --------------------------------------------------------------------------
B, th = 100.0, 60.0
g2 = slab(1e14, 1e6, B, th)
ratios = []
for dlt in (3.0, 4.0, 5.0):
    e = FastElectrons(nb=uniform(1e3), delta=dlt, emin_mev=0.01, emax_mev=100.0)
    for q in (20.0, 50.0):
        fq = q * NU_B_PER_GAUSS * B
        I = rs.make_gs_image(g2, fq, e, free_free=False, **kw).intensity[0, 1, 1]
        eta = (3.3e-24 * 10 ** (-0.52 * dlt) * B * 1e3
               * math.sin(math.radians(th)) ** (-0.43 + 0.65 * dlt) * q ** (1.22 - 0.9 * dlt))
        ratios.append(I / (eta * rs.R_SUN * 100 * 1e-3))
check("thin emissivity vs Dulk & Marsh (1982), delta=3-5, nu/nu_B=20-50",
      0.5 < min(ratios) and max(ratios) < 2.5,
      f"exact/approximate = {np.round(ratios, 2)} (checks units and scalings; the fit "
      f"formulas are only approximate)")

# --------------------------------------------------------------------------
# 3. polarisation sign
# --------------------------------------------------------------------------
e3 = FastElectrons(nb=uniform(1e4), delta=3.5, emin_mev=0.02, emax_mev=5.0)
a = rs.make_gs_image(slab(1e14, 1e6, 200.0, 40.0), "15GHz, 30GHz", e3, **kw)
b = rs.make_gs_image(slab(1e14, 1e6, -200.0, 40.0), "15GHz, 30GHz", e3, **kw)
check("V > 0 for B towards the observer (thin, x-mode); B -> -B gives V -> -V",
      (a.stokes_v[:, 1, 1] > 0).all() and np.allclose(a.stokes_v, -b.stokes_v, rtol=1e-10)
      and np.allclose(a.intensity, b.intensity, rtol=1e-10),
      f"V/I = {np.round(a.polarization[:, 1, 1], 3)} and {np.round(b.polarization[:, 1, 1], 3)}")

# --------------------------------------------------------------------------
# 4. thin free-free
# --------------------------------------------------------------------------
g4 = slab(1e14, 2e6, 30.0, 30.0)
fr4 = np.array([1e9, 3e9, 1e10])
a = rs.make_gs_image(g4, fr4, FastElectrons(), gyrosynchrotron=False, abundance="dulk", **kw)
b = rs.make_gr_image(g4, fr4, mechanisms="ff", gaunt="dulk", **kw)
q = 30.0 * NU_B_PER_GAUSS / fr4 * math.cos(math.radians(30.0))
rI = a.intensity[:, 1, 1] / b.intensity[:, 1, 1]
rP = a.polarization[:, 1, 1] / (2 * q / (1 + q * q))
check("library free-free vs radiosynth (thin): intensity, V/I = 2q/(1+q^2)",
      np.all((rI > 1.0) & (rI < 1.3)) and np.abs(rP - 1).max() < 0.03,
      f"I_lib/I_radiosynth = {np.round(rI, 3)} (different Coulomb logarithm), "
      f"V/I ratio to 2q/(1+q^2) = {np.round(rP, 3)}")

# optically thick free-free: T_b = n^2 T_e (the library's source function is
# n^2 B_nu; for X << 1 this is T_e) - checks the sfu -> intensity conversion
g4b = slab(1e16, 2e6, 10.0, 0.0)
a = rs.make_gs_image(g4b, "5GHz", FastElectrons(), gyrosynchrotron=False, **kw)
n2 = 1 - 1e16 / rs.critical_density(5e9)
rT = a.brightness_temperature[0, 1, 1] / (n2 * 2e6)
check("thick free-free from the library: T_b = n^2 T_e", abs(rT - 1) < 5e-3,
      f"T_b = {a.brightness_temperature[0, 1, 1]:.5g} K, n^2 T_e = {n2 * 2e6:.5g} K")

# --------------------------------------------------------------------------
# 5. body behind empty space
# --------------------------------------------------------------------------
xs = np.linspace(-1.5, -0.5, 11)
g5 = rs.MHDGrid(xs, np.linspace(-0.3, 0.3, 7), np.linspace(-0.3, 0.3, 7), np.zeros((11, 7, 7)),
                np.full((11, 7, 7), 1e6))
g5.add_field(*(np.zeros((11, 7, 7)),) * 3)
im5 = rs.make_gs_image(g5, "5GHz", FastElectrons(), t_body=1e4, observer=(-215, 0, 0),
                       x_size=0.2, y_size=0.2, nx=3, ny=3, verbose=False)
check("body behind empty space: T_b = t_body", np.allclose(im5.brightness_temperature, 1e4, rtol=1e-6),
      f"T_b = {im5.brightness_temperature[0, 1, 1]:.6g} K")

# --------------------------------------------------------------------------
# 6. array-defined distribution
# --------------------------------------------------------------------------
E = np.geomspace(0.02, 5.0, 120)
mu = np.linspace(-1, 1, 31)
F = np.outer(E ** -3.5, np.ones_like(mu))
ea = FastElectrons.from_array(E, mu, F, nb=uniform(1e5))
en = FastElectrons(nb=uniform(1e5), delta=3.5, emin_mev=0.02, emax_mev=5.0)
ia = rs.make_gs_image(gs, freqs, ea, ds=0.05, **kw).intensity[:, 1, 1]
iN = rs.make_gs_image(gs, freqs, en, ds=0.05, **kw).intensity[:, 1, 1]
dA = np.abs(ia / iN - 1).max()
check("array-defined power law = analytical power law", dA < 0.03,
      f"max relative difference {dA:.2e}")

# --------------------------------------------------------------------------
# 8. CLI
# --------------------------------------------------------------------------
from active_region import make_ar_model                                    # noqa: E402
p = os.path.join(OUT, "ar_grid.npz")
make_ar_model(n=60).to_npz(p)
out = os.path.join(OUT, "cli_gs")
r = subprocess.run([sys.executable, "-m", "radiosynth", p, "--mechanism", "gyrosynchrotron",
                    "--freq", "5GHz, 17GHz", "--size", "0.3", "0.3", "--npix", "12", "12",
                    "--nb", "1e6", "--nb-center", "-1.04", "0", "0", "--nb-radius", "0.02",
                    "--delta", "3.5", "--out", out, "--formats", "npz", "fits"],
                   cwd=ROOT, capture_output=True, text=True)
ok_cli = r.returncode == 0 and os.path.exists(out + ".npz") and os.path.exists(out + "_V.fits")
if ok_cli:
    z_ = np.load(out + ".npz")
    ok_cli = z_["intensity"].max() > 0 and z_["stokes_v"].shape == (2, 12, 12)
check("CLI --mechanism gyrosynchrotron", ok_cli,
      (r.stderr.strip().splitlines() or ["ok"])[-1] + " | " +
      " ".join(l for l in r.stdout.splitlines() if "GHz:" in l)[:200])

print(f"Summary: {sum(results)} / {len(results)} passed")
sys.exit(0 if all(results) else 1)
