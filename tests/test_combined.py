# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of the combined folder (free-free + gyroresonance + gyrosynchrotron).

 1. without fast electrons the combined image equals make_gr_image exactly
 2. gyrosynchrotron alone: per-voxel coefficients + radiosynth transfer reproduce
    the fast GS codes' own line-of-sight transfer
 3. per-voxel optical depth independent of the probe intensity
 4. optically thin: combined = sum of the separate contributions; thick: below the sum
 5. a thick gyroresonance layer in front hides a gyrosynchrotron source behind it
 6. contributions add up to 1; B -> -B gives V -> -V
 7. command line --mechanism combined
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
import radiosynth as rs                                               # noqa: E402
from radiosynth.gyrosynchrotron import FastElectrons, uniform, in_sphere, in_box   # noqa: E402
from radiosynth.gyrosynchrotron.library import find_library, load_library          # noqa: E402
from radiosynth.combined import gs_voxel_coefficients                 # noqa: E402
from radiosynth.common.magnetoionic import NU_B_PER_GAUSS             # noqa: E402
from active_region import make_ar_model                               # noqa: E402

OUT = os.path.join(HERE, "output", "combined")
os.makedirs(OUT, exist_ok=True)
results = []


def check(name, cond, msg):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {msg}")


ar = make_ar_model(n=80)
view = dict(observer=(-215, 0, 0), x_size=0.36, y_size=0.36, nx=30, ny=30, verbose=False)

# 1. -----------------------------------------------------------------------
a = rs.make_combined_image(ar, "3GHz, 5GHz, 17GHz", None, **view)
b = rs.make_gr_image(ar, "3GHz, 5GHz, 17GHz", **view)
same = all(np.array_equal(getattr(a, k), getattr(b, k))
           for k in ("intensity", "stokes_v", "tau", "dominant_s", "gr_fraction"))
check("no fast electrons: combined = gyroresonance image", same, "bit-for-bit identical")

try:
    find_library()
except FileNotFoundError:
    print("SKIP gyrosynchrotron parts: fast GS codes not built")
    print(f"Summary: {sum(results)} / {len(results)} passed (gyrosynchrotron tests skipped)")
    sys.exit(0 if all(results) else 1)

x = np.linspace(-3.0, -2.0, 21)
y = np.linspace(-0.2, 0.2, 5)
z = y.copy()
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")


def slab(n, T, B, th_deg, bfun=None):
    g = rs.MHDGrid(x, y, z, np.full(X.shape, n), np.full(X.shape, T))
    th = math.radians(th_deg)
    bm = np.full(X.shape, B) if bfun is None else bfun(X)
    g.add_field(-bm * math.cos(th), -bm * math.sin(th), 0 * X)
    return g


kw = dict(observer=(-1.0, 0, 0), x_size=0.1, y_size=0.1, nx=3, ny=3, distance_pc=1e-6,
          verbose=False)
fr = np.array([1e9, 3e9, 1e10, 3e10])

# 2. -----------------------------------------------------------------------
g = slab(1e15, 2e7, 200.0, 50.0)
e = FastElectrons(nb=uniform(1e5), delta=3.5, emin_mev=0.02, emax_mev=5.0, pitch="gau",
                  alpha_c=60.0, dmu=0.2)
c = rs.make_combined_image(g, fr, e, include=("gs",), ds=0.05, **kw)
d = rs.make_gs_image(g, fr, e, free_free=False, coupling="weak", ds=0.05, **kw)
rel = max(np.abs(c.pol1[:, 1, 1] / d.pol1[:, 1, 1] - 1).max(),
          np.abs(c.pol2[:, 1, 1] / d.pol2[:, 1, 1] - 1).max())
check("gyrosynchrotron alone: per-voxel coefficients = library transfer", rel < 1e-6,
      f"max relative difference {rel:.1e} (T_b = {np.round(c.brightness_temperature[:, 1, 1] / 1e6, 1)} MK, "
      f"thin to thick)")

# 3. -----------------------------------------------------------------------
lib = load_library()
from radiosynth.gyrosynchrotron.image import voxel_parms                     # noqa: E402
fl = dict(te=np.array([2e7]), ne=np.array([1e15]), bmag=np.array([200.0]))
par = e.evaluate(dict(fl, pos=np.zeros((1, 3)), r=np.ones(1), b=np.zeros((1, 3))))
P = voxel_parms(fl, par, np.array([3.5e9]), np.array([True]), np.array([1e5]),
                np.array([50.0]), False, 3, 3, -1, False)
c1 = gs_voxel_coefficients(lib, P, fr)
# tau from two independent one-voxel calls with different incoming intensities
Lp = np.array([1, 1, fr.size, 0, 0, 0, 0, 0, 1, 0, 0, 0], np.int32)
Rp = np.zeros((5, 1), order="F")
Rp[0], Rp[3], Rp[4] = 1e20, 12, 12
taus = []
for I0 in (1.0, 1e4):
    RL0 = np.zeros((7, fr.size, 1), order="F")
    RL0[0] = (fr / 1e9)[:, None]
    lib.get_mw_slice(Lp, Rp, np.asfortranarray(P.reshape(24, 1, 1)), np.zeros(1), np.zeros(1),
                     np.zeros((1, 1, 1, 1), order="F"), RL0)
    RL1 = np.zeros((7, fr.size, 1), order="F")
    RL1[0] = (fr / 1e9)[:, None]
    RL1[1:7] = I0
    lib.get_mw_slice(Lp, Rp, np.asfortranarray(P.reshape(24, 1, 1)), np.zeros(1), np.zeros(1),
                     np.zeros((1, 1, 1, 1), order="F"), RL1)
    with np.errstate(divide="ignore", invalid="ignore"):
        taus.append(-np.log((RL1[2, :, 0] - RL0[2, :, 0]) / I0))
# compare where the voxel is neither opaque nor too thin for the weak probe (3 and 10 GHz)
sel3 = [1, 2]
dt = np.abs(taus[0][sel3] / taus[1][sel3] - 1).max()
dt2 = np.abs(c1["tau_R"][sel3, 0] / taus[1][sel3] - 1).max()
check("per-voxel optical depth independent of the probe intensity", dt < 1e-6 and dt2 < 1e-6,
      f"tau_R (1, 3, 10, 30 GHz) = {c1['tau_R'][:, 0]}; probe 1 vs 1e4 sfu: {dt:.1e}, "
      f"vs coefficients: {dt2:.1e}")

# 4. -----------------------------------------------------------------------
gt = slab(1e13, 2e6, 30.0, 50.0)
et = FastElectrons(nb=uniform(1e3), delta=3.5, emin_mev=0.02, emax_mev=5.0)
full = rs.make_combined_image(gt, "5GHz, 15GHz", et, **kw)
ff = rs.make_combined_image(gt, "5GHz, 15GHz", None, include=("ff",), **kw)
gs = rs.make_combined_image(gt, "5GHz, 15GHz", et, include=("gs",), **kw)
thin_err = np.abs(full.intensity[:, 1, 1] / (ff.intensity + gs.intensity)[:, 1, 1] - 1).max()
gk = slab(1e15, 2e7, 200.0, 50.0)
ek = FastElectrons(nb=uniform(1e6), delta=3.0, emin_mev=0.02, emax_mev=5.0)
fullk = rs.make_combined_image(gk, "1GHz, 3GHz", ek, **kw)
sumk = (rs.make_combined_image(gk, "1GHz, 3GHz", None, **kw).intensity
        + rs.make_combined_image(gk, "1GHz, 3GHz", ek, include=("gs",), **kw).intensity)
ratio_k = fullk.intensity[:, 1, 1] / sumk[:, 1, 1]
check("thin: combined = sum of contributions; thick: below the sum",
      thin_err < 1e-3 and ratio_k[0] < 0.99 and (ratio_k <= 1 + 1e-9).all(),
      f"thin relative difference {thin_err:.1e} (tau = {full.tau[:, 1, 1].round(5)}); "
      f"thick combined/sum = {ratio_k.round(3)}")

# 5. -----------------------------------------------------------------------
# |B| decreasing towards the observer: the s = 3 layer (at |B| = nu/(3 nu_B/G)) lies in
# front of a fast-electron box at the far end of the slab
f5 = 5e9
b3 = f5 / (3 * NU_B_PER_GAUSS)
g5 = slab(3e15, 2e6, 0.0, 60.0, bfun=lambda X_: b3 * (1.0 + 0.6 * (X_ + 2.5)))
e5 = FastElectrons(nb=in_box((-2.25, -1, -1), (-2.0, 1, 1), 1e6), delta=3.5, emin_mev=0.02,
                   emax_mev=5.0)
comb5 = rs.make_combined_image(g5, f5, e5, **kw)
gs5 = rs.make_combined_image(g5, f5, e5, include=("gs",), **kw)
gr5 = rs.make_combined_image(g5, f5, None, include=("gr",), **kw)
T_e = 2e6
check("thick layer in front hides the gyrosynchrotron source",
      gs5.brightness_temperature[0, 1, 1] > 10 * T_e
      and abs(comb5.brightness_temperature[0, 1, 1] / T_e - 1) < 0.05
      and comb5.dominant_mechanism[0, 1, 1] == 1,
      f"T_b: gs alone {gs5.brightness_temperature[0, 1, 1]:.3g} K, layer alone "
      f"{gr5.brightness_temperature[0, 1, 1]:.3g} K, combined {comb5.brightness_temperature[0, 1, 1]:.3g} K "
      f"(dominant: {rs.combined.MECH_NAMES[int(comb5.dominant_mechanism[0, 1, 1])]})")

# 6. -----------------------------------------------------------------------
eb = FastElectrons(nb=in_sphere((-1.04, 0, 0), 0.015, 1e7), delta=3.5, emin_mev=0.02,
                   emax_mev=5.0)
im = rs.make_combined_image(ar, "3GHz, 10GHz", eb, **view)
ssum = im.ff_fraction + im.gr_fraction + im.gs_fraction
okf = np.allclose(ssum[im.intensity > 0], 1.0, atol=1e-9)
ar2 = make_ar_model(n=80)
ar2._bflat = -ar2._bflat
im2 = rs.make_combined_image(ar2, "3GHz, 10GHz", eb, **view)
sym = np.abs(im.stokes_v + im2.stokes_v).max() / np.abs(im.stokes_v).max()
check("fractions add up to 1; B -> -B gives V -> -V", okf and sym < 1e-6,
      f"|V(B)+V(-B)|/max|V| = {sym:.1e}")
im.plot("mechanism", path=os.path.join(OUT, "mechanism.png"))

# 8. free-free refraction option -------------------------------------------
from synthetic_corona import make_model                                       # noqa: E402
cg = make_model(n=81)
cg.add_field(*(np.zeros(cg.shape),) * 3)
k8 = dict(observer=(-215, 0, 0), x_size=5, y_size=5, nx=24, ny=24, verbose=False)
cr = rs.make_combined_image(cg, "80MHz, 2GHz", None, include=("ff",), gaunt=10.0, ds=0.005,
                            ff_refraction=True, trace_kw=dict(step_factor=0.25), **k8)
mr = rs.make_radio_image(cg, "80MHz, 2GHz", gaunt=10.0, transfer="full", step_factor=0.25, **k8)
ms = rs.make_combined_image(cg, "80MHz, 2GHz", None, include=("ff",), gaunt=10.0, ds=0.005, **k8)
r8 = cr.flux_density / mr.flux_density
r8s = ms.flux_density / mr.flux_density
hi = rs.make_combined_image(ar, "17GHz", None, ff_refraction=True, **view)
hs = rs.make_combined_image(ar, "17GHz", None, **view)
dh = np.abs(hi.intensity - hs.intensity).max() / hs.intensity.max()
check("ff_refraction: free-free only = refracted make_radio_image; no effect at 17 GHz",
      np.abs(r8 - 1).max() < 2e-3 and abs(r8s[0] - 1) > 0.05 and dh < 1e-3,
      f"combined(ff, refraction)/refracted tracer flux = {r8.round(4)} "
      f"(straight: {r8s.round(3)}); 17 GHz active region change {dh:.1e}")

# 7. -----------------------------------------------------------------------
p = os.path.join(OUT, "ar_grid.npz")
make_ar_model(n=60).to_npz(p)
out = os.path.join(OUT, "cli")
r = subprocess.run([sys.executable, "-m", "radiosynth", p, "--mechanism", "combined",
                    "--freq", "5GHz", "--size", "0.3", "0.3", "--npix", "12", "12",
                    "--nb", "1e7", "--nb-center", "-1.04", "0", "0", "--nb-radius", "0.015",
                    "--out", out, "--plot", "--formats", "npz"],
                   cwd=ROOT, capture_output=True, text=True)
okc = r.returncode == 0 and os.path.isfile(out + "_mechanism.png")
if okc:
    zz = np.load(out + ".npz", allow_pickle=True)
    okc = "gs_fraction" in zz.files and zz["intensity"].max() > 0
check("CLI --mechanism combined", okc,
      (r.stderr.strip().splitlines() or ["ok"])[-1] + " | " +
      " ".join(l.strip() for l in r.stdout.splitlines() if "GHz:" in l)[:160])

print(f"Summary: {sum(results)} / {len(results)} passed")
sys.exit(0 if all(results) else 1)
