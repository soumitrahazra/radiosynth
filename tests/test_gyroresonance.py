# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of the gyroresonance package and the magnetic-field plumbing.

 1. layer optical depth vs the Maxwellian average of exact Bessel functions
 2. single harmonic layer in a linear field: transfer = S (1 - exp(-tau)), V sign
 3. optically thick layers: T_b of each polarisation = T_e
 4. polarisation sign follows the field; reversing B reverses V exactly
 5. straight-ray free-free (B = 0) = refractive tracer with no_refraction
 6. optically thin magnetised free-free: V/I = 2q/(1+q^2), q = Y |cos(theta)|
 7. SphericalGrid (B_r, B_theta, B_phi) = MHDGrid (B_x, B_y, B_z)
 8. readers: npz round trip with B, Bx/By/Bz units, PLUTO bx1..bx3 with_b
 9. command line --mechanism gyroresonance
"""
import math
import os
import subprocess
import sys

import numpy as np
from scipy import integrate, special

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))
import radiosynth as rs                                          # noqa: E402
from radiosynth import gyroresonance as gr                      # noqa: E402
from radiosynth.common.constants import K_B, M_E, C_LIGHT, M_P  # noqa: E402
from radiosynth.common.magnetoionic import NU_B_PER_GAUSS        # noqa: E402
from active_region import make_ar_model, make_ar_spherical       # noqa: E402
from synthetic_corona import make_model                          # noqa: E402

OUT = os.path.join(HERE, "output", "gyroresonance")
os.makedirs(OUT, exist_ok=True)
results = []


def check(name, cond, msg):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {msg}")


# --------------------------------------------------------------------------
# 1. tau_s = prefactor * s * <J_{s-1}(s beta_perp sin)^2>  to lowest order in beta
# --------------------------------------------------------------------------
te, ne, lb, f, cth = 2e6, 1e15, 1e7, 5e9, 0.6
b2 = K_B * te / (M_E * C_LIGHT**2)
worst = 0.0
for s in range(2, 7):
    st = math.sqrt(1 - cth**2)

    def integrand(bp):     # 2D Maxwellian in beta_perp
        return special.jv(s - 1, s * bp * st) ** 2 * bp / b2 * np.exp(-bp * bp / (2 * b2))
    avg = integrate.quad(integrand, 0, 30 * math.sqrt(b2), limit=200)[0]
    F = gr.mode_factor(0.0, 1.0 / s, cth, "x")
    ref = gr.GR_PREFACTOR * ne * lb / f * s * avg * F
    tau = float(gr.layer_optical_depth(s, ne, te, cth, lb, f, "x"))
    worst = max(worst, abs(tau / ref - 1))
check("layer tau vs Maxwellian <J_{s-1}^2> (s=2..6)", worst < 0.01,
      f"max relative difference {worst:.2e} (O(s^2 beta^2) = {36 * b2:.1e})")

# --------------------------------------------------------------------------
# 2. one layer in a linear field: |B| = b(x), direction fixed at theta to the ray
# --------------------------------------------------------------------------
x = np.linspace(-3.0, -2.0, 41)
y = np.linspace(-0.2, 0.2, 5)
z = np.linspace(-0.2, 0.2, 5)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
f1, s1, theta = 3e9, 3, math.radians(50.0)
b_layer = f1 / (s1 * NU_B_PER_GAUSS)
bmag = b_layer * (1.0 + 0.4 * (X + 2.5))            # 0.8 .. 1.2 b_layer: only s = 3 inside
n0, T0 = 6e11, 3e6
g = rs.MHDGrid(x, y, z, np.full(X.shape, n0), np.full(X.shape, T0), r_body=1.0)
# B . n_obs > 0 (n_obs = -x): B along -x tilted by theta
g.add_field(-bmag * math.cos(theta), -bmag * math.sin(theta), 0 * bmag)
img = rs.make_gr_image(g, f1, observer=(-1.0, 0, 0), x_size=0.1, y_size=0.1, nx=3, ny=3,
                       mechanisms="gr", distance_pc=1e-6, verbose=False)
l_b = b_layer / (0.4 * b_layer) * rs.R_SUN          # |B|/|dB/dl| in metres
Xc = n0 / rs.critical_density(f1)
tx = float(gr.layer_optical_depth(s1, n0, T0, math.cos(theta), l_b, f1, "x", X=Xc))
to = float(gr.layer_optical_depth(s1, n0, T0, math.cos(theta), l_b, f1, "o", X=Xc))
S = f1**2 * K_B * T0 / C_LIGHT**2
eR = img.pol1[0, 1, 1] / (S * -math.expm1(-tx)) - 1
eL = img.pol2[0, 1, 1] / (S * -math.expm1(-to)) - 1
check("single s=3 layer: I_R = S(1-e^-tau_x), I_L = S(1-e^-tau_o)",
      abs(eR) < 1e-6 and abs(eL) < 1e-6 and img.stokes_v[0, 1, 1] > 0,
      f"tau_x={tx:.3f}, tau_o={to:.3f}, rel. errors {eR:.1e}, {eL:.1e}, V/I={img.polarization[0, 1, 1]:+.3f}")

# --------------------------------------------------------------------------
# 3.-4. dipole active region
# --------------------------------------------------------------------------
ar = make_ar_model(n=100)
kw = dict(x_size=0.36, y_size=0.36, nx=60, ny=60, verbose=False)
im = rs.make_gr_image(ar, "5GHz, 17GHz", **kw)
T_e = 2.5e6
t1, t2 = im.tb_pol
thick = (im.tau > 20) & (im.tau2 > 20)
err = max(np.abs(t1[thick] / T_e - 1).max(), np.abs(t2[thick] / T_e - 1).max())
check("optically thick: T_b(R) = T_b(L) = T_e",
      thick.sum() > 100 and err < 1e-3 and im.brightness_temperature.max() <= T_e * (1 + 1e-9),
      f"{thick.sum()} thick pixels, max |T_p/T_e - 1| = {err:.1e}, max T_b = {im.brightness_temperature.max():.4e} K")

# image X = -y for the observer on -x: the positive spot (+y) is on the left
left = im.stokes_v[0][:, :30].sum()
right = im.stokes_v[0][:, 30:].sum()
ar_rev = make_ar_model(n=100)
ar_rev._bflat = -ar_rev._bflat
im_rev = rs.make_gr_image(ar_rev, "5GHz, 17GHz", **kw)
sym = np.abs(im_rev.stokes_v + im.stokes_v).max() / np.abs(im.stokes_v).max()
same = np.abs(im_rev.intensity - im.intensity).max() / im.intensity.max()
check("polarisation sign: RCP over the positive spot, V -> -V when B -> -B",
      left > 0 and right < 0 and sym < 1e-12 and same < 1e-12,
      f"V left {left:+.2e}, right {right:+.2e}; |V(-B)+V(B)|/max = {sym:.1e}, I change {same:.1e}")
im.plot("tb", path=os.path.join(OUT, "ar_tb.png"))
im.plot("pol", path=os.path.join(OUT, "ar_pol.png"))
im.plot("harmonic", path=os.path.join(OUT, "ar_harmonic.png"))
im.save_npz(os.path.join(OUT, "ar.npz"))
check("dominant harmonics at 5 GHz are s = 2-4",
      set(np.unique(im.dominant_s[0][im.gr_fraction[0] > 0.5])) <= {2, 3, 4},
      f"harmonics {sorted(set(np.unique(im.dominant_s[0][im.gr_fraction[0] > 0.5])))}, "
      f"V/I(flux) = {im.flux_density_v[0] / im.flux_density[0]:+.3f}")

# --------------------------------------------------------------------------
# 5. straight-ray free-free (B = 0) vs refractive tracer without refraction
# --------------------------------------------------------------------------
cg = make_model(n=81)
cg.add_field(*(np.zeros(cg.shape),) * 3)
k5 = dict(x_size=4.0, y_size=4.0, nx=24, ny=24, verbose=False)
a = rs.make_gr_image(cg, "300MHz, 2GHz", mechanisms="ff", gaunt=10.0, ds=0.005, **k5)
b = rs.make_radio_image(cg, "300MHz, 2GHz", gaunt=10.0, transfer="full", no_refraction=True,
                        step_factor=0.25, **k5)
dflux = np.abs(a.flux_density / b.flux_density - 1).max()
check("straight-ray free-free = refractive tracer (no_refraction, full transfer)",
      dflux < 2e-3 and np.abs(a.stokes_v).max() == 0,
      f"flux ratio - 1 = {a.flux_density / b.flux_density - 1}")

# --------------------------------------------------------------------------
# 6. thin magnetised free-free: V/I = 2q/(1+q^2)
# --------------------------------------------------------------------------
bth = 30.0
g6 = rs.MHDGrid(x, y, z, np.full(X.shape, 1e13), np.full(X.shape, 2e6), r_body=1.0)
g6.add_field(np.full(X.shape, -bth * math.cos(theta)), np.full(X.shape, -bth * math.sin(theta)),
             0 * X)
f6 = 1e9
im6 = rs.make_gr_image(g6, f6, observer=(-1.0, 0, 0), x_size=0.1, y_size=0.1, nx=3, ny=3,
                       mechanisms="ff", distance_pc=1e-6, verbose=False)
q = bth * NU_B_PER_GAUSS / f6 * math.cos(theta)
p6 = im6.polarization[0, 1, 1]
check("thin magnetised free-free: V/I = 2q/(1+q^2)",
      abs(p6 / (2 * q / (1 + q * q)) - 1) < 1e-3 and im6.tau[0, 1, 1] < 1e-3,
      f"q = {q:.4f}, V/I = {p6:.5f}, expected {2 * q / (1 + q * q):.5f}, tau = {im6.tau[0, 1, 1]:.1e}")

# --------------------------------------------------------------------------
# 7. spherical grid with (B_r, B_theta, B_phi)
# --------------------------------------------------------------------------
sg, edges, fields = make_ar_spherical(nr=300, nth=100, nph=100)
pts = np.array([[-1.05, 0.02, 0.01], [-1.1, -0.05, 0.03], [-1.02, 0.08, -0.04]])
from active_region import dipole_field                          # noqa: E402
bt = np.column_stack(dipole_field(pts[:, 0], pts[:, 1], pts[:, 2]))
bs = sg.sample_field(pts)
berr = np.abs(bs - bt).max() / np.abs(bt).max()
ims = rs.make_gr_image(sg, "5GHz, 17GHz", **kw)
dfl = np.abs(ims.flux_density / im.flux_density - 1).max()
vl = lambda m: m.stokes_v[:, :, :30].sum(axis=(1, 2))           # noqa: E731  V of the + spot
dv = np.abs(vl(ims) / vl(im) - 1).max()
check("SphericalGrid B (r,theta,phi) vs analytic and vs Cartesian image",
      berr < 0.01 and dfl < 0.015 and dv < 0.03,      # 17 GHz source is a few cells wide
      f"field interpolation error {berr:.1e}; flux I ratio - 1 = {dfl:.1e}, "
      f"V of the positive spot: {dv:.1e}")

# --------------------------------------------------------------------------
# 8. readers
# --------------------------------------------------------------------------
p = os.path.join(OUT, "ar_grid.npz")
ar.to_npz(p)
g2 = rs.load_mhd(p)
pp = np.array([[-1.05, 0.03, -0.02]])
ok_npz = g2.has_field and np.allclose(g2.sample_field(pp), ar.sample_field(pp))
sg.to_npz(os.path.join(OUT, "ar_sph.npz"))
g3 = rs.load_mhd(os.path.join(OUT, "ar_sph.npz"))
ok_npz &= np.allclose(g3.sample_field(pp), sg.sample_field(pp))
xs = np.linspace(-1.2, -0.8, 6)
XX, YY, ZZ = np.meshgrid(xs, xs + 1, xs + 1, indexing="ij")
v = {"X [R]": XX.ravel(), "Y [R]": YY.ravel(), "Z [R]": ZZ.ravel(),
     "Rho [g/cm^3]": np.full(XX.size, 1e-15),
     "B_x [nT]": np.full(XX.size, 1e5), "B_y [nT]": np.zeros(XX.size),
     "B_z [nT]": np.full(XX.size, -2e5)}
v = {k.split()[0].lower() if k[0] in "XYZ" else k: a_ for k, a_ in v.items()}
g4 = rs.build_grid(v, with_b=True, te_const=1e6, verbose=False)
b4 = g4.sample_field(np.array([[-1.0, 0.0, 0.0]]))[0]
check("npz round trip with B; Bx/By/Bz [nT] -> gauss", ok_npz and np.allclose(b4, [1.0, 0.0, -2.0]),
      f"B = {b4} G (expected [1, 0, -2])")

# PLUTO spherical .dbl with bx1..bx3 in code units
UD, UL, UV = M_P * 1e3 * 1e8, 6.96e10, 1e7         # g/cm^3 (cgs m_p * 1e8), cm, cm/s
d = os.path.join(OUT, "pluto_ar")
os.makedirs(d, exist_ok=True)
r_e, t_e, p_e = edges
ne_, te_, br, bth_, bph = fields
bfac = math.sqrt(4 * math.pi * UD) * UV
rho = ne_ * M_P * 1e-3 / UD                        # code density (mu_e = 1); M_P in kg
prs = 2 * ne_ * 1e-6 * 1.380649e-16 * te_ / (UD * UV**2)
names = ["rho", "prs", "bx1", "bx2", "bx3"]
with open(os.path.join(d, "grid.out"), "w") as fh:
    fh.write("# PLUTO 4.4 Grid File\n# DIMENSIONS: 3\n# GEOMETRY:   SPHERICAL\n")
    for e in (r_e, t_e, p_e):
        fh.write(f"{e.size - 1}\n")
        for i in range(e.size - 1):
            fh.write(f" {i + 1}  {e[i]:.12e}  {e[i + 1]:.12e}\n")
with open(os.path.join(d, "dbl.out"), "w") as fh:
    fh.write(f"0 0.0 1e-4 0 single_file little {' '.join(names)}\n")
with open(os.path.join(d, "data.0000.dbl"), "wb") as fh:
    for a_ in (rho, prs, br / bfac, bth_ / bfac, bph / bfac):
        fh.write(a_.transpose(2, 1, 0).astype("<f8").tobytes())
gp = rs.load_pluto(d, unit_density=UD, unit_length=UL, unit_velocity=UV, with_b=True,
                   verbose=False)
dpl = np.abs(gp.sample_field(pts) - sg.sample_field(pts)).max() / np.abs(bt).max()
nrel = np.abs(gp.sample(pts)[0] / sg.sample(pts)[0] - 1).max()
check("PLUTO bx1..bx3 (code units) with_b=True -> gauss", dpl < 1e-6 and nrel < 1e-6,
      f"max |B_pluto - B_sph|/|B| = {dpl:.1e}, n_e rel diff {nrel:.1e}")

# --------------------------------------------------------------------------
# 9. CLI
# --------------------------------------------------------------------------
out = os.path.join(OUT, "cli_gr")
r = subprocess.run([sys.executable, "-m", "radiosynth", p, "--mechanism", "gyroresonance",
                    "--freq", "5GHz", "--size", "0.3", "0.3", "--npix", "24", "24",
                    "--out", out, "--formats", "npz", "fits"],
                   cwd=ROOT, capture_output=True, text=True)
ok_cli = r.returncode == 0 and os.path.exists(out + ".npz") and os.path.exists(out + "_V.fits")
if ok_cli:
    z_ = np.load(out + ".npz")
    ok_cli = z_["stokes_v"].shape == (1, 24, 24) and z_["intensity"].max() > 0
r2 = subprocess.run([sys.executable, "-m", "radiosynth", "--list-mechanisms"], cwd=ROOT,
                    capture_output=True, text=True)
check("CLI --mechanism gyroresonance, --list-mechanisms",
      ok_cli and "gyroresonance" in r2.stdout,
      (r.stderr.strip().splitlines() or ["ok"])[-1] + " | " + r2.stdout.strip().replace("\n", "; "))

print(f"Summary: {sum(results)} / {len(results)} passed")
sys.exit(0 if all(results) else 1)
