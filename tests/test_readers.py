# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Round-trip tests for the MHD file readers and the command-line interface.

Writes the analytic corona in BATS-R-US-like formats (dimensional units:
Rho [g/cm^3], P/Pe [dyne/cm^2]), reads them back and checks n_e and T_e.
"""
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))
import radiosynth as rs                                   # noqa: E402
from radiosynth.constants import M_P, K_B                # noqa: E402
from synthetic_corona import newkirk                      # noqa: E402

OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

n = 41
ax = np.linspace(-2.5, 2.5, n)
X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
R = np.sqrt(X**2 + Y**2 + Z**2)
NE = newkirk(np.maximum(R, 0.98))                      # m^-3
TE = 1.2e6 + 0.3e6 * np.tanh(R - 1.5)                  # K
TI = 1.6e6
RHO = NE * M_P / 1e3                                   # g/cm^3
PE = NE * K_B * TE * 10                                # dyne/cm^2
P = NE * K_B * (TE + TI) * 10
UX = np.zeros_like(R)
ok = []


def check(name, grid):
    ne = grid._flat[:, 0].reshape(grid.shape)
    te = grid._flat[:, 4].reshape(grid.shape)
    sel = R > 1.0
    e1 = np.max(np.abs(ne[sel] / NE[sel] - 1))
    e2 = np.max(np.abs(te[sel] / TE[sel] - 1))
    good = e1 < 1e-5 and e2 < 1e-5
    ok.append(good)
    print(f"[{'PASS' if good else 'FAIL'}] {name}: max rel err n_e {e1:.1e}, T_e {e2:.1e}")


# ---- Tecplot ASCII, ordered zone, POINT packing, x fastest
tec = os.path.join(OUT, "box_test.dat")
with open(tec, "w") as fh:
    fh.write('TITLE="BATSRUS: test"\n')
    fh.write('VARIABLES = "X [R]", "Y [R]", "Z [R]", "Rho [g/cm^3]", "U_x [km/s]", '
             '"P [dyne/cm^2]", "Pe [dyne/cm^2]"\n')
    fh.write(f'ZONE T="3D", I={n}, J={n}, K={n}, DATAPACKING=POINT\n')
    cols = [a.transpose(2, 1, 0).ravel() for a in (X, Y, Z, RHO, UX, P, PE)]
    np.savetxt(fh, np.column_stack(cols), fmt="%.9e")
check("Tecplot ASCII (Rho g/cm^3 + Pe)", rs.load_mhd(tec))


# ---- BATS-R-US IDL binary (real4) -- regular 3D box
def frec(fh, payload):
    fh.write(struct.pack("<i", len(payload)))
    fh.write(payload)
    fh.write(struct.pack("<i", len(payload)))


idl = os.path.join(OUT, "box_test.out")
names = "x y z rho ux p pe g rbody"
units = "R R R g/cm3 km/s dyne/cm^2 dyne/cm^2"
with open(idl, "wb") as fh:
    frec(fh, f"{units}".ljust(500).encode())
    frec(fh, struct.pack("<ifiii", 100, 3600.0, 3, 2, 4))
    frec(fh, struct.pack("<3i", n, n, n))
    frec(fh, struct.pack("<2f", 1.6667, 1.0))
    frec(fh, names.ljust(500).encode())
    coord = np.stack([X, Y, Z], axis=-1).astype("<f4")
    frec(fh, coord.ravel(order="F").tobytes())
    for a in (RHO, UX, P, PE):
        frec(fh, a.astype("<f4").ravel(order="F").tobytes())
g = rs.load_mhd(idl)
ne = g._flat[:, 0].reshape(g.shape)
e = np.max(np.abs(ne[R > 1] / NE[R > 1] - 1))
ok.append(e < 1e-5)
print(f"[{'PASS' if e < 1e-5 else 'FAIL'}] IDL real4 binary: max rel err n_e {e:.1e} "
      f"(float32 storage)")

# ---- IDL ascii, unstructured ordering (shuffled) with only P (Te=Ti assumption)
idla = os.path.join(OUT, "box_test_ascii.out")
perm = np.random.default_rng(1).permutation(n**3)
P_eq = 2 * NE * K_B * TE * 10
with open(idla, "w") as fh:
    fh.write("R R R g/cm3 dyne/cm^2\n")
    fh.write(f"0 0.0 -3 0 2\n{n**3} 1 1\n")
    fh.write("x y z rho p\n")
    arr = np.column_stack([X.ravel(), Y.ravel(), Z.ravel(), RHO.ravel(), P_eq.ravel()])[perm]
    np.savetxt(fh, arr, fmt="%.10e")
check("IDL ascii, shuffled points, Te from P/2", rs.load_mhd(idla))

# ---- scattered points -> resample
rng = np.random.default_rng(0)
pts = rng.uniform(-2.5, 2.5, (60000, 3))
rr = np.linalg.norm(pts, axis=1)
v = {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
     "rho [g/cm^3]": newkirk(np.maximum(rr, 0.98)) * M_P / 1e3, "te [K]": np.full(rr.size, 1.5e6)}
g = rs.build_grid(v, resample_shape=(41, 41, 41), resample_method="linear")
gne = g._flat[:, 0].reshape(g.shape)
gx = g.x
GX, GY, GZ = np.meshgrid(gx, g.y, g.z, indexing="ij")
GR = np.sqrt(GX**2 + GY**2 + GZ**2)
sel = (GR > 1.3) & (GR < 2.3)
e = np.median(np.abs(gne[sel] / newkirk(GR[sel]) - 1))
ok.append(e < 0.02)
print(f"[{'PASS' if e < 0.02 else 'FAIL'}] scattered points resampled (linear in log n): "
      f"median rel err {e:.1e}")

# ---- CLI end to end
cmd = [sys.executable, "-m", "radiosynth", tec, "--freq", "100MHz, 1GHz",
       "--npix", "32", "32", "--size", "4", "4", "--out", os.path.join(OUT, "cli_test"),
       "--plot", "--workers", "2"]
r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
good = r.returncode == 0 and os.path.exists(os.path.join(OUT, "cli_test.npz"))
ok.append(good)
print(f"[{'PASS' if good else 'FAIL'}] CLI run")
if not good:
    print(r.stdout[-2000:], r.stderr[-2000:])
z = np.load(os.path.join(OUT, "cli_test.npz"))
print("    CLI image shape", z["intensity"].shape, "Tb max", z["tb"].max(axis=(1, 2)))
print("Summary:", sum(ok), "/", len(ok), "passed")
