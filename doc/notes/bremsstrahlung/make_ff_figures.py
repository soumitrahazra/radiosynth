# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
# Makes ff_rays.pdf and ff_profiles.pdf (figures of radiosynth_bremsstrahlung.tex and
# of the story) and prints the numbers of their tables.
# Usage: python make_ff_figures.py   (from anywhere; uses the radiosynth of this repository)
import sys, math, numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))      # repository root
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import radiosynth as rs
from radiosynth.common.constants import critical_density, K_B, C_LIGHT, M_E, E2_JM
from radiosynth.bremsstrahlung.emission import bremsstrahlung_coefficients
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def newkirk(r): return 4.2e4 * 10 ** (4.32 / r) * 1e6       # m^-3
print("ncr 1GHz", critical_density(1e9))
print("plasma levels (Newkirk): f, ncr[cm-3], r(ncr), r(ncr/4)")
for f in (30e6, 80e6, 150e6, 300e6, 1e9):
    nc = critical_density(f)
    r1 = 4.32 / math.log10(nc / 4.2e10); r4 = 4.32 / math.log10(nc / 4 / 4.2e10)
    print(f"{f/1e6:6.0f} {nc*1e-6:.3e} {r1:.3f} {r4:.3f}")
ne, T = 1e15, 1e6
for f in (80e6, 300e6, 1e9, 5e9):
    k10 = bremsstrahlung_coefficients(np.array([ne]), np.array([T]), f, gaunt=10.0)[0][0]
    kd = bremsstrahlung_coefficients(np.array([ne]), np.array([T]), f, gaunt="dulk")[0][0]
    print(f"kappa n=1e9cm-3 T=1MK f={f/1e6:g}MHz g10 {k10:.3e} dulk {kd:.3e} ratio {kd/k10:.3f}  G={kd/k10*10:.2f}")
print("prefactor const", 4/3*math.sqrt(2*math.pi/3)*E2_JM**3/C_LIGHT/(K_B*M_E)**1.5)

# 1D Newkirk corona
r = np.geomspace(1.0, 10.0, 600)
g = rs.grid_from_profile(r, newkirk(r), np.full(r.size, 1.5e6), r_body=1.0)
# radial optical depth from infinity down to r (straight radial line), g=10, full
for f in (80e6, 150e6, 300e6, 1e9):
    k, _ = bremsstrahlung_coefficients(newkirk(r), np.full(r.size, 1.5e6), f, gaunt=10.0)
    tau = np.concatenate([np.cumsum((0.5*(k[1:]+k[:-1])*np.diff(r)*6.96e8)[::-1])[::-1], [0]])
    i1 = np.argmin(np.abs(tau - 1)) ; print(f"f={f/1e6:g} MHz: radial tau=1 at r={r[i1]:.3f}, tau(1.0)={tau[0]:.3g}")

freqs = ["80MHz", "150MHz", "300MHz", "1GHz"]
prof = {}
for label, kw in (("ref_full", dict(transfer="full")), ("los_full", dict(transfer="full", no_refraction=True)),
                  ("ref_thin", dict(transfer="thin"))):
    s = rs.make_symmetric_image(g, freqs, x_size=5, nx=64, ny=64, n_p=500, p_max=2.5, verbose=False, **kw)
    prof[label] = s.profile
    print(label, "flux sfu", np.round(s.profile["flux"] / 1e-22, 3), "Tb centre MK", np.round(s.profile["tb"][:, 0] / 1e6, 3))
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.3))
for k, f in enumerate(freqs):
    c = f"C{k}"
    ax[0].plot(prof["ref_full"]["p"], prof["ref_full"]["tb"][k] / 1e6, color=c, label=f + " refracted")
    ax[0].plot(prof["los_full"]["p"], prof["los_full"]["tb"][k] / 1e6, color=c, ls="--", label=f + " straight")
ax[0].axvline(1, color="gray", lw=0.6, ls=":")
ax[0].set_xlabel(r"distance from disk centre $p$ [$R_\odot$]"); ax[0].set_ylabel(r"$T_b$ [MK] (full transfer)")
ax[0].set_xlim(0, 2.5); ax[0].legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), columnspacing=0.8); ax[0].set_title("Newkirk corona, 1.5 MK, $g=10$", fontsize=9)
for k, f in enumerate(freqs):
    ax[1].semilogy(prof["ref_thin"]["p"], np.maximum(prof["ref_thin"]["tb"][k], 1) / 1e6, color=f"C{k}", ls="-.", label=f + " thin")
    ax[1].semilogy(prof["ref_full"]["p"], np.maximum(prof["ref_full"]["tb"][k], 1) / 1e6, color=f"C{k}", label=f + " full")
ax[1].axhline(1.5, color="gray", lw=0.6); ax[1].text(2.45, 1.6, r"$T_e$", ha="right", fontsize=8, color="gray")
ax[1].set_xlabel(r"$p$ [$R_\odot$]"); ax[1].set_ylabel(r"$T_b$ [MK]"); ax[1].set_xlim(0, 2.5); ax[1].set_ylim(1e-3, 30)
ax[1].legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), columnspacing=0.8); ax[1].set_title("thin vs full transfer (refracted)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "ff_profiles.pdf"))

# ray paths at 80 MHz in the equatorial plane
fig, ax = plt.subplots(figsize=(5.0, 4.4))
ys = np.linspace(-2.2, 2.2, 23)
res = rs.trace_paths(g, "80MHz", (-215, 0, 0), ys, np.zeros_like(ys))   # image-plane X, Y
for i, pth in res["paths"].items():
    ax.plot(pth[:, 0], pth[:, 1], color="C0", lw=0.8)
t = np.linspace(0, 2*np.pi, 300)
ax.fill(np.cos(t), np.sin(t), color="orange")
nc = critical_density(80e6); rc = 4.32 / math.log10(nc / 4.2e10)
ax.plot(rc*np.cos(t), rc*np.sin(t), "r--", lw=0.8, label=r"critical surface $n_e=n_\mathrm{cr}$")
ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_xlabel(r"$x$ [$R_\odot$] (observer at $x=-215$)"); ax.set_ylabel(r"$y$ [$R_\odot$]")
ax.legend(fontsize=7, loc="upper right"); ax.set_title("80 MHz rays, Newkirk corona", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "ff_rays.pdf"))
print("rc 80MHz", rc)
