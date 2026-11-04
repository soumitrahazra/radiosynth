# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
# Makes tau_theta.pdf (Fig. 2 of radiosynth_gyroresonance.tex, Fig. 1 of the story)
# and prints the numbers of its Tables 2 and 3.
# Usage: python make_gr_figure.py   (from anywhere; uses the radiosynth of this repository)
import sys, math, numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))      # repository root
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
from radiosynth.gyroresonance.opacity import layer_optical_depth, harmonic_field, GR_PREFACTOR, mode_factor
from radiosynth.common.magnetoionic import NU_B_PER_GAUSS, axial_ratio_x
from radiosynth.common.constants import critical_density, K_B, M_E, C_LIGHT
from radiosynth.bremsstrahlung.emission import bremsstrahlung_coefficients
print("nuB/G", NU_B_PER_GAUSS, "pref", GR_PREFACTOR, "ncr(1GHz) m^-3", critical_density(1e9))
print("beta2(1MK)", K_B*1e6/(M_E*C_LIGHT**2))
ne, L, T, f = 1e15, 1e7, 2e6, 5e9   # 1e9 cm^-3, 1e9 cm
for th in (30, 60, 80):
    c = math.cos(math.radians(th))
    row = []
    for s in range(1, 7):
        X = ne/critical_density(f)
        tx = layer_optical_depth(s, ne, T, c, L, f, "x", X=X); to = layer_optical_depth(s, ne, T, c, L, f, "o", X=X)
        row.append((s, float(tx), float(to)))
    print(th, ["s=%d x=%.3g o=%.3g" % r for r in row])
print("X at 5GHz", ne/critical_density(5e9))
for fr in (1e9, 2e9, 5e9, 10e9, 17e9):
    print(fr/1e9, [round(float(harmonic_field(s, fr))) for s in (1,2,3,4)])
# kappa ff at 5 GHz, 1e9 cm^-3, 2 MK, both Gaunt
for g in (10.0, "dulk"):
    print("kff", g, bremsstrahlung_coefficients(np.array([ne]), np.array([T]), f, gaunt=g)[0])
# F factors vs theta at X=0, Y=1/3
for th in (0, 30, 60, 89.9):
    c = math.cos(math.radians(th))
    print("F", th, float(mode_factor(0.0, 1/3, c, "x")), float(mode_factor(0.0, 1/3, c, "o")), float(axial_ratio_x(0.0, 1/3, c)))
# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
th = np.linspace(1, 89.5, 300); c = np.cos(np.radians(th))
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
cols = {2: "C0", 3: "C1", 4: "C2", 5: "C3"}
X = ne/critical_density(f)
for s in (2, 3, 4, 5):
    ax[0].semilogy(th, layer_optical_depth(s, ne*np.ones_like(c), T*np.ones_like(c), c, L, f, "x", X=X*np.ones_like(c)), color=cols[s], label=f"s={s}, x")
    ax[0].semilogy(th, layer_optical_depth(s, ne*np.ones_like(c), T*np.ones_like(c), c, L, f, "o", X=X*np.ones_like(c)), color=cols[s], ls="--", label=f"s={s}, o")
ax[0].axhline(1, color="gray", lw=0.7)
ax[0].set_ylim(1e-6, 1e6); ax[0].set_xlabel(r"$\theta$ [deg]"); ax[0].set_ylabel(r"$\tau_s$")
ax[0].set_title(r"$n_e=10^9$ cm$^{-3}$, $T_e=2$ MK, $L_B=10^9$ cm, 5 GHz", fontsize=9)
ax[0].legend(fontsize=7, ncol=2)
for s, cc in ((2, "C0"), (3, "C1"), (4, "C2")):
    ax[1].plot(th, mode_factor(np.zeros_like(c), 1/s, c, "x"), color=cc, label=f"$F_x$, s={s}")
    ax[1].plot(th, mode_factor(np.zeros_like(c), 1/s, c, "o"), color=cc, ls="--", label=f"$F_o$, s={s}")
ax[1].set_xlabel(r"$\theta$ [deg]"); ax[1].set_ylabel(r"$F_\sigma(\theta)$"); ax[1].legend(fontsize=7); ax[1].set_title("polarisation factor at $Y=1/s$, $X=0$", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "tau_theta.pdf"))
