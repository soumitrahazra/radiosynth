# radiosynth - From MHD to MHz: side view of the model of gyroresonance_simple.py
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
# Usage: python make_side_view.py   -> side_view.pdf
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import gyroresonance_simple as gs

x = np.linspace(-70, 70, 561); z = np.linspace(0, 80, 321)
X, Z = np.meshgrid(x, z)
bx, by, bz = gs.magnetic_field(X, 0 * X, Z)
b = np.sqrt(bx**2 + by**2 + bz**2)
fig, ax = plt.subplots(figsize=(8.6, 3.9))
im = ax.pcolormesh(X, Z, np.log10(b), cmap="Greys", shading="auto", vmin=1.5, vmax=3.5, rasterized=True)
fig.colorbar(im, ax=ax, label=r"$\log_{10}|B|$ [G]", shrink=0.9)
# field lines (streamplot of the x-z components)
ax.streamplot(x, z, bx, bz, color="0.55", density=1.1, linewidth=0.5, arrowsize=0.6)
cols = {2: "C0", 3: "C1", 4: "C2"}
for s, c in cols.items():
    bs = 5e9 / (s * gs.NU_B_PER_GAUSS)
    ax.contour(X, Z, b, levels=[bs], colors=c, linewidths=2)
    ax.plot([], [], color=c, lw=2, label=f"layer s = {s}  (|B| = {bs:.0f} G)")
for xl in (-32, 0, 32):
    ax.annotate("", xy=(xl, 78), xytext=(xl, 2), arrowprops=dict(arrowstyle="->", color="C3", lw=1.2))
ax.text(33.5, 70, "to the\nobserver", color="C3", fontsize=8)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9); ax.set_xlim(-70, 70); ax.set_ylim(0, 80); ax.set_xlabel("x [Mm]"); ax.set_ylabel("height z [Mm]")
ax.set_title("Side view at y = 0: field strength, field lines and the 5 GHz layers", fontsize=10)
fig.tight_layout(); fig.savefig("side_view.pdf", dpi=200); fig.savefig("side_view.png", dpi=90)
