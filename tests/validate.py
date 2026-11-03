# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validation of radiosynth against analytic / independent solutions.

Run:  python tests/validate.py   (writes figures to tests/output/)
"""
import os
import sys

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
import radiosynth as rs                                         # noqa: E402
from radiosynth.constants import R_SUN, critical_density       # noqa: E402
from synthetic_corona import make_model, newkirk               # noqa: E402

OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
OBS = np.array([-215.0, 0.0, 0.0])
results = []


def report(name, ok, detail):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


grid = make_model(half=3.0, n=241)            # 0.025 R_sun cells
print(grid)

# ---------------------------------------------------------------------------
# 1. Bouguer invariant  mu(r) |r x v| = const  and turning point, 80 MHz
# ---------------------------------------------------------------------------
f = 80e6
ncr = critical_density(f)
mu = lambda r: np.sqrt(np.clip(1 - newkirk(r) / ncr, 0, None))   # noqa: E731
ypix = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
res = rs.trace_paths(grid, f, OBS, -ypix * 0, ypix, x_size=6, y_size=6)
worst, worst_turn = 0.0, 0.0
for k, yp in enumerate(ypix):
    P = res["paths"][k]
    v = res["slopes"][k]
    r = np.linalg.norm(P, axis=1)
    inv = mu(r) * np.linalg.norm(np.cross(P, v), axis=1)
    rel = np.abs(inv[2:-1] - inv[2]) / max(inv[2], 1e-3)
    if inv[2] > 0.05:
        worst = max(worst, rel.max())
        # analytic turning point r0: mu(r0) r0 = invariant
        b = inv[2]
        r0 = brentq(lambda rr: mu(rr) * rr - b, 1.0001, 3.0) if mu(1.0001) * 1.0001 < b else np.nan
        if np.isfinite(r0):
            worst_turn = max(worst_turn, abs(r.min() - r0))
report("Bouguer invariant mu*|r x v| conserved (80 MHz)", worst < 0.02,
       f"max relative deviation {worst:.2e}")
report("Turning point vs analytic mu(r0) r0 = b", worst_turn < 0.03,
       f"max |r_min - r0| = {worst_turn:.3e} R_sun")

# ---------------------------------------------------------------------------
# 2. Independent ODE solution of the ray equation (solve_ivp, analytic grad)
# ---------------------------------------------------------------------------
f2 = 150e6
ncr2 = critical_density(f2)


def eps_grad(p):
    r = np.linalg.norm(p)
    n = newkirk(r)
    dn_dr = n * np.log(10.0) * (-4.32 / r**2)
    return 1 - n / ncr2, -(dn_dr / ncr2) * p / r


def rhs(s, u):
    p, v = u[:3], u[3:]
    e, ge = eps_grad(p)
    gl = ge / (2 * e)                            # grad(mu)/mu
    return np.concatenate([v, gl - v * (gl @ v)])   # = v x (gl x v)


ypix2 = np.array([0.3, 0.9, 1.5, 2.2])
res2 = rs.trace_paths(grid, f2, OBS, ypix2 * 0, ypix2, x_size=6, y_size=6)
dev_ang = []
for k in range(len(ypix2)):
    P = res2["paths"][k]
    p0 = P[0]
    v0 = (np.array([0, 0, ypix2[k]]) - OBS)
    v0 /= np.linalg.norm(v0)
    ev = lambda s, u: np.linalg.norm(u[:3]) - 3.0      # noqa: E731
    ev.terminal, ev.direction = True, 1
    sol = solve_ivp(rhs, (0, 30), np.concatenate([p0, v0]), rtol=1e-10, atol=1e-12,
                    events=ev, max_step=0.01)
    v_ref = sol.y[3:, -1] / np.linalg.norm(sol.y[3:, -1])
    v_rs = res2["slopes"][k][-1]
    dev_ang.append(np.degrees(np.arccos(np.clip(v_ref @ v_rs, -1, 1))))
    defl = np.degrees(np.arccos(np.clip(v0 @ v_ref, -1, 1)))
    print(f"    impact {ypix2[k]:.1f}: deflection (ODE) {defl:7.2f} deg, "
          f"radiosynth - ODE direction difference {dev_ang[-1]:.3f} deg")
report("Exit direction vs independent ODE solution (150 MHz)", max(dev_ang) < 1.0,
       f"max angle difference {max(dev_ang):.3f} deg")

# ---------------------------------------------------------------------------
# 3. Straight-line (no refraction) integral vs quadrature on the analytic model
# ---------------------------------------------------------------------------
em = rs.EmissionModel("bremsstrahlung", gaunt=10)
for f3 in (1e9, 17e9):
    errs = []
    for yp in (0.0, 0.6, 1.3, 2.0):
        d = np.array([0, 0, yp]) - OBS
        d /= np.linalg.norm(d)
        r_int = 5.0
        tin, tout = grid.box_intersection(OBS, d[None])
        od = d @ OBS
        disc = r_int**2 - OBS @ OBS + od**2
        t0 = max(-od - np.sqrt(disc), tin[0])
        t1 = min(-od + np.sqrt(disc), tout[0])
        # stop at the photosphere if the line hits it
        disc1 = 1.0 - OBS @ OBS + od**2
        if disc1 > 0:
            t1 = min(t1, -od - np.sqrt(disc1))

        def integrand(t):
            r = np.linalg.norm(OBS + t * d)
            k_, B_ = em(np.array([newkirk(r)]), np.array([1.5e6]), f3)
            return k_[0] * B_[0]
        ref = quad(integrand, t0, t1, limit=400, points=[-od])[0] * R_SUN
        from radiosynth.image import start_points
        pos, st = start_points(grid, OBS, d[None], r_int)
        out = rs.trace_rays(grid, pos, d[None], f3, em, no_refraction=True,
                            r_integration=r_int, status0=st)
        errs.append(abs(out["intensity"][0] / ref - 1))
    report(f"Line-of-sight integral vs quadrature ({f3 / 1e9:g} GHz)", max(errs) < 0.03,
           f"max relative error {max(errs):.2e}")

# ---------------------------------------------------------------------------
# 4. Full radiative transfer: optically thick disk centre -> T_b ~ T_e
# ---------------------------------------------------------------------------
img = rs.make_radio_image(grid, "100MHz", OBS, 4, 4, 41, 41, transfer="full",
                          n_workers=2, verbose=False)
tb = img.brightness_temperature[0]
tc = tb[20, 20]
report("Full transfer: optically thick T_b <= T_e (100 MHz)",
       tb.max() <= 1.5e6 * 1.0001, f"T_b max {tb.max():.3e} K, centre {tc:.3e} K, "
       f"tau centre {img.tau[0, 20, 20]:.2f}")

# ---------------------------------------------------------------------------
# 5. Convergence with step size (step_factor 1 vs 0.25)
# ---------------------------------------------------------------------------
a = rs.make_radio_image(grid, "150MHz", OBS, 4, 4, 41, 41, n_workers=2, verbose=False)
b = rs.make_radio_image(grid, "150MHz", OBS, 4, 4, 41, 41, n_workers=2, verbose=False,
                        step_factor=0.25)
rel = abs(a.flux_density[0] / b.flux_density[0] - 1)
report("Flux converged w.r.t. step size (150 MHz)", rel < 0.02, f"relative change {rel:.2e}")

# ---------------------------------------------------------------------------
# Figures: ray paths (Benkevitch+2010 Fig. 7 style)
# ---------------------------------------------------------------------------
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
yp = np.linspace(-2.4, 2.4, 25)
for ax, fq in zip(axs, (42.3e6, 80e6, 150e6)):
    rr = rs.trace_paths(grid, fq, OBS, yp * 0, yp, x_size=6, y_size=6)
    for k in range(len(yp)):
        P = rr["paths"][k]
        ax.plot(P[:, 0], P[:, 2], lw=0.8, color="tab:blue")
    t = np.linspace(0, 2 * np.pi, 400)
    ax.fill(np.cos(t), np.sin(t), color="orange")
    rc = brentq(lambda r: newkirk(r) - critical_density(fq), 1.0, 5.0) \
        if newkirk(1.0) > critical_density(fq) else None
    if rc:
        ax.plot(rc * np.cos(t), rc * np.sin(t), "r--", lw=1, label=f"critical surface r={rc:.2f}")
        ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_title(f"{fq / 1e6:.1f} MHz ray paths (observer at x=-215)")
    ax.set_xlabel(r"x [$R_\odot$]")
    ax.set_ylabel(r"z [$R_\odot$]")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ray_paths.png"), dpi=120)
print("\nSummary:", sum(ok for _, ok in results), "/", len(results), "passed")
