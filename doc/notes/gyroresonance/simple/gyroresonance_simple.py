# radiosynth - From MHD to MHz: a simple, self-contained gyroresonance calculation
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""
gyroresonance_simple.py  --  thermal gyroresonance + free-free emission of a sunspot,
                             computed step by step, for teaching.

Needs only numpy and matplotlib.  Run:

    python gyroresonance_simple.py      (writes gyroresonance_simple_fig.png and .pdf)

What it does (each STEP below matches a section of gyroresonance_simple.pdf):

  STEP 0  physical constants (cgs units, as in most solar radio papers)
  STEP 1  a simple model of an active region: a box above the photosphere with
          an exponential atmosphere and the field of a buried dipole (two spots)
  STEP 2  one straight, vertical line of sight per pixel, seen from above
  STEP 3  along each line: n_e, T_e, |B|, the angle theta, nu_B, X and Y
  STEP 4  free-free absorption of the two modes (x and o) in every step
  STEP 5  the gyroresonance layers nu = s nu_B crossed in every step, and their
          optical depths
  STEP 6  radiative transfer for right- and left-hand circular polarisation
  STEP 7  brightness temperature, Stokes V, polarisation, flux; figures

This is the same physics as radiosynth.make_gr_image, written as plainly as
possible (and checked against it): a straight, vertical view of a flat box.

TEACHING VERSION.  It is meant for reading and for experiments.  For images of
real MHD models, any observer position, strong mode coupling and the other
options, use radiosynth.make_gr_image.
"""
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# STEP 0.  Constants (cgs)
# =============================================================================
E = 4.80320e-10          # electron charge [esu]
ME = 9.10938e-28         # electron mass [g]
C = 2.99792458e10        # speed of light [cm/s]
KB = 1.380649e-16        # Boltzmann constant [erg/K]
MM = 1.0e8               # 1 Mm in cm

NU_B_PER_GAUSS = E / (2.0 * math.pi * ME * C)     # gyrofrequency per gauss = 2.80e6 Hz/G
GR_PREFACTOR = math.pi * E**2 / (ME * C)          # pi e^2/(m_e c) = 0.02654 cm^2/s

# =============================================================================
# STEP 1.  The model: a box above a sunspot pair
# =============================================================================
#   x, y : horizontal [Mm], z : height above the photosphere [Mm]
N0 = 2.0e9        # electron density at the base [cm^-3]
H_N = 35.0        # density scale height [Mm]
TE = 2.5e6        # coronal temperature [K] (isothermal, to keep it simple)
B_MAX = 2500.0    # field strength at the surface right above the dipole [G]
DEPTH = 35.0      # depth of the buried dipole [Mm]


def density(z):
    """Electron density [cm^-3] at height z [Mm]: n_e = N0 exp(-z/H_N)."""
    return N0 * np.exp(-z / H_N)


def temperature(z):
    """Electron temperature [K]: the same everywhere."""
    return np.full_like(z, TE)


def magnetic_field(x, y, z):
    """Field [G] of a horizontal dipole buried at depth DEPTH, pointing along x.

    A dipole m at the origin gives B = (3 (m.r) r - m r^2) / r^5.  Its two
    poles look like two spots of opposite polarity at x < 0 and x > 0.
    It is scaled so that |B| = B_MAX at the surface right above the dipole."""
    rx, ry, rz = x, y, z + DEPTH                # position relative to the dipole
    r2 = rx**2 + ry**2 + rz**2
    r5 = r2**2.5
    mx = 1.0                                    # dipole moment along x (arbitrary units)
    mdotr = mx * rx
    bx = (3.0 * mdotr * rx - mx * r2) / r5
    by = (3.0 * mdotr * ry) / r5
    bz = (3.0 * mdotr * rz) / r5
    scale = B_MAX * DEPTH**3                    # above the dipole: |B| = 1/DEPTH^3
    return scale * bx, scale * by, scale * bz


# =============================================================================
# STEP 2.  The lines of sight
# =============================================================================
#   We look straight down from above: one vertical line per pixel.  The wave
#   travels upwards, towards us, so the unit vector towards the observer is
#   k = (0, 0, 1).  We sample each line at NZ+1 heights from the top (node 0,
#   nearest to the observer) to the photosphere.
FOV = 140.0        # field of view [Mm] (square)
NPIX = 100         # pixels along each side
Z_TOP = 150.0      # top of the box [Mm]
NZ = 600           # number of steps along each line (step = 0.25 Mm)


def lines_of_sight():
    """Pixel centres X, Y [Mm] and the heights z of the sample points [Mm]."""
    c = -0.5 * FOV + (np.arange(NPIX) + 0.5) * FOV / NPIX
    X, Y = np.meshgrid(c, c)                     # (NPIX, NPIX)
    z = np.linspace(Z_TOP, 0.0, NZ + 1)          # from the top down to the photosphere
    return c, X, Y, z


# =============================================================================
# STEP 3.  The plasma along every line
# =============================================================================
def sample(X, Y, z, freq):
    """n_e, T_e, |B|, cos(theta), nu_B, X = (nu_p/nu)^2, Y = nu_B/nu at every
    sample point; arrays of shape (NPIX, NPIX, NZ+1)."""
    xx = X[:, :, None] + 0.0 * z
    yy = Y[:, :, None] + 0.0 * z
    zz = np.broadcast_to(z, xx.shape)
    ne = density(zz)
    te = temperature(zz)
    bx, by, bz = magnetic_field(xx, yy, zz)
    b = np.sqrt(bx**2 + by**2 + bz**2)
    cos_theta = bz / np.maximum(b, 1e-30)        # angle between B and k = (0, 0, 1)
    nu_b = NU_B_PER_GAUSS * b                    # gyrofrequency [Hz]
    nu_p2 = E**2 * ne / (math.pi * ME)           # plasma frequency squared [Hz^2]
    Xmi = nu_p2 / freq**2                        # X = (nu_p/nu)^2
    Ymi = nu_b / freq                            # Y = nu_B/nu
    return ne, te, b, cos_theta, nu_b, Xmi, Ymi


# =============================================================================
# STEP 4.  Free-free absorption of the two modes
# =============================================================================
def kappa_free_free(ne, te, freq):
    """Thermal free-free absorption coefficient [cm^-1] (Dulk 1985, Eq. 21):
    kappa = 9.78e-3 n_e^2 / (nu^2 T^1.5) * G,  G = 24.5 + ln T - ln nu (T > 2e5 K)."""
    G = np.where(te < 2e5, 18.2 + np.log(te**1.5) - np.log(freq),
                 24.5 + np.log(te) - np.log(freq))
    return 9.78e-3 * ne**2 / (freq**2 * te**1.5) * np.maximum(G, 1.0)


def kappa_two_modes(ne, te, Ymi, cos_theta, freq):
    """In a field the x-mode is absorbed more, the o-mode less:
    kappa_x = kappa/(1 - q)^2, kappa_o = kappa/(1 + q)^2, q = Y |cos(theta)| (<= 0.9)."""
    kappa = kappa_free_free(ne, te, freq)
    q = np.minimum(Ymi * np.abs(cos_theta), 0.9)
    return kappa / (1.0 - q)**2, kappa / (1.0 + q)**2


# =============================================================================
# STEP 5.  Gyroresonance layers and their optical depth
# =============================================================================
def axial_ratio_x(Xmi, Ymi, cos_theta):
    """|T_x|: how circular (1) or linear (0) the x-mode is."""
    c = np.abs(cos_theta)
    s2 = 1.0 - c * c
    a = np.maximum(1.0 - Xmi, 1e-12)
    den = Ymi * s2 + np.sqrt(Ymi**2 * s2**2 + 4.0 * a**2 * c**2)
    return np.where(den > 0, 2.0 * a * c / np.where(den > 0, den, 1.0), 1.0)


def mode_factor(Xmi, s, cos_theta, mode):
    """F_x = (1 + |T_x| |cos|)^2/(1 + T_x^2),  F_o = (|T_x| - |cos|)^2/(1 + T_x^2),
    with T_x evaluated on the layer, where Y = 1/s."""
    c = np.abs(cos_theta)
    tx = axial_ratio_x(Xmi, 1.0 / s, c)
    if mode == "x":
        return (1.0 + tx * c)**2 / (1.0 + tx**2)
    return (tx - c)**2 / (1.0 + tx**2)


def layer_tau(s, ne, te, cos_theta, l_b, freq, Xmi, mode):
    """Optical depth of the s-th harmonic layer (cgs):
    tau = (pi e^2/m_e c) (n_e L_B/nu) (s^2/s!) (s^2 beta^2 sin^2/2)^(s-1) F."""
    beta2 = KB * te / (ME * C**2)
    sin2 = np.maximum(1.0 - cos_theta**2, 0.0)
    F = mode_factor(Xmi, s, cos_theta, mode)
    return (GR_PREFACTOR * ne * l_b / freq * s**2 / math.factorial(s)
            * (0.5 * s**2 * beta2 * sin2)**(s - 1) * F)


def layers_in_steps(ne, te, b, cos_theta, nu_b, Xmi, dl, freq, s_max=6):
    """For every step k (between nodes k and k+1) and every harmonic s, find out
    whether the layer |B| = nu/(s nu_B per gauss) lies inside the step, and add
    its optical depth.  Returns, per step and per mode, the layer optical depth
    and (tau * T) for the source function."""
    shape = dl.shape
    tau_x = np.zeros(shape); tau_o = np.zeros(shape)
    tT_x = np.zeros(shape); tT_o = np.zeros(shape)
    f0, f1 = nu_b[..., :-1], nu_b[..., 1:]                     # nu_B at both ends
    for s in range(1, s_max + 1):
        crosses = (s * f0 - freq) * (s * f1 - freq) < 0        # sign change -> a layer
        if not crosses.any():
            continue
        u = (freq / s - f0) / np.where(crosses, f1 - f0, 1.0)  # where inside the step
        u = np.where(crosses, u, 0.0)
        at = lambda a: a[..., :-1] + u * (a[..., 1:] - a[..., :-1])   # value at the layer
        ne_c, te_c, cos_c, X_c = at(ne), at(te), at(cos_theta), at(Xmi)
        b_s = freq / (s * NU_B_PER_GAUSS)                       # |B| on the layer
        l_b = b_s * dl / np.maximum(np.abs(b[..., 1:] - b[..., :-1]), 1e-30)   # L_B [cm]
        tx = np.where(crosses, layer_tau(s, ne_c, te_c, cos_c, l_b, freq, X_c, "x"), 0.0)
        to = np.where(crosses, layer_tau(s, ne_c, te_c, cos_c, l_b, freq, X_c, "o"), 0.0)
        if s == 1:                                             # the x-mode cannot reach s = 1
            tx = np.where(X_c > 0, 0.0, tx)
        tau_x += tx; tau_o += to
        tT_x += tx * te_c; tT_o += to * te_c
    return tau_x, tau_o, tT_x, tT_o


# =============================================================================
# STEP 6.  Radiative transfer for right- and left-hand circular polarisation
# =============================================================================
def transfer(dtau, tT, cutoff):
    """Walk along the line from the observer inwards and add the light of every
    step, dimmed by everything in front of it:

        T_p += T_step (1 - exp(-dtau)) exp(-tau),    tau += dtau

    T_step = sum(tau_i T_i)/sum(tau_i) is the temperature the step glows at.
    Works in temperature units (T_p = I_p c^2/(k nu^2), one polarisation).
    A polarisation stops at its first cutoff node."""
    npx, npy, nk = dtau.shape
    t_pol = np.zeros((npx, npy))
    tau = np.zeros((npx, npy))
    alive = ~cutoff[..., 0]
    for k in range(nk):                                         # step k: node k -> k+1
        alive &= ~cutoff[..., k + 1]
        d = np.where(alive, dtau[..., k], 0.0)
        t_step = np.where(d > 0, tT[..., k] / np.where(d > 0, d, 1.0), 0.0)
        t_pol += t_step * (-np.expm1(-d)) * np.exp(-tau)
        tau += d
    return t_pol, tau


def gyroresonance_image(freq, s_max=6):
    """Everything together for one frequency [Hz].  Returns a dict of maps."""
    c, X, Y, z = lines_of_sight()
    ne, te, b, cos_theta, nu_b, Xmi, Ymi = sample(X, Y, z, freq)
    dl = np.abs(np.diff(z)) * MM * np.ones_like(ne[..., 1:])   # step lengths [cm]

    # mid-step values for the free-free part
    mid = lambda a: 0.5 * (a[..., 1:] + a[..., :-1])
    kx, ko = kappa_two_modes(mid(ne), mid(te), mid(Ymi), mid(cos_theta), freq)
    dtau_x, dtau_o = kx * dl, ko * dl
    tT_x, tT_o = dtau_x * mid(te), dtau_o * mid(te)

    # the layers
    gx, go, gTx, gTo = layers_in_steps(ne, te, b, cos_theta, nu_b, Xmi, dl, freq, s_max)
    dtau_x, dtau_o = dtau_x + gx, dtau_o + go
    tT_x, tT_o = tT_x + gTx, tT_o + gTo

    # right-hand (R) light is the x-mode where B points towards us (cos > 0), else the o-mode
    pos = mid(cos_theta) > 0
    dtau_R, tT_R = np.where(pos, dtau_x, dtau_o), np.where(pos, tT_x, tT_o)
    dtau_L, tT_L = np.where(pos, dtau_o, dtau_x), np.where(pos, tT_o, tT_x)
    cut_x, cut_o = Xmi >= 1.0 - Ymi, Xmi >= 1.0                # where the modes cannot exist
    cut_R = np.where(cos_theta > 0, cut_x, cut_o)
    cut_L = np.where(cos_theta > 0, cut_o, cut_x)
    t_R, tau_R = transfer(dtau_R, tT_R, cut_R)
    t_L, tau_L = transfer(dtau_L, tT_L, cut_L)

    # =========================================================================
    # STEP 7.  What the telescope sees
    # =========================================================================
    tb = 0.5 * (t_R + t_L)                                     # T_b of Stokes I
    tb_v = 0.5 * (t_R - t_L)                                   # T_b of Stokes V
    pol = np.where(tb > 0, tb_v / np.where(tb > 0, tb, 1.0), 0.0)
    # flux density at the Earth: S = (2 k nu^2/c^2) sum(T_b) dOmega
    d_omega = (FOV / NPIX * MM / 1.496e13)**2                  # pixel solid angle [sr]
    flux = 2.0 * KB * freq**2 / C**2 * tb.sum() * d_omega / 1e-19    # in sfu
    flux_v = 2.0 * KB * freq**2 / C**2 * tb_v.sum() * d_omega / 1e-19
    return dict(x=c, tb=tb, tb_v=tb_v, pol=pol, tb_R=t_R, tb_L=t_L,
                tau_R=tau_R, tau_L=tau_L, flux_sfu=flux, flux_v_sfu=flux_v)


# =============================================================================
if __name__ == "__main__":
    freqs = [5e9, 17e9]
    res = {f: gyroresonance_image(f) for f in freqs}
    for f, r in res.items():
        print(f"{f / 1e9:5.1f} GHz:  max T_b = {r['tb'].max() / 1e6:5.2f} MK,  "
              f"flux = {r['flux_sfu']:7.2f} sfu,  V flux = {r['flux_v_sfu']:+7.2f} sfu")

    ext = [-0.5 * FOV, 0.5 * FOV, -0.5 * FOV, 0.5 * FOV]
    fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.6))
    panels = [(res[5e9]["tb"] / 1e6, "5 GHz: $T_b$ [MK]", "inferno", 0, None),
              (res[5e9]["pol"], "5 GHz: $V/I$", "RdBu_r", -1, 1),
              (res[17e9]["tb"] / 1e6, "17 GHz: $T_b$ [MK]", "inferno", 0, None),
              (res[17e9]["pol"], "17 GHz: $V/I$", "RdBu_r", -1, 1)]
    for a, (data, title, cmap, lo, hi) in zip(ax.ravel(), panels):
        im = a.imshow(data, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi)
        a.set_title(title, fontsize=10)
        a.set_xlabel("x [Mm]")
        a.set_ylabel("y [Mm]")
        fig.colorbar(im, ax=a, shrink=0.85)
    fig.tight_layout()
    fig.savefig("gyroresonance_simple_fig.png", dpi=130)
    fig.savefig("gyroresonance_simple_fig.pdf")
    print("written: gyroresonance_simple_fig.png / .pdf")
