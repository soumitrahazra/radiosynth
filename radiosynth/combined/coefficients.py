# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-voxel gyrosynchrotron coefficients from the fast GS codes.

The fast GS codes (Kuznetsov & Fleishman 2021) return intensities at the end
of a line of sight, not absorption and emission coefficients.  For a single
homogeneous voxel of optical depth tau and source function S (per circular
polarisation) the formal solution is

    I_out = I_in exp(-tau) + S (1 - exp(-tau)) = I_in exp(-tau) + E.

Calling the codes for every voxel as a one-voxel line of sight, once with
I_in = 0 (gives E) and once with a large known I_in (gives exp(-tau)), yields
tau and E for both circular polarisations.  Only the fast electrons are
included in these calls (the codes' free-free is switched off and the thermal
plasma enters through radiosynth's own free-free and gyroresonance
coefficients), so the three mechanisms can be combined in one transfer.

All voxels of a chunk are sent in one GET_MW_SLICE call (one "line of sight"
per voxel; OpenMP inside the codes).  The weak-coupling output rows are used:
for a single voxel there is no mode coupling to treat.
"""
import numpy as np

from ..common.constants import AU, C_LIGHT, K_B

SFU = 1e-22
_SRC = K_B / C_LIGHT**2
S_CM2 = 1e20                       # nominal source area of the one-voxel calls [cm^2]


def gs_voxel_coefficients(lib, P, freqs, E_arr=None, mu_arr=None, f_arr=None, fc=12.0,
                          fwh=12.0, nnodes=0, q_optimization=True, log_key=0):
    """Optical depth and emitted intensity of single voxels.

    lib   : GSLibrary;  P : (24, Nvox) voxel parameters (gyrosynchrotron.voxel_parms,
            with free-free off);  freqs : [Hz]
    E_arr, mu_arr, f_arr : array-defined distribution (f_arr: (NE, Nmu, Nvox)) or None
    Returns dict tau_R, tau_L, E_R, E_L of shape (nf, Nvox); E in W m^-2 Hz^-1 sr^-1.
    """
    nv = P.shape[1]
    nf = freqs.size
    out = {k: np.zeros((nf, nv)) for k in ("tau_R", "tau_L", "E_R", "E_L")}
    if nv == 0:
        return out
    to_int = SFU * AU**2 / (S_CM2 * 1e-4)
    arr = f_arr is not None
    ne_ = E_arr.size if arr else 0
    nmu = mu_arr.size if arr else 0
    Lp = np.array([nv, 1, nf, ne_, nmu, int(nnodes), 0, 0 if q_optimization else 1,
                   2 if arr else 1, log_key, 0, 0], np.int32)
    Rp = np.zeros((5, nv), order="F")
    Rp[0], Rp[3], Rp[4] = S_CM2, fc, fwh
    Pm = np.asfortranarray(P.reshape(24, 1, nv))
    Ea = E_arr if arr else np.zeros(1)
    ma = mu_arr if arr else np.zeros(1)
    fa = np.asfortranarray(f_arr.reshape(ne_, nmu, 1, nv)) if arr else np.zeros((1, 1, 1, 1),
                                                                                  order="F")

    def run(I0):
        RL = np.zeros((7, nf, nv), order="F")
        RL[0] = (freqs / 1e9)[:, None]
        if I0 is not None:
            RL[1:7] = I0[None]
        lib.get_mw_slice(Lp, Rp, Pm, Ea, ma, fa, RL)
        return RL[1].copy(), RL[2].copy()            # L, R (weak coupling rows)

    E_L, E_R = run(None)
    I0 = 1e6 * np.maximum(np.maximum(E_L, E_R), 1e-30) + 1e-20
    O_L, O_R = run(I0)
    for pol, E, O in (("L", E_L, O_L), ("R", E_R, O_R)):
        att = np.clip((O - E) / I0, 1e-300, 1.0)
        out["tau_" + pol] = np.minimum(-np.log(att), 700.0)
        out["E_" + pol] = np.maximum(E, 0.0) * to_int
    return out


def source_weights(coef, freqs):
    """tau x effective temperature [K] of every voxel and polarisation, the
    quantity added to the step sums of the polarised transfer:
        tau T_eff = tau E / ((1 - exp(-tau)) nu^2 k/c^2)   (-> E/(nu^2 k/c^2) as tau -> 0)."""
    f2 = (freqs ** 2)[:, None] * _SRC
    out = {}
    for pol in ("R", "L"):
        t, E = coef["tau_" + pol], coef["E_" + pol]
        with np.errstate(invalid="ignore", divide="ignore"):
            fac = np.where(t > 1e-12, t / -np.expm1(-np.maximum(t, 1e-300)), 1.0)
        out["st_" + pol] = E * fac / f2
    return out
