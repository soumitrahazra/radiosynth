# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Thermal gyroresonance emission (with magnetised free-free emission).

* opacity.py  : optical depth of the harmonic layers nu = s nu_B for the x- and
                o-modes (Zheleznyakov 1970; White & Kundu 1997)
* transfer.py : polarised transfer along straight rays: layers + two-mode
                free-free, Stokes I and V, weak or strong mode coupling
* image.py    : make_gr_image and the GRImage container (I, V, per-polarisation
                T_b, dominant harmonic, emission radius, GR fraction)

Uses from common/: grids (with the magnetic field), readers, image-plane
geometry, straight-ray sampler, magneto-ionic theory, figures; from
bremsstrahlung/: the magnetised free-free coefficients.

>>> import radiosynth as rs
>>> grid = rs.load_mhd("3d__mhd_ar.dat", with_b=True)
>>> img = rs.gyroresonance.make_image(grid, "5GHz, 17GHz", x_size=1, y_size=1)
>>> img.plot("tb"); img.plot("pol"); img.plot("harmonic")
"""
from ..common.mechanism import Mechanism, register
from .opacity import layer_optical_depth, mode_factor, harmonic_field, GR_PREFACTOR
from .transfer import transfer_chunk
from .image import make_gr_image, GRImage

make_image = make_gr_image

MECHANISM = register(Mechanism(
    "gyroresonance", coefficients=None, make_image=make_gr_image,
    description="thermal gyroresonance layers + magnetised free-free, straight rays, "
                "Stokes I and V (needs B)",
    options=("mechanisms", "s_max", "coupling", "gaunt", "t_body", "ds")))

__all__ = ["layer_optical_depth", "mode_factor", "harmonic_field", "GR_PREFACTOR",
           "transfer_chunk", "make_gr_image", "make_image", "GRImage", "MECHANISM"]
