# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Combined image: free-free + gyroresonance + gyrosynchrotron in one transfer.

* coefficients.py : per-voxel gyrosynchrotron optical depth and emission
                    extracted from the fast GS codes (one-voxel calls)
* image.py        : make_combined_image and CombinedImage (Stokes I and V,
                    contribution of each mechanism, dominant-mechanism map)

Uses the polarised straight-ray transfer of gyroresonance/ (with the
magnetised free-free of bremsstrahlung/) and the fast-electron prescription
and library interface of gyrosynchrotron/.

>>> import radiosynth as rs
>>> from radiosynth.gyrosynchrotron import FastElectrons, in_sphere
>>> e = FastElectrons(nb=in_sphere((-1.04, 0, 0), 0.015, 1e7), delta=3.5)
>>> img = rs.combined.make_image(grid, "3GHz, 10GHz, 17GHz", e, x_size=0.4, y_size=0.4)
>>> img.plot("tb"); img.plot("mechanism"); img.plot("pol")
"""
from ..common.mechanism import Mechanism, register
from .coefficients import gs_voxel_coefficients, source_weights
from .image import make_combined_image, CombinedImage, MECH_NAMES

make_image = make_combined_image

MECHANISM = register(Mechanism(
    "combined", coefficients=None, make_image=make_combined_image,
    description="free-free + gyroresonance + gyrosynchrotron in one polarised transfer "
                "(straight rays; gyrosynchrotron needs the fast GS codes and a "
                "FastElectrons prescription)",
    options=("electrons", "include", "coupling", "s_max", "gaunt", "t_body", "ds", "library")))

__all__ = ["gs_voxel_coefficients", "source_weights", "make_combined_image", "make_image",
           "CombinedImage", "MECH_NAMES", "MECHANISM"]
