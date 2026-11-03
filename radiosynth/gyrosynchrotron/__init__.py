# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gyrosynchrotron emission of fast electrons, computed with the public fast
GS codes (Fleishman & Kuznetsov 2010; Kuznetsov & Fleishman 2021 - the engine
of GX Simulator), which are called as a compiled library.

* library.py   : finds, builds (git clone + make) and loads the GS codes
                 (https://github.com/kuznetsov-radio/gyrosynchrotron, GPL-3.0,
                 not bundled with radiosynth)
* electrons.py : FastElectrons - the prescription of the fast electrons
                 (distribution type, n_b rules, pitch-angle distribution,
                 array-defined distributions, energy check)
* image.py     : make_gs_image and GSImage (Stokes I and V)

Uses from common/: grids (with B and scalar fields such as the current
density, common/fields.py), readers, image geometry, straight-ray sampler.

>>> import radiosynth as rs
>>> from radiosynth.gyrosynchrotron import FastElectrons, in_sphere
>>> rs.gyrosynchrotron.build_library()                      # once
>>> e = FastElectrons(nb=in_sphere((-1.05, 0, 0), 0.02, 1e7), delta=3.5)
>>> img = rs.gyrosynchrotron.make_image(grid, "3GHz, 10GHz, 17GHz", e,
...                                     x_size=0.4, y_size=0.4, nx=80, ny=80)
"""
from ..common.mechanism import Mechanism, register
from .electrons import (FastElectrons, uniform, fraction_of_thermal, power_of, in_sphere,
                        in_box, product)
from .library import build_library, find_library, load_library, library_version
from .image import make_gs_image, GSImage

make_image = make_gs_image

MECHANISM = register(Mechanism(
    "gyrosynchrotron", coefficients=None, make_image=make_gs_image,
    description="fast-electron gyrosynchrotron + free-free via the public fast GS codes "
                "(Kuznetsov & Fleishman 2021), straight rays, Stokes I and V (needs B "
                "and a FastElectrons prescription)",
    options=("electrons", "coupling", "free_free", "abundance", "t_body", "ds", "library")))

__all__ = ["FastElectrons", "uniform", "fraction_of_thermal", "power_of", "in_sphere",
           "in_box", "product", "build_library", "find_library", "load_library",
           "library_version", "make_gs_image", "make_image", "GSImage", "MECHANISM"]
