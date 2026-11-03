# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""radiosynth - Synthetic Radio Imaging of MHD Solar-Stellar Coronal Model Output
and Comparison with Observations.

Package layout
--------------
radiosynth/
    common/          machinery used by every mechanism: grids, readers (BATS-R-US,
                     PLUTO, pyPLUTO), refractive ray tracer, straight-ray sampler,
                     image driver, magneto-ionic theory, figures, solar orientation
                     and comparison with observations, stellar observers, FITS, CLI
    simplistic/      Benkevitch et al. (2010) ad-hoc emissivity
    bremsstrahlung/  thermal free-free emission (Moschou et al. 2018), two-mode
                     magnetised free-free, analytic wind solution
    gyroresonance/   thermal gyroresonance (harmonic layers s nu_B = nu) combined
                     with magnetised free-free, Stokes I and V
    gyrosynchrotron/ fast-electron gyrosynchrotron + free-free, computed with the
                     public fast GS codes (Kuznetsov & Fleishman 2021) as a library
    combined/        free-free + gyroresonance + gyrosynchrotron in one polarised
                     radiative transfer

Works with BATS-R-US (Tecplot / IDL output), PLUTO (.dbl/.flt/.vtk, Cartesian
or spherical) and any model given as numpy arrays, for the Sun and for other
stars.  A Python re-implementation of the BATS-R-US radio-wave imaging tool
(Benkevitch et al. 2010, ApJ 725, 447; Moschou et al. 2018, ApJ 867, 51) that
works on saved MHD output instead of running inside BATS-R-US.

Quick start
-----------
>>> import radiosynth as rs
>>> grid = rs.load_mhd("box_mhd_t001000.dat")              # or rs.MHDGrid(x, y, z, ne, te)
>>> img = rs.make_radio_image(grid, "80MHz, 150MHz, 17GHz",
...                           observer=(-215, 0, 0), x_size=6, y_size=6,
...                           nx=256, ny=256, n_workers=8)
>>> img.plot("tb", path="radio.png"); img.save_npz("radio.npz")

Gyroresonance (needs the magnetic field):
>>> grid = rs.load_mhd("box_mhd_t001000.dat", with_b=True)
>>> gr = rs.gyroresonance.make_image(grid, "5GHz, 17GHz", observer=(-215, 0, 0),
...                                  x_size=1, y_size=1, nx=200, ny=200)
>>> gr.plot("tb"); gr.plot("v")
"""
import sys as _sys

from .common import (constants, emission, grid, raytrace, image, readers, pluto,
                     fitsio, figures, stellar, mechanism, magnetoionic, straight, fields,
                     symmetric)
from .common.constants import critical_density, plasma_frequency, R_SUN, AU_RSUN
from .common.emission import EmissionModel
from .common.grid import MHDGrid, SphericalGrid
from .common.raytrace import trace_rays, STATUS_NAMES
from .common.image import (make_radio_image, RadioImage, trace_paths, image_geometry,
                           parse_frequencies, compare_refraction)
from .common.readers import (load_mhd, build_grid, grid_from_arrays, read_tecplot_ascii,
                             read_pytecplot, read_batsrus_idl, derive_ne_te, points_to_grid)
from .common.pluto import (load_pluto, read_pluto_binary, read_pluto_vtk, read_pluto_pypluto,
                           parse_definitions)
from .common.stellar import (stellar_observer, make_stellar_image, stellar_spectrum,
                             rotational_modulation, spectral_index)
from .common.mechanism import get_mechanism, available_mechanisms
from . import simplistic, bremsstrahlung, gyroresonance, gyrosynchrotron, combined
from .combined import make_combined_image, CombinedImage
from .gyroresonance import make_gr_image, GRImage
from .gyrosynchrotron import make_gs_image, GSImage, FastElectrons
from .common.fields import current_density
from .common.symmetric import (grid_from_profile, read_profile, make_symmetric_image,
                               plot_profile)

# short module paths: "import radiosynth.grid", "from radiosynth.figures import ..."
for _name, _mod in (("constants", constants), ("emission", emission), ("grid", grid),
                    ("raytrace", raytrace), ("image", image), ("readers", readers),
                    ("pluto", pluto), ("fitsio", fitsio), ("figures", figures),
                    ("stellar", stellar)):
    _sys.modules.setdefault(f"{__name__}.{_name}", _mod)
from .common import contrib as _contrib  # noqa: E402
_sys.modules.setdefault(f"{__name__}.contrib", _contrib)
del _name, _mod

__version__ = "2.2.0"
